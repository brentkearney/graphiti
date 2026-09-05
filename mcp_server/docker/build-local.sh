#!/usr/bin/env bash
# Build and push multi-arch MCP server image from the local graphiti-core sources.
#
# Produces ghcr.io/brentkearney/graphiti-mcp:feat-bearer-auth (and a SHA tag).
#
# Two-step flow:
#   1. buildx builds the multi-arch image into a local OCI tarball
#   2. oras copies the tarball to GHCR from the host (bypassing Docker
#      Desktop's VM, which has been known to wedge on large outbound pushes)
#
# Prerequisites:
#   - `docker login ghcr.io -u <username>` with a PAT that has write:packages
#     (oras reads credentials from ~/.docker/config.json automatically)
#   - A buildx builder that supports multi-platform (the 'multiplatform' builder
#     in ~/.docker is typically preconfigured)
#   - `oras` on PATH: brew install oras
#
# Usage:
#   mcp_server/docker/build-local.sh            # build + push via oras
#   PUSH=0 mcp_server/docker/build-local.sh     # build only (single-arch, --load into docker images)

set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/brentkearney/graphiti-mcp}"
# Release version lives in mcp_server/docker/VERSION. Bump it for every published build:
# the tag is immutable, so re-running with an unchanged VERSION is refused below.
# "bk-" distinguishes this image from atvenu/graphiti-custom, which has its own version line.
BK_VERSION="${BK_VERSION:-$(cat "$(dirname "$0")/VERSION")}"
TAG="${TAG:-bk-v${BK_VERSION}}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
PUSH="${PUSH:-1}"
BUILDER="${BUILDER:-multiplatform}"
OCI_TARBALL="${OCI_TARBALL:-/tmp/graphiti-mcp-build.tar}"

# cd to repo root so the build context is correct
cd "$(dirname "$0")/../.."

VCS_REF="$(git rev-parse --short HEAD)"
BUILD_DATE="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
# The upstream mcp_server version comes from its pyproject, never a hardcoded default:
# a stale default silently mislabels provenance, which is the one thing this image's
# labels exist to get right.
MCP_SERVER_VERSION="$(grep -m1 '^version' mcp_server/pyproject.toml | sed 's/.*"\(.*\)".*/\1/')"

# Assert the built image's provenance labels match the sources they are derived from,
# BEFORE the tag is spent. Tags are immutable, so anything discovered after the push
# costs a version number to correct — bk-v1.1.0 was burned exactly that way, published
# with a mcp-server-version label frozen at a hardcoded Dockerfile default. The OCI
# tarball exists locally before `oras copy`, so this is checkable while the tag is free.
verify_labels() {
  python3 - "$OCI_TARBALL" "$BK_VERSION" "$MCP_SERVER_VERSION" "$VCS_REF" <<'PY'
import json, sys, tarfile

tarball, bk_version, mcp_version, vcs_ref = sys.argv[1:5]

expected = {
    'org.opencontainers.image.version': bk_version,
    'ca.netmojo.graphiti.mcp-server-version': mcp_version,
    'org.opencontainers.image.revision': vcs_ref,
}

problems = []
checked = 0

with tarfile.open(tarball) as tf:
    def blob(digest):
        path = 'blobs/' + digest.replace(':', '/')
        for name in (path, './' + path):
            try:
                return json.load(tf.extractfile(name))
            except KeyError:
                continue
        raise KeyError(f'blob {digest} missing from {tarball}')

    # index.json usually points at an image index, which in turn lists the platforms.
    manifests = []
    for entry in json.load(tf.extractfile('index.json'))['manifests']:
        doc = blob(entry['digest'])
        if 'manifests' in doc:
            manifests.extend(doc['manifests'])
        else:
            manifests.append(entry)

    for m in manifests:
        platform = m.get('platform', {})
        arch = platform.get('architecture', '?')
        # Attestation manifests carry no runtime config and no labels.
        if 'unknown' in (platform.get('os'), arch):
            continue
        labels = (blob(blob(m['digest'])['config']['digest']).get('config') or {}).get('Labels') or {}
        checked += 1
        for key, want in expected.items():
            got = labels.get(key)
            if got != want:
                problems.append(f'  {arch}: {key} = {got!r}, expected {want!r}')

if not checked:
    problems.append('  no platform manifests found — nothing was verified')

if problems:
    sys.stderr.write('ERROR: image labels disagree with the sources they are derived from.\n')
    sys.stderr.write('\n'.join(problems) + '\n')
    sys.stderr.write('       Refusing to push: a wrong label would cost a version to correct.\n')
    sys.exit(1)

print(f'    labels verified on {checked} platform manifest(s)')
PY
}

echo "==> Building ${IMAGE}:${TAG}"
echo "    platforms: ${PLATFORMS}"
echo "    VCS ref:   ${VCS_REF}"
echo "    mcp server: ${MCP_SERVER_VERSION}"
echo "    push:      ${PUSH}"
echo

if [ "${PUSH}" = "1" ]; then
  if ! command -v oras >/dev/null 2>&1; then
    echo "ERROR: 'oras' is required for the push path. Install with: brew install oras" >&2
    exit 1
  fi
  # Immutable-tag guard. Publishing two different images under one tag is how a
  # graphiti-core downgrade once shipped unnoticed: the floating tag moved while the
  # host stayed pinned to an older digest, so both answered to the same name.
  if EXISTING=$(oras manifest fetch --descriptor "${IMAGE}:${TAG}" 2>/dev/null); then
    echo "ERROR: ${IMAGE}:${TAG} already exists on the registry." >&2
    echo "       ${EXISTING}" >&2
    echo "       Bump mcp_server/docker/VERSION (currently ${BK_VERSION}) and re-run." >&2
    echo "       To retag an existing digest instead: oras tag ${IMAGE}:<existing> <new-tag>" >&2
    exit 1
  fi

  echo "==> Building OCI tarball at ${OCI_TARBALL}"
  docker buildx build \
    --builder "${BUILDER}" \
    --file mcp_server/docker/Dockerfile.local \
    --platform "${PLATFORMS}" \
    --build-arg "BUILD_DATE=${BUILD_DATE}" \
    --build-arg "VCS_REF=${VCS_REF}" \
    --build-arg "BK_VERSION=${BK_VERSION}" \
    --build-arg "MCP_SERVER_VERSION=${MCP_SERVER_VERSION}" \
    --tag "${IMAGE}:${TAG}" \
    --output "type=oci,dest=${OCI_TARBALL}" \
    .

  echo "==> Verifying provenance labels before spending the tag"
  verify_labels

  echo
  echo "==> Pushing ${IMAGE}:${TAG} via oras"
  oras copy --from-oci-layout "${OCI_TARBALL}:${TAG}" "${IMAGE}:${TAG}"

  echo "==> Tagging ${IMAGE}:${TAG}-${VCS_REF} on registry"
  oras tag "${IMAGE}:${TAG}" "${TAG}-${VCS_REF}"
else
  HOST_PLATFORM="linux/$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')"
  echo "==> PUSH=0, building single-arch (${HOST_PLATFORM}) with --load"
  docker buildx build \
    --builder "${BUILDER}" \
    --file mcp_server/docker/Dockerfile.local \
    --platform "${HOST_PLATFORM}" \
    --build-arg "BUILD_DATE=${BUILD_DATE}" \
    --build-arg "VCS_REF=${VCS_REF}" \
    --build-arg "BK_VERSION=${BK_VERSION}" \
    --build-arg "MCP_SERVER_VERSION=${MCP_SERVER_VERSION}" \
    --tag "${IMAGE}:${TAG}" \
    --tag "${IMAGE}:${TAG}-${VCS_REF}" \
    --load \
    .
fi

echo
echo "==> Done."
if [ "${PUSH}" = "1" ]; then
  echo "    Pushed ${IMAGE}:${TAG}"
  echo "    Pushed ${IMAGE}:${TAG}-${VCS_REF}"
  echo
  echo "To get the manifest digest:"
  echo "  docker buildx imagetools inspect ${IMAGE}:${TAG}"
fi
