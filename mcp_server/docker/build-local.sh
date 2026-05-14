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
TAG="${TAG:-feat-bearer-auth}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
PUSH="${PUSH:-1}"
BUILDER="${BUILDER:-multiplatform}"
OCI_TARBALL="${OCI_TARBALL:-/tmp/graphiti-mcp-build.tar}"

# cd to repo root so the build context is correct
cd "$(dirname "$0")/../.."

VCS_REF="$(git rev-parse --short HEAD)"
BUILD_DATE="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

echo "==> Building ${IMAGE}:${TAG}"
echo "    platforms: ${PLATFORMS}"
echo "    VCS ref:   ${VCS_REF}"
echo "    push:      ${PUSH}"
echo

if [ "${PUSH}" = "1" ]; then
  if ! command -v oras >/dev/null 2>&1; then
    echo "ERROR: 'oras' is required for the push path. Install with: brew install oras" >&2
    exit 1
  fi
  echo "==> Building OCI tarball at ${OCI_TARBALL}"
  docker buildx build \
    --builder "${BUILDER}" \
    --file mcp_server/docker/Dockerfile.local \
    --platform "${PLATFORMS}" \
    --build-arg "BUILD_DATE=${BUILD_DATE}" \
    --build-arg "VCS_REF=${VCS_REF}" \
    --tag "${IMAGE}:${TAG}" \
    --output "type=oci,dest=${OCI_TARBALL}" \
    .

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
