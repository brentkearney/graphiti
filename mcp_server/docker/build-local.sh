#!/usr/bin/env bash
# Build and push multi-arch MCP server image from the local graphiti-core sources.
#
# Produces ghcr.io/brentkearney/graphiti-mcp:feat-bearer-auth (and a SHA tag).
#
# Prerequisites:
#   - `docker login ghcr.io -u <username>` with a PAT that has write:packages
#   - A buildx builder that supports multi-platform (the 'multiplatform' builder
#     in ~/.docker is typically preconfigured)
#
# Usage:
#   mcp_server/docker/build-local.sh            # build + push
#   PUSH=0 mcp_server/docker/build-local.sh     # build only, no push

set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/brentkearney/graphiti-mcp}"
TAG="${TAG:-feat-bearer-auth}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
PUSH="${PUSH:-1}"
BUILDER="${BUILDER:-multiplatform}"

# cd to repo root so the build context is correct
cd "$(dirname "$0")/../.."

VCS_REF="$(git rev-parse --short HEAD)"
BUILD_DATE="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

echo "==> Building ${IMAGE}:${TAG}"
echo "    platforms: ${PLATFORMS}"
echo "    VCS ref:   ${VCS_REF}"
echo "    push:      ${PUSH}"
echo

PUSH_FLAG=""
if [ "${PUSH}" = "1" ]; then
  PUSH_FLAG="--push"
else
  PUSH_FLAG="--load"
  PLATFORMS="linux/$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')"
  echo "==> PUSH=0, building single-arch (${PLATFORMS}) with --load"
fi

docker buildx build \
  --builder "${BUILDER}" \
  --file mcp_server/docker/Dockerfile.local \
  --platform "${PLATFORMS}" \
  --build-arg "BUILD_DATE=${BUILD_DATE}" \
  --build-arg "VCS_REF=${VCS_REF}" \
  --tag "${IMAGE}:${TAG}" \
  --tag "${IMAGE}:${TAG}-${VCS_REF}" \
  ${PUSH_FLAG} \
  .

echo
echo "==> Done."
if [ "${PUSH}" = "1" ]; then
  echo "    Pushed ${IMAGE}:${TAG}"
  echo "    Pushed ${IMAGE}:${TAG}-${VCS_REF}"
  echo
  echo "To get the manifest digest:"
  echo "  docker buildx imagetools inspect ${IMAGE}:${TAG}"
fi
