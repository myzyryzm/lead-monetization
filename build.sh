#!/usr/bin/env bash
# Build the lead-monetization image and push it to ECR.
#
# Usage:
#   ./build.sh                # build + push :latest
#   ./build.sh v2             # build + push :v2
#   ./build.sh --build-only   # build only (no ECR login/tag/push)
#   ./build.sh v2 --build-only
set -euo pipefail

AWS_REGION="us-west-1"
AWS_ACCOUNT_ID="282873277208"
IMAGE_NAME="lead-monetization"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_URI="${ECR_REGISTRY}/${IMAGE_NAME}"

TAG="latest"
PUSH=true
for arg in "$@"; do
  case "$arg" in
    --build-only) PUSH=false ;;
    *) TAG="$arg" ;;
  esac
done

cd "$(dirname "$0")"

# linux/amd64 explicitly: Fargate runs x86_64 by default, and a native build
# on Apple Silicon would produce an arm64 image that fails on ECS.
echo "==> Building ${IMAGE_NAME}:${TAG} (linux/amd64)"
docker buildx build --platform linux/amd64 -t "${IMAGE_NAME}:${TAG}" --load .

if [ "$PUSH" = false ]; then
  echo "==> Build-only mode: skipping ECR login/tag/push"
  exit 0
fi

echo "==> Logging in to ${ECR_REGISTRY}"
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ECR_REGISTRY"

echo "==> Tagging ${IMAGE_NAME}:${TAG} -> ${ECR_URI}:${TAG}"
docker tag "${IMAGE_NAME}:${TAG}" "${ECR_URI}:${TAG}"

echo "==> Pushing ${ECR_URI}:${TAG}"
docker push "${ECR_URI}:${TAG}"

echo "==> Done: ${ECR_URI}:${TAG}"
