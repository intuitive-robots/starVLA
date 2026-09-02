#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
DEFINITION="${SCRIPT_DIR}/libero_plus_arm64.def"
OUTPUT_DIR="${REPO_ROOT}/playground/sims/sif"
OUTPUT_IMAGE="${OUTPUT_DIR}/libero-plus-v0.5.0-arm64.sif"
BUILD_PARENT="${APPTAINER_BUILD_PARENT:-${TMPDIR:-/tmp}}"
ASSET_CACHE=/tmp/starvla-libero-plus-assets-dd2bd61b7d9a6fef1abc52d606e983b41886a149.zip
ASSET_SHA256=96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf
ASSET_URL='https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/dd2bd61b7d9a6fef1abc52d606e983b41886a149/assets.zip?download=true'

if [[ $(uname -m) != "aarch64" ]]; then
    echo "ERROR: this recipe must be built natively on aarch64." >&2
    exit 1
fi

if [[ -e "${OUTPUT_IMAGE}" ]]; then
    echo "ERROR: output already exists: ${OUTPUT_IMAGE}" >&2
    echo "Remove or rename it explicitly before rebuilding." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

if ! printf '%s  %s\n' "${ASSET_SHA256}" "${ASSET_CACHE}" | sha256sum --check --status 2>/dev/null; then
    echo "Staging resumable LIBERO-Plus asset archive: ${ASSET_CACHE}"
    wget --continue --progress=dot:giga "${ASSET_URL}" -O "${ASSET_CACHE}"
    printf '%s  %s\n' "${ASSET_SHA256}" "${ASSET_CACHE}" | sha256sum --check
fi

BUILD_ROOT=$(mktemp -d "${BUILD_PARENT%/}/starvla-libero-plus-apptainer.XXXXXX")

cleanup() {
    build_status=$?
    if [[ -n ${BUILD_ROOT:-} && -d ${BUILD_ROOT} && ${BUILD_ROOT} == "${BUILD_PARENT%/}"/starvla-libero-plus-apptainer.* ]]; then
        if [[ ${KEEP_APPTAINER_BUILD:-0} == 1 ]]; then
            echo "Keeping Apptainer build workspace: ${BUILD_ROOT}"
        else
            rm -rf -- "${BUILD_ROOT}"
        fi
    fi
    if [[ ${build_status} == 0 && ${KEEP_ASSET_CACHE:-0} != 1 ]]; then
        rm -f -- "${ASSET_CACHE}"
    elif [[ -f ${ASSET_CACHE} ]]; then
        echo "Keeping resumable asset archive after unsuccessful build: ${ASSET_CACHE}"
    fi
    return "${build_status}"
}
trap cleanup EXIT

export APPTAINER_TMPDIR="${BUILD_ROOT}/tmp"
export APPTAINER_CACHEDIR="${BUILD_ROOT}/cache"
mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"

echo "Definition : ${DEFINITION}"
echo "Output     : ${OUTPUT_IMAGE}"
echo "Build temp : ${BUILD_ROOT}"

apptainer build --fakeroot "${OUTPUT_IMAGE}" "${DEFINITION}"
apptainer inspect "${OUTPUT_IMAGE}"

echo "Built ${OUTPUT_IMAGE}"
