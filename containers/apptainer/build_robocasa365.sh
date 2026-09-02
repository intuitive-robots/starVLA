#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
DEFINITION="${SCRIPT_DIR}/robocasa365_arm64.def"
OUTPUT_DIR="${REPO_ROOT}/playground/sims/sif"
OUTPUT_IMAGE="${OUTPUT_DIR}/robocasa365-main-arm64.sif"
BUILD_PARENT="${APPTAINER_BUILD_PARENT:-${TMPDIR:-/tmp}}"
ASSET_CACHE=/tmp/starvla-robocasa365-assets-main

declare -A ASSET_URLS=(
    [textures.zip]='https://utexas.box.com/shared/static/4i85ileasdvstmlln5sbvzptz7keuoy1.zip'
    [generative_textures.zip]='https://utexas.box.com/shared/static/ebaad09k82tmfmlq6ohdkmrh8izl9vn5.zip'
    [fixtures.zip]='https://utexas.box.com/shared/static/idbncsadpnaz1jfl4i6m8qejawk7p9pi.zip'
    [objaverse.zip]='https://utexas.box.com/shared/static/03eionyo8fk3a9dsksq9jb8du5lqfw8h.zip'
    [aigen_objs.zip]='https://utexas.box.com/shared/static/nwi1vrn5pgbo95kushkasa3nx1i012ff.zip'
    [lightwheel.zip]='https://utexas.box.com/shared/static/vckqvvkh1z8t69k8qcpcmee6k66stii4.zip'
)

if [[ $(uname -m) != "aarch64" ]]; then
    echo "ERROR: this recipe must be built natively on aarch64." >&2
    exit 1
fi

if [[ -e "${OUTPUT_IMAGE}" ]]; then
    echo "ERROR: output already exists: ${OUTPUT_IMAGE}" >&2
    echo "Remove or rename it explicitly before rebuilding from current upstream main." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${ASSET_CACHE}"

for asset_name in \
    textures.zip \
    generative_textures.zip \
    fixtures.zip \
    objaverse.zip \
    aigen_objs.zip \
    lightwheel.zip; do
    asset_path="${ASSET_CACHE}/${asset_name}"
    if ! unzip -tq "${asset_path}" >/dev/null 2>&1; then
        echo "Staging resumable RoboCasa asset archive: ${asset_name}"
        wget --continue --progress=dot:giga "${ASSET_URLS[${asset_name}]}" -O "${asset_path}"
        unzip -tq "${asset_path}" >/dev/null
    fi
done

(
    cd "${ASSET_CACHE}"
    sha256sum \
        textures.zip \
        generative_textures.zip \
        fixtures.zip \
        objaverse.zip \
        aigen_objs.zip \
        lightwheel.zip \
        > SHA256SUMS
)

BUILD_ROOT=$(mktemp -d "${BUILD_PARENT%/}/starvla-robocasa365-apptainer.XXXXXX")

cleanup() {
    build_status=$?
    if [[ -n ${BUILD_ROOT:-} && -d ${BUILD_ROOT} && ${BUILD_ROOT} == "${BUILD_PARENT%/}"/starvla-robocasa365-apptainer.* ]]; then
        if [[ ${KEEP_APPTAINER_BUILD:-0} == 1 ]]; then
            echo "Keeping Apptainer build workspace: ${BUILD_ROOT}"
        else
            rm -rf -- "${BUILD_ROOT}"
        fi
    fi
    if [[ ${build_status} == 0 && ${KEEP_ASSET_CACHE:-0} != 1 ]]; then
        rm -rf -- "${ASSET_CACHE}"
    elif [[ -d ${ASSET_CACHE} ]]; then
        echo "Keeping resumable asset archives after unsuccessful build: ${ASSET_CACHE}"
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
echo "Source     : current robocasa/main and robosuite/master at build time"

apptainer build --fakeroot "${OUTPUT_IMAGE}" "${DEFINITION}"
apptainer inspect "${OUTPUT_IMAGE}"
apptainer exec "${OUTPUT_IMAGE}" sh -c \
    'printf "RoboCasa %s (%s)\nRoboSuite %s (%s)\n" \
        "$(cat /opt/starvla-build-info/robocasa-version)" \
        "$(cat /opt/starvla-build-info/robocasa-revision)" \
        "$(cat /opt/starvla-build-info/robosuite-version)" \
        "$(cat /opt/starvla-build-info/robosuite-revision)"'

echo "Built ${OUTPUT_IMAGE}"
