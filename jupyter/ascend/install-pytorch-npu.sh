#!/bin/bash
set -Eeuxo pipefail

TUNA_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple
TSINGHUA_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
GITHUB_HTTPS_PROXY=http://192.168.144.12:7890
ATB_SET_ENV_ARGS="${ATB_SET_ENV_ARGS:---cxx_abi=0}"
TORCH_VERSION="${TORCH_VERSION:-2.9.0}"
TORCH_NPU_RELEASE="${TORCH_NPU_RELEASE:-v7.3.0-pytorch2.9.0}"
MSMODELSLIM_TRANSFORMERS_VERSION="${MSMODELSLIM_TRANSFORMERS_VERSION:-5.2.0}"
MSMODELSLIM_TORCHVISION_VERSION="${MSMODELSLIM_TORCHVISION_VERSION:-0.24.0}"
MSMODELSLIM_MISTRAL_COMMON_VERSION="${MSMODELSLIM_MISTRAL_COMMON_VERSION:-1.11.0}"
MSMODELSLIM_HUGGINGFACE_HUB_VERSION="${MSMODELSLIM_HUGGINGFACE_HUB_VERSION:-1.10.2}"
MSMODELSLIM_EASYDICT_VERSION="${MSMODELSLIM_EASYDICT_VERSION:-1.13}"
MSMODELSLIM_WCMATCH_VERSION="${MSMODELSLIM_WCMATCH_VERSION:-10.1}"
MSMODELSLIM_BRACEX_VERSION="${MSMODELSLIM_BRACEX_VERSION:-2.6}"

detect_python_tag() {
    python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")'
}

detect_python_version() {
    python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
}

source_ascend_env() {
    set +u
    source /usr/local/Ascend/cann/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh "${ATB_SET_ENV_ARGS}"
    set -u
}

cd /tmp

PYTHON_VERSION=$(detect_python_version)
PYTHON_TAG=$(detect_python_tag)

case "${PYTHON_VERSION}" in
    3.11 | 3.12)
        ;;
    *)
        echo "Unsupported Python version for Ascend PyTorch install: ${PYTHON_VERSION}" >&2
        exit 1
        ;;
esac

TORCH_WHEEL="torch-${TORCH_VERSION}%2Bcpu-${PYTHON_TAG}-${PYTHON_TAG}-manylinux_2_28_aarch64.whl"
TORCH_NPU_WHEEL="torch_npu-${TORCH_VERSION}-${PYTHON_TAG}-${PYTHON_TAG}-manylinux_2_28_aarch64.whl"

wget -q -O "${TORCH_WHEEL}" "https://download.pytorch.org/whl/cpu/${TORCH_WHEEL}"
pip3 install --no-cache-dir "${TORCH_WHEEL}"
rm -f "${TORCH_WHEEL}"

wget -q -O "${TORCH_NPU_WHEEL}" "https://gitcode.com/Ascend/pytorch/releases/download/${TORCH_NPU_RELEASE}/${TORCH_NPU_WHEEL}"
pip3 install --no-cache-dir "${TORCH_NPU_WHEEL}"
rm -f "${TORCH_NPU_WHEEL}"

ls -la /usr/local/Ascend/

source_ascend_env

git clone --depth 1 https://gitcode.com/ascend/MindSpeed.git
cd MindSpeed
pip3 install --no-cache-dir -r requirements.txt -i "${TUNA_INDEX_URL}"
# Avoid an editable install because the source tree is removed later in the build.
pip3 install --no-cache-dir . --no-deps -i "${TUNA_INDEX_URL}"
cd ..

git clone --depth 1 https://gitcode.com/ascend/MindSpeed-LLM.git
export HTTPS_PROXY="${GITHUB_HTTPS_PROXY}"
export https_proxy="${GITHUB_HTTPS_PROXY}"
git clone --depth 1 --branch core_v0.12.1 https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip3 install --no-cache-dir . --no-deps
cp -r megatron ../MindSpeed-LLM/
# pip only packages megatron.core, copy full megatron/ to site-packages for megatron.training etc.
SITE_PACKAGES=$(python3 -c 'import site; print(site.getsitepackages()[0])')
mkdir -p "${SITE_PACKAGES}/megatron"
cp -rn megatron/* "${SITE_PACKAGES}/megatron/"
cd ../MindSpeed-LLM
git checkout master

# Install ray from Tsinghua mirror (Huawei mirror doesn't have ray 2.10.0)
pip3 install --no-cache-dir "ray==2.54.0" -i "${TSINGHUA_INDEX_URL}"

grep -vE "^(ray==|triton-ascend)" requirements.txt > requirements_filtered.txt
pip3 install --no-cache-dir -r requirements_filtered.txt -i "${TUNA_INDEX_URL}"
# Fix: add missing __init__.py so setuptools.find_packages() includes all sub-packages
find mindspeed_llm -mindepth 1 -type d -exec sh -c 'test ! -f "$1/__init__.py" && touch "$1/__init__.py"' _ {} \;
# Avoid an editable install because the source tree is removed later in the build.
pip3 install --no-cache-dir . --no-deps -i "${TUNA_INDEX_URL}"
cd ..

# Restore click to a version compatible with odh-elyra
# MindSpeed's typer dependency pulls in click 8.3.1, but odh-elyra requires click==8.1.8
pip3 install --no-cache-dir "click==8.1.8" -i "${TUNA_INDEX_URL}"

# Pin pyarrow and datasets: pyarrow>=15 on aarch64 requires libprotobuf.so.25
# which is absent in the CANN base image. Other packages (sklearn, transformers,
# feast) may silently upgrade pyarrow, so we force it back here at the very end.
pip3 install --no-cache-dir "pyarrow==14.0.1" "datasets<2.20" -i "${TSINGHUA_INDEX_URL}"

if [ "${INSTALL_MSMODELSLIM:-false}" = "true" ]; then
    # msmodelslim wheels have shipped a bogus data-scheme requirements.txt entry.
    # Depending on the release, it may be a file or a directory tree. pip can
    # fail with EISDIR when trying to install it into /opt/app-root/requirements.txt,
    # so strip that path from the wheel before installing it.
    MSMODELSLIM_VERSION="${MSMODELSLIM_VERSION:-26.0.0a2}"
    MSMODELSLIM_WHEEL_URL="${MSMODELSLIM_WHEEL_URL:-}"
    rm -rf /tmp/ms /tmp/ms-extract
    mkdir -p /tmp/ms /tmp/ms-extract
    if [ -n "${MSMODELSLIM_WHEEL_URL}" ]; then
        MS_WHEEL="/tmp/ms/$(basename "${MSMODELSLIM_WHEEL_URL}")"
        curl -LfsS "${MSMODELSLIM_WHEEL_URL}" -o "${MS_WHEEL}"
    else
        pip3 download --no-cache-dir --no-deps -d /tmp/ms \
            "msmodelslim==${MSMODELSLIM_VERSION}" -i "${TUNA_INDEX_URL}"
    fi
    MS_WHEEL=$(ls /tmp/ms/msmodelslim-*-py3-none-any.whl | head -n1)
    MS_WHEEL_NAME=$(basename "${MS_WHEEL}")
    ( cd /tmp/ms-extract && unzip -q "${MS_WHEEL}" )
    DATA_REQUIREMENTS_PATH=$(find /tmp/ms-extract -path "*.data/data/requirements.txt" \( -type f -o -type d \) -print -quit)
    if [ -n "${DATA_REQUIREMENTS_PATH}" ]; then
        rm -rf "${DATA_REQUIREMENTS_PATH}"
    fi
    MS_RECORD=$(find /tmp/ms-extract -path "*.dist-info/RECORD" -type f -print -quit)
    if [ -f "${MS_RECORD}" ]; then
        sed -i '/\.data\/data\/requirements\.txt/d' "${MS_RECORD}"
    fi
    rm -f "/tmp/ms/${MS_WHEEL_NAME}"
    ( cd /tmp/ms-extract && zip -qr "/tmp/ms/${MS_WHEEL_NAME}" . )
    pip3 install --no-cache-dir --no-deps "/tmp/ms/${MS_WHEEL_NAME}"
    rm -rf /tmp/ms /tmp/ms-extract
    # Keep the final runtime stack compatible with the Qwen3.5 multimodal one-click quant path.
    # We strip the broken bundled requirements payload from the wheel above, so install the
    # known-good runtime set explicitly here. Some transitive installs in the Ascend stack
    # can also silently replace huggingface-hub with an older release that breaks
    # `from huggingface_hub import is_offline_mode` during `import transformers`.
    pip3 install --no-cache-dir --no-deps \
        "bracex==${MSMODELSLIM_BRACEX_VERSION}" \
        "easydict==${MSMODELSLIM_EASYDICT_VERSION}" \
        "huggingface-hub==${MSMODELSLIM_HUGGINGFACE_HUB_VERSION}" \
        "mistral-common==${MSMODELSLIM_MISTRAL_COMMON_VERSION}" \
        "transformers==${MSMODELSLIM_TRANSFORMERS_VERSION}" \
        "torchvision==${MSMODELSLIM_TORCHVISION_VERSION}" \
        "wcmatch==${MSMODELSLIM_WCMATCH_VERSION}" \
        -i "${TSINGHUA_INDEX_URL}"
fi

rm -rf /tmp/MindSpeed /tmp/MindSpeed-LLM /tmp/Megatron-LM
