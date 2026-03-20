#!/bin/bash
set -Eeuo pipefail

CORE_ROOT="${MINDSPEED_CORE_MS_PATH:-/opt/app-root/share/MindSpeed-Core-MS}"
SET_PATH_SCRIPT="${CORE_ROOT}/tests/scripts/set_path.sh"
CONVERT_ENTRY="${CORE_ROOT}/MindSpeed-LLM/mindspeed_llm/mindspore/convert_ckpt.py"
KEEP_TEMP="${HF2MCORE_KEEP_TEMP:-0}"

source_runtime_env() {
    set +u
    if [ -f /usr/local/Ascend/cann/set_env.sh ]; then
        source /usr/local/Ascend/cann/set_env.sh
    fi
    if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
        source /usr/local/Ascend/nnal/atb/set_env.sh "${ATB_SET_ENV_ARGS:---cxx_abi=0}"
    fi
    if [ -f "${SET_PATH_SCRIPT}" ]; then
        source "${SET_PATH_SCRIPT}"
    fi
    set -u
}

build_clean_pythonpath() {
    local clean=()
    local entry
    IFS=':' read -r -a path_entries <<< "${PYTHONPATH:-}"
    for entry in "${path_entries[@]}"; do
        [ -z "${entry}" ] && continue
        case "${entry}" in
            */MindSpeed-LLM|*/MindSpeed-LLM/*|*/msadapter|*/msadapter/*|*/MSAdapter|*/MSAdapter/*|*/msa_thirdparty|*/msa_thirdparty/*)
                continue
                ;;
        esac
        clean+=("${entry}")
    done
    local joined=""
    for entry in "${clean[@]}"; do
        if [ -n "${joined}" ]; then
            joined="${joined}:"
        fi
        joined="${joined}${entry}"
    done
    printf '%s' "${joined}"
}

prepare_bin_checkpoint() {
    local src_dir="$1"
    local dst_dir="$2"
    local clean_pythonpath
    local src_dir_abs
    local dst_dir_abs

    clean_pythonpath="$(build_clean_pythonpath)"
    src_dir_abs="$(cd "${src_dir}" && pwd -P)"
    mkdir -p "${dst_dir}"
    dst_dir_abs="$(cd "${dst_dir}" && pwd -P)"
    (
        # Keep the prepare step out of MindSpeed-LLM so vendored transformers
        # from the caller cwd cannot shadow the system torch/transformers stack.
        cd "${dst_dir_abs}"
        CLEAN_PYTHONPATH="${clean_pythonpath}" \
        SRC_MODEL_DIR="${src_dir_abs}" \
        DST_MODEL_DIR="${dst_dir_abs}" \
        PYTHONPATH="${clean_pythonpath}" \
        python - <<'PY'
import os
import shutil
from pathlib import Path

from transformers import AutoModelForCausalLM
import torch

src = Path(os.environ["SRC_MODEL_DIR"])
dst = Path(os.environ["DST_MODEL_DIR"])
dst.mkdir(parents=True, exist_ok=True)

for path in src.iterdir():
    if not path.is_file():
        continue
    if path.name.endswith(".safetensors") or path.name.endswith(".safetensors.index.json"):
        continue
    if path.name.startswith("pytorch_model") and (
        path.name.endswith(".bin") or path.name.endswith(".bin.index.json")
    ):
        continue
    shutil.copy2(path, dst / path.name)

model = AutoModelForCausalLM.from_pretrained(
    str(src),
    device_map="cpu",
    trust_remote_code=True,
    local_files_only=True,
    low_cpu_mem_usage=False,
)
torch.save(model.state_dict(), dst / "pytorch_model.bin")
print(f"Prepared torch checkpoint: {dst / 'pytorch_model.bin'}")
PY
    )
}

rewrite_load_dir_args() {
    local new_load_dir="$1"
    shift
    local rewritten=()
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --load-dir)
                if [ "$#" -lt 2 ]; then
                    echo "--load-dir requires a value" >&2
                    exit 1
                fi
                rewritten+=("$1" "${new_load_dir}")
                shift 2
                ;;
            *)
                rewritten+=("$1")
                shift
                ;;
        esac
    done
    printf '%s\0' "${rewritten[@]}"
}

main() {
    source_runtime_env

    if [ ! -f "${CONVERT_ENTRY}" ]; then
        echo "MindSpeed MindSpore convert entry not found: ${CONVERT_ENTRY}" >&2
        exit 1
    fi

    if [ "$#" -eq 0 ]; then
        python "${CONVERT_ENTRY}" --help
        exit 0
    fi

    local original_args=("$@")
    local load_dir=""
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --help|-h)
                python "${CONVERT_ENTRY}" --help
                exit 0
                ;;
            --load-dir)
                shift
                load_dir="${1:-}"
                break
                ;;
        esac
        shift
    done

    if [ -z "${load_dir}" ]; then
        echo "Missing required --load-dir argument" >&2
        exit 1
    fi
    if [ ! -d "${load_dir}" ]; then
        echo "Load directory does not exist: ${load_dir}" >&2
        exit 1
    fi

    local effective_load_dir="${load_dir}"
    local temp_dir=""
    if find "${load_dir}" -maxdepth 1 \( -name '*.safetensors' -o -name '*.safetensors.index.json' \) | grep -q .; then
        temp_dir="$(mktemp -d "${TMPDIR:-/tmp}/hf2mcore.XXXXXX")"
        effective_load_dir="${temp_dir}"
        echo "Preparing PyTorch checkpoint mirror in ${effective_load_dir}"
        prepare_bin_checkpoint "${load_dir}" "${effective_load_dir}"
    fi

    local rewritten_args=()
    while IFS= read -r -d '' arg; do
        rewritten_args+=("${arg}")
    done < <(rewrite_load_dir_args "${effective_load_dir}" "${original_args[@]}")

    local status=0
    (
        cd "${CORE_ROOT}/MindSpeed-LLM"
        python -u - "${CONVERT_ENTRY}" "${rewritten_args[@]}" <<'PY'
import importlib.metadata as importlib_metadata
import json
import runpy
import sys

entry = sys.argv[1]
args = sys.argv[2:]

real_metadata_version = importlib_metadata.version


def patched_metadata_version(pkg_name):
    if pkg_name == "msadapter":
        import torch

        return torch.__version__
    return real_metadata_version(pkg_name)


importlib_metadata.version = patched_metadata_version

import transformers.modeling_utils as modeling_utils
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel, nn


def patched_to_json_string(self, use_diff=True):
    payload = self.to_diff_dict() if use_diff else self.to_dict()
    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"


def patched_tie_or_clone_weights(self, output_embeddings, input_embeddings):
    if self.config.torchscript:
        output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
    else:
        output_embeddings.weight.data.copy_(input_embeddings.weight.data)
    if getattr(output_embeddings, "bias", None) is not None:
        output_embeddings.bias.data = nn.functional.pad(
            output_embeddings.bias.data,
            (0, output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0]),
            "constant",
            0,
        )
    if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
        output_embeddings.out_features = input_embeddings.num_embeddings


PretrainedConfig.to_json_string = patched_to_json_string
modeling_utils.check_torch_load_is_safe = lambda: None
PreTrainedModel._tie_or_clone_weights = patched_tie_or_clone_weights

sys.argv = [entry, *args]
runpy.run_path(entry, run_name="__main__")
PY
    ) || status=$?

    if [ -n "${temp_dir}" ] && [ "${KEEP_TEMP}" != "1" ]; then
        rm -rf "${temp_dir}"
    fi

    return "${status}"
}

main "$@"
