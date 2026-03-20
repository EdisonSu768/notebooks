from __future__ import annotations

import atexit
import builtins
import importlib
import importlib.metadata as importlib_metadata
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

try:
    import importlib_metadata as backport_importlib_metadata
except ImportError:
    backport_importlib_metadata = None

_DISABLED_VALUES = {"0", "false", "no", "off"}
_TEMP_DIRS: list[Path] = []
_REAL_IMPORT = builtins.__import__

_PREPARE_BIN_CODE = r"""
import json
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

trust_remote_code_env = os.environ.get("ODH_HF_TRUST_REMOTE_CODE")
if trust_remote_code_env is None:
    config_path = src / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        trust_remote_code = bool(config.get("auto_map"))
    else:
        trust_remote_code = False
else:
    trust_remote_code = trust_remote_code_env.strip().lower() not in {"0", "false", "no", "off"}

model = AutoModelForCausalLM.from_pretrained(
    str(src),
    device_map="cpu",
    trust_remote_code=trust_remote_code,
    local_files_only=True,
    low_cpu_mem_usage=False,
)
torch.save(model.state_dict(), dst / "pytorch_model.bin")
print(f"Prepared torch checkpoint: {dst / 'pytorch_model.bin'}")
"""


def _env_enabled(name: str, *, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in _DISABLED_VALUES


def build_clean_pythonpath(pythonpath: str | None = None) -> str:
    clean: list[str] = []
    for entry in (pythonpath or os.environ.get("PYTHONPATH") or "").split(":"):
        if not entry:
            continue
        if any(
            marker in entry
            for marker in (
                "/MindSpeed-LLM",
                "/msadapter",
                "/MSAdapter",
                "/msa_thirdparty",
            )
        ):
            continue
        clean.append(entry)
    return ":".join(clean)


def _get_arg_value(argv: Sequence[str], flag: str) -> str | None:
    for index, arg in enumerate(argv):
        if arg != flag:
            continue
        if index + 1 >= len(argv):
            return None
        return argv[index + 1]
    return None


def _rewrite_arg_value(argv: Sequence[str], flag: str, value: str) -> list[str]:
    rewritten = list(argv)
    for index, arg in enumerate(rewritten):
        if arg != flag:
            continue
        if index + 1 < len(rewritten):
            rewritten[index + 1] = value
        return rewritten
    return rewritten


def _should_patch_convert(argv: Sequence[str]) -> bool:
    if not argv:
        return False
    if Path(argv[0]).name != "convert_ckpt.py":
        return False
    if "--help" in argv or "-h" in argv:
        return False
    if _get_arg_value(argv, "--load-model-type") != "hf":
        return False
    if _get_arg_value(argv, "--ai-framework") != "mindspore":
        return False
    return True


def _has_safetensors(load_dir: str | None) -> bool:
    if not load_dir:
        return False
    model_dir = Path(load_dir)
    if not model_dir.is_dir():
        return False
    return any(
        path.is_file()
        for pattern in ("*.safetensors", "*.safetensors.index.json")
        for path in model_dir.glob(pattern)
    )


def _cleanup_temp_dirs() -> None:
    if not _env_enabled("HF2MCORE_KEEP_TEMP", default=False):
        for path in reversed(_TEMP_DIRS):
            shutil.rmtree(path, ignore_errors=True)


def _register_cleanup() -> None:
    if getattr(_register_cleanup, "_odh_msadapter_compat__", False):
        return
    atexit.register(_cleanup_temp_dirs)
    _register_cleanup._odh_msadapter_compat__ = True


def prepare_bin_checkpoint(
    src_dir: str | Path,
    *,
    python_executable: str | None = None,
    pythonpath: str | None = None,
) -> Path:
    src_dir_abs = Path(src_dir).resolve()
    dst_dir = Path(tempfile.mkdtemp(prefix="hf2mcore."))
    clean_pythonpath = build_clean_pythonpath(pythonpath)
    env = os.environ.copy()
    env["SRC_MODEL_DIR"] = str(src_dir_abs)
    env["DST_MODEL_DIR"] = str(dst_dir)
    env["PYTHONNOUSERSITE"] = "1"
    if clean_pythonpath:
        env["PYTHONPATH"] = clean_pythonpath
    else:
        env.pop("PYTHONPATH", None)

    print(f"Preparing PyTorch checkpoint mirror in {dst_dir}", flush=True)
    subprocess.run(
        [python_executable or sys.executable, "-c", _PREPARE_BIN_CODE],
        check=True,
        cwd=str(dst_dir),
        env=env,
        text=True,
    )
    return dst_dir


def maybe_prepare_convert_ckpt(
    argv: Sequence[str],
    *,
    python_executable: str | None = None,
    pythonpath: str | None = None,
) -> list[str]:
    if not _should_patch_convert(argv):
        return list(argv)

    load_dir = _get_arg_value(argv, "--load-dir")
    if not _has_safetensors(load_dir):
        return list(argv)

    temp_dir = prepare_bin_checkpoint(
        load_dir,
        python_executable=python_executable,
        pythonpath=pythonpath,
    )
    _TEMP_DIRS.append(temp_dir)
    _register_cleanup()
    return _rewrite_arg_value(argv, "--load-dir", str(temp_dir))


def patch_msadapter_metadata_version() -> None:
    if getattr(importlib_metadata.version, "__odh_msadapter_compat__", False):
        return

    real_version = importlib_metadata.version

    def patched_version(package_name: str) -> str:
        if package_name == "msadapter":
            return importlib.import_module("torch").__version__
        return real_version(package_name)

    patched_version.__odh_msadapter_compat__ = True
    importlib_metadata.version = patched_version

    if backport_importlib_metadata is None:
        return

    if not getattr(backport_importlib_metadata.version, "__odh_msadapter_compat__", False):
        backport_importlib_metadata.version = patched_version


def patch_mindspore_ones_like() -> bool:
    mindspore_module = sys.modules.get("mindspore")
    if mindspore_module is None:
        return False

    mint_module = getattr(mindspore_module, "mint", None)
    if mint_module is None:
        return False

    real_ones_like = getattr(mint_module, "ones_like", None)
    if real_ones_like is None or getattr(real_ones_like, "__odh_msadapter_compat__", False):
        return True

    if "memory_format" in inspect.signature(real_ones_like).parameters:
        return True

    ops_module = importlib.import_module("mindspore.ops")
    sentinel = object()

    def compat_ones_like(input, *, dtype=None, memory_format=sentinel):
        if memory_format is not sentinel:
            effective_dtype = input.dtype if dtype is None else dtype
            return ops_module.ones_like(input, dtype=effective_dtype)
        return real_ones_like(input, dtype=dtype)

    compat_ones_like.__odh_msadapter_compat__ = True
    mint_module.ones_like = compat_ones_like
    return True


def _install_mindspore_import_hook() -> None:
    if getattr(builtins.__import__, "__odh_msadapter_compat__", False):
        return

    if patch_mindspore_ones_like():
        return

    def hooked_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = _REAL_IMPORT(name, globals, locals, fromlist, level)
        if name == "mindspore" or name.startswith("mindspore."):
            if patch_mindspore_ones_like():
                builtins.__import__ = _REAL_IMPORT
        return module

    hooked_import.__odh_msadapter_compat__ = True
    builtins.__import__ = hooked_import


def patch_transformers_for_convert() -> None:
    modeling_utils = importlib.import_module("transformers.modeling_utils")
    configuration_utils = importlib.import_module("transformers.configuration_utils")
    PretrainedConfig = configuration_utils.PretrainedConfig
    PreTrainedModel = modeling_utils.PreTrainedModel
    nn = modeling_utils.nn

    if not getattr(PretrainedConfig.to_json_string, "__odh_msadapter_compat__", False):
        def patched_to_json_string(self, use_diff: bool = True) -> str:
            payload = self.to_diff_dict() if use_diff else self.to_dict()
            return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"

        patched_to_json_string.__odh_msadapter_compat__ = True
        PretrainedConfig.to_json_string = patched_to_json_string

    if not getattr(modeling_utils.check_torch_load_is_safe, "__odh_msadapter_compat__", False):
        def patched_check_torch_load_is_safe() -> None:
            return None

        patched_check_torch_load_is_safe.__odh_msadapter_compat__ = True
        modeling_utils.check_torch_load_is_safe = patched_check_torch_load_is_safe

    if not getattr(PreTrainedModel._tie_or_clone_weights, "__odh_msadapter_compat__", False):
        def patched_tie_or_clone_weights(self, output_embeddings, input_embeddings) -> None:
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

        patched_tie_or_clone_weights.__odh_msadapter_compat__ = True
        PreTrainedModel._tie_or_clone_weights = patched_tie_or_clone_weights


def bootstrap(argv: Sequence[str] | None = None) -> None:
    if not _env_enabled("ODH_MINDSPORE_MSADAPTER_COMPAT", default=True):
        return

    patch_msadapter_metadata_version()
    _install_mindspore_import_hook()

    active_argv = list(sys.argv if argv is None else argv)
    if not _should_patch_convert(active_argv):
        return

    rewritten_argv = maybe_prepare_convert_ckpt(active_argv)
    patch_transformers_for_convert()

    if argv is None:
        sys.argv[:] = rewritten_argv
