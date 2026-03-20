from __future__ import annotations

import importlib.util
import json
import stat
import sys
import textwrap
from pathlib import Path

from tests import PROJECT_ROOT

MODULE_PATH = (
    PROJECT_ROOT
    / "jupyter/ascend/mindspore/ubi9-python-3.12/odh_mindspore_msadapter_compat.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("odh_mindspore_msadapter_compat_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_file(path: Path, content: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_maybe_prepare_convert_ckpt_rewrites_load_dir_without_cwd_shadowing(tmp_path: Path) -> None:
    compat = load_module()

    clean_site = tmp_path / "clean-site"
    bad_vendor_root = tmp_path / "MindSpeed-LLM"
    src_model_dir = tmp_path / "hf-model"

    write_file(
        clean_site / "torch/__init__.py",
        """
        from __future__ import annotations

        import json
        from pathlib import Path

        __version__ = "2.4.1"


        def save(obj, path):
            Path(path).write_text(json.dumps(obj, sort_keys=True))
        """,
    )
    write_file(
        clean_site / "transformers/__init__.py",
        """
        from __future__ import annotations

        __version__ = "4.55.2"


        class _FakeModel:
            def state_dict(self):
                return {"weight": "ok"}


        class AutoModelForCausalLM:
            @classmethod
            def from_pretrained(cls, model_dir, **kwargs):
                return _FakeModel()
        """,
    )
    write_file(
        bad_vendor_root / "transformers/__init__.py",
        """
        raise RuntimeError("compat bootstrap should not import transformers from MindSpeed-LLM")
        """,
    )

    src_model_dir.mkdir(parents=True, exist_ok=True)
    (src_model_dir / "config.json").write_text("{}")
    (src_model_dir / "model.safetensors").write_text("stub")
    (src_model_dir / "tokenizer.json").write_text("{}")

    argv = [
        str(bad_vendor_root / "mindspeed_llm/mindspore/convert_ckpt.py"),
        "--load-model-type",
        "hf",
        "--ai-framework",
        "mindspore",
        "--load-dir",
        str(src_model_dir),
        "--save-dir",
        str(tmp_path / "converted"),
    ]
    dirty_pythonpath = ":".join((str(bad_vendor_root), str(clean_site)))

    rewritten = compat.maybe_prepare_convert_ckpt(
        argv,
        python_executable=sys.executable,
        pythonpath=dirty_pythonpath,
    )

    assert rewritten != argv
    prepared_dir = Path(rewritten[rewritten.index("--load-dir") + 1])
    assert prepared_dir != src_model_dir
    assert (prepared_dir / "config.json").exists()
    assert (prepared_dir / "tokenizer.json").exists()
    payload = json.loads((prepared_dir / "pytorch_model.bin").read_text())
    assert payload == {"weight": "ok"}


def test_build_clean_pythonpath_strips_mindspeed_entries() -> None:
    compat = load_module()

    pythonpath = (
        "/tmp/keep:/opt/app-root/share/MindSpeed-Core-MS/MindSpeed-LLM:"
        "/opt/app-root/share/MindSpeed-Core-MS/MSAdapter:/tmp/keep2:"
        "/opt/app-root/share/MindSpeed-Core-MS/msadapter/msa_thirdparty"
    )

    assert compat.build_clean_pythonpath(pythonpath) == "/tmp/keep:/tmp/keep2"
