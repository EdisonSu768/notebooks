from __future__ import annotations

import json
import os
import stat
import subprocess
import textwrap
from pathlib import Path

from tests import PROJECT_ROOT

WRAPPER = (
    PROJECT_ROOT
    / "jupyter/ascend/mindspore/ubi9-python-3.12/hf_to_mcore_msadapter_compat.sh"
)


def write_file(path: Path, content: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_hf_to_mcore_wrapper_isolates_prepare_step_from_cwd_packages(tmp_path: Path) -> None:
    core_root = tmp_path / "MindSpeed-Core-MS"
    clean_site = tmp_path / "clean-site"
    bad_cwd = tmp_path / "bad-cwd"
    src_model_dir = tmp_path / "hf-model"
    save_dir = tmp_path / "converted"

    write_file(
        core_root / "tests/scripts/set_path.sh",
        f"""#!/bin/bash
        export PYTHONPATH="{clean_site}"
        """,
        executable=True,
    )

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
        clean_site / "transformers/configuration_utils.py",
        """
        class PretrainedConfig:
            def to_diff_dict(self):
                return {}

            def to_dict(self):
                return {}
        """,
    )
    write_file(
        clean_site / "transformers/modeling_utils.py",
        """
        def check_torch_load_is_safe():
            return None


        class _Functional:
            @staticmethod
            def pad(data, *_args, **_kwargs):
                return data


        class _NN:
            functional = _Functional()

            @staticmethod
            def Parameter(value):
                return value


        nn = _NN()


        class PreTrainedModel:
            pass
        """,
    )

    write_file(
        bad_cwd / "transformers/__init__.py",
        """
        raise RuntimeError("wrapper should not import transformers from the caller cwd")
        """,
    )

    write_file(
        core_root / "MindSpeed-LLM/mindspeed_llm/mindspore/convert_ckpt.py",
        """
        from __future__ import annotations

        import argparse
        import json
        from pathlib import Path

        parser = argparse.ArgumentParser()
        parser.add_argument("--load-dir", required=True)
        parser.add_argument("--save-dir", required=True)
        args, _unknown = parser.parse_known_args()

        load_dir = Path(args.load_dir)
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "result.json").write_text(
            json.dumps(
                {
                    "load_dir": str(load_dir),
                    "has_config": (load_dir / "config.json").exists(),
                    "has_pytorch_bin": (load_dir / "pytorch_model.bin").exists(),
                },
                sort_keys=True,
            )
        )
        """,
    )

    src_model_dir.mkdir(parents=True, exist_ok=True)
    (src_model_dir / "config.json").write_text("{}")
    (src_model_dir / "model.safetensors").write_text("stub")
    (src_model_dir / "tokenizer.json").write_text("{}")

    env = os.environ.copy()
    env["MINDSPEED_CORE_MS_PATH"] = str(core_root)

    result = subprocess.run(
        ["bash", str(WRAPPER), "--load-dir", str(src_model_dir), "--save-dir", str(save_dir)],
        cwd=bad_cwd,
        capture_output=True,
        env=env,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Prepared torch checkpoint:" in result.stdout

    payload = json.loads((save_dir / "result.json").read_text())
    assert payload["has_config"] is True
    assert payload["has_pytorch_bin"] is True

