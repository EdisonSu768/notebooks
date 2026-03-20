import re
import tomllib

from packaging.markers import Marker

from tests import PROJECT_ROOT


MODELSLIM_LOCKFILE = PROJECT_ROOT / "jupyter/ascend/modelslim/ubi9-python-3.11/uv.lock.d/pylock.cann.toml"
MODELSLIM_TARGET_ENV = {
    "extra": "",
    "implementation_name": "cpython",
    "os_name": "posix",
    "platform_machine": "aarch64",
    "platform_system": "Linux",
    "python_full_version": "3.11.13",
    "python_version": "3.11",
    "sys_platform": "linux",
}
MODELSLIM_COMPATIBLE_WHEEL_RE = re.compile(
    r"-(?:cp311-|cp3(?:8|9|10)-abi3-|cp311-abi3-|(?:py3|py2\.py3)-none-(?:any|[^/]*aarch64))"
)


def _load_modelslim_lock() -> dict:
    with MODELSLIM_LOCKFILE.open("rb") as f:
        return tomllib.load(f)


def _wheel_supports_modelslim_target(url: str) -> bool:
    return bool(MODELSLIM_COMPATIBLE_WHEEL_RE.search(url.rsplit("/", 1)[-1]))


def _find_modelslim_package(lock: dict, name: str) -> dict:
    for package in lock["packages"]:
        if package["name"] == name:
            return package
    raise AssertionError(f"{name} not found in {MODELSLIM_LOCKFILE}")


def test_ascend_modelslim_containerfile_pins_verified_runtime_stack():
    containerfile = PROJECT_ROOT / "jupyter/ascend/modelslim/ubi9-python-3.11/Containerfile.cann"
    text = containerfile.read_text()

    assert "MSMODELSLIM_VERSION=26.0.0a2" in text
    assert (
        "MSMODELSLIM_WHEEL_URL=https://gitcode.com/Ascend/msmodelslim/releases/download/"
        "tag_mindstudio_26.0.0.alpha02/msmodelslim-26.0.0a2-py3-none-any.whl"
    ) in text
    assert "MSMODELSLIM_BRACEX_VERSION=2.6" in text
    assert "MSMODELSLIM_EASYDICT_VERSION=1.13" in text
    assert "MSMODELSLIM_HUGGINGFACE_HUB_VERSION=1.10.2" in text
    assert "MSMODELSLIM_TORCHVISION_VERSION=0.24.0" in text
    assert "MSMODELSLIM_MISTRAL_COMMON_VERSION=1.11.0" in text
    assert "MSMODELSLIM_WCMATCH_VERSION=10.1" in text


def test_ascend_modelslim_install_script_reinstalls_multimodal_runtime_dependencies():
    install_script = PROJECT_ROOT / "jupyter/ascend/install-pytorch-npu.sh"
    text = install_script.read_text()

    assert 'MSMODELSLIM_BRACEX_VERSION="${MSMODELSLIM_BRACEX_VERSION:-2.6}"' in text
    assert 'MSMODELSLIM_EASYDICT_VERSION="${MSMODELSLIM_EASYDICT_VERSION:-1.13}"' in text
    assert 'MSMODELSLIM_HUGGINGFACE_HUB_VERSION="${MSMODELSLIM_HUGGINGFACE_HUB_VERSION:-1.10.2}"' in text
    assert 'MSMODELSLIM_WHEEL_URL="${MSMODELSLIM_WHEEL_URL:-}"' in text
    assert 'MSMODELSLIM_WCMATCH_VERSION="${MSMODELSLIM_WCMATCH_VERSION:-10.1}"' in text
    assert 'curl -LfsS "${MSMODELSLIM_WHEEL_URL}" -o "${MS_WHEEL}"' in text
    assert (
        'find /tmp/ms-extract -path "*.data/data/requirements.txt" \\( -type f -o -type d \\) -print -quit'
    ) in text
    assert 'find /tmp/ms-extract -path "*.dist-info/RECORD" -type f -print -quit' in text
    assert '"bracex==${MSMODELSLIM_BRACEX_VERSION}"' in text
    assert '"easydict==${MSMODELSLIM_EASYDICT_VERSION}"' in text
    assert '"huggingface-hub==${MSMODELSLIM_HUGGINGFACE_HUB_VERSION}"' in text
    assert '"torchvision==${MSMODELSLIM_TORCHVISION_VERSION}"' in text
    assert '"mistral-common==${MSMODELSLIM_MISTRAL_COMMON_VERSION}"' in text
    assert '"wcmatch==${MSMODELSLIM_WCMATCH_VERSION}"' in text


def test_ascend_pytorch_image_does_not_enable_modelslim_install_path():
    pytorch_containerfile = PROJECT_ROOT / "jupyter/ascend/pytorch/ubi9-python-3.12/Containerfile.cann"
    text = pytorch_containerfile.read_text()

    assert "INSTALL_MSMODELSLIM=true" not in text
    assert "MSMODELSLIM_VERSION=" not in text
    assert "MSMODELSLIM_WHEEL_URL=" not in text


def test_ascend_modelslim_pyproject_declares_qwen35_multimodal_dependencies():
    pyproject = PROJECT_ROOT / "jupyter/ascend/modelslim/ubi9-python-3.11/pyproject.toml"
    text = pyproject.read_text()

    assert '"bracex==2.6"' in text
    assert '"easydict==1.13"' in text
    assert '"huggingface-hub==1.10.2"' in text
    assert '"torchvision==0.24.0"' in text
    assert '"mistral-common==1.11.0"' in text
    assert '"wcmatch==10.1"' in text


def test_ascend_modelslim_notebook_preflights_model_permissions_before_quant():
    notebook = PROJECT_ROOT / "jupyter/ascend/modelslim/ubi9-python-3.11/qwen35_modelslim_quant_verify.ipynb"
    text = notebook.read_text()

    assert "def prepare_msmodelslim_permissions(model_path: Path, save_path: Path) -> None:" in text
    assert "for path in (model_path.parent, save_path.parent):" in text
    assert "def print_msmodelslim_permission_report(model_path: Path, save_path: Path) -> bool:" in text
    assert "MODEL_PATH still has group/other writable bits after permission prep." in text


def test_ascend_modelslim_pylock_pins_known_working_huggingface_hub():
    lock = _load_modelslim_lock()
    package = _find_modelslim_package(lock, "huggingface-hub")

    assert package["version"] == "1.10.2"


def test_ascend_modelslim_pylock_pins_missing_msmodelslim_runtime_dependencies():
    lock = _load_modelslim_lock()

    assert _find_modelslim_package(lock, "bracex")["version"] == "2.6"
    assert _find_modelslim_package(lock, "easydict")["version"] == "1.13"
    assert _find_modelslim_package(lock, "wcmatch")["version"] == "10.1"


def test_ascend_modelslim_pylock_has_python311_linux_aarch64_candidates():
    lock = _load_modelslim_lock()
    incompatible_packages = []

    for package in lock["packages"]:
        marker = package.get("marker")
        if marker and not Marker(marker).evaluate(MODELSLIM_TARGET_ENV):
            continue

        has_sdist = "sdist" in package
        has_compatible_wheel = any(
            _wheel_supports_modelslim_target(wheel["url"]) for wheel in package.get("wheels", [])
        )
        if not has_sdist and not has_compatible_wheel:
            incompatible_packages.append(package["name"])

    assert not incompatible_packages, incompatible_packages


def test_ascend_modelslim_pylock_keeps_cp311_wheels_for_native_runtime_packages():
    lock = _load_modelslim_lock()

    for package_name in ("aiohttp", "numpy", "pandas", "pydantic-core"):
        package = _find_modelslim_package(lock, package_name)
        assert any(_wheel_supports_modelslim_target(wheel["url"]) for wheel in package["wheels"]), package_name
