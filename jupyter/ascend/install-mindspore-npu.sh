#!/bin/bash
set -Eeuxo pipefail

MINDSPORE_INDEX_URL=https://repo.mindspore.cn/pypi/simple
MINDSPORE_EXTRA_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
TUNA_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
DEFAULT_GIT_HTTPS_PROXY=http://192.168.144.12:7890
MINDSPORE_DOWNLOAD_RETRIES="${MINDSPORE_DOWNLOAD_RETRIES:-20}"
MINDSPORE_DOWNLOAD_TIMEOUT="${MINDSPORE_DOWNLOAD_TIMEOUT:-60}"
MINDSPEED_CORE_MS_REPO="${MINDSPEED_CORE_MS_REPO:-https://gitcode.com/ascend/MindSpeed-Core-MS.git}"
MINDSPEED_CORE_MS_BRANCH="${MINDSPEED_CORE_MS_BRANCH:-master}"
MINDSPEED_CORE_MS_COMMIT="${MINDSPEED_CORE_MS_COMMIT:-15c32460f9ba04d4d4ca7b50cbe07344cdb4f12c}"
MINDSPEED_CORE_MS_BUILD_PATH="${MINDSPEED_CORE_MS_PATH:-/opt/app-root/share/MindSpeed-Core-MS}"
MSADAPTER_COMMIT="${MSADAPTER_COMMIT:-79d4868844313341da6c4da6f22708eea0f34a84}"
MSADAPTER_BUILD_PATH="${MSADAPTER_BUILD_PATH:-${MINDSPEED_CORE_MS_BUILD_PATH}/MSAdapter}"
ATB_SET_ENV_ARGS="${ATB_SET_ENV_ARGS:---cxx_abi=0}"
# Keep the default on the latest publicly available MindSpore wheel.
MS_VERSION="${MINDSPORE_VERSION:-2.8.0}"

source_ascend_env() {
    set +u
    source /usr/local/Ascend/cann/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh "${ATB_SET_ENV_ARGS}"
    set -u
}

configure_git_proxy() {
    local proxy="${GIT_HTTPS_PROXY:-${HTTPS_PROXY:-${https_proxy:-${DEFAULT_GIT_HTTPS_PROXY}}}}"
    if [ -n "${proxy}" ]; then
        export HTTPS_PROXY="${proxy}"
        export https_proxy="${proxy}"
    fi
}

install_mindspore() {
    local wheel_dir
    local wheels

    wheel_dir="$(mktemp -d)"
    # Download the wheel first so pip can use more resume attempts before install.
    pip3 download --no-cache-dir --no-deps \
        --dest "${wheel_dir}" \
        --only-binary=:all: \
        --retries "${MINDSPORE_DOWNLOAD_RETRIES}" \
        --resume-retries "${MINDSPORE_DOWNLOAD_RETRIES}" \
        --timeout "${MINDSPORE_DOWNLOAD_TIMEOUT}" \
        --trusted-host repo.mindspore.cn \
        -i "${MINDSPORE_INDEX_URL}" \
        --extra-index-url "${MINDSPORE_EXTRA_INDEX_URL}" \
        "mindspore==${MS_VERSION}"

    wheels=("${wheel_dir}"/mindspore-"${MS_VERSION}"-*.whl)
    if [ "${#wheels[@]}" -ne 1 ]; then
        echo "Expected exactly one downloaded MindSpore wheel in ${wheel_dir}, found ${#wheels[@]}" >&2
        exit 1
    fi

    pip3 install --no-cache-dir "${wheels[0]}"
    rm -rf "${wheel_dir}"
}

clone_pinned_branch() {
    local repo="$1"
    local branch="$2"
    local commit="$3"
    local dest="$4"
    local actual_commit

    git clone --depth 1 --branch "${branch}" "${repo}" "${dest}"
    actual_commit="$(git -C "${dest}" rev-parse HEAD)"
    if [ -n "${commit}" ] && [ "${actual_commit}" != "${commit}" ]; then
        echo "Pinned commit mismatch for ${repo}: expected ${commit}, got ${actual_commit}" >&2
        exit 1
    fi
}

checkout_pinned_commit() {
    local repo_dir="$1"
    local commit="$2"

    if [ -z "${commit}" ]; then
        return 0
    fi

    if [ "$(git -C "${repo_dir}" rev-parse HEAD)" != "${commit}" ]; then
        git -C "${repo_dir}" fetch --depth 1 origin "${commit}"
        git -C "${repo_dir}" checkout --detach "${commit}"
    fi

    if [ "$(git -C "${repo_dir}" rev-parse HEAD)" != "${commit}" ]; then
        echo "Pinned commit mismatch for ${repo_dir}: expected ${commit}" >&2
        exit 1
    fi
}

install_mindspore

rm -rf "${MINDSPEED_CORE_MS_BUILD_PATH}" "${MSADAPTER_BUILD_PATH}"
configure_git_proxy
mkdir -p "$(dirname "${MINDSPEED_CORE_MS_BUILD_PATH}")"
clone_pinned_branch "${MINDSPEED_CORE_MS_REPO}" "${MINDSPEED_CORE_MS_BRANCH}" "${MINDSPEED_CORE_MS_COMMIT}" "${MINDSPEED_CORE_MS_BUILD_PATH}"

source_ascend_env

cd "${MINDSPEED_CORE_MS_BUILD_PATH}"

pip3 install --no-cache-dir -r requirements.txt -i "${TUNA_INDEX_URL}"

set +u
source auto_convert.sh llm
set -u

MSADAPTER_BUILD_PATH="${MINDSPEED_CORE_MS_BUILD_PATH}/MSAdapter"
MINDSPEED_LLM_BUILD_PATH="${MINDSPEED_CORE_MS_BUILD_PATH}/MindSpeed-LLM"

checkout_pinned_commit "${MSADAPTER_BUILD_PATH}" "${MSADAPTER_COMMIT}"

ln -sfn MSAdapter "${MINDSPEED_CORE_MS_BUILD_PATH}/msadapter"

cd "${MSADAPTER_BUILD_PATH}"

# Patch upstream MSAdapter packaging/import issues before building the wheel.
python3 - <<'PY'
from pathlib import Path

setup_path = Path("setup.py")
setup_text = setup_path.read_text()
setup_old = '    packages=find_packages(include=["msadapter", "msa_thirdparty"]),\n'
setup_new = '    packages=find_packages(include=["msadapter*", "msa_thirdparty*"]),\n'
if setup_old not in setup_text:
    raise SystemExit("expected packages line not found in setup.py")
setup_path.write_text(setup_text.replace(setup_old, setup_new))

linalg_dir = Path("msadapter/linalg")
legacy_init = linalg_dir / "__int__.py"
init_py = linalg_dir / "__init__.py"
if legacy_init.exists() and not init_py.exists():
    init_py.write_text(legacy_init.read_text())

proxy_path = Path("msadapter/proxy.py")
proxy_text = proxy_path.read_text()
finder_old = '''class RedirectFinder(importlib.abc.MetaPathFinder):
\tdef __init__(self, redirect_map):
\t\tself.redirect_map = redirect_map

\tdef find_spec(self, fullname, path, target=None):
\t\tfor proxy_prefix, target_prefix in self.redirect_map.items():
\t\t\tif fullname == proxy_prefix or fullname.startswith(proxy_prefix + "."):
\t\t\t\ttarget_name = fullname.replace(proxy_prefix, target_prefix, 1)
\t\t\t\ttry:
\t\t\t\t\timportlib.import_module(target_name)
\t\t\t\texcept Exception as e:
\t\t\t\t\traise e

\t\t\t\treturn importlib.machinery.ModuleSpec(
\t\t\t\t\tname=fullname,
\t\t\t\t\tloader=RedirectLoader(target_name),
\t\t\t\t\tis_package=self._is_package(target_name),
\t\t\t\t)
\t\treturn None

\tdef _is_package(self, module_name):
\t\ttry:
\t\t\tmodule = importlib.import_module(module_name)
\t\t\treturn hasattr(module, "__path__")
\t\texcept ImportError:
\t\t\treturn False
'''
finder_new = '''class RedirectFinder(importlib.abc.MetaPathFinder):
\tdef __init__(self, redirect_map):
\t\tself.redirect_map = redirect_map

\tdef find_spec(self, fullname, path, target=None):
\t\tfor proxy_prefix, target_prefix in self.redirect_map.items():
\t\t\tif fullname == proxy_prefix or fullname.startswith(proxy_prefix + "."):
\t\t\t\ttarget_name = fullname.replace(proxy_prefix, target_prefix, 1)
\t\t\t\tif target_name in sys.modules:
\t\t\t\t\tis_package = hasattr(sys.modules[target_name], "__path__")
\t\t\t\telse:
\t\t\t\t\tspec = importlib.machinery.PathFinder.find_spec(target_name)
\t\t\t\t\tif spec is None:
\t\t\t\t\t\treturn None
\t\t\t\t\tis_package = spec.submodule_search_locations is not None

\t\t\t\treturn importlib.machinery.ModuleSpec(
\t\t\t\t\tname=fullname,
\t\t\t\t\tloader=RedirectLoader(target_name),
\t\t\t\t\tis_package=is_package,
\t\t\t\t)
\t\treturn None

\tdef _is_package(self, module_name):
\t\tif module_name in sys.modules:
\t\t\treturn hasattr(sys.modules[module_name], "__path__")
\t\tspec = importlib.machinery.PathFinder.find_spec(module_name)
\t\treturn spec is not None and spec.submodule_search_locations is not None
'''
getattr_old = '''\t\t\t@functools.lru_cache(maxsize=None)
\t\t\tdef __getattr__(_, name):
\t\t\t\ttry:
\t\t\t\t\ttarget_module = importlib.import_module(self.target_name)
\t\t\t\texcept ImportError as e:
\t\t\t\t\traise AttributeError(f"Target module {self.target_name} could not be imported: {e}") from e
\t\t\t\texcept Exception as e:
\t\t\t\t\traise e

\t\t\t\tif hasattr(target_module, name):
\t\t\t\t\treturn getattr(target_module, name)

\t\t\t\ttry:
\t\t\t\t\tsubmodule_name = f"{self.target_name}.{name}"
\t\t\t\t\treturn importlib.import_module(submodule_name)
\t\t\t\texcept ImportError as e:
\t\t\t\t\traise AttributeError(
\t\t\t\t\t\tf"Module '{self.target_name}' has no attribute '{name}'"
\t\t\t\t\t)

\t\t\tdef __setattr__(_, name, value):
\t\t\t\ttry:
\t\t\t\t\ttarget_module = importlib.import_module(self.target_name)
\t\t\t\t\tif not hasattr(target_module, name):
\t\t\t\t\t\treturn
\t\t\t\texcept Exception as e:
\t\t\t\t\traise e
\t\t\t\treturn super().__setattr__(name, value)
'''
getattr_new = '''\t\t\t@functools.lru_cache(maxsize=None)
\t\t\tdef __getattr__(_, name):
\t\t\t\ttarget_module = sys.modules.get(self.target_name)
\t\t\t\tif target_module is None:
\t\t\t\t\ttry:
\t\t\t\t\t\ttarget_module = importlib.import_module(self.target_name)
\t\t\t\t\texcept ImportError as e:
\t\t\t\t\t\traise AttributeError(f"Target module {self.target_name} could not be imported: {e}") from e
\t\t\t\t\texcept Exception as e:
\t\t\t\t\t\traise e

\t\t\t\ttarget_dict = getattr(target_module, "__dict__", {})
\t\t\t\tif name in target_dict:
\t\t\t\t\treturn target_dict[name]

\t\t\t\ttry:
\t\t\t\t\tsubmodule_name = f"{self.target_name}.{name}"
\t\t\t\t\tsubmodule = sys.modules.get(submodule_name)
\t\t\t\t\tif submodule is not None:
\t\t\t\t\t\treturn submodule
\t\t\t\t\treturn importlib.import_module(submodule_name)
\t\t\t\texcept ImportError as e:
\t\t\t\t\traise AttributeError(
\t\t\t\t\t\tf"Module '{self.target_name}' has no attribute '{name}'"
\t\t\t\t\t)

\t\t\tdef __setattr__(_, name, value):
\t\t\t\ttry:
\t\t\t\t\ttarget_module = sys.modules.get(self.target_name)
\t\t\t\t\tif target_module is None:
\t\t\t\t\t\ttarget_module = importlib.import_module(self.target_name)
\t\t\t\t\tif name not in getattr(target_module, "__dict__", {}):
\t\t\t\t\t\treturn
\t\t\t\texcept Exception as e:
\t\t\t\t\traise e
\t\t\t\treturn super().__setattr__(name, value)
'''
if finder_old not in proxy_text:
    raise SystemExit("expected RedirectFinder block not found in proxy.py")
if getattr_old not in proxy_text:
    raise SystemExit("expected __getattr__ block not found in proxy.py")
proxy_text = proxy_text.replace(finder_old, finder_new).replace(getattr_old, getattr_new)
proxy_path.write_text(proxy_text)
PY

bash scripts/build.sh
pip3 install --no-cache-dir dist/*.whl
cd "${MINDSPEED_CORE_MS_BUILD_PATH}"

bash tools/convert/convert.sh MindSpeed-LLM
cp -f "${MSADAPTER_BUILD_PATH}/msadapter/proxy.py" "${MINDSPEED_LLM_BUILD_PATH}/msadapter/proxy.py"

# Patch the MindSpore conversion entry so HF->MG conversion gets the same
# LoRA defaults as the generic convert_ckpt.py entrypoint.
MINDSPEED_LLM_BUILD_PATH="${MINDSPEED_LLM_BUILD_PATH}" python3 - <<'PY'
import os
from pathlib import Path

convert_ckpt = Path(os.environ["MINDSPEED_LLM_BUILD_PATH"]) / "mindspeed_llm" / "mindspore" / "convert_ckpt.py"
text = convert_ckpt.read_text()
marker = "    known_args, _ = parser.parse_known_args()\n"
insert = """    parser.add_argument('--lora-target-modules', nargs='+', type=str, default=[],\n                       help='Lora target modules.')\n""" + marker
if "--lora-target-modules" not in text:
    if marker not in text:
        raise SystemExit(f"expected parse_known_args marker not found in {convert_ckpt}")
    text = text.replace(marker, insert, 1)
    convert_ckpt.write_text(text)
PY

cd "${MINDSPEED_LLM_BUILD_PATH}"

find mindspeed_llm msadapter_npu -mindepth 1 -type d -exec sh -c 'test ! -f "$1/__init__.py" && touch "$1/__init__.py"' _ {} \;

set +u
source "${MINDSPEED_CORE_MS_BUILD_PATH}/tests/scripts/set_path.sh"
set -u

# Verify the converted runtime tree resolves imports the same way as set_path.sh.
MINDSPEED_CORE_MS_BUILD_PATH="${MINDSPEED_CORE_MS_BUILD_PATH}" python3 - <<'PY'
import importlib
import os
from pathlib import Path

modules = (
    "msadapter",
    "msadapter.utils.cpp_extension",
    "msadapter_npu.utils.utils",
    "msadapter_npu.contrib.transfer_to_npu",
    "mindspeed",
    "mindspeed.features_manager.affinity.affinity",
    "mindspeed.te.pytorch.module.layernorm",
    "mindspeed_llm",
)

core_root = Path(os.environ["MINDSPEED_CORE_MS_BUILD_PATH"])
expected_roots = {
    # Official set_path.sh prepends MindSpeed-LLM ahead of msadapter/MSAdapter,
    # so runtime imports may resolve msadapter from the converted model tree first.
    "msadapter": (
        core_root / "MindSpeed-LLM",
        core_root / "msadapter",
        core_root / "MSAdapter",
    ),
    "msadapter_npu": (core_root / "MindSpeed-LLM",),
    # convert.sh copies mindspeed into MindSpeed-LLM, and set_path.sh places
    # MindSpeed-LLM before the standalone MindSpeed source tree.
    "mindspeed": (
        core_root / "MindSpeed-LLM",
        core_root / "MindSpeed",
    ),
    "mindspeed_llm": (core_root / "MindSpeed-LLM",),
}

for name in modules:
    module = importlib.import_module(name)
    module_file = Path(getattr(module, "__file__", ""))
    top_level = name.split(".", 1)[0]
    allowed_roots = expected_roots[top_level]
    if module_file and not any(root == module_file or root in module_file.parents for root in allowed_roots):
        allowed_roots_str = ", ".join(str(root) for root in allowed_roots)
        raise SystemExit(
            f"{name} loaded from unexpected location: {module_file} "
            f"(expected under one of: {allowed_roots_str})"
        )
    print(f"OK: {name} -> {module_file}")

runtime_model_cfg = core_root / "MindSpeed-LLM" / "configs" / "checkpoint" / "model_cfg.json"
if not runtime_model_cfg.exists():
    raise SystemExit(f"MindSpeed-LLM checkpoint config missing after install: {runtime_model_cfg}")
print(f"OK: {runtime_model_cfg}")

mindspore_convert_ckpt = core_root / "MindSpeed-LLM" / "mindspeed_llm" / "mindspore" / "convert_ckpt.py"
if "--lora-target-modules" not in mindspore_convert_ckpt.read_text():
    raise SystemExit(
        "MindSpeed-LLM MindSpore convert_ckpt.py is missing the "
        "lora-target-modules compatibility argument"
    )
print(f"OK: {mindspore_convert_ckpt}")
PY

cd /
