from tests import PROJECT_ROOT


def test_ascend_modelslim_hardens_msmodelslim_package_after_fix_permissions():
    containerfile = PROJECT_ROOT / "jupyter/ascend/modelslim/ubi9-python-3.11/Containerfile.cann"
    text = containerfile.read_text()

    fix_permissions = "fix-permissions /opt/app-root -P"
    hardening = 'chmod -R go-w "${MSMODELSLIM_PACKAGE_DIR}"'

    assert fix_permissions in text
    assert 'importlib.util.find_spec("msmodelslim")' in text
    assert hardening in text
    assert text.index(fix_permissions) < text.index(hardening), (
        "msmodelslim package permissions must be tightened after the final fix-permissions call"
    )
