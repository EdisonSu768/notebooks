#!/usr/bin/env bash

# Load bash libraries
SCRIPT_DIR=/opt/app-root/bin
source ${SCRIPT_DIR}/utils/process.sh

if [ -f "/opt/app-root/bin/activate" ]; then
  source /opt/app-root/bin/activate
fi

if [ -f "/usr/local/Ascend/cann/set_env.sh" ]; then
  source /usr/local/Ascend/cann/set_env.sh
fi

if [ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]; then
  source /usr/local/Ascend/nnal/atb/set_env.sh ${ATB_SET_ENV_ARGS}
fi

resolve_mindspeed_core_ms_path() {
  local candidate

  for candidate in \
    "${MINDSPEED_CORE_MS_PATH:-}" \
    /opt/app-root/share/MindSpeed-Core-MS
  do
    if [ -n "${candidate}" ] && [ -f "${candidate}/tests/scripts/set_path.sh" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

if MINDSPEED_CORE_MS_PATH_RESOLVED="$(resolve_mindspeed_core_ms_path)"; then
  export MINDSPEED_CORE_MS_PATH="${MINDSPEED_CORE_MS_PATH_RESOLVED}"
  source "${MINDSPEED_CORE_MS_PATH}/tests/scripts/set_path.sh"
fi

# Interactive shells spawned by Jupyter only see exported variables.
# Without exporting PS1 here, they fall back to the default host prompt
# even though the app-root virtual environment is already active.
if [ -n "${VIRTUAL_ENV:-}" ]; then
  if [ -z "${PS1:-}" ]; then
    PS1="${VIRTUAL_ENV_PROMPT:-"(app-root) "}"'\w\$ '
  fi
  export PS1
fi

if [ -f "${SCRIPT_DIR}/utils/setup-elyra.sh" ]; then
  source ${SCRIPT_DIR}/utils/setup-elyra.sh
fi

# Initialize notebooks arguments variable
NOTEBOOK_PROGRAM_ARGS=""

# Set default ServerApp.port value if NOTEBOOK_PORT variable is defined
if [ -n "${NOTEBOOK_PORT}" ]; then
    NOTEBOOK_PROGRAM_ARGS+="--ServerApp.port=${NOTEBOOK_PORT} "
fi

# Set default ServerApp.base_url value if NOTEBOOK_BASE_URL variable is defined
if [ -n "${NOTEBOOK_BASE_URL}" ]; then
    NOTEBOOK_PROGRAM_ARGS+="--ServerApp.base_url=${NOTEBOOK_BASE_URL} "
fi

# Set default ServerApp.root_dir value if NOTEBOOK_ROOT_DIR variable is defined
if [ -n "${NOTEBOOK_ROOT_DIR}" ]; then
    NOTEBOOK_PROGRAM_ARGS+="--ServerApp.root_dir=${NOTEBOOK_ROOT_DIR} "
else
    NOTEBOOK_PROGRAM_ARGS+="--ServerApp.root_dir=${HOME} "
fi

# Add additional arguments if NOTEBOOK_ARGS variable is defined
if [ -n "${NOTEBOOK_ARGS}" ]; then
    NOTEBOOK_PROGRAM_ARGS+=${NOTEBOOK_ARGS}
fi

# Start the JupyterLab notebook
start_process jupyter lab ${NOTEBOOK_PROGRAM_ARGS} \
    --ServerApp.ip="" \
    --ServerApp.allow_origin="*" \
    --ServerApp.open_browser=False
