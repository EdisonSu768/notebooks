---
name: ascend-build
description: Sync Ascend code, push to build repo, trigger CI pipeline via browser, and get the built image name
---

# Ascend Image Build & Deploy Skill

This skill syncs Ascend notebook code to the build repository, triggers the CI pipeline, and retrieves the built image name.

## Parameters

- `$IMAGE_TARGET` (required): The build target name, e.g. `mindspore-cann`, `pytorch-cann`, `codeserver-cpu`
- `$SOURCE_DIR` (optional): Source directory to sync. Defaults to `jupyter/ascend`

## Step 1: Sync code to build repository

```bash
cp -r jupyter/ascend /Users/szg/alauda/odh-workbench-images/jupyter
```

## Step 2: Git push in build repository

```bash
cd /Users/szg/alauda/odh-workbench-images && git add . && git commit --amend --no-edit && ggf
```

## Step 3: Browser automation to trigger CI build

### 3a: Open browser with login state

Check if saved auth state exists at `/Users/szg/Github/odh/notebooks/.claude/edge-auth.json`.

If auth state file exists:
```bash
playwright-cli open --headed
playwright-cli state-load /Users/szg/Github/odh/notebooks/.claude/edge-auth.json
playwright-cli goto "https://edge.alauda.cn/console-devops/workspace/aml/ci?buildName=workbench-arm-images&cluster=business-build&namespace=aml-dev"
```

If auth state file does NOT exist:
```bash
playwright-cli open --headed "https://edge.alauda.cn/console-devops/workspace/aml/ci?buildName=workbench-arm-images&cluster=business-build&namespace=aml-dev"
```
Then **ASK the user to log in manually in the browser window**. Wait for the user to confirm they've logged in, then:
```bash
playwright-cli state-save /Users/szg/Github/odh/notebooks/.claude/edge-auth.json
```

### 3b: Trigger the build pipeline

Use `playwright-cli snapshot` to inspect the page and find the correct element refs. Then:

1. Take a snapshot to find the "执行" (Execute) button and click it:
   ```bash
   playwright-cli snapshot
   ```
   Find the execute button ref, then click it:
   ```bash
   playwright-cli click "<execute-button-ref>"
   ```

2. In the dialog that appears, find the "构建对象" (Build Target) dropdown/select and set it to branch mode:
   - Use snapshot to find the select element for 构建对象
   - Select "分支" (branch) option using `playwright-cli select`

3. Find the branch dropdown and select `feat/ascend`:
   - Use snapshot to find the branch select
   - Select `feat/ascend` using `playwright-cli select`

4. Find the targets input field and enter the target name:
   ```bash
   playwright-cli fill "<targets-input-ref>" "$IMAGE_TARGET"
   ```
   Then press Enter to confirm:
   ```bash
   playwright-cli press Enter
   ```

5. Click the "确定" (Confirm) button to start the build:
   ```bash
   playwright-cli click "<confirm-button-ref>"
   ```

**IMPORTANT**: At each step, use `playwright-cli snapshot` to discover the current element refs. Do NOT guess element selectors - always snapshot first, identify the correct ref, then act on it.

### 3c: Wait for pipeline completion

After triggering, the pipeline takes approximately 25-30 minutes. Use `playwright-cli snapshot` periodically to check status.

Polling strategy:
1. Go back to the pipeline list page if needed
2. Take a snapshot to check the latest pipeline runs
3. Look for the pipeline with `$IMAGE_TARGET` in its parameters
4. Check its status - wait until it shows success/failure
5. Poll every 3-5 minutes using `playwright-cli snapshot`

### 3d: Get the built image name

Once the pipeline completes successfully:

1. In the pipeline list table, find the row where the execution parameters contain `$IMAGE_TARGET`
2. Hover over the artifact icon (the `img` with `src="assets/icons/pipeline/badges/artifact.svg"`) next to the execution name:
   ```bash
   playwright-cli hover "<artifact-icon-ref>"
   ```
3. Wait for the overlay to appear, then take a snapshot:
   ```bash
   playwright-cli snapshot
   ```
4. Find and click the first item in the overlay to copy the image address:
   ```bash
   playwright-cli click "<first-overlay-item-ref>"
   ```
5. Read the clipboard content to get the image name:
   ```bash
   playwright-cli eval "navigator.clipboard.readText()" 
   ```
   Or use macOS pbpaste:
   ```bash
   pbpaste
   ```

## Step 4: Generate workspacekind patch

Using the image name obtained in Step 3, generate the JSON patch file:

```bash
cd /Users/szg/alauda/odh-workbench-images && ./scripts/generate-workspacekind-patch.sh --image "<FULL_IMAGE_NAME>"
```

Where `<FULL_IMAGE_NAME>` is the complete image reference from Step 3d (e.g. `192.168.111.127:11443/mlops/workbench-images/alauda-workbench-jupyter-mindspore-cann-py312-ubi9:v0.0.0-feat.4.g64ef4a24-feat-asc`).

This script writes the patch to `resources/patch-workspacekind.json`.

## Step 5: Ensure NPU K8s tunnel is connected

Check if the SSH tunnel to the NPU K8s cluster is already running:

```bash
lsof -Pi :6443 -sTCP:LISTEN -t >/dev/null 2>&1
```

If port 6443 is NOT listening, start the tunnel:

```bash
/Users/szg/alauda/odh-workbench-images/scripts/connect-npu-k8s.sh
```

**IMPORTANT**: This script blocks and runs in the foreground. It MUST be run in the background:

```bash
bash /Users/szg/alauda/odh-workbench-images/scripts/connect-npu-k8s.sh &
```

Wait a few seconds, then verify the tunnel is up:

```bash
lsof -Pi :6443 -sTCP:LISTEN -t >/dev/null 2>&1 && echo "tunnel OK" || echo "tunnel NOT ready"
```

If the tunnel fails to start or the probe fails, ask the user to check SSH connectivity to jumpserver and npu.

## Step 6: Patch the workspacekind resource

Once the tunnel is confirmed active, apply the patch to update the workspace image config:

```bash
cd /Users/szg/alauda/odh-workbench-images/resources && k patch workspacekind jupyterlab-internal-v0-1-6 --type='json' --patch-file patch-workspacekind.json
```

Verify the patch was applied:

```bash
k get workspacekind jupyterlab-internal-v0-1-6 -o jsonpath='{.spec.podTemplate.options.imageConfig.values[-1].spec.image}'
```

This should output the new image name.

## Step 7: Report result

Report the final result to the user. Example output:
```
Build & deploy completed successfully.
Image: 192.168.111.127:11443/mlops/workbench-images/alauda-workbench-jupyter-mindspore-cann-py312-ubi9:v0.0.0-feat.4.g64ef4a24-feat-asc
Workspacekind jupyterlab-internal-v0-1-6 patched.
```

If any step failed, report the failure and suggest the user check the relevant logs.

## Notes

- Always use `playwright-cli snapshot` before interacting with any element
- Use `--headed` flag when opening browser so the user can observe
- The auth state may expire; if page shows login, ask user to re-login and re-save state
- `ggf` is a shell alias for `git push --force origin <current-branch>`
- The SSH tunnel script must run in the background; check port 6443 before starting a new one
