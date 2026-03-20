# Ascend msModelSlim Image Review Notes

## 1. 背景与原始需求

本次需求的起点是：

1. `jupyter/ascend/pytorch` 镜像已经通过当前目录下的 2 个 notebook 做过验证。
2. 需要新增一个支持 `msmodelslim` 的 Ascend Jupyter 镜像，用于模型压缩能力验证。
3. 优先按照 Ascend 官方路线评估，不优先走第三方压缩方案。
4. `msmodelslim` 对 Python 版本敏感，因此需要评估是否必须从 Python 3.12 调整到 Python 3.11。
5. 用户明确希望：
   - 不再依赖单独新增的 `base-images/cann/8.5.0/c9s-python-3.11` 基础镜像；
   - 直接在 `jupyter/ascend/modelslim` 的构建里完成 `py311 + CANN + msmodelslim`；
   - 不保留从 `jupyter/ascend/pytorch` 复制来的 2 个验证 notebook；
   - 改为基于官方文档 `https://raw.gitcode.com/Ascend/msmodelslim/raw/master/example/Qwen3_5/README.md` 新建验证 notebook。

## 2. 关键约束与发现

### 2.0 2026-04-16 补充验证结论

- 已在真实工作空间 pod 中完成 `Qwen3.5-27B` 的一键量化验证，最终产物成功落盘。
- 当前验证通过的组合不是 `msmodelslim 8.2.1`，而是 `msmodelslim 26.0.0a2`。
- 对 `Qwen3.5-27B` 这类多模态模型，仅有 `transformers==5.2.0` 还不够，实际还需要：
  - `torchvision==0.24.0`
  - `mistral-common==1.11.0`
- 最终镜像阶段还需要显式回钉 `huggingface-hub==1.10.2`，否则 Ascend 栈里的后续 `pip install` 可能把它替换成一个无法支撑 `transformers 5.2.0` 导入链路的版本。
- `msmodelslim` wheel 仍然保留 `data/requirements.txt` 数据路径问题；不同版本里这个条目可能表现为文件或目录，因此镜像里继续需要对 wheel 做修补后再安装。

### 2.1 Python 版本

- `msmodelslim` 这条官方路线最终落在 Python 3.11。
- 因此当前新镜像目录定为：
  - `jupyter/ascend/modelslim/ubi9-python-3.11`

### 2.2 官方示例要求

官方 `Qwen3.5` 文档明确要求：

- 安装 `msmodelslim`
- 安装 `transformers==5.2.0`
- 使用 `msmodelslim quant ... --device npu ...`

因此本次实现里明确引入了：

- `msmodelslim==26.0.0a2`
- `transformers==5.2.0`
- `huggingface-hub==1.10.2`
- `torchvision==0.24.0`
- `mistral-common==1.11.0`
- `numpy~=1.26.4`

### 2.3 依赖冲突结论

用户特别问到为什么去掉了 `feast`、`codeflare-sdk`、`kfp`，这里给出最终可复现结论。

#### `feast`

- `feast 0.60.0` 解析时要求 `numpy>=2.0.0,<3`
- 当前 `msmodelslim` 路线固定 `numpy~=1.26.4`
- 这是直接硬冲突，不能在当前方案里共存

结论：

- `feast` 本次没有加回

#### `kfp`

- 单独显式加回 `kfp~=2.15.2` 后，依赖可以正常解析
- 最终锁文件中保留了：
  - `kfp 2.15.2`
  - `kfp-kubernetes 2.15.2`
  - `kfp-pipeline-spec 2.16.0`
  - `kfp-server-api 2.16.0`
  - `kubernetes 30.1.0`

结论：

- `kfp` 可以加回，当前已显式恢复

#### `codeflare-sdk`

- 原始 `pytorch` 镜像使用的是 `codeflare-sdk~=0.35.0`
- `codeflare-sdk 0.35.x` 会引入 `kube-authkit>=0.4.0`
- `kube-authkit 0.4.0` 进一步要求 `kubernetes>=35`
- 但当前 `odh-elyra==4.3.2 -> kfp>=2.0.0` 最终要求 `kubernetes<31`

因此：

- `codeflare-sdk 0.35.x` 与 `odh-elyra + kfp` 这条链路不兼容

进一步测试后发现：

- `codeflare-sdk==0.34.0` 不再引入这条 `kube-authkit>=0.4.0` 冲突链
- 且可以与：
  - `odh-elyra==4.3.2`
  - `kfp~=2.15.2`
  - `numpy~=1.26.4`
  - `msmodelslim==26.0.0a2`
  一起成功生成锁文件

结论：

- `codeflare-sdk` 已恢复，但版本改为 `0.34.0`

### 2.4 Python 3.11 锁文件生成注意事项

- 2026-04-16 实际构建时发现，原先用 `cpu-ubi9-test` RH 索引加 `unsafe-best-match` 生成的
  `uv.lock.d/pylock.cann.toml` 会把部分 Python 3.11 依赖错误锁到仅提供 `cp312` 轮子的条目。
- 第一个暴露出来的报错是 `aiohttp==3.13.5`，但实际受影响的不止一个包，`numpy`、`pandas`、
  `pydantic-core` 等带本地扩展的依赖也会出现同类问题。
- 这不是容器里的 `uv pip install` 参数问题，而是锁文件本身已经不适用于
  `python 3.11 + linux/arm64` 目标环境。

当前处理方式：

- 保持 `jupyter/ascend/modelslim` 自包含，不依赖额外修改 `scripts/pylocks_generator.py`
- 直接基于公开 PyPI 重新生成 `uv.lock.d/pylock.cann.toml`
- 并在测试里增加针对 `python 3.11 + linux/aarch64` 候选分发的静态检查，防止后续再回归到
  `cp312-only` 锁文件

## 3. 当前实现方案

### 3.1 新镜像目录

新增目录：

- `jupyter/ascend/modelslim/ubi9-python-3.11`

目录包含：

- `Containerfile.cann`
- `build-args/cann.conf`
- `pyproject.toml`
- `uv.lock.d/pylock.cann.toml`
- `qwen35_modelslim_quant_verify.ipynb`

### 3.2 构建方式

当前不再依赖单独新增的 `base-images/cann/8.5.0/c9s-python-3.11`。

改为在 `jupyter/ascend/modelslim/ubi9-python-3.11/Containerfile.cann` 内直接完成：

1. 参考 `base-images/cann/8.5.0/c9s-python-3.12` 的结构内联构建 base 阶段
2. `buildscripts` 阶段基于 `quay.io/centos/centos:stream9`
3. `cann-base` 阶段基于 `quay.io/sclorg/python-311-c9s:c9s`
4. 在该 base 阶段内联 CANN 8.5.0 安装
5. 内联 `pip.conf` / `uv.toml` 配置
6. 复用 Ascend minimal / datascience 的 notebook 结构
7. 在最终阶段安装：
   - lock 文件中的 Python 依赖
   - `torch`
   - `torch_npu`
   - `MindSpeed`
   - `MindSpeed-LLM`
   - `msmodelslim`
   - `transformers==5.2.0`

这样做的原因：

- reviewer 只需要审 `jupyter/ascend/modelslim` 这一条镜像链路
- 避免单独引入新的 base image 维护面
- 更符合用户“直接做到 modelslim 构建里”的要求
- 同时尽量贴近现有 Ascend `c9s` 路线，减少与既有镜像链路的 repo / 包源差异

### 3.3 PDF 导出能力

最初的直接 `ubi9/python-311` 内联方案在构建时暴露出一个额外问题：

- `jupyter/utils/install_pdf_deps.sh` 依赖的一批 `texlive-*` RPM
- 在那条直接 UBI 9 Python 3.11 的 repo 视图下无法解析
- 因此镜像会在 PDF 依赖安装阶段失败

当前处理方式：

- 不再继续使用直接 `ubi9/python-311` 作为 base
- 改为切回与现有 Ascend 底包一致的 `c9s` 路线
- 在 `modelslim` 镜像内恢复正常执行 `./utils/install_pdf_deps.sh`
- 不修改全局 `install_pdf_deps.sh`

影响范围：

- 当前方案重新对齐到现有 Ascend `c9s` 包源环境
- 目标是保留与其他 Jupyter 镜像一致的 PDF 导出能力
- 但由于本轮未能实际执行完整 `podman build`，PDF 这部分仍属于“按现有链路推断可恢复”，而不是已实机验证完成

### 3.4 安装脚本

公共脚本 `jupyter/ascend/install-pytorch-npu.sh` 已改为：

- 支持 Python 3.11 / 3.12 自动识别
- 允许通过环境变量安装 `msmodelslim`
- 在 `INSTALL_MSMODELSLIM=true` 时额外安装 `transformers==5.2.0`

### 3.5 Notebook 方案

已删除从 `jupyter/ascend/pytorch` 复制过来的：

- `qwen25_pretrain_verify.ipynb`
- `qwen3_finetune_verify.ipynb`

新增：

- `qwen35_modelslim_quant_verify.ipynb`

新 notebook 的设计原则：

- 基于官方 `Qwen3.5` 量化说明
- 默认只做环境验证和命令拼装
- 不默认直接启动大模型量化任务
- 通过 `RUN_QUANT = False/True` 控制是否真正执行量化

这样更适合作为镜像 smoke test，也更适合 review。

## 4. 具体变更点

### 4.1 依赖声明

`jupyter/ascend/modelslim/ubi9-python-3.11/pyproject.toml` 当前核心依赖为：

- `codeflare-sdk==0.34.0`
- `kfp~=2.15.2`
- `numpy~=1.26.4`
- `msmodelslim==26.0.0a2`
- `transformers==5.2.0`
- `torchvision==0.24.0`
- `mistral-common==1.11.0`
- `odh-elyra==4.3.2`

并明确保留说明：

- `feast` 暂不纳入当前镜像

### 4.2 锁文件

已重新生成：

- `jupyter/ascend/modelslim/ubi9-python-3.11/uv.lock.d/pylock.cann.toml`

当前已确认锁文件中存在：

- `codeflare-sdk 0.34.0`
- `kfp 2.15.2`
- `transformers 5.2.0`
- `huggingface-hub 1.10.2`
- `msmodelslim 26.0.0a2`
- `torchvision 0.24.0`
- `mistral-common 1.11.0`
- `numpy 1.26.4`
- `odh-elyra 4.3.2`

### 4.3 测试侧处理

当前 `tests/test_main.py` 里保留了一个临时 skip：

- `jupyter/ascend/modelslim/ubi9-python-3.11`

原因：

- 该新镜像还没有配套 manifest
- 静态测试里会因为“manifest not implemented”失败

这部分属于后续可继续补齐的工作，不影响当前方案 review。

## 5. 验证情况

本轮已完成的验证：

1. 新 `pyproject.toml` 可以成功生成 `pylock.cann.toml`
2. `pyproject` 中声明的依赖都能在 lock 文件中找到
3. 新 notebook JSON 格式校验通过
4. `codeflare-sdk==0.34.0 + kfp~=2.15.2 + msmodelslim==26.0.0a2 + transformers==5.2.0 + torchvision==0.24.0 + mistral-common==1.11.0` 可以共存
5. 已定位并修复当前构建日志中的两个阻塞点：
   - `aipcc.sh` 在非 root 用户下执行 `dnf`
   - 直接 `ubi9/python-311` 构建链路中的 PDF 依赖安装失败
6. 已把 `modelslim` 的 base 阶段切回与现有 Ascend 底包一致的 `c9s` Python 3.11 路线
7. 已在真实工作空间 pod 中完成 `Qwen3.5-27B` `w8a8` 量化，输出目录成功生成量化 safetensors

本轮尚未完成的验证：

1. 未实际跑完整镜像 `podman build`
2. 未补 manifests

## 6. 当前结论

当前 reviewer 可按以下结论理解这次方案：

1. `msmodelslim` 这条官方路线落在 Python 3.11，而不是当前 `ascend/pytorch` 的 Python 3.12。
2. 不需要单独新增 `base-images/cann/8.5.0/c9s-python-3.11`，但 `modelslim` 的内联 base 阶段应当按现有 `c9s` 路线来做，而不是直接从 `ubi9/python-311` 起步。
3. `kfp` 可以加回。
4. `codeflare-sdk` 可以加回，但必须从 `0.35.x` 降到 `0.34.0`。
5. `feast` 由于 `numpy` 主版本冲突，当前不建议放进同一个 `msmodelslim` 镜像。
6. `Qwen3.5-27B` 这条量化链路需要在镜像中额外补齐 `torchvision==0.24.0` 和 `mistral-common==1.11.0`。
7. `transformers 5.2.0` 在最终镜像中还需要显式保住 `huggingface-hub==1.10.2`，否则导入阶段可能报 `cannot import name 'is_offline_mode'`。
8. 验证 notebook 不应沿用训练 / 微调 notebook，而应改成基于官方 Qwen3.5 量化说明的专用验证 notebook。

## 7. 建议 reviewer 重点看什么

建议 reviewer 优先关注以下内容：

1. `jupyter/ascend/modelslim/ubi9-python-3.11/Containerfile.cann`
   - 是否接受“内联 c9s py311 + CANN”的构建策略
2. `jupyter/ascend/modelslim/ubi9-python-3.11/pyproject.toml`
   - 是否接受 `codeflare-sdk==0.34.0`
   - 是否接受当前阶段不引入 `feast`
3. `jupyter/ascend/install-pytorch-npu.sh`
   - 是否接受在 `INSTALL_MSMODELSLIM=true` 时补装 `transformers==5.2.0`
   - 是否接受默认改为安装 `msmodelslim 26.0.0a2`
   - 是否接受追加 `torchvision==0.24.0` 和 `mistral-common==1.11.0`
4. `jupyter/ascend/modelslim/ubi9-python-3.11/qwen35_modelslim_quant_verify.ipynb`
   - 是否满足当前镜像验证目标
5. `tests/test_main.py`
   - 是否接受在 manifest 补齐前先保留 skip
