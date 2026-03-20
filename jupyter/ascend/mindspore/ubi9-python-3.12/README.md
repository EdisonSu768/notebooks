# Qwen3 MindSpore HF->MCore 与训练运行时踩坑记录

本文记录 `jupyter/ascend/mindspore/ubi9-python-3.12/qwen3_0.6b_finetune_verify.ipynb` 在调试 Qwen3 HF -> MindSpeed/MCore 权重转换时的完整踩坑过程、验证结论和最终修复方案。

时间基线：
- 首轮集中排障与最终验证时间：`2026-04-14`
- 实际验证 pod：`zgsu-global-ns1/ws-mindspore-wzsk4-0`
- 容器：`main`

## 背景

notebook 第 4 步会调用 MindSpeed-LLM 的 MindSpore 转换入口：

```bash
/opt/app-root/share/MindSpeed-Core-MS/MindSpeed-LLM/mindspeed_llm/mindspore/convert_ckpt.py
```

目标是把 HuggingFace 的 `Qwen3-0.6B` 模型权重转换成 MindSpeed/MCore 训练可用的 checkpoint。

最初在 notebook 中看到的失败点是：

```text
building GPT model ...
ERROR:root:Loader exited, exiting saver
```

这类报错信息太浅，必须进入 pod 逐段拆解。

## 最开始的误判

最初看日志，最容易怀疑的是：
- 镜像里没有正确的 `torch`
- `transformers` 版本不对
- MindSpore/MSAdapter 路径被覆盖

这个方向只对了一小半。

### 实际验证结果

在 pod 中清掉 MindSpeed vendored 路径后，系统级 HF 栈是正常的：

```text
torch: 2.4.1
transformers: 4.55.2
config ok: Qwen3Config qwen3 torch.bfloat16
model ok: Qwen3ForCausalLM torch.float32 cpu
```

这一步直接证明：

- 问题不是系统 `torch` 缺失
- 问题不是系统 `transformers` 无法加载 Qwen3
- 问题不在原始 HF 模型目录本身

真正有问题的是 MindSpeed vendored 运行时和 `msadapter` 代理栈的组合。

## 踩坑过程

### 1. `msadapter` 版本探测把 vendored transformers 卡死了

MindSpeed vendored `transformers` 不直接用真实 `torch` 包做版本探测，而是把 `msadapter` 当成 `torch backend`。

但 pod 里实际状态是：

```text
importlib.metadata.version("msadapter") -> 0.7.0
vendored torch.__version__ -> 2.1.1+dev
```

于是 vendored `transformers` 在导入期把 `_torch_available` 判成了 `False`。

这解释了最早 notebook 里那种看起来像“明明有 torch 却说没有”的怪现象。

### 2. 只修版本探测后，下一层炸在 BFloat 配置序列化

把：

```python
importlib.metadata.version("msadapter")
```

临时 monkeypatch 成真实 `torch.__version__` 后，转换流程可以继续往下走。

但紧接着就碰到：

```text
TypeError: Object of type BFloat is not JSON serializable
```

根因是 vendored `transformers.configuration_utils.PretrainedConfig.to_json_string()` 在当前栈下会把 `BFloat` 之类对象直接送进 `json.dumps()`。

### 3. 再修 JSON 序列化后，炸在 safetensors + `UntypedStorage`

再把：

```python
json.dumps(..., default=str)
```

补上之后，流程继续推进，随后出现：

```text
AttributeError: Module 'msadapter' has no attribute 'UntypedStorage'
```

这一步发生在 safetensors 读取阶段，本质上是：

- vendored `transformers` 走了 `safe_open(..., framework="pt")`
- 但这里的“pt”实际上被代理到了 `msadapter`
- `msadapter` 并没有 PyTorch 那套完整的 storage 兼容面

进一步验证也确认了：

```text
hasattr(msadapter, "UntypedStorage") == False
hasattr(torch, "UntypedStorage") == True   # 仅系统真实 torch
```

### 4. 尝试彻底绕开 vendored 路径，不现实

我们试过把 `MindSpeed-LLM/transformers`、`msadapter` 等 vendored 路径从 `PYTHONPATH` 中清掉，只保留系统 `torch+transformers`。

结果是 HF 侧能跑，但 `mindspeed_llm` 自身又会依赖：

- `msadapter_npu`
- `torch_npu`
- `MindSpeed` 内部 patch
- `FeatureAdaptor.execute()` 的一整套导入副作用

继续混用系统栈和 vendored 栈，会引出更多新的兼容性问题，包括但不限于：

- `ImportError: cannot import name 'PreTrainedModel'`
- `ModuleNotFoundError: No module named 'torch.distributed.tensor'`
- `ModuleNotFoundError: No module named 'torch.library'`
- `protobuf` / `mindspore` 组合在特定导入路径下的 `_pb2.py` 描述符异常

这条路太脆，不能作为修复方案。

### 5. 把 HF `safetensors` 先转成 `.bin`，能绕开 `UntypedStorage`

继续在 pod 内验证后发现，问题并不是 HF 模型不能被读取，而是：

- vendored `from_pretrained()` 读取 `safetensors` 时依赖了 PyTorch storage 兼容层

所以我们改为先用系统级 `torch+transformers` 把：

```text
model.safetensors
```

转成：

```text
pytorch_model.bin
```

这样就把 `safetensors` 这颗雷提前排掉了。

### 6. `.bin` 路线又炸在 `tie_weights()`

`.bin` 成功绕开 `UntypedStorage` 以后，下一步又报：

```text
TypeError: cannot assign '<class 'abc.Parameter'>' as parameter 'weight'
(msadapter.nn.Parameter or None expected)
```

根因是 vendored `transformers.modeling_utils._tie_or_clone_weights()` 默认会做：

```python
output_embeddings.weight = input_embeddings.weight
```

这对标准 PyTorch 是可行的，但对 `msadapter.nn.Parameter` 不成立。

### 7. `.bin` 路线还会被 `torch.load >= 2.6` 安全闸门拦住

同一条 `.bin` 路径还遇到另一个显式版本门槛：

```text
ValueError: Due to a serious vulnerability issue in `torch.load` ...
require users to upgrade torch to at least v2.6
```

这不是功能错误，而是 vendored `transformers` 对 `torch.load` 的安全限制。

但当前代理栈报告出来的“torch 版本”根本不可能满足这个约束，所以必须显式绕过。

### 8. 直接把 `tie_weights()` 整个关掉也不行

尝试把 `tie_weights()` 直接置空虽然能继续跑，但会导致 embedding 权重没有按预期落到模型里。

实际观察到：

```text
unexpected_keys ['model.embed_tokens.weight']
```

这说明“简单禁用”不是正确修法。

## 最终确认的根因

根因不是单点，而是一串串联兼容性问题：

1. MindSpeed vendored `transformers` 用 `msadapter` 的 distribution version 充当 torch 版本探测，和真实代理行为不一致
2. vendored 配置序列化对 `BFloat` 等对象不兼容
3. vendored safetensors 加载依赖 PyTorch storage API，而 `msadapter` 不具备完整兼容面
4. vendored `.bin` 加载又会被 `torch.load` 安全门槛卡住
5. `tie_weights()` 默认的 Parameter 赋值方式不兼容 `msadapter.nn.Parameter`

一句话概括：

> 不是“没有 torch”，而是 MindSpeed vendored HF loader 和 `msadapter` 代理栈在 `safetensors`、`torch.load`、`tie_weights()` 这三层同时不兼容。

## 最终采用的修复方案

不继续在 notebook 里硬塞 monkeypatch，而是新增一个专用 wrapper：

- 文件：`hf_to_mcore_msadapter_compat.sh`

这个脚本做两件事：

### 第一步：如果输入是 safetensors，先用系统 HF 栈转成 `.bin`

脚本会：

- 用干净的 `PYTHONPATH` 排除 MindSpeed vendored 路径
- 调用系统 `torch+transformers`
- 把 HF 模型目录镜像到临时目录
- 生成 `pytorch_model.bin`

这样就避开了 `msadapter` 对 `safetensors` 的不兼容。

### 第二步：调用 vendored `convert_ckpt.py`，但只打四个必要补丁

wrapper 在真正调用：

```bash
MindSpeed-LLM/mindspeed_llm/mindspore/convert_ckpt.py
```

前，会注入以下 4 个补丁：

1. `importlib.metadata.version("msadapter") -> 真实 torch.__version__`
2. `PretrainedConfig.to_json_string()` 改为 `json.dumps(..., default=str)`
3. `modeling_utils.check_torch_load_is_safe = lambda: None`
4. `PreTrainedModel._tie_or_clone_weights()` 改为 `.data.copy_()`，不直接替换 `Parameter` 对象

这 4 个补丁是 pod 内实际验证过的最小可行闭环。

## 为什么不直接去 patch vendored transformers 源码

可以 patch，但不优先。

原因有三点：

1. 失败点是“沿路移动的”
   - 修完版本探测，炸 JSON
   - 修完 JSON，炸 safetensors
   - 绕开 safetensors，炸 `torch.load`
   - 绕开 `torch.load`，炸 `tie_weights()`

2. 这些 patch 分散在多处 vendored 源码里，构建期 patch 可维护性差

3. wrapper 把兼容逻辑限制在 HF->MCore 这一个入口，不污染其他路径

所以这次优先选择：

> 入口级 wrapper，而不是在 notebook 单元格里再打一层运行时补丁，也不是直接改一大片 vendored transformers 源码。

## pod 内的最终验证结果

在 `2026-04-14`，pod `zgsu-global-ns1/ws-mindspore-wzsk4-0` 中做了等价全链路验证。

最终成功产出：

```text
/tmp/qwen3_mcore_test/iter_0000001
/tmp/qwen3_mcore_test/latest_checkpointed_iteration.txt
```

关键成功日志：

```text
saving checkpoint at iteration       1 to /tmp/qwen3_mcore_test in msadapter format
successfully saved checkpoint from iteration       1 to /tmp/qwen3_mcore_test
INFO:root:Done!
```

这说明最终方案不是“理论上可行”，而是已经在真实 pod 中跑通了转换闭环。

## 这次顺手踩到的环境坑

### 1. Ascend 环境脚本和 `set -u` 不兼容

会看到类似：

```text
/usr/local/Ascend/nnal/atb/set_env.sh: line 43: ZSH_VERSION: unbound variable
```

或者：

```text
/usr/local/Ascend/cann/set_env.sh: line 48: PYTHONPATH: unbound variable
```

所以 source Ascend 环境脚本时必须先：

```bash
set +u
source ...
set -u
```

wrapper 里已经按这个方式处理。

### 2. `protobuf 6.33.6` 是潜在风险，但不是这次主因

调试过程中确实见过 `mindspore/train/checkpoint_pb2.py` 的 descriptor 报错。

但这个问题只在特定混合导入路径下触发，并不是当前 HF->MCore 转换失败的主因。当前真正阻塞转换的是前面那 5 个兼容点。

## 训练阶段新增问题与修复

HF -> MCore 转换打通后，notebook 第 6 步继续训练时，又暴露出两个新的运行时兼容问题。

### 1. DataCollator 在首个 batch 直接报 “PyTorch is not installed”

训练日志中的首个致命错误是：

```text
ImportError: Unable to convert output to PyTorch tensors format, PyTorch is not installed.
```

这一层不是缺系统 `torch`，而是：

- `mindspeed_llm/legacy/data/data_samplers.py` 对指令数据固定使用
  `DataCollatorForSeq2Seq(..., return_tensors='pt')`
- MindSpeed vendored `transformers/utils/import_utils.py` 仍然用
  `importlib.metadata.version("msadapter")` 充当 torch 版本探测
- pod 里 `msadapter` distribution version 是 `0.7.0`
- 所以 vendored `transformers` 又一次把 `_torch_available` 判成了 `False`

也就是说，训练阶段和转换阶段命中了同一类根因，只是触发点从 checkpoint loader 换成了 HF data collator。

### 2. 修完 DataCollator 后，首个 backward 又炸在 `ones_like(memory_format=...)`

把 `msadapter` 版本探测补齐后，训练可以继续推进到真正的前向和反向。

新的失败点变成了：

```text
TypeError: ones_like_ext() got an unexpected keyword argument 'memory_format'
```

具体调用点在：

```text
megatron/core/pipeline_parallel/schedules.py
```

MindSpeed/Megatron 的这段代码调用了：

```python
mindspore.mint.ones_like(output, memory_format=msadapter.preserve_format)
```

但当前 pod 里的 `mindspore.mint.ones_like` 签名只有：

```text
(input, *, dtype=None)
```

所以这不是模型配置问题，而是当前 MindSpore runtime API 与这段移植代码之间的直接不兼容。

## 训练阶段最终修法

最初为了快速验证，我们新增过一个受控 runtime wrapper：

- `jupyter/ascend/mindspore/ubi9-python-3.12/mindspeed_msadapter_runtime_compat.sh`

它负责两件事：

1. 把 `importlib.metadata.version("msadapter")` 补成真实 `torch.__version__`
2. 当调用 `mindspore.mint.ones_like(..., memory_format=...)` 时，改走兼容的 `mindspore.ops.ones_like(...)`

但仅靠 wrapper 仍然要求用户“知道应该走哪条命令”，对通用镜像不够友好。

因此后续又把同一层兼容逻辑下沉进镜像默认 Python 启动流程：

- `jupyter/ascend/mindspore/ubi9-python-3.12/odh_mindspore_msadapter_compat.py`
- `jupyter/ascend/mindspore/ubi9-python-3.12/odh_mindspore_msadapter_compat.pth`

当前行为是：

- 直接运行官方 `convert_ckpt.py` 时，会自动识别 `hf + mindspore + safetensors` 场景，先准备 `.bin` 镜像，再补齐必要的 vendored `transformers` 兼容补丁
- 直接运行 `msrun posttrain_gpt.py` 等训练入口时，会自动带上 `msadapter` 版本探测和 `ones_like(memory_format=...)` 兼容

也就是说，用户按官方入口跑时已经不需要显式知道 runtime wrapper 的存在。

## 训练阶段 pod 验证结果

在 `2026-04-14`，对真实 pod：

- namespace: `zgsu-global-ns1`
- pod: `ws-mindspore-wzsk4-0`
- container: `main`

做了 2 卡、1 step 的真实 `msrun` 验证。

最终结果：

```text
iteration        1/       1
successfully saved checkpoint from iteration       1 to /tmp/qwen3_train_probe
```

这说明：

- 原始的 `DataCollatorForSeq2Seq -> PyTorch is not installed` 已被消除
- 后续的 `ones_like(memory_format=...)` backward 兼容问题也已被消除
- notebook 第 6 步的训练主路径在 pod 中已经真实跑通，而不是只停留在静态分析

## 本次仓库改动

### 新增

- `jupyter/ascend/mindspore/ubi9-python-3.12/hf_to_mcore_msadapter_compat.sh`
- `jupyter/ascend/mindspore/ubi9-python-3.12/mindspeed_msadapter_runtime_compat.sh`
- `jupyter/ascend/mindspore/ubi9-python-3.12/odh_mindspore_msadapter_compat.py`
- `jupyter/ascend/mindspore/ubi9-python-3.12/odh_mindspore_msadapter_compat.pth`

### 更新

- `jupyter/ascend/mindspore/ubi9-python-3.12/Containerfile.cann`
  - 把两个 wrapper 复制进镜像 `/opt/app-root/bin/`
  - 把默认 compat 模块安装进 `/opt/app-root/lib/python3.12/site-packages/`

- `jupyter/ascend/mindspore/ubi9-python-3.12/qwen3_0.6b_finetune_verify.ipynb`
  - notebook 第 4 步改回调用官方 `convert_ckpt.py`
  - notebook 第 6 步改回直接启动 `msrun`
  - 由镜像默认 compat 层兜底，不再要求 notebook 显式调用 wrapper

## 后续使用方式

### 直接在镜像里用 notebook

重新 build 新镜像后，直接运行 notebook 即可，转换和训练都会自动命中镜像内置 compat 逻辑。

### 手工调用

也可以继续在 pod 内直接执行官方入口：

```bash
cd /opt/app-root/share/MindSpeed-Core-MS/MindSpeed-LLM
python ./mindspeed_llm/mindspore/convert_ckpt.py \
  --use-mcore-models \
  --model-type GPT \
  --load-model-type hf \
  --save-model-type mg \
  --target-tensor-parallel-size 1 \
  --target-pipeline-parallel-size 1 \
  --load-dir /opt/app-root/src/models/Qwen3-0.6B \
  --save-dir /opt/app-root/src/Qwen3-0.6B-work-dir/model_weights/qwen3_mcore_tp1_pp1 \
  --tokenizer-model /opt/app-root/src/models/Qwen3-0.6B/tokenizer.json \
  --params-dtype bf16 \
  --model-type-hf qwen3 \
  --ai-framework mindspore \
  --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec
```

旧 wrapper 仍然保留，主要用于兼容已有 notebook 或手工脚本：

```bash
/opt/app-root/bin/hf_to_mcore_msadapter_compat.sh \
  --use-mcore-models \
  --model-type GPT \
  --load-model-type hf \
  --save-model-type mg \
  --target-tensor-parallel-size 1 \
  --target-pipeline-parallel-size 1 \
  --load-dir /opt/app-root/src/models/Qwen3-0.6B \
  --save-dir /opt/app-root/src/Qwen3-0.6B-work-dir/model_weights/qwen3_mcore_tp1_pp1 \
  --tokenizer-model /opt/app-root/src/models/Qwen3-0.6B/tokenizer.json \
  --params-dtype bf16 \
  --model-type-hf qwen3 \
  --ai-framework mindspore \
  --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec
```

## 最终结论

这次问题的本质不是“镜像里缺 torch”，而是：

- MindSpeed vendored HF loader
- `msadapter` 的 torch 代理行为
- HF safetensors / bin 加载路径
- `tie_weights()` 的 Parameter 绑定方式
- HF data collator 的 `return_tensors='pt'` 路径
- MindSpore `ones_like` 与 Megatron backward helper 的 API 差异

这些层叠在一起形成的兼容性断裂。

最终修法不是继续在 notebook 里补丁，而是：

> 把 safetensors 预转 `.bin`、vendored `transformers` 补丁和 runtime 兼容层都下沉到镜像默认入口，让用户继续走官方 `convert_ckpt.py` / `msrun` 命令即可。

这个方案已经在真实 pod 中验证通过。
