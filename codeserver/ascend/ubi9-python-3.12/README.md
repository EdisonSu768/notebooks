# codeserver/ascend/ubi9-python-3.12

Ascend/ARM64 variant of the code-server image with Python 3.12 on UBI 9.

这个目录已经精简为只产出 Ascend/ARM64 镜像的版本：
- 只保留 `Containerfile.cpu` 这一条构建路径
- 只支持 `linux/arm64`
- 构建时使用仓库根目录作为 build context
- 直接下载官方 `code-server` arm64 release tarball
- Python 依赖通过 `uv.lock.d/pylock.cpu.toml` 安装

## Build

```bash
podman build \
  -f codeserver/ascend/ubi9-python-3.12/Containerfile.cpu \
  --platform linux/arm64 \
  -t codeserver-ascend:test \
  --build-arg BASE_IMAGE=quay.io/opendatahub/odh-base-image-cpu-py312-c9s:latest \
  --build-arg PYLOCK_FLAVOR=cpu \
  .
```

也可以用 docker/buildx（同样只构建 arm64）：

```bash
docker buildx build \
  -f codeserver/ascend/ubi9-python-3.12/Containerfile.cpu \
  --platform linux/arm64 \
  -t codeserver-ascend:test \
  --build-arg BASE_IMAGE=quay.io/opendatahub/odh-base-image-cpu-py312-c9s:latest \
  --build-arg PYLOCK_FLAVOR=cpu \
  .
```

## Notes

- 当前实现里 “Ascend” 主要表示 ARM64/Ascend 变体命名，不包含专门的 CANN/NPU 运行时。
- 该目录现在按 arm64-only 维护，不再保留多架构下载分支。
- 如果后续要重新接回源码构建链路，再考虑用 submodule + 流水线 checkout 的方式，而不是把大目录直接提交进 git。