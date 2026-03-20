# 制作 CANN 版本的 base-image

## 任务说明
- 参考 `/Users/szg/Github/odh/notebooks/base-images/cuda/13.0/c9s-python-3.12` 这个是怎么做的
- 在 `/Users/szg/Github/odh/notebooks/base-images/cann/8.5.0/c9s-python-3.12` 下创建 `Containerfile.cann` 文件
- `CANN` 安装命令见本文档，用 `wget` 方式获取并 `bash` 安装
- `Containerfile.cann` 只需要有 `arm` 架构
- `Containerfile.cann` 里关于 `python` 和其他一些依赖参考 `/Users/szg/Github/odh/notebooks/base-images/cuda/13.0/c9s-python-3.12` 安装了什么

## CANN 安装命令 (芯片型号 910B)
```bash
#groupadd HwHiAiUser
#useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash

#sudo yum makecache
#sudo yum install -y kernel-headers-$(uname -r) kernel-devel-$(uname -r)
#sudo curl https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo && yum makecache

#sudo yum install -y Ascend-cann-toolkit-8.5.0 --install-path=/usr/local/Ascend/cann
#sudo yum install -y Ascend-cann-910b-ops-8.5.0 --install-path=/usr/local/Ascend/ops


# NNAL
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-nnal_8.5.0_linux-aarch64.run
bash ./Ascend-cann-nnal_8.5.0_linux-aarch64.run --install

# toolkit
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-toolkit_8.5.0_linux-aarch64.run
bash ./Ascend-cann-toolkit_8.5.0_linux-aarch64.run --install

# ops
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.T63/Ascend-cann-910b-ops_8.5.0_linux-aarch64.run
bash ./Ascend-cann-910b-ops_8.5.0_linux-aarch64.run --install

# 安装完执行
ls -la /usr/local/Ascend/
```
