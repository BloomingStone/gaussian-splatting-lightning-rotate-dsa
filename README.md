# Gaussian Splatting

## 使用方法

此项目依赖 [pixi](https://pixi.sh/) 配置python环境。系统环境为(仅供参考)

```txt
OS: Ubuntu 22.04.3 LTS (Jammy Jellyfish) x86_64
Kernel: Linux 6.8.0-136-generic
GPU: NVIDIA GeForce RTX 3090
cuda-runtime:
    NVIDIA-SMI 580.173.02, Driver
    Version: 580.173.02
    CUDA Version: 13.0
cuda-toolkit(nvcc): release 12.0, V12.0.76
```

解压项目后，然后进入到项目目录中，使用 `pixi install --frozen` 从 `pixi.lock` 安装环境。该方法确保环境的一致性，避免因依赖版本变化引起的问题。

如果安装失败，可能是由于某些依赖包在当前环境中无法安装。可以尝试以下步骤：

+ 使用 `pixi install` 重新安装，这将允许pixi根据当前环境自动调整依赖版本。
+ 如果仍然无法安装，检查错误日志，确定是哪个包导致安装失败，并修改 `pixi.toml` 中对应包的版本

安装完成后，可以使用 `pixi shell` 进入环境，或者使用 `pixi run <command>` 直接运行命令，例如 `pixi run python XXX.py` 可以运行python脚本。

随项目附带了一例测试用数据，可以使用以下命令运行：

``` bash
CUDA_VISIBLE_DEVICES=3 pixi run python main.py fit --config configs/gen_4d_output_all/motion.yaml --data.path data/test_data -n recon_test -v v0
```