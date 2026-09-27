# MNIST Flow Matching：从噪声生成手写数字

一个可训练、可采样的 PyTorch 入门项目。模型学习随时间变化的**速度场**，将标准高斯噪声连续地推向 MNIST 手写数字图像。项目使用直线路径的 Conditional Flow Matching（CFM）目标和小型 U-Net，训练时不必反向传播穿过 ODE 求解器；生成时可选 Euler 或 Heun 积分。

> **任务边界：**这是无条件图像**生成**，不是数字分类。采样得到的是形似 MNIST 的图像，并不保证指定的数字类别。仓库不包含预训练权重，也没有宣称尚未运行的训练结果或生成质量指标。

## 快速开始

准备 Python 3.10+。以下命令以 macOS/Linux 的终端为例；Windows PowerShell 用 `.venv\Scripts\Activate.ps1` 激活虚拟环境。

```bash
git clone https://github.com/Zmw2006/mnist-flow-matching.git
cd mnist-flow-matching
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m pytest -q
```

安装 PyTorch 遇到 CUDA 平台差异时，先按 [PyTorch 官方安装指引](https://pytorch.org/get-started/locally/) 安装适合本机的 `torch` 与 `torchvision`，然后执行 `python -m pip install -e '.[test]'`。首次训练会自动下载 MNIST 到 `data/`，需要访问数据源。

先进行一次短流程检查（CPU 也能运行，实际耗时因机器而异）：

```bash
python -m mnist_flow.train --epochs 1 --batch-size 128 --channels 8 --time-dim 32 --preview-steps 10 --device cpu --output-dir outputs/smoke
python -m mnist_flow.sample --checkpoint outputs/smoke/checkpoint.pt --output outputs/smoke/samples.png --num-samples 16 --steps 10 --device cpu
```

短流程的目的仅是确认下载、训练、保存和生成都能运行，**不能**期待一轮小模型给出清晰手写数字。正式训练建议从默认模型开始：

```bash
python -m mnist_flow.train --epochs 20 --batch-size 128 --device auto --output-dir outputs/run1
python -m mnist_flow.sample --checkpoint outputs/run1/checkpoint.pt --output outputs/run1/samples.png --num-samples 64 --steps 100 --method heun
```

如有 GPU，`--device auto` 优先选择 CUDA，其次 MPS，最后 CPU。默认训练的计算量不小；可根据显存调整 `--batch-size`。运行 `python -m mnist_flow.train --help` 或 `python -m mnist_flow.sample --help` 查看所有选项。安装后也可以使用 `mnist-flow-train` 和 `mnist-flow-sample` 命令。

## 原理：网络到底学什么？

将原图 `x₁` 的像素从 `[0,1]` 映射到 `[-1,1]`，同时从标准正态分布采样同形状的噪声 `x₀`。对每张训练图随机采样 `t ~ Uniform(0,1)`，构造

```text
xₜ = (1 − t) x₀ + t x₁
目标速度 uₜ = x₁ − x₀
损失 L = E[ ||vθ(xₜ, t) − uₜ||² ]
```

`vθ` 是 U-Net 预测的速度场。它只看到插值图像 `xₜ` 和时间 `t`，不知道单独采样到的 `x₀` 和 `x₁`；用均方误差拟合配对路径速度。这里的“conditional”指**给定一对噪声与图像时的路径**，并非给定数字类别。每次迭代都会重新采样噪声和时间。

采样时从新噪声 `z ~ N(0,I)` 出发，求解常微分方程

```text
dx/dt = vθ(x,t),     x(0) = z,     t: 0 → 1
```

Euler 每步用当前斜率更新；Heun 先预测下一位置，再平均两处斜率，因此每步需要两次网络前向计算。最后只在保存图片时将像素裁剪到 `[-1,1]` 并映射到 `[0,1]`；积分过程中不裁剪状态。

## 项目结构

| 路径 | 用途 |
| --- | --- |
| `src/mnist_flow/model.py` | 时间嵌入、残差模块与 28×28 U-Net 速度场 |
| `src/mnist_flow/flow.py` | CFM 损失和 Euler/Heun 求解器 |
| `src/mnist_flow/train.py` | MNIST 下载、训练、EMA、预览及断点保存 |
| `src/mnist_flow/sample.py` | 加载权重并保存生成图片网格 |
| `src/mnist_flow/utils.py` | 设备、种子和原子化保存辅助函数 |
| `tests/test_core.py` | 梯度、输出维度和 ODE 解的 CPU 测试 |
| `.github/workflows/tests.yml` | 在 GitHub Actions 上运行单元测试 |

## 输出、断点续训和复现

每轮训练会在输出目录写入 `epoch_001.png` 等 4×4 预览图，更新 `checkpoint.pt`。checkpoint 保存当前模型、指数移动平均（EMA）模型、优化器状态、已完成的轮数、模型结构参数与 CPU/CUDA 随机状态。独立采样默认使用 EMA 权重。预览图片仅用于观察趋势；请不要用其代替定量评测。

训练中断后，例如已经完成 10 轮、目标总轮数为 20：

```bash
python -m mnist_flow.train --resume outputs/run1/checkpoint.pt --output-dir outputs/run1 --epochs 20 --batch-size 128 --channels 32 --time-dim 128
```

`--epochs` 表示**累计总轮数**，续训时 `--channels` 和 `--time-dim` 必须与保存的结构一致。可以用 `--lr` 改变续训学习率。`--seed` 固定常见的随机源，但多进程数据加载、不同设备与底层算子仍可能导致结果不完全逐位一致。**仅加载自己信任的 `.pt` 文件**：训练用的优化器 checkpoint 使用 PyTorch 的对象反序列化机制。

`data/`、`outputs/` 与权重文件已通过 `.gitignore` 排除，不会意外上传数据或大型模型。完整训练和下载 MNIST **不会**在 CI 中执行；CI 仅检查模型、损失与积分逻辑是否正常。

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--epochs` | 20 | 训练的累计总轮数 |
| `--batch-size` | 128 | 每步图片数，显存不足时减小 |
| `--lr` | 0.0002 | AdamW 学习率 |
| `--channels` | 32 | U-Net 初始通道数，必须是 8 的倍数 |
| `--time-dim` | 128 | 时间嵌入维数，至少 4 且为偶数 |
| `--ema-decay` | 0.999 | EMA 衰减系数 |
| `--preview-steps` | 50 | 每轮预览图的 Heun 积分步数 |
| 采样 `--steps` | 100 | ODE 积分步数；增大更慢 |
| 采样 `--method` | heun | `euler` 或 `heun` |
| 采样 `--num-samples` | 64 | 输出网格的图像数量 |

## 可继续扩展

加入类别条件嵌入可以按 0–9 指定数字；加入独立评估流程可以检验生成数字的质量与多样性。损失下降只能说明模型更接近训练目标，本身不能作为生成图像质量的保证。
