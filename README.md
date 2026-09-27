# MNIST Flow Matching：从噪声生成手写数字

[![Python tests](https://github.com/Zmw2006/mnist-flow-matching/actions/workflows/tests.yml/badge.svg)](https://github.com/Zmw2006/mnist-flow-matching/actions/workflows/tests.yml)

一个可训练、可采样的 PyTorch 入门项目。模型学习随时间变化的**速度场**，将标准高斯噪声连续地推向 MNIST 手写数字图像。项目使用直线路径的 Conditional Flow Matching（CFM）目标和小型 U-Net，训练时不必反向传播穿过 ODE 求解器；生成时可选 Euler 或 Heun 积分。

> **任务边界：**这是无条件图像**生成**，不是数字分类。采样得到的是形似 MNIST 的图像，并不保证指定的数字类别。仓库源码不包含预训练权重；README 也不把测试通过说成模型已经训练完成。

## 阅读路线

1. 想立即试运行：看[快速开始](#快速开始)。
2. 想理解算法：看[从公式到代码](#从公式到代码)。
3. 想在 GitHub 上训练和下载模型：看[使用-github-actions-训练](#使用-github-actions-训练)。
4. 想判断训练是否有效：看[结果怎么看](#结果怎么看)。

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
python -m mnist_flow.train --epochs 1 --batch-size 128 --channels 8 --time-dim 32 --max-train-samples 512 --max-val-samples 128 --preview-steps 10 --device cpu --output-dir outputs/smoke
python -m mnist_flow.sample --checkpoint outputs/smoke/best.pt --output outputs/smoke/samples.png --num-samples 16 --steps 10 --device cpu
```

短流程的目的仅是确认下载、训练、保存和生成都能运行，**不能**期待一轮小模型给出清晰手写数字。正式训练建议从默认模型开始：

```bash
python -m mnist_flow.train --epochs 20 --batch-size 128 --device auto --output-dir outputs/run1
python -m mnist_flow.sample --checkpoint outputs/run1/best.pt --output outputs/run1/samples.png --num-samples 64 --steps 100 --method heun
python -m mnist_flow.evaluate --checkpoint outputs/run1/best.pt --output outputs/run1/test_metrics.json
```

如有 GPU，`--device auto` 优先选择 CUDA，其次 MPS，最后 CPU。默认训练的计算量不小；可根据显存调整 `--batch-size`。运行 `python -m mnist_flow.train --help`、`python -m mnist_flow.sample --help`、`python -m mnist_flow.evaluate --help` 查看所有选项。安装后也可以使用 `mnist-flow-train`、`mnist-flow-sample` 和 `mnist-flow-evaluate` 命令。

第一次运行 `train` 会下载 MNIST；如果下载失败，检查本机网络或将通过 torchvision 获取的 MNIST 文件放入 `--data-dir` 对应目录，再运行命令。

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

## 从公式到代码

假设一张图片里某个像素原本是 `x₁=0.8`，抽到的噪声是 `x₀=-0.2`。当 `t=0.25` 时，路径上的像素是 `xₜ=0.75×(-0.2)+0.25×0.8=0.05`，目标速度是 `0.8-(-0.2)=1.0`。训练对整张 28×28 图做同样的计算，再让网络预测每个像素的速度。这里的线性插值描述的是**配对样本的训练路径**；生成时网络看到的是当前图像和时间，逐步推断该走向哪里。

| 实现位置 | 做的事情 |
| --- | --- |
| `flow.py:flow_matching_loss` | 每张图片各抽一个噪声图和时间，构造 `xₜ` 并计算速度预测的均方误差 |
| `model.py:TimeEmbedding` | 把标量时间变成网络能够使用的向量 |
| `model.py:VelocityUNet` | 用残差卷积、下采样、上采样和跳接预测 28×28 速度图 |
| `train.py:main` | 更新网络参数，并以指数移动平均维护 EMA 参数 |
| `flow.py:sample` | 从噪声出发，按 `dx/dt=vθ(x,t)` 的预测轨迹积分 |

EMA 模型可以理解为训练过程中对参数做平滑：每次更新后取 `EMA ← decay×EMA + (1−decay)×当前参数`。预览、验证与独立采样默认使用 EMA 参数。网络宽度可通过 `--channels` 调整；缩小宽度适合检查流程，但通常会降低生成能力。

### 为什么不直接把训练损失当作生成质量？

CFM 均方误差测的是速度预测误差。采样图像是否清晰、数字是否多样，还受到网络容量、训练时长与 ODE 步数等影响。因此项目同时保存数值记录和图片供检查，但**没有**把 CFM MSE 称为准确率、FID 或人眼质量分数。

### 数据集如何划分？

| 用途 | 数据来源 | 默认规模 | 是否用于参数更新 |
| --- | --- | ---: | --- |
| 训练 | MNIST 官方训练集前 90% | 54,000 | 是 |
| 验证 | MNIST 官方训练集后 10% | 取前 1,000 张；可改为全部 6,000 张 | 否，用于选择 `best.pt` |
| 独立评估 | MNIST 官方测试集 | 10,000 | 否；仅在单独运行 `evaluate` 时使用 |

这个顺序切分固定且易于复现。`--max-train-samples` 和 `--max-val-samples` 分别从对应部分取前 N 张；设置为 `0` 表示使用对应部分的全部图像。用于快速检查的小样本运行会让估计更不稳定，不能当作正式实验结论。

## 项目结构

| 路径 | 用途 |
| --- | --- |
| `src/mnist_flow/model.py` | 时间嵌入、残差模块与 28×28 U-Net 速度场 |
| `src/mnist_flow/flow.py` | CFM 损失和 Euler/Heun 求解器 |
| `src/mnist_flow/train.py` | MNIST 下载、训练、EMA、预览及断点保存 |
| `src/mnist_flow/sample.py` | 加载权重并保存生成图片网格 |
| `src/mnist_flow/evaluate.py` | 用保留的测试集评估 CFM 均方误差并输出 JSON |
| `src/mnist_flow/utils.py` | 设备、种子和原子化保存辅助函数 |
| `tests/test_core.py` | 梯度、ODE 解、训练续跑和生成命令的 CPU 测试 |
| `.github/workflows/tests.yml` | 在 GitHub Actions 上运行单元测试 |
| `.github/workflows/train.yml` | 手动启动 MNIST 训练，并上传生成的模型和结果 |

## 输出、断点续训和复现

每轮训练会在输出目录写入 `epoch_001.png` 等预览图，并更新 `checkpoint.pt`。同一训练运行内的预览图使用**同一组起始噪声**，因此更容易看出图像随轮次的变化。训练还会把每轮数值写入 `metrics.csv`，当验证误差创下新低时保存 `best.pt`。checkpoint 包含当前模型、EMA 模型、优化器、已完成轮数、模型结构参数以及 CPU/CUDA 和数据打乱随机状态；独立采样默认使用 EMA 权重。

| 文件 | 作用 | 什么时候查看 |
| --- | --- | --- |
| `metrics.csv` | `epoch,train_mse,val_ema_mse`；逐轮训练与验证 CFM MSE | 看误差是否总体改善、是否过拟合 |
| `epoch_001.png` 等 | 每轮用固定起始噪声生成的图片网格 | 观察图片轮廓与多样性 |
| `checkpoint.pt` | 最近一轮的完整断点，含优化器 | 继续训练 |
| `best.pt` | 验证 CFM MSE 最低时的完整断点 | 优先用于采样和独立评估 |
| `samples.png` | 独立采样命令生成的图片网格 | 检查最终生成效果 |
| `test_metrics.json` | 独立测试集 CFM MSE、测试样本数和对应轮次 | 报告保留集上的数值诊断 |

一个 `metrics.csv` 文件的格式示例（下面是**字段说明示意值，不是本项目实测结果**）：

```csv
epoch,train_mse,val_ema_mse
1,1.23456789,1.34567890
2,1.12345678,1.23456789
```

验证集使用固定的时间与噪声抽样设置，重复训练时指标具有可比性；批量大小、数据上限和种子在续训时需要保持一致。训练与验证目标上的 MSE 仍不能独立证明生成数字清晰或类别均衡。

训练中断后，例如已经完成 10 轮、目标总轮数为 20：

```bash
python -m mnist_flow.train --resume outputs/run1/checkpoint.pt --output-dir outputs/run1 --epochs 20 --batch-size 128 --channels 32 --time-dim 128
```

`--epochs` 表示**累计总轮数**，续训时 `--channels`、`--time-dim`、`--seed`、`--batch-size` 与数据上限必须与原运行一致；`--output-dir` 必须是 checkpoint 所在目录，以保留完整指标和最佳模型。可以用 `--lr` 改变续训学习率，但这样会改变实验条件。`--seed` 固定常见的随机源，但多进程数据加载、不同设备与底层算子仍可能导致结果不完全逐位一致。加载 `.pt` 时使用 PyTorch 的 `weights_only=True` 安全加载模式。

`data/`、`outputs/` 与权重文件已通过 `.gitignore` 排除，不会意外上传数据或大型模型。自动触发的 Python tests **不会**下载和完整训练 MNIST；训练工作流必须手动启动。

## 使用 GitHub Actions 训练

不想在本机安装环境时，可以通过仓库的 [Train and sample MNIST 工作流](https://github.com/Zmw2006/mnist-flow-matching/actions/workflows/train.yml) 运行一次 CPU 训练：

1. 打开链接，在右侧点击 **Run workflow**，选择 `main`。
2. 保持默认值可以做一次小规模流程验证：1 轮、8 通道、1,024 张训练图片、256 张验证图片。**这只是流程检查，不能期待清晰的最终样本。**
3. 需要更大训练时，调高 `epochs` 和 `channels`，把 `max_train_samples`、`max_val_samples` 设置为 `0` 可使用全部 54,000/6,000 张划分数据。GitHub 托管的运行器是 CPU，工作流设置了 180 分钟超时；完整 20 轮可能无法在此限制内完成，建议使用本机或云端 GPU。
4. 点击运行记录，待状态为绿色成功后，在页面下方 **Artifacts** 下载 `mnist-flow-model-and-results`。其中包含 `best.pt`、`checkpoint.pt`、`metrics.csv`、预览图、`samples.png` 和 `test_metrics.json`。Artifact 设置了 7 天保留期；请及时下载。

工作流会从公开数据源下载 MNIST，因此下载失败会使任务失败；页面内可以展开对应步骤查看错误日志。无论是 Python tests 通过，还是训练工作流只跑完默认的一轮，都不能推断生成效果已经达到高质量。

## 结果怎么看

建议按这个顺序检查：

1. 看 `metrics.csv` 的 `train_mse` 是否随训练总体下降，再看 `val_ema_mse` 是否也改善。如果训练下降、验证长期变差，应检查训练轮数或实验设置。两列来自不同模型（当前权重与 EMA 权重），不要拿单轮两列的差值当作严格的泛化差距。
2. 按文件名顺序查看 `epoch_001.png`、`epoch_002.png` 等图，观察相同噪声的输出是否逐渐出现笔画、轮廓及多种数字。
3. 用 `best.pt` 生成更多图片、尝试不同的 `--seed`，检查是否反复出现同一种形状；图片边缘值已裁剪用于显示。
4. 通过 `python -m mnist_flow.evaluate --checkpoint outputs/run1/best.pt` 得到 `test_metrics.json`。这是官方**测试集的速度预测误差**，并非分类准确率或 FID；不要根据它断言视觉质量。

如果需要正式比较不同方法，先固定相同的数据划分、种子、网络大小、训练预算和采样步数，再加入单独的生成质量评估；这个入门项目没有内置 FID，因此不会制造无法复核的分数。

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
| `--preview-count` | 16 | 每轮预览图片数量 |
| `--max-train-samples` | 0 | `0` 为训练划分的全部 54,000 张 |
| `--max-val-samples` | 1,000 | 用于选最佳模型的验证张数；`0` 为全部 6,000 张 |
| `--num-workers` | 0 | 数据加载子进程数 |
| 采样 `--steps` | 100 | ODE 积分步数；增大更慢 |
| 采样 `--method` | heun | `euler` 或 `heun` |
| 采样 `--num-samples` | 64 | 输出网格的图像数量 |
| 评估 `--max-test-samples` | 0 | `0` 为官方测试集全部 10,000 张 |

### 常见问题

**问：运行几轮后图片仍像噪声，是否程序错了？** 首先核对下载、训练和保存步骤是否成功，然后检查训练轮数、训练样本数与图像预览。默认的 Actions 流程只有小模型和少量图片，目的就是验证链路；质量不佳并不意味着算法已被充分训练。

**问：为什么生成图像没有标签？** 模型没有把 0–9 的类别当作条件输入，因此不能要求它生成指定数字；MNIST 标签在训练时被忽略。

**问：保存的是 28×28 图片还是网格？** 网络生成的单张张量大小为 `[1,28,28]`。`samples.png` 将多张图按网格排版，网格图片的尺寸自然大于 28×28。

**问：`best.pt` 和 `checkpoint.pt` 应该用哪个？** 前者是当前验证 CFM MSE 最低的轮次，建议用来生成；后者是最近一轮，适合用 `--resume` 续训。模型质量还需结合图片检查。

## 可继续扩展

加入类别条件嵌入可以按 0–9 指定数字；加入独立评估流程可以检验生成数字的质量与多样性。损失下降只能说明模型更接近训练目标，本身不能作为生成图像质量的保证。
