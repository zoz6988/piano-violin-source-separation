# Piano/Violin Source Separation Mini Demo

This repository contains a minimal, runnable reproduction of the main pipeline
from `Improving Music Source Separation Based on Deep Neural Networks Through
Data Augmentation and Network Blending`.

The original paper targets DSD100 music stems with large FNN/BLSTM systems. This
demo keeps the testable core idea in a small Python script:

- synthesize piano and violin harmony stems;
- create random augmented training mixtures;
- train one small feed-forward network per source on STFT magnitudes;
- train two context variants and blend their raw magnitude estimates with
  `lambda = 0.25`;
- apply Wiener filtering so the separated estimates add up consistently with the
  mixture;
- write WAV files and check SDR improvement.

## Run

Use the bundled Codex Python if `python` is not on PATH:

```powershell
& 'C:\Users\91453\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' demo\piano_violin_separation_demo.py
```

Or, with a normal Python that has NumPy:

```powershell
python demo\piano_violin_separation_demo.py
```

Open the upload-style desktop UI:

```powershell
& 'C:\Users\91453\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' demo\piano_violin_separation_demo.py --ui
```

The UI lets you choose a WAV file and an output folder. It then writes separated
`*_piano_estimate.wav` and `*_violin_estimate.wav` files to that folder.

You can also separate a WAV directly from the command line:

```powershell
& 'C:\Users\91453\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' demo\piano_violin_separation_demo.py --input path\to\harmony.wav --output outputs\my_separation
```

The script writes:

- `outputs/piano_violin_demo/mixture.wav`
- `outputs/piano_violin_demo/piano_estimate.wav`
- `outputs/piano_violin_demo/violin_estimate.wav`
- reference stems and `metrics.json`

The run passes when both separated sources improve over the raw mixture baseline
by at least `1.0 dB`.

## Notes

This is not a full DSD100 reproduction. It is a compact proof-of-concept for the
paper's separation recipe, specialized to synthetic piano/violin harmony so it
can be trained and tested locally without external datasets or neural-network
frameworks.
太棒了，你已经成功进入了 `E:\Codex20260517\music_instrument\demo` 文件夹！

现在的报错“系统找不到指定的路径”，是因为你现在的命令里多写了一个 `demo\`。因为你**已经在** `demo` 文件夹里了，直接叫文件名就行。

### 最终正确的运行命令

请直接输入这一行：

```bash
python piano_violin_separation_demo.py --ui

```

---

### 为什么之前的命令不行？

你之前的命令是：`demo\piano_violin_separation_demo.py`
这相当于你在“客厅”里找“客厅里的电视机”，电脑会去 `demo` 文件夹下面再找一个叫 `demo` 的子文件夹，结果发现没有，所以报错了。

**只要输入上面那行简单的命令，界面就会弹出来啦！** 如果运行后提示“找不到 python”，记得确认一下你的 Python 环境是否安装好这段代码是一个基于 **NumPy** 实现的紧凑型钢琴/小提琴音频源分离（Source Separation）演示程序。它的算法架构启发自深度学习音频分离领域的经典论文 *Uhlich et al. 2017*，但为了摆脱对 PyTorch、TensorFlow 或 Librosa 等重量级第三方库的依赖，它完全使用底层的数学公式和 NumPy 矩阵运算手写了整个管线。

为了让你更轻松地理解它，我为你整理了这份结构化的核心功能注解：

---

## 核心算法管线解析

这个脚本实现的是一个典型的 **基于掩膜（Masking）与维纳滤波（Wiener Filtering）** 的时频域音频分离系统。它的完整工作流如下：

### 1. 时频域转换（STFT / iSTFT）

音频在时域（波形图）中是交织在一起的，很难直接分离。代码通过手写的短时傅里叶变换 `stft()` 将音频转换到时频域（频谱图），并在分离完成后通过逆短时傅里叶变换 `istft()` 将其还原为听得见的音频波形。

### 2. 局部上下文特征提取（Context Features）

单看某一个时间帧的频谱很难准确判断属于哪种乐器。

* `context_features(mag, context)` 函数通过在当前帧的前后各拼接 `context` 帧的幅度谱，来为神经网络提供局部的上下文信息，帮助模型捕捉声音的动态变化趋势。

### 3. 轻量化前馈神经网络（TinyFNN）

代码中完全用 NumPy 手写了一个带有单隐藏层的神经网络：

* **前向传播**：输入对数压缩后的频谱特征 $\log(1 + x)$，经过标准化，通过隐藏层（带 `ReLU` 激活函数 `np.maximum(x @ w1 + b1, 0.0)`），最后经过 `Sigmoid` 激活函数生成一个范围在 $0$ 到 $1$ 之间的 **软掩膜（Soft Mask）**。
* **反向传播与训练**：在 `train_fnn()` 中，利用链式法则手写了梯度计算，并使用随机梯度下降（SGD）来更新权重 $W$ 和偏置 $b$。训练的目标是让模型学会根据混合频谱预测目标乐器所占的能量比例。

### 4. 模型融合（Ensemble Blending）

为了提高分离的鲁棒性，系统同时训练了两个结构稍有不同的网络：

* 一个网络查看较窄的上下文（`context=1`），另一个查看较宽的上下文（`context=2`）。
* 在 `Separator.separate()` 中，通过 `blend` 参数（默认 `0.25`）将这两个网络的输出进行线性加权融合。

### 5. 维纳滤波后处理（Wiener Separation）

神经网络直接预测的幅度谱可能存在伪影。

* `wiener_separate()` 函数利用网络预测的钢琴和小提琴功率谱密度（PSD），在混合信号的原始复数频谱上计算比值：

$$ \text{Gain}_{\text{piano}} = \frac{\text{PSD}_{\text{piano}}}{\text{PSD}_{\text{piano}} + \text{PSD}_{\text{violin}} + \epsilon} $$


* 这种方法能够完美保留原始混合音频的**相位信息**，极大地减少了声音失真。

---

## 代码模块与关键函数说明

| 函数 / 类名 | 功能描述 | 核心数学/实现细节 |
| --- | --- | --- |
| `piano_note` / `violin_note` | **合成乐器音色** | 钢琴使用指数衰减包络和轻微泛音泛音列；小提琴引入了低频 LFO 调制的**颤音（Vibrato）**和弓弦摩擦噪声。 |
| `collect_training_data` | **动态生成训练集** | 随机组合不同的和弦音轨，计算它们各自的短时傅里叶幅度，并生成监督学习所需的理想掩膜（Target）。 |
| `sdr` | **性能评估指标** | 计算**信号失真比（Signal-to-Distortion Ratio, SDR）**。通过比较参考原声与分离估计声的均方误差，来以分贝（dB）衡量分离质量。 |
| `run_ui` | **图形用户界面** | 基于 Python 内置的 `tkinter` 构建。为了防止模型训练阻塞界面导致卡死，它使用 `threading.Thread` 将计算任务放到了后台线程。 |

---

## 运行模式

该脚本设计了两种运行模式，可以通过命令行参数进行切换：

1. **自测试模式（默认）**：
直接运行 `python demo.py`。它会自己合成一段 4 秒左右的钢琴小提琴合奏音频作为测试集，就地训练模型并进行分离，最后将原声、混合声、分离声以及评估指标（SDR 提升进度）保存到 `outputs/` 目录下。
2. **桌面 UI 交互模式**：
运行 `python demo.py --ui`。会弹出一个精简的中文界面，允许用户选择自己本地的 `.wav` 混合音频文件，并一键完成分离。

> **友情提示**：为了确保能跑出结果，代码默认将采样率（`--sr`）降到了 `8000Hz`，主要是为了让纯 NumPy 手写的神经网络在没有 GPU 加速的情况下也能在几秒钟内快速训练并运行完毕。。



