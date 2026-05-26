# 测试教程：钢琴/小提琴和声分离与简谱输出

本文档用于快速测试当前 demo：输入一段钢琴和小提琴和声音频，输出分离后的钢琴、小提琴音频，以及简谱分析结果。

## 1. 环境准备

如果系统命令行没有 `python`，使用 Codex 自带 Python：

```powershell
& 'C:\Users\91453\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' --version
```

本项目只依赖 NumPy 和 Python 标准库，不需要额外安装 PyTorch、librosa 或 soundfile。

## 2. 生成一段测试和声音频

先运行默认 demo，它会自动生成一段钢琴/小提琴合成和声，并完成一次分离和简谱分析：

```powershell
& 'C:\Users\91453\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' demo\piano_violin_separation_demo.py
```

成功后会看到 `PASS`，并生成这些文件：

- `outputs\piano_violin_demo\mixture.wav`：测试用和声音频
- `outputs\piano_violin_demo\piano_estimate.wav`：分离后的钢琴
- `outputs\piano_violin_demo\violin_estimate.wav`：分离后的小提琴
- `outputs\piano_violin_demo\jianpu.txt`：简谱文本
- `outputs\piano_violin_demo\jianpu.json`：结构化简谱数据
- `outputs\piano_violin_demo\staff.pdf`：由简谱/音高分析渲染出的五线谱 PDF
- `outputs\piano_violin_demo\metrics.json`：分离指标

## 3. 使用 UI 测试上传和输出目录

打开 UI：

```powershell
& 'C:\Users\91453\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' demo\piano_violin_separation_demo.py --ui
```

界面使用步骤：

1. 点击 `选择音频`。
2. 选择 `outputs\piano_violin_demo\mixture.wav`，或选择你自己的 WAV 文件。
3. 点击 `选择目录`，选择分离结果保存位置，例如 `outputs\ui_test`。
4. 点击 `开始分离并生成简谱`。
5. 等待进度条停止并弹出完成提示。

输出目录中会生成：

- `<输入文件名>_model_input_mono_8000hz.wav`
- `<输入文件名>_piano_estimate.wav`
- `<输入文件名>_violin_estimate.wav`
- `<输入文件名>_jianpu.txt`
- `<输入文件名>_jianpu.json`
- `<输入文件名>_staff.pdf`
- `<输入文件名>_separation_result.json`

## 4. 命令行方式测试自定义音频

不打开 UI，也可以直接指定输入音频和输出目录：

```powershell
& 'C:\Users\91453\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' demo\piano_violin_separation_demo.py --input outputs\piano_violin_demo\mixture.wav --output outputs\cli_test
```

## 5. 查看简谱结果

打开 `jianpu.txt`，内容类似：

```text
钢琴/小提琴分离与简谱分析

调性: B minor
估计速度: 75.0 BPM
量化网格: eighth-note grid

钢琴简谱:
[#1,, 4,, 6,]x2 ...

小提琴简谱:
[6' 1']x2 ...
```

符号说明：

- `0` 表示休止。
- `[1 3 5]` 表示同一个时间片检测到和声音。
- `'` 表示高八度，`,` 表示低八度。
- `x2` 表示同一音型连续重复 2 个量化时间片。

## 6. 查看五线谱 PDF

简谱输出后，程序会继续把检测到的音高序列渲染成两行五线谱 PDF：

- 上方谱表：Violin
- 下方谱表：Piano
- 每个量化时间片绘制一个音符、和声音或休止
- 每 8 个时间片绘制一个小节线

默认 demo 的 PDF 路径：

```text
outputs\piano_violin_demo\staff.pdf
```

UI 或命令行自定义输入的 PDF 路径：

```text
<输出目录>\<输入文件名>_staff.pdf
```

## 7. 注意事项

- 当前 demo 的输入音频格式建议使用 WAV。
- 多声道音频会自动转成单声道。
- 输入音频会重采样到模型使用的 `8000 Hz`，输出也会保存为 `8000 Hz` WAV。
- 这是最小可运行复现，适合验证论文流程，不等同于商业级通用分离模型。
