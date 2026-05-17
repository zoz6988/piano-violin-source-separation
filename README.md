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

**只要输入上面那行简单的命令，界面就会弹出来啦！** 如果运行后提示“找不到 python”，记得确认一下你的 Python 环境是否安装好。

