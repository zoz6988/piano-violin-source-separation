# 完整工作流介绍：钢琴/小提琴和声分离、简谱分析与五线谱 PDF 输出

本文档说明本项目从输入音频到输出结果的完整流程，并把每一步对应的论文原理、代码位置和使用示例标注清楚。

核心代码文件：

- `demo/piano_violin_separation_demo.py`
- 测试说明：`TESTING_README.md`
- 默认输出目录：`outputs/piano_violin_demo`

## 1. 总体流程

本项目实现的是一个最小可运行 demo，目标是：

1. 输入一段钢琴和小提琴和声音频。
2. 将和声音频分离为钢琴轨道和小提琴轨道。
3. 对分离后的音频做音高分析。
4. 输出简谱文本。
5. 根据音高分析结果渲染五线谱 PDF。

整体流程如下：

```text
输入 WAV 音频
  -> 读取与预处理
  -> 训练最小 FNN 分离模型
  -> STFT 频谱分析
  -> 钢琴/小提琴幅度谱估计
  -> FNN raw output blending
  -> Wiener mask 后处理
  -> 输出钢琴 WAV / 小提琴 WAV
  -> 音高、调性、节拍网格分析
  -> 输出简谱 TXT / JSON
  -> 输出五线谱 PDF
```

## 2. 输入与预处理

### 原理

音频源分离通常先把时域音频转换到统一格式，保证后续 STFT 和模型输入维度一致。本项目把输入音频统一为：

- 单声道
- `8000 Hz` 采样率
- 浮点幅度范围 `[-1, 1]`

### 代码位置

```python
def read_wav(path: Path) -> tuple[np.ndarray, int]
def resample_linear(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray
def normalize_audio(x: np.ndarray, peak: float = 0.98) -> np.ndarray
```

### 代码运用事例

```python
mixture, input_sr = read_wav(input_path)
model_mix = normalize_audio(resample_linear(mixture, input_sr, args.sr), peak=0.9)
```

作用：

- `read_wav()` 读取 WAV。
- 多声道输入会自动转为单声道。
- `resample_linear()` 把采样率转为模型使用的 `args.sr`。
- `normalize_audio()` 避免音频过载削波。

## 3. 合成训练数据

### 原理

论文的源分离方法依赖训练数据学习“混合音频 -> 目标声源”的映射。本项目没有外部数据集，因此用可控合成方式生成钢琴和小提琴训练样本：

- 钢琴：快速 attack、指数衰减、多个泛音。
- 小提琴：持续包络、vibrato、多个泛音和轻微 bow noise。
- 随机和弦、随机音量、随机相位形成数据增强。

### 代码位置

```python
def piano_note(...)
def violin_note(...)
def render_progression(...)
def make_random_pair(...)
def collect_training_data(...)
```

### 代码运用事例

```python
piano, violin = make_random_pair(duration, sr, rng)
mix, (piano, violin) = mix_and_scale([piano, violin], peak=0.9)
mix_mag = np.abs(stft(mix, n_fft, hop)).astype(np.float32)
piano_mag = np.abs(stft(piano, n_fft, hop)).astype(np.float32)
violin_mag = np.abs(stft(violin, n_fft, hop)).astype(np.float32)
```

作用：

- 合成训练用和声。
- 计算混合音频、钢琴、 小提琴的 STFT 幅度谱。
- 构造监督训练目标。

## 4. STFT 频域表示

### 原理

论文中的 DNN 源分离是在 STFT 域进行的：模型输入混合音频的频谱幅度，输出目标声源频谱幅度或 mask。这样可以利用不同乐器在频谱上的结构差异。

### 代码位置

```python
def stft(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray
def istft(spec: np.ndarray, n_fft: int, hop: int, length: int) -> np.ndarray
```

### 代码运用事例

```python
mix_spec = stft(mixture, self.n_fft, self.hop)
mix_mag = np.abs(mix_spec).astype(np.float32)
```

输出：

- `mix_spec`：复数 STFT，包含幅度和相位。
- `mix_mag`：幅度谱，作为神经网络输入。

## 5. FNN 源分离模型

### 原理

论文中使用了 feed-forward network 和 BLSTM network，并对不同声源分别训练模型。本项目保留最核心思想：每个声源一个小型 FNN。

本项目让网络预测 soft mask：

```text
mask_piano = FNN_piano(|X|)
mask_violin = FNN_violin(|X|)
```

再得到目标声源幅度估计：

```text
|S_piano| = mask_piano * |X|
|S_violin| = mask_violin * |X|
```

### 代码位置

```python
@dataclass
class TinyFNN

def train_fnn(...)
```

### 代码运用事例

```python
h = np.maximum(x @ self.w1 + self.b1, 0.0)
mask = 1.0 / (1.0 + np.exp(-(h @ self.w2 + self.b2)))
return (mask * mix_mag).astype(np.float32)
```

说明：

- 第一层使用 ReLU。
- 输出层使用 sigmoid，得到 `0-1` 之间的 soft mask。
- mask 乘以混合幅度谱，得到对应声源的幅度估计。

## 6. 多上下文网络与 Blending

### 原理

论文提出把不同网络结构的 raw output 做线性融合：

```text
S_blend = lambda * S_FNN + (1 - lambda) * S_BLSTM
```

本项目没有实现大型 BLSTM，而是用两个不同上下文窗口的 FNN 来模拟“不同模型输出融合”：

- context 1：较短上下文
- context 2：较长上下文

融合权重默认：

```text
lambda = 0.25
```

### 代码位置

```python
@dataclass
class Separator

def train_separator(...)
```

### 代码运用事例

```python
piano_mag = self.blend * self.piano_context_1.predict(mix_mag)
piano_mag += (1.0 - self.blend) * self.piano_context_2.predict(mix_mag)

violin_mag = self.blend * self.violin_context_1.predict(mix_mag)
violin_mag += (1.0 - self.blend) * self.violin_context_2.predict(mix_mag)
```

作用：

- 模拟论文中的 network blending。
- 减少单一模型预测偏差。
- 为后续 Wiener 后处理提供更稳的幅度估计。

## 7. Wiener Mask 后处理

### 原理

论文中使用 Wiener filtering 保证各声源相加后与原混合音频一致，并减少干扰。本项目用简化版 Wiener mask：

```text
M_i = |S_i|^2 / (|S_piano|^2 + |S_violin|^2)
```

然后：

```text
S_i_complex = M_i * X_complex
```

也就是说，使用混合音频的相位，结合估计出来的幅度比例重建声源。

### 代码位置

```python
def wiener_separate(...)
```

### 代码运用事例

```python
piano_psd = np.maximum(piano_mag, 0.0) ** power
violin_psd = np.maximum(violin_mag, 0.0) ** power
denom = piano_psd + violin_psd + EPS
piano_spec = mix_spec * (piano_psd / denom)
violin_spec = mix_spec * (violin_psd / denom)
```

输出：

- `piano_spec`
- `violin_spec`

之后通过 `istft()` 转回时域 WAV。

## 8. 输出分离音频

### 原理

频域分离完成后，需要用 inverse STFT 回到时域，再保存为 WAV 文件。

### 代码位置

```python
def write_wav(...)
def Separator.separate(...)
```

### 代码运用事例

```python
piano = istft(piano_spec, self.n_fft, self.hop, len(mixture))
violin = istft(violin_spec, self.n_fft, self.hop, len(mixture))

write_wav(piano_path, normalize_audio(piano_est), args.sr)
write_wav(violin_path, normalize_audio(violin_est), args.sr)
```

生成文件示例：

```text
mixture_piano_estimate.wav
mixture_violin_estimate.wav
```

## 9. 简谱分析

### 原理

`AUTOMATIC_GENERATION.pdf` 的主线是自动生成 lead sheet：

- 估计 beat grid。
- 估计 key。
- 把 melody notes 映射到节拍网格。
- 对 onset 和 duration 做量化。
- 渲染可读乐谱。

本项目用最小实现完成类似流程：

1. 从混合音频估计调性。
2. 从频谱 flux 估计节拍时长。
3. 把时间切成八分音符网格。
4. 对每个网格检测钢琴/小提琴音高。
5. 把 MIDI 音高转为简谱。

### 代码位置

```python
def estimate_key(...)
def estimate_beat_duration(...)
def detect_segment_pitches(...)
def midi_to_jianpu(...)
def transcribe_to_jianpu(...)
def write_jianpu_report(...)
```

### 代码运用事例

```python
transcription = transcribe_to_jianpu(
    piano_est,
    violin_est,
    model_mix,
    args.sr,
    args.n_fft,
    args.hop,
)

write_jianpu_report(jianpu_path, transcription)
```

简谱输出示例：

```text
钢琴简谱:
[#1,, 4,, 6,]x2 [#4,, 7, #1,]x2 ...

小提琴简谱:
[6' 1']x2 [7' #1']x2 ...
```

符号说明：

- `0`：休止。
- `[1 3 5]`：同一时间片检测到多个音，即和声音。
- `'`：高八度。
- `,`：低八度。
- `x2`：连续重复两个时间片。

## 10. 五线谱 PDF 输出

### 原理

简谱分析阶段已经得到每个时间片的 MIDI 音高。本项目继续把 MIDI 音高映射到五线谱坐标：

```text
MIDI note -> staff y coordinate
```

再用 PDF 画布绘制：

- 标题
- 调性和 BPM
- Violin 五线谱
- Piano 五线谱
- 音符
- 休止符
- 小节线
- 加线

### 代码位置

```python
def midi_to_staff_y(...)
def draw_staff(...)
def draw_note(...)
def draw_rest(...)
def draw_ledger_lines(...)
def write_staff_pdf(...)
```

### 代码运用事例

```python
write_staff_pdf(staff_pdf_path, transcription)
```

输出文件示例：

```text
mixture_staff.pdf
staff.pdf
```

## 11. UI 工作流

### 原理

UI 只是对命令行流程的封装，让用户可以通过窗口选择输入音频和输出目录。

### 代码位置

```python
def run_ui(default_args: argparse.Namespace) -> None
```

### UI 操作流程

```text
打开 UI
  -> 点击“选择音频”
  -> 选择 WAV 文件
  -> 点击“选择目录”
  -> 选择输出路径
  -> 点击“开始分离并生成简谱”
  -> 等待进度条
  -> 查看 WAV / TXT / JSON / PDF 输出
```

### 运行命令

```powershell
& 'C:\Users\91453\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' demo\piano_violin_separation_demo.py --ui
```

## 12. 命令行工作流

### 默认测试

运行默认测试会生成合成和声，并完成分离、简谱和五线谱 PDF 输出：

```powershell
& 'C:\Users\91453\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' demo\piano_violin_separation_demo.py
```

默认输出：

```text
outputs\piano_violin_demo\mixture.wav
outputs\piano_violin_demo\piano_estimate.wav
outputs\piano_violin_demo\violin_estimate.wav
outputs\piano_violin_demo\jianpu.txt
outputs\piano_violin_demo\jianpu.json
outputs\piano_violin_demo\staff.pdf
outputs\piano_violin_demo\metrics.json
```

### 自定义输入

```powershell
& 'C:\Users\91453\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' demo\piano_violin_separation_demo.py --input outputs\piano_violin_demo\mixture.wav --output outputs\cli_test
```

自定义输出：

```text
outputs\cli_test\mixture_model_input_mono_8000hz.wav
outputs\cli_test\mixture_piano_estimate.wav
outputs\cli_test\mixture_violin_estimate.wav
outputs\cli_test\mixture_jianpu.txt
outputs\cli_test\mixture_jianpu.json
outputs\cli_test\mixture_staff.pdf
outputs\cli_test\mixture_separation_result.json
```

## 13. 参数说明

主要参数在 `parse_args()` 中定义：

```python
parser.add_argument("--ui", action="store_true")
parser.add_argument("--input", default="")
parser.add_argument("--output", default="outputs/piano_violin_demo")
parser.add_argument("--sr", type=int, default=8000)
parser.add_argument("--n-fft", type=int, default=512)
parser.add_argument("--hop", type=int, default=128)
parser.add_argument("--examples", type=int, default=120)
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--blend", type=float, default=0.25)
```

常用参数：

- `--ui`：打开图形界面。
- `--input`：指定需要分离的 WAV 文件。
- `--output`：指定输出目录。
- `--examples`：合成训练样本数量。
- `--epochs`：训练轮数。
- `--blend`：两个上下文网络的融合权重。

## 14. 验收方式

### 语法检查

```powershell
& 'C:\Users\91453\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m py_compile demo\piano_violin_separation_demo.py
```

### 默认 demo 验收

```powershell
& 'C:\Users\91453\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' demo\piano_violin_separation_demo.py
```

成功时会出现类似：

```text
PASS: both sources improved by at least 1.00 dB.
```

### 输出检查

确认这些文件存在：

```text
piano_estimate.wav
violin_estimate.wav
jianpu.txt
jianpu.json
staff.pdf
```

## 15. 项目边界

这是一个最小可运行复现，重点是展示论文流程如何落到代码中：

- 可验证源分离流程。
- 可输出简谱。
- 可输出五线谱 PDF。
- 可通过 UI 选择输入和输出目录。

它不是商业级通用音乐转谱系统，真实复杂录音的准确率会受到乐器音色、噪声、混响、速度变化和复调复杂度影响。
