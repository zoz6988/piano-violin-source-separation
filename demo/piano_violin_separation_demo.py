"""Minimal piano/violin source-separation demo inspired by Uhlich et al. 2017.

The paper trains one DNN per source on STFT magnitudes, blends raw network
outputs, then applies Wiener filtering.  This compact reproduction keeps that
pipeline but uses synthetic piano/violin harmony and a small NumPy FNN so it can
run without heavyweight audio or deep-learning dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from tkinter import StringVar, Tk, filedialog, messagebox, ttk

import numpy as np


EPS = 1e-8


def midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def normalize_audio(x: np.ndarray, peak: float = 0.98) -> np.ndarray:
    max_abs = float(np.max(np.abs(x)))
    if max_abs < EPS:
        return x.astype(np.float32)
    return (x / max_abs * peak).astype(np.float32)


def mix_and_scale(sources: list[np.ndarray], peak: float = 0.9) -> tuple[np.ndarray, list[np.ndarray]]:
    mixture = np.sum(sources, axis=0)
    max_abs = float(np.max(np.abs(mixture)))
    scale = 1.0 if max_abs < EPS else peak / max_abs
    scaled_sources = [(src * scale).astype(np.float32) for src in sources]
    return np.sum(scaled_sources, axis=0).astype(np.float32), scaled_sources


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio.astype(np.float32), sample_rate


def resample_linear(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio.astype(np.float32)
    duration = len(audio) / source_sr
    source_times = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    target_len = max(1, int(round(duration * target_sr)))
    target_times = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(target_times, source_times, audio).astype(np.float32)


def piano_note(freq: float, duration: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    signal = np.zeros_like(t)
    partials = [1.0, 0.58, 0.32, 0.18, 0.10, 0.06]
    for k, amp in enumerate(partials, start=1):
        detune = 1.0 + rng.normal(0.0, 0.0005)
        signal += amp * np.sin(2 * np.pi * freq * k * detune * t + rng.uniform(0, 2 * np.pi))
    attack = np.minimum(t / 0.012, 1.0)
    decay = np.exp(-3.2 * t / duration)
    envelope = attack * decay
    transient = np.exp(-t / 0.018) * rng.normal(0.0, 0.15, size=t.shape)
    return (signal * envelope + transient).astype(np.float32)


def violin_note(freq: float, duration: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    signal = np.zeros_like(t)
    partials = [1.0, 0.82, 0.58, 0.40, 0.27, 0.18, 0.12, 0.08]
    vibrato = 1.0 + 0.006 * np.sin(2 * np.pi * rng.uniform(4.5, 6.2) * t)
    phase = np.cumsum(2 * np.pi * freq * vibrato / sr)
    for k, amp in enumerate(partials, start=1):
        signal += amp * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
    bow_noise = rng.normal(0.0, 0.025, size=t.shape)
    attack = np.minimum(t / 0.18, 1.0)
    release = np.minimum((duration - t) / 0.16, 1.0)
    envelope = np.clip(attack * release, 0.0, 1.0)
    return ((signal + bow_noise) * envelope).astype(np.float32)


def render_progression(
    chords: list[list[int]],
    instrument: str,
    duration_per_chord: float,
    sr: int,
    rng: np.random.Generator,
) -> np.ndarray:
    rendered: list[np.ndarray] = []
    synth = piano_note if instrument == "piano" else violin_note
    octave_shift = -12 if instrument == "piano" else 0
    for chord in chords:
        notes = [synth(midi_to_hz(m + octave_shift), duration_per_chord, sr, rng) for m in chord]
        chord_audio = np.sum(notes, axis=0) / max(len(notes), 1)
        rendered.append(chord_audio)
    return normalize_audio(np.concatenate(rendered), peak=0.7)


def make_random_pair(duration: float, sr: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    chord_count = max(2, int(duration / 0.75))
    root_choices = [48, 50, 52, 53, 55, 57, 59, 60]
    qualities = [[0, 4, 7], [0, 3, 7], [0, 5, 9], [0, 4, 9]]
    piano_chords = []
    violin_chords = []
    for _ in range(chord_count):
        root = int(rng.choice(root_choices))
        quality = list(rng.choice(qualities))
        piano_chords.append([root + step for step in quality])
        lead = root + int(rng.choice([12, 14, 16, 19, 21]))
        violin_chords.append([lead, lead + int(rng.choice([3, 4, 7]))])
    piano = render_progression(piano_chords, "piano", duration / chord_count, sr, rng)
    violin = render_progression(violin_chords, "violin", duration / chord_count, sr, rng)
    return piano, violin


def make_test_song(sr: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    piano_chords = [[48, 52, 55], [53, 57, 60], [55, 59, 62], [52, 55, 59], [48, 52, 55]]
    violin_chords = [[67, 71], [69, 72], [74, 78], [71, 76], [67, 72]]
    piano = render_progression(piano_chords, "piano", 0.8, sr, rng)
    violin = render_progression(violin_chords, "violin", 0.8, sr, rng)
    return piano, violin


def stft(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    window = np.hanning(n_fft).astype(np.float32)
    padded = np.pad(x.astype(np.float32), (0, n_fft), mode="constant")
    frame_count = 1 + (len(padded) - n_fft) // hop
    frames = np.stack([padded[i * hop : i * hop + n_fft] * window for i in range(frame_count)])
    return np.fft.rfft(frames, axis=1)


def istft(spec: np.ndarray, n_fft: int, hop: int, length: int) -> np.ndarray:
    window = np.hanning(n_fft).astype(np.float32)
    out = np.zeros((spec.shape[0] - 1) * hop + n_fft, dtype=np.float32)
    norm = np.zeros_like(out)
    frames = np.fft.irfft(spec, n=n_fft, axis=1).astype(np.float32)
    for i, frame in enumerate(frames):
        start = i * hop
        out[start : start + n_fft] += frame * window
        norm[start : start + n_fft] += window * window
    out /= np.maximum(norm, EPS)
    return out[:length]


def context_features(mag: np.ndarray, context: int) -> np.ndarray:
    padded = np.pad(mag, ((context, context), (0, 0)), mode="edge")
    chunks = [padded[i : i + mag.shape[0]] for i in range(2 * context + 1)]
    return np.concatenate(chunks, axis=1)


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


@dataclass
class TinyFNN:
    context: int
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    scaler: Standardizer

    def predict(self, mix_mag: np.ndarray) -> np.ndarray:
        x = np.log1p(context_features(mix_mag, self.context))
        x = self.scaler.transform(x)
        h = np.maximum(x @ self.w1 + self.b1, 0.0)
        mask = 1.0 / (1.0 + np.exp(-(h @ self.w2 + self.b2)))
        return (mask * mix_mag).astype(np.float32)


@dataclass
class Separator:
    piano_context_1: TinyFNN
    violin_context_1: TinyFNN
    piano_context_2: TinyFNN
    violin_context_2: TinyFNN
    blend: float
    n_fft: int
    hop: int

    def separate(self, mixture: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mix_spec = stft(mixture, self.n_fft, self.hop)
        mix_mag = np.abs(mix_spec).astype(np.float32)
        piano_mag = self.blend * self.piano_context_1.predict(mix_mag)
        piano_mag += (1.0 - self.blend) * self.piano_context_2.predict(mix_mag)
        violin_mag = self.blend * self.violin_context_1.predict(mix_mag)
        violin_mag += (1.0 - self.blend) * self.violin_context_2.predict(mix_mag)
        piano_spec, violin_spec = wiener_separate(mix_spec, piano_mag, violin_mag)
        piano = istft(piano_spec, self.n_fft, self.hop, len(mixture))
        violin = istft(violin_spec, self.n_fft, self.hop, len(mixture))
        return piano, violin


def train_fnn(
    features: np.ndarray,
    targets: np.ndarray,
    context: int,
    hidden: int,
    epochs: int,
    batch_size: int,
    lr: float,
    rng: np.random.Generator,
) -> TinyFNN:
    x = np.log1p(features).astype(np.float32)
    y = targets.astype(np.float32)
    scaler = Standardizer(x.mean(axis=0, keepdims=True), x.std(axis=0, keepdims=True) + 1e-5)
    x = scaler.transform(x)
    w1 = (rng.normal(0.0, math.sqrt(2 / x.shape[1]), size=(x.shape[1], hidden))).astype(np.float32)
    b1 = np.zeros((1, hidden), dtype=np.float32)
    w2 = (rng.normal(0.0, math.sqrt(2 / hidden), size=(hidden, y.shape[1]))).astype(np.float32)
    b2 = np.zeros((1, y.shape[1]), dtype=np.float32)
    n = x.shape[0]
    for _ in range(epochs):
        for start in range(0, n, batch_size):
            idx = rng.choice(n, size=min(batch_size, n), replace=False)
            xb = x[idx]
            yb = y[idx]
            h_pre = xb @ w1 + b1
            h = np.maximum(h_pre, 0.0)
            logits = h @ w2 + b2
            pred = 1.0 / (1.0 + np.exp(-logits))
            grad = (2.0 / xb.shape[0]) * (pred - yb)
            grad *= pred * (1.0 - pred)
            dw2 = h.T @ grad
            db2 = grad.sum(axis=0, keepdims=True)
            dh = grad @ w2.T
            dh[h_pre <= 0.0] = 0.0
            dw1 = xb.T @ dh
            db1 = dh.sum(axis=0, keepdims=True)
            w2 -= lr * dw2
            b2 -= lr * db2
            w1 -= lr * dw1
            b1 -= lr * db1
    return TinyFNN(context, w1, b1, w2, b2, scaler)


def collect_training_data(
    examples: int,
    duration: float,
    sr: int,
    n_fft: int,
    hop: int,
    context: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    piano_targets: list[np.ndarray] = []
    violin_targets: list[np.ndarray] = []
    for _ in range(examples):
        piano, violin = make_random_pair(duration, sr, rng)
        piano *= rng.uniform(0.55, 1.15)
        violin *= rng.uniform(0.55, 1.15)
        if rng.random() < 0.5:
            piano = -piano
        mix, (piano, violin) = mix_and_scale([piano, violin], peak=0.9)
        mix_mag = np.abs(stft(mix, n_fft, hop)).astype(np.float32)
        piano_mag = np.abs(stft(piano, n_fft, hop)).astype(np.float32)
        violin_mag = np.abs(stft(violin, n_fft, hop)).astype(np.float32)
        xs.append(context_features(mix_mag, context))
        total = piano_mag + violin_mag + EPS
        piano_targets.append(piano_mag / total)
        violin_targets.append(violin_mag / total)
    return np.vstack(xs), np.vstack(piano_targets), np.vstack(violin_targets)


def train_separator(args: argparse.Namespace, rng: np.random.Generator) -> Separator:
    train_a = collect_training_data(args.examples, 1.5, args.sr, args.n_fft, args.hop, 1, rng)
    train_b = collect_training_data(args.examples, 1.5, args.sr, args.n_fft, args.hop, 2, rng)
    return Separator(
        piano_context_1=train_fnn(train_a[0], train_a[1], 1, args.hidden, args.epochs, args.batch_size, args.lr, rng),
        violin_context_1=train_fnn(train_a[0], train_a[2], 1, args.hidden, args.epochs, args.batch_size, args.lr, rng),
        piano_context_2=train_fnn(train_b[0], train_b[1], 2, args.hidden, args.epochs, args.batch_size, args.lr, rng),
        violin_context_2=train_fnn(train_b[0], train_b[2], 2, args.hidden, args.epochs, args.batch_size, args.lr, rng),
        blend=args.blend,
        n_fft=args.n_fft,
        hop=args.hop,
    )


def wiener_separate(
    mix_spec: np.ndarray,
    piano_mag: np.ndarray,
    violin_mag: np.ndarray,
    power: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    piano_psd = np.maximum(piano_mag, 0.0) ** power
    violin_psd = np.maximum(violin_mag, 0.0) ** power
    denom = piano_psd + violin_psd + EPS
    piano_spec = mix_spec * (piano_psd / denom)
    violin_spec = mix_spec * (violin_psd / denom)
    return piano_spec, violin_spec


def sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    length = min(len(reference), len(estimate))
    ref = reference[:length]
    est = estimate[:length]
    return 10.0 * np.log10((np.sum(ref * ref) + EPS) / (np.sum((ref - est) ** 2) + EPS))


def separate_file(args: argparse.Namespace, separator: Separator) -> dict[str, str]:
    input_path = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    mixture, input_sr = read_wav(input_path)
    model_mix = normalize_audio(resample_linear(mixture, input_sr, args.sr), peak=0.9)
    piano_est, violin_est = separator.separate(model_mix)
    stem = input_path.stem
    piano_path = output / f"{stem}_piano_estimate.wav"
    violin_path = output / f"{stem}_violin_estimate.wav"
    mix_path = output / f"{stem}_model_input_mono_{args.sr}hz.wav"
    write_wav(mix_path, model_mix, args.sr)
    write_wav(piano_path, normalize_audio(piano_est), args.sr)
    write_wav(violin_path, normalize_audio(violin_est), args.sr)
    result = {
        "input": str(input_path.resolve()),
        "model_input": str(mix_path.resolve()),
        "piano_estimate": str(piano_path.resolve()),
        "violin_estimate": str(violin_path.resolve()),
        "sample_rate": str(args.sr),
    }
    (output / f"{stem}_separation_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_demo(args: argparse.Namespace) -> dict[str, float]:
    rng = np.random.default_rng(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    separator = train_separator(args, rng)

    if args.input:
        result = separate_file(args, separator)
        print(json.dumps(result, indent=2))
        print(f"Separated WAV files written to: {Path(args.output).resolve()}")
        return {}

    piano_ref, violin_ref = make_test_song(args.sr, rng)
    piano_ref *= 0.85
    violin_ref *= 0.95
    mixture, (piano_ref, violin_ref) = mix_and_scale([piano_ref, violin_ref], peak=0.9)

    piano_est, violin_est = separator.separate(mixture)

    mix_piano_sdr = sdr(piano_ref, mixture)
    mix_violin_sdr = sdr(violin_ref, mixture)
    est_piano_sdr = sdr(piano_ref, piano_est)
    est_violin_sdr = sdr(violin_ref, violin_est)
    metrics = {
        "mixture_as_piano_sdr_db": float(mix_piano_sdr),
        "separated_piano_sdr_db": float(est_piano_sdr),
        "piano_improvement_db": float(est_piano_sdr - mix_piano_sdr),
        "mixture_as_violin_sdr_db": float(mix_violin_sdr),
        "separated_violin_sdr_db": float(est_violin_sdr),
        "violin_improvement_db": float(est_violin_sdr - mix_violin_sdr),
    }

    write_wav(output / "mixture.wav", mixture, args.sr)
    write_wav(output / "piano_reference.wav", normalize_audio(piano_ref), args.sr)
    write_wav(output / "violin_reference.wav", normalize_audio(violin_ref), args.sr)
    write_wav(output / "piano_estimate.wav", normalize_audio(piano_est), args.sr)
    write_wav(output / "violin_estimate.wav", normalize_audio(violin_est), args.sr)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    if metrics["piano_improvement_db"] < args.min_improvement or metrics["violin_improvement_db"] < args.min_improvement:
        raise SystemExit(
            f"Demo did not pass: expected both improvements >= {args.min_improvement:.2f} dB"
        )
    print(f"PASS: both sources improved by at least {args.min_improvement:.2f} dB.")
    print(f"WAV files written to: {output.resolve()}")
    return metrics


def run_ui(default_args: argparse.Namespace) -> None:
    root = Tk()
    root.title("Piano / Violin Separator")
    root.geometry("620x250")
    root.resizable(False, False)

    input_var = StringVar()
    output_var = StringVar(value=str(Path(default_args.output).resolve()))
    status_var = StringVar(value="请选择一个 WAV 音频文件和分离结果保存目录。")

    def choose_input() -> None:
        path = filedialog.askopenfilename(
            title="选择需要分离的音频",
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
        )
        if path:
            input_var.set(path)

    def choose_output() -> None:
        path = filedialog.askdirectory(title="选择分离后存储位置")
        if path:
            output_var.set(path)

    def set_busy(is_busy: bool) -> None:
        state = "disabled" if is_busy else "normal"
        start_button.configure(state=state)
        input_button.configure(state=state)
        output_button.configure(state=state)

    def worker() -> None:
        try:
            args = argparse.Namespace(**vars(default_args))
            args.input = input_var.get()
            args.output = output_var.get()
            rng = np.random.default_rng(args.seed)
            separator = train_separator(args, rng)
            result = separate_file(args, separator)

            def done() -> None:
                set_busy(False)
                status_var.set("分离完成。")
                messagebox.showinfo(
                    "分离完成",
                    "已生成文件：\n"
                    f"钢琴: {result['piano_estimate']}\n"
                    f"小提琴: {result['violin_estimate']}",
                )

            root.after(0, done)
        except Exception as exc:
            error_message = str(exc)

            def failed() -> None:
                set_busy(False)
                status_var.set("分离失败，请检查输入文件。")
                messagebox.showerror("分离失败", error_message)

            root.after(0, failed)

    def start() -> None:
        if not input_var.get():
            messagebox.showwarning("缺少音频", "请先选择需要分离的 WAV 音频。")
            return
        if not output_var.get():
            messagebox.showwarning("缺少目录", "请先选择分离后存储位置。")
            return
        set_busy(True)
        status_var.set("正在训练最小模型并分离音频，请稍等...")
        threading.Thread(target=worker, daemon=True).start()

    frame = ttk.Frame(root, padding=18)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="待分离音频").grid(row=0, column=0, sticky="w", pady=(0, 8))
    ttk.Entry(frame, textvariable=input_var, width=64).grid(row=0, column=1, sticky="ew", pady=(0, 8))
    input_button = ttk.Button(frame, text="选择音频", command=choose_input)
    input_button.grid(row=0, column=2, padx=(8, 0), pady=(0, 8))

    ttk.Label(frame, text="保存位置").grid(row=1, column=0, sticky="w", pady=(0, 8))
    ttk.Entry(frame, textvariable=output_var, width=64).grid(row=1, column=1, sticky="ew", pady=(0, 8))
    output_button = ttk.Button(frame, text="选择目录", command=choose_output)
    output_button.grid(row=1, column=2, padx=(8, 0), pady=(0, 8))

    start_button = ttk.Button(frame, text="开始分离", command=start)
    start_button.grid(row=2, column=1, sticky="e", pady=(18, 10))
    ttk.Label(frame, textvariable=status_var).grid(row=3, column=0, columnspan=3, sticky="w")
    frame.columnconfigure(1, weight=1)
    root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and test a tiny piano/violin separator.")
    parser.add_argument("--ui", action="store_true", help="Open a small desktop UI.")
    parser.add_argument("--input", default="", help="Optional WAV file to separate instead of running synthetic test.")
    parser.add_argument("--output", default="outputs/piano_violin_demo")
    parser.add_argument("--sr", type=int, default=8000)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop", type=int, default=128)
    parser.add_argument("--examples", type=int, default=120)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--blend", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-improvement", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.ui:
        run_ui(parsed_args)
    else:
        run_demo(parsed_args)
