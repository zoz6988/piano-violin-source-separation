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
import wave
from dataclasses import dataclass
from pathlib import Path

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


def run_demo(args: argparse.Namespace) -> dict[str, float]:
    rng = np.random.default_rng(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    train_a = collect_training_data(args.examples, 1.5, args.sr, args.n_fft, args.hop, 1, rng)
    train_b = collect_training_data(args.examples, 1.5, args.sr, args.n_fft, args.hop, 2, rng)

    piano_a = train_fnn(train_a[0], train_a[1], 1, args.hidden, args.epochs, args.batch_size, args.lr, rng)
    violin_a = train_fnn(train_a[0], train_a[2], 1, args.hidden, args.epochs, args.batch_size, args.lr, rng)
    piano_b = train_fnn(train_b[0], train_b[1], 2, args.hidden, args.epochs, args.batch_size, args.lr, rng)
    violin_b = train_fnn(train_b[0], train_b[2], 2, args.hidden, args.epochs, args.batch_size, args.lr, rng)

    piano_ref, violin_ref = make_test_song(args.sr, rng)
    piano_ref *= 0.85
    violin_ref *= 0.95
    mixture, (piano_ref, violin_ref) = mix_and_scale([piano_ref, violin_ref], peak=0.9)

    mix_spec = stft(mixture, args.n_fft, args.hop)
    mix_mag = np.abs(mix_spec).astype(np.float32)
    piano_mag = args.blend * piano_a.predict(mix_mag) + (1.0 - args.blend) * piano_b.predict(mix_mag)
    violin_mag = args.blend * violin_a.predict(mix_mag) + (1.0 - args.blend) * violin_b.predict(mix_mag)
    piano_spec, violin_spec = wiener_separate(mix_spec, piano_mag, violin_mag)
    piano_est = istft(piano_spec, args.n_fft, args.hop, len(mixture))
    violin_est = istft(violin_spec, args.n_fft, args.hop, len(mixture))

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and test a tiny piano/violin separator.")
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
    run_demo(parse_args())
