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

For a complete workflow explanation, see [WORKFLOW.md](WORKFLOW.md).
For a step-by-step Chinese testing guide, see [TESTING_README.md](TESTING_README.md).

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
`*_piano_estimate.wav`, `*_violin_estimate.wav`, `*_jianpu.txt`, and
`*_jianpu.json`, and `*_staff.pdf` files to that folder.

You can also separate a WAV directly from the command line:

```powershell
& 'C:\Users\91453\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' demo\piano_violin_separation_demo.py --input path\to\harmony.wav --output outputs\my_separation
```

The script writes:

- `outputs/piano_violin_demo/mixture.wav`
- `outputs/piano_violin_demo/piano_estimate.wav`
- `outputs/piano_violin_demo/violin_estimate.wav`
- `outputs/piano_violin_demo/jianpu.txt`
- `outputs/piano_violin_demo/jianpu.json`
- `outputs/piano_violin_demo/staff.pdf`
- reference stems and `metrics.json`

The numbered-notation report follows the lead-sheet paper's idea at demo scale:
after source separation it estimates key from chroma energy, estimates a beat
grid, quantizes notes to eighth-note slots, and prints piano/violin notes as
simple numbered notation. `0` means rest; brackets mean simultaneous harmony
notes in one slot.

The demo also renders the detected notes as a simple two-staff PDF score with
separate Piano and Violin systems.

The run passes when both separated sources improve over the raw mixture baseline
by at least `1.0 dB`.

## Notes

This is not a full DSD100 reproduction. It is a compact proof-of-concept for the
paper's separation recipe, specialized to synthetic piano/violin harmony so it
can be trained and tested locally without external datasets or neural-network
frameworks.
