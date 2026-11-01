# Contributing

Thanks for helping! Bug reports, test footage descriptions, translations and pull requests are all welcome.

## Setup

```bash
git clone https://github.com/canardu1/shorts-studio.git
cd shorts-studio
python -m venv .venv && source .venv/bin/activate
pip install -e ".[app]" pytest scikit-image
QT_QPA_PLATFORM=offscreen python -m pytest
```

On Linux without an audio device, add `QT_AUDIO_BACKEND=pulseaudio` if Qt Multimedia crashes in the GUI tests.

## Layout

- `src/face_reframe/` — the engine (detection, tracking, cuts, audio, colour, restoration) and the CLI.
- `src/face_reframe/app/` — Shorts Studio (PySide6). `jobs.py` is the Qt-free render pipeline; `window.py` is the main window.
- `resolve/`, `lua/` — DaVinci Resolve launcher scripts; `install.py` copies them into Resolve.
- `tests/` — pytest suite; `make_sample.py` builds synthetic clips, so no real footage is needed.

## Guidelines

- Keep heavy work off the GUI thread (use `MainWindow._run`), and make it cancellable.
- New processing steps should be cheap per frame (shorts are usually rendered on laptops); cache anything expensive per file with `app/diskcache.py`.
- Add a test for every fix or feature. GUI tests run offscreen.
- Models: only add ones whose licence allows redistribution, and list them in `THIRD_PARTY_NOTICES.md`.
- UI strings are Italian for now; if you add an English translation, wrap strings so both can coexist.

## Reporting bugs

Please include your OS, Python version, the app log (bottom panel), and a description of the footage (resolution, codec, camera). Short sample clips help a lot.
