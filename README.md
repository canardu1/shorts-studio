# Shorts Studio

**Turn long horizontal videos into finished vertical shorts — on your own computer, no subscription, no upload.**

Face-following reframe, automatic jump cuts, word-by-word captions, cinematic audio, face restoration, skin-tone fix, social camera moves and click-worthy thumbnails, in one desktop app. The same engine also ships as **free scripts for DaVinci Resolve** (they work on the free version).

<!-- Demo media: see docs/media/README.md to (re)generate these from your own footage -->
<p align="center">
  <img src="docs/media/reframe.gif" width="32%" alt="Face-following 16:9 → 9:16 reframe">
  <img src="docs/media/captions.gif" width="32%" alt="Word-by-word captions and punch-in">
  <img src="docs/media/thumbnails.jpg" width="32%" alt="Generated thumbnails">
</p>

## Why

Tools like OpusClip, Submagic or CapCut Pro do this in the cloud for $15–30 a month. Shorts Studio runs **locally** (your footage never leaves your machine), works **offline** after the first model download, and processes a whole folder in **batch**.

## Features

### Reframe
- **Face-following 16:9 → 9:16** crop with smooth "camera operator" motion (dead zone, smoothing, max pan speed) and a hard jump at every cut.
- **Shot detection:** close-ups and wide shots are recognised automatically; wide shots get a punch-in so the speaker fills the vertical frame.
- **Who to follow:** the main person, locked per shot (no jitter between two people), or **whoever is talking** — mouth-movement heuristic or neural **voice ID** (ECAPA embeddings match the voice to the speaker; voice changes are marked on the timeline).
- **Follow object:** in shots without faces the crop follows the most salient thing (products, pets, B-roll) via U²-Net-p instead of staying centred.
- **Pick an object:** draw a box on any frame and NanoTrack follows that exact thing — cars, gadgets, animals — in both directions.
- **Person picker:** thumbnails of everyone in the video, matched across shots with face recognition — tick the person to follow, even when they appear in both close-up and wide shots.
- **Hand corrections:** drag the crop in the preview to re-aim from any frame, "auto from here" to hand control back.
- **Look room:** optional crop shift toward where the face appears to be looking (from the eye/nose landmarks), so the subject isn't nose-to-edge.
- **Split screen** for two-person podcasts (both stacked when both are in shot).
- **Multi-format:** 9:16, 4:5, 1:1 and 16:9 from one analysis.

### Automatic editing
- **Cut pauses** with a selectable speech detector — neural voice-activity detection (Silero VAD, bundled) or a classic level/dB threshold (automatic fallback if the model is missing) — plus **cut breaths**, **cut filler words** (*uhm, ehm, uh…*) and **cut retakes** (repeated lines — the last take wins).
- **Best moments finder:** proposes the strongest 30–60 s clips of a long video (dense speech, a hook in the first sentence, ends on a full sentence) — **YAMNet audio events** boost clips with laughter, applause or music and mark them on the timeline.
- **Clip ranges** and **batch processing** of a whole folder.

### Captions
- **Whisper transcription** running locally, cached per file.
- **TikTok-style word-by-word captions** with the spoken word highlighted, or classic lines; colour, position, uppercase.
- **Subtitle editor:** fix misheard words, find & replace (names, jargon), timings re-spread automatically, restore the original.
- **English translation** or **bilingual** captions (original + English).

### Camera moves & effects
- **Punch-in** on every new phrase.
- **Handheld** drift, **slow push-in**, **crash zoom** on strong words, **impact shake** on voice peaks.
- **Transitions** on cuts: whip pan, zoom, glitch or random.
- **Stabilisation** (optical flow, per shot).
- **Slow motion** ×2/×4: **RIFE** invents the in-between frames (export only), or FFmpeg's instant blend; audio is retimed to match.

### Image
- **Face restoration** with AI models: GPEN-256 (fast), GPEN-512, GFPGAN 1.4, CodeFormer (best quality).
- **Skin-tone normalisation:** fixes skin that the camera renders too red or saturated, touching only skin pixels; exportable as a `.cube` LUT for Resolve.
- **Beauty:** skin smoothing, blemish removal, wrinkle and under-eye softening — a neural face-parsing mask (89 MB, downloads on first use) keeps eyes, lips, beard, hair and glasses razor sharp.
- **Colour looks:** warm, cool, cinema, pop, vintage, black & white, plus **auto white balance** and exposure — and a manual **grade** (temperature, tint, exposure, contrast, saturation, shadows, highlights, vignette) baked into the same fast LUT.
- **Lift dark shots:** neural low-light boost — SG-LLIE (fast) or URetinex (quality) — with a strength slider.
- **Enhance:** fast sharpening or Real-ESRGAN AI upscaling of the crop.
- **Blur background:** portrait-lens look — a recurrent RVM matte keeps the subject sharp and blurs the rest.

### Audio
- **Cinematic dialogue** in one click: denoise (**RNNoise**, **DeepFilterNet** — strongest on changing noise — or classic FFT), de-ess, de-reverb, rumble/boxiness cut, compression, warmth/presence/air EQ, −14 LUFS.
- **Background music** ducked under the voice with a sidechain compressor.
- **Live preview of the treated audio** while you tune, with **A/B** against the original.

### Publishing
- **Hook title** on screen for the first seconds (automatic from the first sentence, or your own).
- **Logo / watermark** and a closing **call-to-action** card.
- **Click-worthy thumbnails** for YouTube (16:9) and Shorts (9:16): best expressive frame (faceless videos rank candidates by saliency + sharpness), subject cut-out with outline and shadow, 4 styles, 2–4 big words with one highlighted.
- **Cover image** for each short.
- **Title, description and hashtags** drafted from the transcript.

### Workflow
- **Live preview** with source and result side by side, and a timeline showing cuts, shots, corrections, voice changes and sound events.
- **Safe-zone overlay** for the TikTok, Reels and Shorts buttons.
- **Undo / redo**, **projects** (`.shorts`) with **autosave** and session restore.
- **Presets** (Interview, 2-person podcast, split-screen podcast, Vlog, slow PC) plus your own.
- **Disk cache** for face analysis, transcripts and stabilisation: re-opening a video is instant.
- **Proxies** for heavy 4K / HEVC 10-bit camera files, so playback stays smooth.
- **Hardware encoding** (VideoToolbox on the Mac).

### DaVinci Resolve scripts (free version included)
Face Reframe (non-destructive Fusion keyframes or render), Cut Pauses, Cut Fillers, Cut Breaths, Cut Retakes, Auto Subtitles, Punch In, Cinematic Audio, Normalize Audio, Normalize Color, Enhance, Duck Music, Sync Audio (align external recorder audio), Auto Markers, Split At Cuts and a one-dialog **Shorts Pipeline**.

The app's interface is in **English** by default; Italian is available from the Export tab (**UI language**, applies on restart).

## Quick start

### Shorts Studio on macOS — one command

Open Terminal and paste:

```bash
curl -fsSL https://raw.githubusercontent.com/canardu1/shorts-studio/main/install.sh | bash
```

It installs everything in `~/.shorts-studio` (a private Python, no admin rights) and creates **Shorts Studio** in your user Applications folder. Run it again to update; uninstall with `rm -rf ~/.shorts-studio ~/Applications/"Shorts Studio.app"`. [install.sh](install.sh) is short if you want to read it first.

### From source (macOS, Linux; Windows untested)

```bash
git clone https://github.com/canardu1/shorts-studio.git
cd shorts-studio
pip install -e ".[app]"
face-reframe-app
```

To build a standalone `.app` bundle: `packaging/build_mac.sh`.

Typical flow: add videos → **Analyze faces** → tune on the live preview → **Create short**.
The full option guide is in [docs/shorts-studio.md](docs/shorts-studio.md).

### DaVinci Resolve scripts

```bash
python install.py --deps
```

Then restart Resolve and use **Workspace › Scripts**: Face Reframe, Cut Pauses, Cut Fillers, Cut Breaths, Cut Retakes, Auto Subtitles, Punch In, Cinematic Audio, Normalize Audio/Color, Enhance, Duck Music, Sync Audio, Auto Markers, Split At Cuts and a one-dialog Shorts Pipeline. Details: [docs/resolve-tools.md](docs/resolve-tools.md).

### Command line

```bash
face-reframe interview.mp4 -o short.mp4 --subject speaker
python -m face_reframe.cut in.mp4 -o out.mp4
python -m face_reframe.cinematic in.mp4 -o out.mp4
```

## Performance

Measured on an Apple M1, 1080p source:

| Step | Time |
|---|---|
| Face restoration GPEN-256 / GFPGAN / CodeFormer | 0.15 / 2.1 / 5 s per face per frame |
| Beauty, skin fix, looks, camera moves | a few ms per frame |
| Thumbnails (4 variants) | ~1.5 s |
| Encoding (VideoToolbox vs x264 medium) | ~2× faster |

A "PC lento" preset keeps everything usable on older machines.

## Models

Small ONNX models run on the CPU through OpenCV / onnxruntime. Some ship with the repo, the others download on first use into `~/Library/Application Support/ShortsStudio/models` (macOS) or the platform equivalent — face restore (GPEN/GFPGAN/CodeFormer, 24–370 MB), BiSeNet beauty mask (89 MB), RVM matte (14 MB), U²-Net-p saliency (5 MB), SG-LLIE + URetinex low light (0.05–1.6 MB), YAMNet audio events (16 MB), ECAPA voice ID (84 MB), RIFE interpolation (21 MB), NanoTrack tracking (1.7 MB), DeepFilterNet binary (15 MB). Licences are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) — note that **CodeFormer and GPEN are for non-commercial use only**.

## Development

```bash
pip install -e ".[app]" pytest scikit-image
QT_QPA_PLATFORM=offscreen python -m pytest
```

The Resolve/Fusion logic is tested against a fake Resolve API; the app has offscreen GUI tests. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Support the project

If Shorts Studio saves you time, you can [sponsor its development](https://github.com/sponsors/canardu1) — or just star the repo and share a short you made with it.

## License

[GPL-3.0](LICENSE) © 2026 Carlo Anardu. Third-party components keep their own licences, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
