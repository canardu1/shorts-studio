# DaVinci Resolve tools

Turns horizontal footage into vertical shorts (9:16, 4:5, 1:1) and keeps the speaker's face centred. It works like Resolve Studio's Smart Reframe, but it only follows faces, it runs on the **free** version, and you can tune the camera motion.

How Face Reframe works:

1. **Detect.** OpenCV's YuNet face detector runs on every 2nd frame. The model ships with the repo. Hard cuts are found with a colour-histogram check.
2. **Classify shots.** Hard cuts are found from colour changes, from sudden changes in picture layout (so a cut from close-up to wide shot in the same set is caught), and from the face suddenly changing size. Each shot is labelled **close** or **wide** based on how big the face is. A wide shot is one where the face is under 12% of the frame height, for example the speaker seated on a stool. By default wide shots get an automatic punch-in, capped at 2×, so the speaker fills the vertical frame. The frame then follows the face vertically too, with the face 40% from the top. Choose "Keep full height" to leave wide shots uncropped vertically.
3. **Track.** The main face (largest, most confident, closest to the previous one) is followed through each shot. A median filter removes glitches. Then a dead zone, Gaussian smoothing and a pan speed limit turn the track into smooth "camera operator" motion. At a cut the frame jumps straight to the new position; it doesn't pan across.
4. **Apply.** One of two modes:
   - **Fusion keyframes (default, non-destructive):** each clip gets a Fusion comp `MediaIn → Merge (FG) over Background (timeline res) → MediaOut`, with `FaceReframeMerge.Center` keyframed. You can open the Fusion page and adjust any keyframe by hand.
   - **Render new file:** writes `<clip>_reframed_1080x1920_<start>.mp4` with ffmpeg (audio included), imports it and builds a new vertical timeline.

## Install

You need **Python 3** (the version Resolve uses for scripting), plus `numpy` and `opencv-python`. Render mode also needs **ffmpeg** on PATH.

```bash
python install.py --deps     # copies the script and vendors numpy + opencv into it
```

Dependencies are vendored inside the script's own folder (`face_reframe_lib/deps`), so they don't depend on which Python Resolve uses internally. Only the wheels for your OS and CPU are vendored — run the install on the machine where Resolve runs.

The script goes into the Resolve `Fusion/Scripts/Utility` folder:

| OS | Folder |
|---|---|
| Windows | `%APPDATA%\Blackmagic Design\DaVinci Resolve\Support\Fusion\Scripts\Utility` |
| macOS | `~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Utility` |
| Linux | `~/.local/share/DaVinciResolve/Fusion/Scripts/Utility` |

After installing, restart Resolve. If you moved Resolve or use both the free and Studio versions, reinstall for the other app data folder. If Resolve can't find Python, set the path in **Preferences > System > General > External scripting using** (or install Python 3 from python.org).

## Use in Resolve

1. Put your clip on a timeline. Move the playhead over it, or choose "All clips on track".
2. Open **Workspace > Scripts > Face Reframe**.
3. Pick an aspect and a mode, then click **Reframe**. Progress is shown in **Workspace > Console**.

In Fusion mode the timeline is switched to 1080×1920 (you can turn this off in the dialog). Each clip then gets a keyframed Fusion comp. Running the script again on the same clip replaces the old keyframes.

| Setting | Meaning |
|---|---|
| Smoothing (s) | How slowly the camera follows. 0.3 is snappy, 1.0 is cinematic. |
| Dead zone | How far the face can drift (as a fraction of the crop width) before the camera moves. 0 means always exactly centred. |
| Max pan speed | Upper limit on pan speed, in source widths per second. |
| Zoom | Punch-in on top of the 9:16 crop, applied to every shot. Combine it with "Follow face vertically". |
| Wide shots | "Zoom in on speaker" (default) or "Keep full height". |
| Wide if face < | Face height, as a fraction of the frame height, below which a shot counts as wide. |
| Max wide zoom | Upper limit on the wide-shot punch-in. With 1080p sources, values above about 1.5× get soft; 4K sources handle 2× or more. |

## Command line (no Resolve needed)

```bash
pip install -e .
face-reframe interview.mp4 -o short.mp4                      # 1080x1920, audio kept
face-reframe interview.mp4 --aspect 4:5 --smoothing 0.8 --dead-zone 0
face-reframe interview.mp4 --zoom 1.3 --follow-vertical --face-pos 0.4
face-reframe interview.mp4 --wide keep                       # don't punch in on wide shots
face-reframe interview.mp4 --max-zoom 1.5 --wide-threshold 0.1
face-reframe interview.mp4 --no-render --json track.json     # crop track + shot list only
```

`--debug` draws the tracked face position on the output.

### "mach-o file, but is an incompatible architecture"

Your Resolve runs the other CPU variant than the wheels you vendored. Re-run `--deps` with the arch from the error (`need 'x86_64'` -> `--arch x86_64`).

## Cut Pauses

A second pair of scripts removes silent pauses (auto jump-cuts). Run **Workspace > Scripts > Cut Pauses** (playhead clip) or **Cut Pauses (All Tracks)**. It analyses each clip's audio, finds the spoken segments, and builds a new timeline called `<name> (no pauses)` with only those segments — the original timeline is untouched.

- The silence threshold adapts to the clip's own noise floor (+8 dB margin). Pauses shorter than 0.35 s are kept, and 0.1 s of audio is kept around each speech segment so it doesn't sound clipped.
- Clips with no detected speech (b-roll, title cards) are kept whole.
- Needs ffmpeg to decode audio — `install.py --deps` vendors one (via imageio-ffmpeg), or `brew install ffmpeg`.

Standalone CLI:

```bash
python -m face_reframe.cut in.mp4 -o out.mp4 --min-pause 0.4 --padding 0.05 --json spans.json
```

## Cut Fillers

**Cut Fillers** / **Cut Fillers (All Tracks)** work like Cut Pauses but also remove filler words — *uhm, ehm, uh, em, ah, hmm* — using word-level Whisper transcription. They create a new `<name> (no pauses)` timeline; chain it after Cut Pauses on the generated timeline for a fully cleaned cut. The first run downloads the Whisper model (~460 MB for `small`) — you need internet once. Model choice: set `FACE_REFRAME_WHISPER_MODEL` (default `small`; `tiny`/`base` are faster but less accurate, `medium` is more accurate but much slower).

## Auto Subtitles

**Auto Subtitles** transcribes every clip on the timeline and imports a subtitle track — short, TikTok-style captions of up to 3 words / 1.6 s each. If subtitle import fails it writes an .srt you can import via File > Import > Subtitle (the path is printed to the Console). First run downloads the Whisper model.

## Punch In

**Punch In** / **Punch In (All Tracks)** add the classic shorts zoom: a Fusion `PunchInZoom` Transform node that toggles between the normal frame and a 1.18× zoom with a quick ease — **synced to the voice** (the zoom state flips at each detected phrase start, so it emphasises what is being said). When no speech is found it falls back to a fixed 3.5 s interval. It works on any clip, reframed or not — run it after Face Reframe and the zoom composes on top while the face stays centred. Re-running replaces the keyframes, no duplicates.

## Normalize Audio

**Normalize Audio** / **Normalize Audio (All Tracks)** measure each clip's integrated loudness with ffmpeg `loudnorm` and set the clip gain so everything lands at −14 LUFS. Resolve older than 21.1 exposes no clip-gain API, so there the script falls back to baking a normalized copy of each clip (video stream-copied losslessly, gain + safety limiter on the audio) and builds a `<name> (normalized)` timeline from it.

## Cinematic Audio

**Cinematic Audio** / **Cinematic Audio (All Tracks)** clean up dialogue and give it a filmic polish in one pass: adaptive FFT denoise, a de-esser, a downward expander that ducks the room/reverb tail between words (keyed on the pre-compression signal, so the compressor can't pump the room back up), rumble and boxiness cuts, gentle compression, then a warmth/presence/air EQ and a loudness trim to −14 LUFS behind a safety limiter. Resolve exposes no audio-FX scripting API, so the script always bakes a sweetened copy (video stream-copied, lossless) and rebuilds the timeline as `<name> (cinematic)`; the original timeline is untouched.

How the bake is organised:

- Segments sharing a source file — e.g. the dozens of segments a `(no pauses)` timeline holds — are processed **once per file**, not once per segment. The noise floor and speech level are measured on the *whole* source file: after Cut Pauses the true noise bed survives only there, so measuring on the segments would over-estimate it and make the denoise (and the expander threshold) too aggressive. The loudness for the −14 LUFS trim is measured on the union of the ranges the timeline actually uses.
- The "All Tracks" variant also sweeps **audio tracks**, so separately recorded dialogue (lav/boom on A2) gets cleaned too — the linked audio halves of A/V clips are skipped so nothing is processed twice. Items land back at their original timeline positions (`recordFrame`), so gaps and track stacking are preserved where the Resolve version honours it — otherwise they pack in order on track 1.
- Baked copies are written next to the source file, or into a temp folder when the source sits on a read-only mount (external drives, NAS).
- On ffmpeg older than 5 the `afftdn` gain-smoothing option (`gs`, which suppresses the metallic residue) doesn't exist — it's probed once and dropped automatically.

The dereverb stage tightens a roomy recording (big studios, halls) but doesn't deconvolve true echo — for severe reverb, run the clip through an ML dereverb tool first. Set `dereverb_db` (default 12 dB of max tail attenuation) or `--dereverb 0` to disable.

Standalone CLI:

```bash
python -m face_reframe.cinematic in.mp4 -o out.mp4 --denoise 15 --dereverb 15 --warmth 2 --presence 2.5
```

## Cut Breaths

**Cut Breaths** / **Cut Breaths (All Tracks)** are Cut Pauses with an extra pass: audible inhales/sighs inside the kept pauses (bursts above the noise floor but too soft to be speech) are cut too, leaving cleaner jump cuts. Heavier breathing is left alone.

## Split At Cuts

**Split At Cuts** / **Split At Cuts (All Tracks)** detect hard cuts inside each clip (histogram + structural change, frame-precise) and rebuild the timeline as `<name> (split)` with one item per shot — handy for grading or reframing shot by shot. Shots shorter than 0.5 s are merged back.

## Normalize Color

**Normalize Color** / **Normalize Color (All Tracks)** compute a per-clip auto white-balance + exposure correction (gray-world on sampled frames, targeting mid-grey luminance) and bake it into a small `.cube` LUT. Application order tried: the clip's LUT slot, then a `FaceReframeLUT` node in the clip's Fusion comp. LUTs are written to Resolve's own `Fusion/LUTs/face_reframe/` folder, so even when neither apply path exists they're listed in Effects > LUT > face_reframe and can be dragged onto the clip.

**No dialogs on Resolve Free 19.1+** — Blackmagic disabled `UIManager` on the free version, so script dialogs can't appear there; launchers fall back to options saved in `~/.face_reframe_prefs.json`. Run **Face Reframe Settings** (Workspace > Scripts) to open that file in a text editor — edit it, save, re-run the tool. Each tool reads its own section (`face_reframe`, `enhance`, `markers`, `pipeline`). On Resolve Studio the dialogs appear and write their choices back to the same file, so the '(All Tracks)' shortcuts reuse your last settings.

**Enhance** (a choice in the Face Reframe dialog, render mode only) sharpens the upscaled crop — not the whole frame, so it stays fast. `Fast` is lanczos + an unsharp mask with extra sharpening around the tracked face (cost: a few ms per frame). `AI` runs the compact Real-ESRGAN model (`realesr-general-x4v3`, vendored in `models/`) on a quarter-size input — the network's learned denoise/deblur does the restoration, then a lanczos pass reaches the output size. On Apple Silicon onnxruntime uses the CoreML/Neural Engine provider (~0.1–0.5 s per frame); on plain CPU it's several times slower — fine for shorts, painful for long clips. Needs `onnxruntime` (vendored by `install.py --deps`); without it the mode falls back to `fast`. CLI: `--enhance fast|ai`.

## Enhance (standalone)

**Enhance** applies the same sharpen/AI pass to the clips on the timeline *as they are*, at their own resolution — no re-crop, no face analysis. Run it after Face Reframe (on the rendered or reframed timeline), on native vertical clips, or on anything that needs a lift. `AI full` feeds the model input it needs to output native resolution — a real restoration pass, slow on CPU-only machines. The result lands in a `<name> (enhanced)` timeline; standalone audio items are carried over untouched.

Dependencies are vendored into `face_reframe_deps/<arch>` inside the Fusion directory — deliberately outside the Scripts folder, because Resolve scans Scripts/ and stalls on the tens of thousands of vendored files.

## Auto Markers

**Auto Markers** transcribes each clip (word-level Whisper) and drops timeline markers on it: **red** cue markers on configurable keywords said out loud while recording (`taglia`, `rifacciamo`, `retake`, `daccapo`, `ricominciamo` — editable in the dialog), and **blue** topic markers where speech resumes after a long pause, titled with the first words of the new block. Run it on the rough cut to find takes and chapters before editing.

## Cut Retakes

**Cut Retakes** / **Cut Retakes (All Tracks)** transcribe each clip and find *repeated phrases*: the same line said again right after (the classic "ok, again... <same line>"). The earlier take plus everything up to the keeper is cut, so the **last** take wins. Matching is fuzzy (difflib ratio ≥ 0.65) and conservative — short blocks ride along, unrelated content in between breaks the chain and is never cut. Same `(no pauses)` rebuild as the other cut tools; combine with `Cut Pauses` via `CutOptions(remove_retakes=True, ...)`.

## Sync Audio

**Sync Audio** aligns standalone audio items (a lav/boom recorder dumped on A2+) to the camera track — the job Resolve free can't do. For each unattached audio item it rebuilds the camera audio under the item's span, finds the offset by GCC-PHAT cross-correlation (robust to room tone and level differences, windows capped so hour-long files don't explode), then writes a shifted WAV copy and relinks the media pool clip to it — every use of that clip is then in sync. Low-confidence matches are reported, not applied. A fixed offset is assumed; clock drift between recorder and camera over very long takes isn't modelled yet.

## Duck Music

**Duck Music** lowers the music bed under the voice with a real sidechain compressor: the voice track is rebuilt in memory at its timeline positions and fed to ffmpeg `sidechaincompress` as the key, so the music ducks exactly where someone speaks. Each standalone audio item (music on its own track) is baked to a ducked copy and relinked in place — no Resolve audio-FX needed (none are scriptable). Depth/rate knobs are in `DuckParams`.

## Shorts Pipeline

**Shorts Pipeline** is the whole chain in one dialog: check the steps you want (`Cut Pauses → Cinematic Audio → Face Reframe → Punch In → Auto Subtitles`, in the order the README recommends), pick an aspect, click Run. Each bake step produces a new current timeline, so the next tool always works on the right one — that's also why Cut/Cinematic come *before* Face Reframe's Fusion comps. Console shows each step's progress.

## Recommended order

Tools that bake files or rebuild the timeline — Cut Pauses/Fillers/Breaths, Split At Cuts, Cinematic Audio, and Normalize Audio when it falls back to baking — rebuild the timeline from media pool items. Fusion comps (Face Reframe, Punch In), grading, titles and other tracks' items aren't carried over, so run them **first**, on the rough cut, then finish on the generated timeline. A typical short-form chain:

**Cut Pauses → Cinematic Audio → Face Reframe → Punch In → Normalize Color → Auto Subtitles**

## Updating

Run `python install.py` again (the vendored `deps` folder is kept). Or run it inside Resolve once — Workspace > Scripts only shows `.py` files you copied there, but `install.py` installs itself next to its own location when run from inside Resolve.

## Limitations

- Shot type is decided once per shot. A slow zoom by the camera operator inside one shot is followed horizontally, but the crop size doesn't change.
- Clips with speed changes or retiming, and clips whose frame rate differs from the timeline, may drift (the script warns about frame-rate mismatches).
- Only faces are tracked. If a shot has no face, the frame holds the last known position (or the centre).
- With several people in shot it follows the main face, or the person talking with *Follow: speaker* (from mouth movement) — the standalone app also lets you pick who to follow.

