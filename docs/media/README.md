# Demo media

The main README shows files from this folder.

## App UI shots (no footage needed)

`app.png` (full window) and `timeline.gif` (manual cut/restore) are rendered by
driving the real window offscreen on a synthetic clip:

```bash
python scripts/shoot_ui.py
```

## Feature demos (from your own footage)

Make them from your own footage — people in the videos must agree to appear:

| File | What | How |
|---|---|---|
| `reframe.gif` | original 16:9 next to the vertical short | `python scripts/make_demo.py compare original.mp4 short.mp4 -o docs/media/reframe.gif --start 12 --dur 6` |
| `captions.gif` | a few seconds of a finished short (captions, punch-in, moves) | `python scripts/make_demo.py clip short.mp4 -o docs/media/captions.gif --start 3 --dur 5` |
| `thumbnails.jpg` | the four thumbnails from *Copertine…* | `python scripts/make_demo.py grid a.jpg b.jpg c.jpg d.jpg -o docs/media/thumbnails.jpg` |

Keep each GIF under ~5 MB (lower `--fps`, `--height` or `--dur` if needed) so the README loads fast.
