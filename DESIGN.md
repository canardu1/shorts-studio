---
name: Shorts Studio
description: Local, offline-first long-form-to-shorts video editor for creators
colors:
  ink-deep: "#0b0b0e"
  ink-base: "#0e0e12"
  ink-panel: "#141419"
  ink-raise: "#1c1c23"
  ink-lift: "#202027"
  ink-edge: "#232329"
  ink-line: "#2e2e37"
  coral: "#ff4d6d"
  coral-soft: "#ff8fa3"
  coral-deep: "#e2405c"
  coral-pale: "#ffd9df"
  amber: "#ff8a3d"
  amber-deep: "#e07530"
  tint-selected: "#2a141b"
  tint-hover: "#38222a"
  text-primary: "#f0f0f4"
  text-secondary: "#9a9aa8"
  text-faint: "#9494aa"
  success: "#00b140"
  warn: "#f5c542"
  pick-yellow: "#FFD400"
typography:
  body:
    fontFamily: "Manrope, sans-serif"
    fontSize: "12px"
    fontWeight: 400
  title:
    fontFamily: "Manrope, sans-serif"
    fontSize: "12px"
    fontWeight: 700
    letterSpacing: "0.04em"
  label:
    fontFamily: "Manrope, sans-serif"
    fontSize: "11px"
    fontWeight: 700
    letterSpacing: "0.06em"
  hint:
    fontFamily: "Manrope, sans-serif"
    fontSize: "11px"
    fontWeight: 400
  mono:
    fontFamily: "Menlo, Consolas, monospace"
    fontSize: "11px"
    fontWeight: 400
rounded:
  sm: "4px"
  md: "6px"
  lg: "8px"
spacing:
  xs: "4px"
  sm: "6px"
  md: "10px"
  lg: "14px"
components:
  button-accent:
    backgroundColor: "{colors.coral}"
    textColor: "#ffffff"
    rounded: "{rounded.md}"
    padding: "8px 16px"
  button-ghost:
    backgroundColor: "{colors.ink-lift}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.md}"
    padding: "7px 12px"
  input-field:
    backgroundColor: "{colors.ink-raise}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.sm}"
    padding: "5px 6px"
  list-item-selected:
    backgroundColor: "{colors.tint-selected}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.sm}"
  rail-tab-checked:
    backgroundColor: "{colors.tint-selected}"
    textColor: "{colors.coral}"
    rounded: "{rounded.md}"
  checkbox-checked:
    backgroundColor: "{colors.coral}"
    textColor: "#ffffff"
    rounded: "{rounded.sm}"
---

# Design System: Shorts Studio

## Overview

**Creative North Star: "Neon Editorial"**

A deep-black editing room lit by a single neon sign. The app is an Operate
surface — creators come to finish a job — so the dark, low-distraction shell
does the heavy lifting while one electric accent (coral bleeding into amber)
marks only the moments that matter: the render button, progress, the current
selection, the word being spoken. Restraint is the aesthetic; the glow is the
reward.

Everything sits on a stack of near-black surfaces separated by ~4% luminance
steps rather than borders. Hierarchy comes from tonal layering and typographic
weight, not from chrome. Controls are compact but never cramped; long-form
tools tolerate density because their users learn the surface once.

**Key Characteristics:**
- Near-black tonal layering, borders only where surfaces meet
- One accent gradient (coral → amber) reserved for *moment* elements
- Painted stroke icons, consistent weight, coral when active
- Manrope everywhere; Menlo only for the log (data, not decoration)
- English UI, Italian selectable — strings must stay short enough for both

## Colors

### Primary
- **Signal Coral** (`#ff4d6d`): the voice of action — primary buttons, checked
  states, active rail icon, focus accents, clip score marks.
- **Ember Amber** (`#ff8a3d`): never alone — only as the warm end of the accent
  gradient with coral.

### Neutral
- **Ink Deep** (`#0b0b0e`): window ground.
- **Ink Panel** (`#141419`): card/group surfaces, preview frame.
- **Ink Raise** (`#1c1c23`): inputs, segmented controls, raised rows.
- **Ink Edge** (`#232329`): hairline borders between surfaces.
- **Text Primary** (`#f0f0f4`): all readable content.
- **Text Secondary** (`#9a9aa8`): labels, secondary copy, status text.
- **Text Faint** (`#9494aa`): hints, placeholders, disabled — ~6:1 on the
  darkest surfaces; still prefer Secondary when the text carries an action.

### Accent Tints
- **Tint Selected** (`#2a141b`), **Tint Hover** (`#38222a`): coral-dyed darks
  for selected list items and rail tabs — the accent whispered, not shouted.

### State
- **Success** (`#00b140`), **Warn** (`#f5c542`), **Pick Yellow** (`#FFD400`):
  semantic only — status marks, warnings, the draggable crop rectangle.

### Named Rules
**The One Voice Rule.** Coral+amber appear on ≤10% of any screen. If two
elements on screen both demand the gradient, one of them is wrong — demote it
to ghost or a tint.

**The Tint, Not Paint Rule.** Selection and hover states dye surfaces with
coral-tinted darks (`tint-*`), never solid coral fills behind dark text.

## Typography

**Display/Body Font:** Manrope (embedded variable weight, OFL — bundled in
the app, packaged in the PyInstaller spec)
**Mono Font:** Menlo (system) — log output only

**Character:** A single geometric grotesque at humane sizes. Weight and
size steps do the work; there is no second voice.

### Hierarchy
- **Title** (700, 12px, tracked +0.04em, uppercase where used): group labels,
  section headers, the wordmark.
- **Body** (400, 12px): controls, list items, buttons.
- **Label** (700, 11px, tracked +0.06em): tab names, field labels.
- **Hint** (400, 11px, `text-faint`): explainer blocks under dense settings.
- **Mono** (400, 11px): the render log — measurement data, not a tech costume.

### Named Rules
**The Two-Size Rule.** UI lives at 11–13px. Anything larger must earn it
(the preview and the Create button do; nothing else has yet).

## Layout

Three-column shell: queue (left, ~230px) · preview + timeline (center, fluid) ·
settings rail + tab panel (right, ~340px scrollable). Bottom status bar spans
the window: stage label → update link → progress → Create → Stop.

- Settings rail is a vertical icon tab bar; the panel scrolls independently.
- Form rows use a label column + control column; hint paragraphs span full
  width under dense groups.
- Density: compact rows (~30px), `spacing.sm` between related controls,
  `spacing.lg` before group boundaries.

## Elevation & Depth

Flat by design. Depth is conveyed by the ~4% luminance steps between
`ink-deep → ink-panel → ink-raise → ink-lift` plus 1px `ink-edge` hairlines.
No drop shadows in QSS — Qt widgets don't carry them and faking them reads
as decoration. The "glow" of focus is a coral hairline + one-step lift, not
a shadow.

### Named Rules
**The Hairline Rule.** Focus and emphasis arrive as a 1px coral border and a
one-step surface lift. If you want a shadow, you want a tint instead.

## Shapes

Soft rectangles throughout: `sm` (4px) for list items, checkboxes, inputs;
`md` (6px) for buttons, rail pills, segmented controls; `lg` (8px) for cards
and panels. No circles except the slider handle and progress caps. Segmented
controls are pill groups on an `ink-raise` track; the active segment is an
`ink-lift` tile, not a coral fill.

## Components

### Buttons
- **Shape:** `rounded.md`, ~34px min height for primaries.
- **Primary (`button-accent`):** coral→amber *gradient* only for the two true
  moment actions — "Create short" and "Analyze faces". Everything else coral
  would break the One Voice Rule.
- **Ghost (`button-ghost`):** `ink-lift` surface, `text-primary`; hover lifts
  one tonal step.
- **Hover/Focus:** hover = one tonal step brighter or a tint; keyboard focus =
  1px coral hairline + `ink-lift` fill.

### Segmented Control (signature)
- Track: `ink-raise`, `rounded.md`. Options: transparent, `text-secondary`.
- Active option: `ink-lift` tile + `text-primary` — never a coral fill;
  the rail pill is the only accent-dyed tab.
- Must elide or wrap within the panel — overflow clipping is a defect.

### List Items (queue)
- Default: transparent, `text-primary`, `○` status mark prefix.
- Selected: `tint-selected` surface + 3px coral bar at the leading edge.
- Data glyphs (`↳ 🔥 👤`) are content markers, allowed inline in labels.

### Rail Tabs
- 56px icon + 11px label column; checked = `tint-selected` pill + coral icon.
- Icons are painted strokes, one weight, coral when active.

### Inputs
- `ink-raise` fill, `ink-edge` border, `rounded.sm`; focus swaps border to
  coral and fill to `ink-lift`.
- Checkboxes: `ink-raise` square, coral fill + white check when on.

### Progress
- Track: `ink-raise`; fill: the coral→amber gradient — the other licensed
  gradient moment, matching the Create button it feeds.

## Do's and Don'ts

### Do:
- **Do** reserve the coral→amber gradient for render-bound moments: Create
  button, progress fill, slider groove, caption highlight.
- **Do** express selection as `tint-*` surfaces with a coral edge, not fills.
- **Do** keep every interactive label short enough to fit a ~340px panel in
  both English and Italian — segmented controls elide with the full label
  in the tooltip.
- **Do** name every per-tab template row "<Scope> template" with the
  "— template —" placeholder; the word *preset* belongs to the global
  content preset only.
- **Do** tint secondary text from the surface hue; pure gray on tinted
  surfaces reads muddy.

### Don't:
- **Don't** let segmented controls or template rows overflow the settings
  panel — elide, wrap, or shorten the label.
- **Don't** use `text-faint` for information the user needs to act on —
  hints explain, secondary text instructs.
- **Don't** add a second accent hue for decoration; semantic colors only.
- **Don't** put borders above 1px or offset shadows on cards — depth is
  tonal layering, not chrome.
- **Don't** grow the type past 13px for chrome; the preview is the display.
