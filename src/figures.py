"""Generate the report's charts as theme-aware inline SVG.

Each chart is emitted twice over: colours are written as
`var(--series-N, #hex)`, so the file renders correctly on its own (the hex
fallback) and picks up light/dark tokens when inlined into an HTML page that
defines them.  No chart library and no CDN dependency.

Palette: the dataviz reference categorical slots 1-3, validated all-pairs in
both modes (`validate_palette.js`).  Slot 3 (aqua) sits below 3:1 on the light
surface, so every chart here carries visible direct labels -- the relief rule.
"""
from __future__ import annotations

import html
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

FIG = os.path.join(os.path.dirname(__file__), "..", "output", "figures")

# role -> (light fallback, dark value); dark is emitted by the host page's CSS.
INK = {
    "primary": "var(--text-primary, #0b0b0b)",
    "secondary": "var(--text-secondary, #52514e)",
    "muted": "var(--text-muted, #78766f)",
    "grid": "var(--grid, #e4e2dc)",
    "surface": "var(--surface-1, #fcfcfb)",
}
SERIES = ["var(--series-1, #2a78d6)", "var(--series-2, #eb6834)",
          "var(--series-3, #1baf7a)"]
NEG = "var(--series-2, #eb6834)"

# Resolves to the host page's data face when one is defined, so charts inlined
# into the report share its typography; falls back to system UI standalone.
FONT = ("var(--chart-font, system-ui, -apple-system, 'Segoe UI', Roboto, "
        "'Helvetica Neue', Arial, sans-serif)")


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


class Svg:
    """Minimal SVG builder with a fixed viewBox and no external deps."""

    def __init__(self, w: int, h: int, title: str, subtitle: str = ""):
        self.w, self.h = w, h
        self.parts: list[str] = []
        self.title, self.subtitle = title, subtitle

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(self, x, y, s, size=12, fill=None, anchor="start",
             weight="400", opacity=1.0):
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
            f'font-family="{FONT}" fill="{fill or INK["secondary"]}" '
            f'text-anchor="{anchor}" font-weight="{weight}" '
            f'opacity="{opacity}">{_esc(s)}</text>')

    def rect(self, x, y, w, h, fill, rx=0, extra=""):
        if w <= 0 or h <= 0:
            return
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
                 f'height="{h:.1f}" fill="{fill}" rx="{rx}" {extra}/>')

    def line(self, x1, y1, x2, y2, stroke=None, width=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
                 f'y2="{y2:.1f}" stroke="{stroke or INK["grid"]}" '
                 f'stroke-width="{width}"{d}/>')

    def path(self, d, stroke=None, width=2, fill="none", cap="round"):
        self.add(f'<path d="{d}" fill="{fill}" stroke="{stroke or SERIES[0]}" '
                 f'stroke-width="{width}" stroke-linecap="{cap}" '
                 f'stroke-linejoin="round"/>')

    def circle(self, cx, cy, r, fill, ring=True):
        if ring:
            self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r + 2:.1f}" '
                     f'fill="{INK["surface"]}"/>')
        self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}"/>')

    def render(self) -> str:
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} '
                f'{self.h}" width="100%" role="img" '
                f'aria-label="{_esc(self.title)}" '
                f'style="max-width:100%;height:auto;display:block">')
        body = "".join(self.parts)
        return head + body + "</svg>"

    def save(self, name: str) -> str:
        os.makedirs(FIG, exist_ok=True)
        path = os.path.join(FIG, name)
        with open(path, "w") as f:
            f.write(self.render())
        return path


# ------------------------------------------------------------------ helpers
def _nice_ticks(vmax: float, n: int = 4) -> list[float]:
    if vmax <= 0:
        return [0.0]
    raw = vmax / n
    mag = 10 ** int(f"{raw:e}".split("e")[1])
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if step >= raw:
            break
    ticks, t = [], 0.0
    while t <= vmax * 1.0001 + step * 0.001:
        ticks.append(t)
        t += step
    return ticks


def _range_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Ticks inside [lo, hi] -- for series that never come near zero."""
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / n
    mag = 10 ** int(f"{raw:e}".split("e")[1])
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if step >= raw:
            break
    t = (int(lo / step) + (1 if lo > 0 else 0)) * step
    out = []
    while t <= hi:
        out.append(t)
        t += step
    return out


def _fmt(v: float, unit: str = "") -> str:
    if v == 0:
        return f"0{unit}"
    if abs(v) >= 1000:
        s = f"{v:,.0f}".replace(",", " ")
    elif abs(v) >= 100:
        s = f"{v:,.0f}"
    elif abs(v) >= 10:
        s = f"{v:,.1f}".replace(".", ",")
    else:
        s = f"{v:,.2f}".replace(".", ",")
    return f"{s}{unit}"


def hbar(name: str, title: str, subtitle: str, rows: list[tuple[str, float]],
         unit: str = "", label_w: int = 260, colours: list[str] | None = None,
         note: str = "") -> str:
    """Horizontal bars with direct value labels.  One series, no legend."""
    n = len(rows)
    bar_h, gap = 22, 12
    top = 54 if not subtitle else 74
    bottom = 44 + (18 if note else 0)
    h = top + n * (bar_h + gap) + bottom
    w = 860
    plot_x = label_w + 12
    plot_w = w - plot_x - 96

    s = Svg(w, h, title, subtitle)
    s.text(0, 22, title, size=15, fill=INK["primary"], weight="600")
    if subtitle:
        s.text(0, 42, subtitle, size=12, fill=INK["secondary"])

    vmax = max(max(v for _, v in rows), 0.0)
    vmin = min(min(v for _, v in rows), 0.0)
    span = (vmax - vmin) or 1.0
    zero_x = plot_x + (0 - vmin) / span * plot_w

    for t in _nice_ticks(vmax):
        x = plot_x + (t - vmin) / span * plot_w
        s.line(x, top - 8, x, top + n * (bar_h + gap) - gap + 4,
               stroke=INK["grid"], width=1)
        s.text(x, h - bottom + 26, _fmt(t), size=11, fill=INK["muted"], anchor="middle")

    for i, (lab, v) in enumerate(rows):
        y = top + i * (bar_h + gap)
        col = (colours or SERIES)[i % len(colours or SERIES)]
        s.text(label_w, y + bar_h * 0.72, lab, size=12, fill=INK["primary"],
               anchor="end")
        if v >= 0:
            s.rect(zero_x, y, (v - 0) / span * plot_w, bar_h, col, rx=4)
            s.text(zero_x + (v / span) * plot_w + 8, y + bar_h * 0.72,
                   _fmt(v, unit), size=12, fill=INK["primary"], weight="600")
        else:
            width = abs(v) / span * plot_w
            s.rect(zero_x - width, y, width, bar_h, NEG, rx=4)
            s.text(zero_x - width - 8, y + bar_h * 0.72, _fmt(v, unit), size=12,
                   fill=INK["primary"], anchor="end", weight="600")
    s.line(zero_x, top - 8, zero_x, top + n * (bar_h + gap) - gap + 4,
           stroke=INK["secondary"], width=1)
    if note:
        s.text(0, h - 6, note, size=11, fill=INK["muted"])
    return s.save(name)


def waterfall(name: str, title: str, subtitle: str,
              steps: list[tuple[str, float, str]], unit: str = "",
              note: str = "") -> str:
    """Waterfall: each step is (label, value, kind) with kind in
    {'start', 'delta', 'total'}."""
    w = 860
    n = len(steps)
    left = 64
    col_w = min(96, (w - left - 16) / n - 14)
    gapx = (w - left - 16 - n * col_w) / max(n - 1, 1)
    top, plot_h = 82, 260
    h = top + plot_h + 96
    left = 64

    running = 0.0
    bars = []
    for lab, v, kind in steps:
        if kind in ("start", "total"):
            bars.append((lab, 0.0, v, kind, v))
            running = v
        else:
            bars.append((lab, running, running + v, kind, v))
            running += v
    lo = min(min(b[1], b[2]) for b in bars)
    hi = max(max(b[1], b[2]) for b in bars)
    lo = min(lo, 0.0)
    span = (hi - lo) or 1.0

    def yv(v):
        return top + plot_h - (v - lo) / span * plot_h

    s = Svg(w, h, title, subtitle)
    s.text(0, 22, title, size=15, fill=INK["primary"], weight="600")
    if subtitle:
        s.text(0, 44, subtitle, size=12, fill=INK["secondary"])

    for t in _nice_ticks(hi):
        y = yv(t)
        s.line(left, y, w - 8, y, stroke=INK["grid"], width=1)
        s.text(left - 8, y + 4, _fmt(t), size=11, fill=INK["muted"], anchor="end")

    prev_end = None
    for i, (lab, y0, y1, kind, v) in enumerate(bars):
        x = left + i * (col_w + gapx)
        top_y, bot_y = min(yv(y0), yv(y1)), max(yv(y0), yv(y1))
        col = SERIES[0] if kind in ("start", "total") else (
            SERIES[2] if v >= 0 else NEG)
        # 2px surface gap keeps adjacent fills legible
        s.rect(x, top_y, col_w, max(bot_y - top_y, 2), col, rx=4)
        if prev_end is not None and kind == "delta":
            s.line(prev_end[0], prev_end[1], x, prev_end[1],
                   stroke=INK["muted"], width=1, dash="3 3")
        prev_end = (x + col_w, yv(y1))
        s.text(x + col_w / 2, top_y - 8 if v >= 0 else bot_y + 16,
               _fmt(v, unit), size=12, fill=INK["primary"], anchor="middle",
               weight="600")
        # wrapped category label
        words, line, lines = lab.split(), "", []
        for wd in words:
            if len(line + " " + wd) > 15 and line:
                lines.append(line)
                line = wd
            else:
                line = (line + " " + wd).strip()
        lines.append(line)
        for j, ln in enumerate(lines[:3]):
            s.text(x + col_w / 2, top + plot_h + 20 + j * 13, ln, size=11,
                   fill=INK["secondary"] if kind == "delta" else INK["primary"],
                   anchor="middle",
                   weight="600" if kind in ("start", "total") else "400")
    s.line(left, yv(0), w - 8, yv(0), stroke=INK["secondary"], width=1)
    if note:
        s.text(0, h - 8, note, size=11, fill=INK["muted"])
    return s.save(name)


def grouped_bars(name: str, title: str, subtitle: str, cats: list[str],
                 series: list[tuple[str, list[float]]], unit: str = "",
                 note: str = "", label_w: int = 300) -> str:
    """Horizontal grouped bars, two series, legend + direct labels."""
    n, k = len(cats), len(series)
    bar_h, inner, gap = 18, 2, 18
    top, bottom = 96, 48 + (18 if note else 0)
    h = top + n * (k * bar_h + (k - 1) * inner + gap) + bottom
    w = 880
    plot_x = label_w + 12
    plot_w = w - plot_x - 110

    s = Svg(w, h, title, subtitle)
    s.text(0, 22, title, size=15, fill=INK["primary"], weight="600")
    if subtitle:
        s.text(0, 44, subtitle, size=12, fill=INK["secondary"])

    lx = 0
    for i, (lab, _) in enumerate(series):
        s.rect(lx, 60, 11, 11, SERIES[i], rx=3)
        s.text(lx + 17, 70, lab, size=12, fill=INK["secondary"])
        lx += 17 + len(lab) * 6.6 + 22

    vmax = max(max(v for v in vals) for _, vals in series) or 1.0
    for t in _nice_ticks(vmax):
        x = plot_x + t / vmax * plot_w
        s.line(x, top - 10, x, h - bottom + 2, stroke=INK["grid"], width=1)
        s.text(x, h - bottom + 24, _fmt(t), size=11, fill=INK["muted"],
               anchor="middle")

    row_h = k * bar_h + (k - 1) * inner + gap
    for i, cat in enumerate(cats):
        y0 = top + i * row_h
        s.text(label_w, y0 + (k * bar_h) / 2 + 4, cat, size=12,
               fill=INK["primary"], anchor="end")
        for j, (_, vals) in enumerate(series):
            y = y0 + j * (bar_h + inner)
            bw = vals[i] / vmax * plot_w
            s.rect(plot_x, y, bw, bar_h, SERIES[j], rx=4)
            s.text(plot_x + bw + 8, y + bar_h * 0.75, _fmt(vals[i], unit),
                   size=11, fill=INK["primary"], weight="600")
    s.line(plot_x, top - 10, plot_x, h - bottom + 2, stroke=INK["secondary"], width=1)
    if note:
        s.text(0, h - 8, note, size=11, fill=INK["muted"])
    return s.save(name)


def dot_compare(name: str, title: str, subtitle: str, rows: list[tuple[str, float, float]],
                labels: tuple[str, str], unit: str = "", note: str = "",
                label_w: int = 240) -> str:
    """Two-value dot plot per row: shows one measure moving while the other does not."""
    n = len(rows)
    row_h = 30
    top, bottom = 96, 48 + (18 if note else 0)
    h = top + n * row_h + bottom
    w = 880
    plot_x = label_w + 16
    plot_w = w - plot_x - 80

    s = Svg(w, h, title, subtitle)
    s.text(0, 22, title, size=15, fill=INK["primary"], weight="600")
    if subtitle:
        s.text(0, 44, subtitle, size=12, fill=INK["secondary"])
    lx = 0
    for i, lab in enumerate(labels):
        s.circle(lx + 6, 66, 5, SERIES[i], ring=False)
        s.text(lx + 17, 70, lab, size=12, fill=INK["secondary"])
        lx += 17 + len(lab) * 6.6 + 24

    vmax = max(max(a, b) for _, a, b in rows) * 1.08
    for t in _nice_ticks(vmax):
        x = plot_x + t / vmax * plot_w
        s.line(x, top - 10, x, h - bottom + 2, stroke=INK["grid"], width=1)
        s.text(x, h - bottom + 24, _fmt(t), size=11, fill=INK["muted"], anchor="middle")

    for i, (lab, a, b) in enumerate(rows):
        y = top + i * row_h + row_h / 2
        xa, xb = plot_x + a / vmax * plot_w, plot_x + b / vmax * plot_w
        s.text(label_w, y + 4, lab, size=12, fill=INK["primary"], anchor="end")
        s.line(min(xa, xb), y, max(xa, xb), y, stroke=INK["grid"], width=2)
        s.circle(xb, y, 5, SERIES[1])
        s.circle(xa, y, 5, SERIES[0])
        s.text(max(xa, xb) + 12, y + 4, _fmt(a, unit), size=11,
               fill=INK["primary"], weight="600")
    s.line(plot_x, top - 10, plot_x, h - bottom + 2, stroke=INK["secondary"], width=1)
    if note:
        s.text(0, h - 8, note, size=11, fill=INK["muted"])
    return s.save(name)


def line_chart(name: str, title: str, subtitle: str, x: list, series: list[tuple[str, list]],
               unit: str = "", note: str = "", marker_last: bool = True,
               hlines: list[tuple[float, str]] | None = None) -> str:
    w, h = 860, 400
    left, right, top, bottom = 52, 118, 84, 60
    pw, ph = w - left - right, h - top - bottom

    s = Svg(w, h, title, subtitle)
    s.text(0, 22, title, size=15, fill=INK["primary"], weight="600")
    if subtitle:
        s.text(0, 44, subtitle, size=12, fill=INK["secondary"])

    allv = [v for _, ys in series for v in ys if v is not None]
    lo, hi = min(allv), max(allv)
    pad = (hi - lo) * 0.12 or 1
    lo, hi = lo - pad, hi + pad

    def px(i):
        return left + i / max(len(x) - 1, 1) * pw

    def py(v):
        return top + ph - (v - lo) / (hi - lo) * ph

    for t in _range_ticks(lo, hi, 5):
        s.line(left, py(t), left + pw, py(t), stroke=INK["grid"], width=1)
        s.text(left - 8, py(t) + 4, _fmt(t), size=11, fill=INK["muted"], anchor="end")
    step = max(1, len(x) // 8)
    for i in range(0, len(x), step):
        s.text(px(i), h - bottom + 22, x[i], size=11, fill=INK["muted"], anchor="middle")

    for hv, hl in (hlines or []):
        s.line(left, py(hv), left + pw, py(hv), stroke=INK["secondary"],
               width=1, dash="4 4")
        s.text(left + 6, py(hv) - 6, hl, size=11, fill=INK["secondary"])

    for j, (lab, ys) in enumerate(series):
        d = ""
        for i, v in enumerate(ys):
            if v is None:
                continue
            d += ("M" if not d else "L") + f"{px(i):.1f},{py(v):.1f}"
        s.path(d, stroke=SERIES[j], width=2)
        if marker_last:
            last = max(i for i, v in enumerate(ys) if v is not None)
            s.circle(px(last), py(ys[last]), 4.5, SERIES[j])
            s.text(px(last) + 10, py(ys[last]) + 4, lab, size=12,
                   fill=INK["primary"], weight="600")
    if note:
        s.text(0, h - 8, note, size=11, fill=INK["muted"])
    return s.save(name)
