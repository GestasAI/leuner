"""Generate the favicon set and the social share images for leuner.es.

Draws the dendrite mark of assets/favicon.svg with Pillow primitives (no SVG rasteriser is
installed), in the editorial visual system of the page: white ground, ink strokes, blue spines.
The favicon is the mark in negative (white strokes on ink) so it stays visible on any tab colour.
Fonts come from the system: Georgia Bold for display and Consolas for the small line
(the web page itself loads Manrope and Newsreader from Google Fonts).

Usage:  python make_images.py [--out dist]
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FONT_DIR = Path("C:/Windows/Fonts")

WHITE = (255, 255, 255)
INK = (14, 18, 22)
INK_2 = (74, 83, 97)
LINE = (227, 230, 234)
ACCENT = (26, 86, 196)
ACCENT_LIGHT = (127, 171, 236)


def font(name, size):
    return ImageFont.truetype(str(FONT_DIR / name), size)


def draw_mark(draw, x, y, size, stroke=None, ink=INK, ground=WHITE, spine=ACCENT):
    """The dendrite mark in a box of `size` px at (x, y). Geometry from assets/favicon.svg (64 units)."""
    s = size / 64.0
    w = stroke or max(2, round(4.5 * s))

    def p(px, py):
        return (x + px * s, y + py * s)

    def seg(a, b):
        draw.line([p(*a), p(*b)], fill=ink, width=w)
        for q in (a, b):  # round caps
            cx, cy = p(*q)
            draw.ellipse([cx - w / 2, cy - w / 2, cx + w / 2, cy + w / 2], fill=ink)

    seg((41, 41), (27, 27))
    seg((27, 27), (23, 8))
    seg((27, 27), (8, 23))
    cx, cy = p(47, 47)
    r = 7 * s
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=ground, outline=ink, width=w)
    for sx, sy in [(17.4, 13.1), (31.6, 17.1), (19, 32), (11.5, 30.4), (38.6, 29.4)]:
        cx, cy = p(sx, sy)
        r = 3.4 * s
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=spine)


def icon(size):
    """El símbolo libre sobre fondo transparente. Un halo blanco fino bajo el trazo lo mantiene
    visible en pestañas y pantallas oscuras sin encerrarlo en ningún recuadro."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    stroke = max(2, round(4.5 * size / 64))
    halo = stroke + max(2, round(size / 20))
    draw_mark(d, 0, 0, size, stroke=halo, ink=(255, 255, 255, 230), ground=(0, 0, 0, 0), spine=(0, 0, 0, 0))
    draw_mark(d, 0, 0, size, stroke=stroke, ink=INK, ground=(0, 0, 0, 0), spine=ACCENT)
    return img


SKETCH = (208, 214, 222)        # trazo del dibujo de fondo
SKETCH_SPINE = (176, 200, 236)  # espinas del dibujo de fondo


def draw_dendrite_sketch(draw, x0, y0, length, angle, depth, width, rng):
    """Árbol dendrítico a trazos finos: cada rama se divide en dos, más cortas y más finas.
    Las espinas aparecen en las ramas finas, como en una dendrita real."""
    import math
    x1 = x0 + length * math.cos(angle)
    y1 = y0 + length * math.sin(angle)
    draw.line([(x0, y0), (x1, y1)], fill=SKETCH, width=max(1, round(width)))
    if depth >= 3:
        for t in (0.35, 0.7):
            if rng.random() < 0.7:
                sx, sy = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
                side = rng.choice((-1, 1))
                ox, oy = -math.sin(angle) * side * 9, math.cos(angle) * side * 9
                r = 4
                draw.ellipse([sx + ox - r, sy + oy - r, sx + ox + r, sy + oy + r], fill=SKETCH_SPINE)
    if depth == 0 or length < 18:
        return
    for sign in (-1, 1):
        spread = math.radians(rng.uniform(18, 42))
        draw_dendrite_sketch(draw, x1, y1, length * rng.uniform(0.62, 0.78), angle + sign * spread,
                             depth - 1, width * 0.72, rng)


def background_dendrite(draw, soma, length, angle, seed=7):
    import math
    import random
    rng = random.Random(seed)
    sx, sy = soma
    r = 30
    draw.ellipse([sx - r, sy - r, sx + r, sy + r], outline=SKETCH, width=3, fill=WHITE)
    for a in (angle - 0.45, angle, angle + 0.5):
        draw_dendrite_sketch(draw, sx + r * math.cos(a), sy + r * math.sin(a), length, a, 6, 5, rng)


def share_image(w, h):
    import math
    img = Image.new("RGB", (w, h), WHITE)
    d = ImageDraw.Draw(img)
    if w > h:  # 1200 x 630: marca arriba a la izquierda, titular debajo, dendrita a trazos por detrás
        background_dendrite(d, (w - 150, h - 90), h * 0.34, math.radians(-118))
        tx = 90
        draw_mark(d, tx - 6, 84, 116, stroke=9)  # el símbolo libre, en tinta, sin recuadro
        d.text((tx + 128, 84), "Leuner", font=font("georgiab.ttf", 112), fill=INK)
        d.text((tx, 262), "Inteligencia artificial emocional", font=font("georgiab.ttf", 58), fill=INK)
        write_with_bold(d, (tx, 352), [("Un programa de investigación de ", False), ("GestasAI", True), (" que construye", False)],
                        font("georgia.ttf", 29), font("georgiab.ttf", 29), INK_2)
        d.text((tx, 390), "una dendrita artificial dentro de un modelo de lenguaje abierto.", font=font("georgia.ttf", 29), fill=INK_2)
        d.line([(tx, 470), (tx + 420, 470)], fill=LINE, width=2)
        d.text((tx, 492), "gestasai.com     leuner.es     juancarlosaguirre.es", font=font("consola.ttf", 22), fill=INK_2)
    else:  # 1200 x 1200: marca centrada arriba, titular debajo, dendrita a trazos por detrás
        background_dendrite(d, (w - 200, h - 120), 300, math.radians(-125))
        mark = 240
        draw_mark(d, (w - mark) // 2, 130, mark, stroke=18)  # el símbolo libre, en tinta, sin recuadro
        d.text((w / 2, 480), "Leuner", font=font("georgiab.ttf", 156), fill=INK, anchor="mm")
        d.text((w / 2, 640), "Inteligencia artificial", font=font("georgiab.ttf", 74), fill=INK, anchor="mm")
        d.text((w / 2, 728), "emocional", font=font("georgiab.ttf", 74), fill=INK, anchor="mm")
        write_with_bold(d, (w / 2, 860), [("Un programa de investigación de ", False), ("GestasAI", True), (" que construye", False)],
                        font("georgia.ttf", 34), font("georgiab.ttf", 34), INK_2, centered=True)
        d.text((w / 2, 906), "una dendrita artificial dentro de un modelo de lenguaje abierto.", font=font("georgia.ttf", 34), fill=INK_2, anchor="mm")
        d.line([(w / 2 - 220, 990), (w / 2 + 220, 990)], fill=LINE, width=2)
        d.text((w / 2, 1035), "gestasai.com     leuner.es     juancarlosaguirre.es", font=font("consola.ttf", 26), fill=INK_2, anchor="mm")
    return img


def write_with_bold(draw, pos, runs, regular, bold, fill, centered=False):
    """Escribe una línea con tramos en negrita. `runs` = [(texto, es_negrita), ...]."""
    widths = [(bold if b else regular).getlength(t) for t, b in runs]
    x, y = pos
    if centered:
        x -= sum(widths) / 2
        y -= regular.size / 2
    for (t, b), wdt in zip(runs, widths):
        draw.text((x, y), t, font=bold if b else regular, fill=fill)
        x += wdt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "dist"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    for size, name in [(16, "favicon-16.png"), (32, "favicon-32.png"), (180, "apple-touch-icon.png"),
                       (192, "icon-192.png"), (512, "icon-512.png")]:
        icon(size).save(out / name)
    icon(256).save(out / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])
    (out / "favicon.svg").write_bytes((HERE / "assets" / "favicon.svg").read_bytes())
    share_image(1200, 630).save(out / "og-image.png", optimize=True)
    share_image(1200, 1200).save(out / "og-image-square.png", optimize=True)
    for f in sorted(out.iterdir()):
        if f.suffix in (".png", ".ico", ".svg"):
            print(f"{f.name:<24}{f.stat().st_size / 1024:7.1f} KB")


if __name__ == "__main__":
    main()
