"""Draws the Auxein logo: every file in this directory is written by this script, from the same geometry.

    uv run --no-project --with fonttools --with pillow python docs/logo/generate.py

The mark is a phyllotaxis, the golden-angle spiral that the plant hormone auxin sets as a plant grows: each seed is a candidate,
the outermost ones the latest generation, and the seeds are joined along the two families of spirals a sunflower grows
(parastichies, every 13th and 21st seed). One seed of the outer ring, the champion, is in green. Small sizes (up to 48 px) use
a sparser spiral of 21 seeds joined every 5th and 8th, which stays legible as a favicon.

The wordmark is Archivo at width 125 and weight 600, and the tagline under it Space Mono (both SIL Open Font License),
converted to outlines so that the files need no installed font. The fonts are downloaded from the google/fonts repository when
the script runs.
"""

import io
import math
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ARCHIVO = "https://github.com/google/fonts/raw/main/ofl/archivo/Archivo%5Bwdth,wght%5D.ttf"
SPACE_MONO = "https://github.com/google/fonts/raw/main/ofl/spacemono/SpaceMono-Regular.ttf"
TAGLINE = "EVOLVING AGENTS"  # no version number: the files would have to change at every release
GOLD = math.radians(137.50776)
SIZE = 256  # the mark's viewBox


@dataclass(frozen=True)
class Palette:
    ground: str  # the background the mark is drawn for (the links are mixed towards it)
    ink: str
    dim: str
    accent: str


ON_DARK = Palette(ground="#0a0c0f", ink="#c9d0d6", dim="#5d6670", accent="#5ef2b0")
ON_LIGHT = Palette(ground="#ffffff", ink="#151a20", dim="#9aa3ab", accent="#0b9a62")


@dataclass(frozen=True)
class Spiral:
    seeds: int
    radius: float
    steps: tuple[tuple[int, float], ...]  # (parastichy step, relative line weight)
    first_link: int  # the innermost links would be long spokes
    seed_radius: tuple[float, float]  # innermost and outermost seed, in viewBox units
    champion_radius: float
    line_width: tuple[float, float]


FULL = Spiral(89, 100, ((13, 1.0), (21, 0.7)), 8, (1.2, 3.8), 6.0, (0.5, 1.6))
SMALL = Spiral(21, 98, ((5, 1.0), (8, 1.0)), 5, (5.0, 13.0), 18.0, (6.0, 10.0))


def mix(a: str, b: str, t: float) -> str:
    ca = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb, strict=True))


Line = tuple[float, float, float, float, float, str]  # x1, y1, x2, y2, width, colour
Dot = tuple[float, float, float, str]  # x, y, radius, colour


def mark(spiral: Spiral, palette: Palette) -> tuple[list[Line], list[Dot]]:
    """The mark's links and seeds, in a 256 x 256 box centred on (128, 128)."""
    n, c = spiral.seeds, spiral.radius / math.sqrt(spiral.seeds)
    points = [(128 + c * math.sqrt(k) * math.cos(k * GOLD), 128 - c * math.sqrt(k) * math.sin(k * GOLD)) for k in range(1, n + 1)]
    outer = range(n - 12, n + 1)  # the champion: the seed of the outer ring closest to the upper right
    champion = min(outer, key=lambda k: abs(math.atan2(128 - points[k - 1][1], points[k - 1][0] - 128) - math.radians(40)))
    lines: list[Line] = []
    for step, weight in spiral.steps:
        for k in range(spiral.first_link, n + 1 - step):
            (x1, y1), (x2, y2) = points[k - 1], points[k + step - 1]
            t = (k + step) / n
            width = (spiral.line_width[0] + (spiral.line_width[1] - spiral.line_width[0]) * t) * weight
            lines.append((x1, y1, x2, y2, width, mix(palette.ground, palette.ink, 0.25 + 0.45 * t)))
    dots: list[Dot] = []
    for k, (x, y) in enumerate(points, start=1):
        t = k / n
        if k == champion:
            dots.append((x, y, spiral.champion_radius, palette.accent))
        else:
            r = spiral.seed_radius[0] + (spiral.seed_radius[1] - spiral.seed_radius[0]) * t
            dots.append((x, y, r, mix(palette.dim, palette.ink, t**0.7)))
    return lines, dots


def svg_body(lines: list[Line], dots: list[Dot], dx: float = 0, dy: float = 0, scale: float = 1) -> str:
    def f(v: float) -> str:
        return f"{v:.2f}".rstrip("0").rstrip(".")

    out = [
        f'<line x1="{f(dx + x1 * scale)}" y1="{f(dy + y1 * scale)}" x2="{f(dx + x2 * scale)}" y2="{f(dy + y2 * scale)}" '
        f'stroke="{colour}" stroke-width="{f(w * scale)}" stroke-linecap="round"/>'
        for x1, y1, x2, y2, w, colour in lines
    ]
    out += [f'<circle cx="{f(dx + x * scale)}" cy="{f(dy + y * scale)}" r="{f(r * scale)}" fill="{colour}"/>' for x, y, r, colour in dots]
    return "\n".join(out)


def svg(width: float, height: float, body: str, background: str | None = None, title: str = "Auxein") -> str:
    back = f'<rect width="{width:g}" height="{height:g}" fill="{background}"/>\n' if background else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:g} {height:g}" width="{width:g}" height="{height:g}" '
        f'role="img" aria-label="{title}">\n<title>{title}</title>\n{back}{body}\n</svg>\n'
    )


def png(lines: list[Line], dots: list[Dot], size: int, background: str | None = None, inset: float = 0) -> Image.Image:
    """The mark rasterised at `size` pixels, drawn 8 times larger and reduced, for anti-aliasing."""
    over = 8
    big = size * over
    image = Image.new("RGBA", (big, big), background or (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    scale = big * (1 - 2 * inset) / SIZE
    offset = big * inset

    def p(v: float) -> float:
        return offset + v * scale

    for x1, y1, x2, y2, w, colour in lines:
        width = max(1, round(w * scale))
        draw.line((p(x1), p(y1), p(x2), p(y2)), fill=colour, width=width)
        for x, y in ((x1, y1), (x2, y2)):  # round caps
            draw.ellipse((p(x) - width / 2, p(y) - width / 2, p(x) + width / 2, p(y) + width / 2), fill=colour)
    for x, y, r, colour in dots:
        draw.ellipse((p(x) - r * scale, p(y) - r * scale, p(x) + r * scale, p(y) + r * scale), fill=colour)
    return image.resize((size, size), Image.Resampling.LANCZOS)


def ico(images: list[Image.Image]) -> bytes:
    """A .ico of PNG images (one per size), which every current browser reads."""
    blobs = []
    for image in images:
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        blobs.append(buffer.getvalue())
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries = b""
    for image, blob in zip(images, blobs, strict=True):
        side = image.width % 256  # 0 means 256
        entries += struct.pack("<BBBBHHII", side, side, 0, 0, 1, 32, len(blob), offset)
        offset += len(blob)
    return header + entries + b"".join(blobs)


def outline(text: str, url: str, tracking: float, axes: dict[str, float] | None = None) -> tuple[str, float, float]:
    """`text` as one SVG path in font units, with its width and cap height (the cap height at y = 0, y pointing down)."""
    with urllib.request.urlopen(url) as response:
        font = TTFont(io.BytesIO(response.read()))
    if axes:
        font = instantiateVariableFont(font, axes)
    glyphs, cmap, metrics = font.getGlyphSet(), font.getBestCmap(), font["hmtx"].metrics
    em = font["head"].unitsPerEm
    cap = font["OS/2"].sCapHeight
    pen = SVGPathPen(glyphs)
    x = 0.0
    for i, char in enumerate(text):
        name = cmap[ord(char)]
        glyphs[name].draw(TransformPen(pen, (1, 0, 0, -1, x, cap)))
        x += metrics[name][0] + (tracking * em if i < len(text) - 1 else 0)
    return pen.getCommands(), x, cap


def main() -> None:
    word, word_width, cap = outline("AUXEIN", ARCHIVO, 0.18, {"wdth": 125, "wght": 600})
    tag, tag_width, tag_cap = outline(TAGLINE, SPACE_MONO, 0.21)
    for name, palette in (("on-dark", ON_DARK), ("on-light", ON_LIGHT)):
        full, small = mark(FULL, palette), mark(SMALL, palette)
        (HERE / f"auxein-mark-{name}.svg").write_text(svg(SIZE, SIZE, svg_body(*full)))
        (HERE / f"auxein-mark-small-{name}.svg").write_text(svg(SIZE, SIZE, svg_body(*small)))
        for size in (16, 32, 48):
            png(*small, size).save(HERE / f"auxein-mark-{name}-{size}.png")
        for size in (64, 128, 256, 512, 1024):
            png(*full, size).save(HERE / f"auxein-mark-{name}-{size}.png")
        # the lockup: the mark, then the word (its cap height about a quarter of the mark's) over the tagline, centred on the mark
        height = 200.0
        word_cap = height * 0.27
        tag_cap_px = word_cap * 0.26
        line_gap = word_cap * 0.52
        top = (height - (word_cap + line_gap + tag_cap_px)) / 2
        left = height * 1.2
        s, ts = word_cap / cap, tag_cap_px / tag_cap
        width = left + max(word_width * s, tag_width * ts)
        body = svg_body(*full, scale=height / SIZE)
        body += f'\n<path transform="translate({left:g} {top:.2f}) scale({s:.5f})" fill="{palette.ink}" d="{word}"/>'
        place = f"translate({left + word_cap * 0.03:.2f} {top + word_cap + line_gap:.2f}) scale({ts:.5f})"
        body += f'\n<path transform="{place}" fill="{palette.dim}" d="{tag}"/>'
        (HERE / f"auxein-lockup-{name}.svg").write_text(svg(round(width, 2), height, body))
    # the avatar and the favicon carry their own ground, so they read on any background
    full, small = mark(FULL, ON_DARK), mark(SMALL, ON_DARK)
    (HERE / "auxein-avatar.svg").write_text(svg(SIZE, SIZE, svg_body(*full, dx=25.6, dy=25.6, scale=0.8), background=ON_DARK.ground))
    png(*full, 1024, background=ON_DARK.ground, inset=0.1).save(HERE / "auxein-avatar-1024.png")
    (HERE / "favicon.svg").write_text(svg(SIZE, SIZE, svg_body(*small, dx=12.8, dy=12.8, scale=0.9), background=ON_DARK.ground))
    (HERE / "favicon.ico").write_bytes(ico([png(*small, s, background=ON_DARK.ground, inset=0.05) for s in (16, 32, 48)]))


if __name__ == "__main__":
    main()
