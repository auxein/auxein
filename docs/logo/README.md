# The Auxein logo

The mark is a phyllotaxis: the golden-angle spiral that auxin, the plant hormone Auxein is named after, sets as a plant grows.
Each seed is a candidate, the outer ring the latest generation, and the seeds are joined along the spirals a sunflower grows
(every 13th and 21st seed), so the lattice is computed, not drawn. The green seed is the champion, the best candidate found.

Every file here is written by [`generate.py`](generate.py); change the script, not the files:

```sh
uv run --no-project --with fonttools --with pillow python docs/logo/generate.py
```

| File | Use |
|---|---|
| `auxein-lockup-on-light.svg`, `auxein-lockup-on-dark.svg` | the mark, the wordmark and the tagline ("EVOLVING AGENTS", deliberately without a version), for headers (the README uses both, switched by the theme) |
| `auxein-mark-on-light.svg`, `auxein-mark-on-dark.svg` | the mark alone, transparent, at 64 px and above |
| `auxein-mark-small-on-light.svg`, `auxein-mark-small-on-dark.svg` | the simplified mark (21 seeds), for 48 px and below |
| `auxein-mark-on-{light,dark}-{16,32,48,64,128,256,512,1024}.png` | the mark as PNGs, transparent; 16 to 48 px are the simplified mark |
| `auxein-avatar.svg`, `auxein-avatar-1024.png` | the mark on its graphite ground, for a GitHub or social avatar |
| `favicon.svg`, `favicon.ico` | the simplified mark on its ground; the `.ico` holds 16, 32 and 48 px |

"On light" files are for light backgrounds (dark ink), "on dark" for dark ones (silver ink).

| | On dark | On light |
|---|---|---|
| Ground | `#0a0c0f` | `#ffffff` |
| Ink | `#c9d0d6` | `#151a20` |
| Dim (inner seeds) | `#5d6670` | `#9aa3ab` |
| Champion | `#5ef2b0` | `#0b9a62` |

The wordmark is [Archivo](https://fonts.google.com/specimen/Archivo) at width 125 and weight 600, tracked 0.18 em, and the
tagline [Space Mono](https://fonts.google.com/specimen/Space+Mono), tracked 0.21 em, in the dim colour (both SIL Open Font
License), converted to outlines.
