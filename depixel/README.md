# Depixel

Pixel art → smooth, scalable vector art (SVG), in the browser. It's a pure-algorithm implementation of Kopf & Lischinski, *Depixelizing Pixel Art* (SIGGRAPH 2011): no AI, no server, and no build step.

Open `index.html` (works from `file://`), pick a sample or drop/paste a sprite, and download the SVG.

## Pipeline (`depixel.js`)
1. **Similarity graph.** Each pixel is linked to its 8 neighbours when their colours are close in YUV (hqx thresholds 48/7/6). Transparent pixels are only similar to each other.
2. **Diagonal resolution.** Fully similar 2×2 blocks drop both diagonals. Crossing diagonals are decided by the paper's votes: curve length, sparse pixels (8×8 window) and islands (weight 5).
3. **Reshaped cells.** Where a diagonal survives, the two pixels share a ¼-pixel edge and the other two cells are cut.
4. **Curves.** The visible edges between different colour regions are chained into curves, splitting at junctions, the image border and corners sharper than `sharpAngle`.
5. **Smoothing.** Quadratic B-spline control points are relaxed by randomised search. It minimises the squared turning angle (scale-invariant, so 1-pixel details don't shrink) plus `positionWeight·|Δp|⁴`.
6. **SVG.** There's one closed path per colour region, and shared curves keep neighbouring shapes gap-free.

Extras: automatic detection of integer upscaling (`unscale`), and a transparent margin for sprites on transparent backgrounds so their outlines can round off.

## Tests
```bash
node --test depixel/test/depixel.test.js
```

## Not yet implemented
- The paper's colour diffusion (smooth shading).
- Explicit corner patterns for keeping 90° corners sharp.
