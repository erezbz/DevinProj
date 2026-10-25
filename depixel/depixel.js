/*
 * Depixelizing Pixel Art (Kopf & Lischinski, SIGGRAPH 2011), pure JavaScript.
 *
 * Pipeline: similarity graph -> resolve crossing diagonals (curves / sparse
 * pixels / islands heuristics) -> reshaped pixel cells -> visible boundary
 * curves -> quadratic B-splines smoothed by energy minimisation -> SVG.
 *
 * Works in the browser (window.Depixel) and in Node (require).
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.Depixel = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // 8 directions clockwise from east; y grows downwards.
  const DX = [1, 1, 0, -1, -1, -1, 0, 1];
  const DY = [0, 1, 1, 1, 0, -1, -1, -1];
  const BIT_E = 1, BIT_SE = 2, BIT_S = 4, BIT_SW = 8;

  const DEFAULTS = {
    yThreshold: 48, uThreshold: 7, vThreshold: 6, // hqx similarity thresholds (YUV)
    sparseWindow: 8,
    islandWeight: 5,
    smoothIterations: 24,
    positionWeight: 10,
    sharpAngle: 100, // degrees of turn above which a boundary corner stays sharp
    seed: 1,
  };

  function popcount(v) { let c = 0; while (v) { v &= v - 1; c++; } return c; }

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ---------------------------------------------------------------- graph

  function similarityGraph(img, o) {
    const { width: w, height: h, data } = img;
    const n = w * h;
    const Y = new Float32Array(n), U = new Float32Array(n), V = new Float32Array(n);
    const clear = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const r = data[4 * i], g = data[4 * i + 1], b = data[4 * i + 2];
      clear[i] = data[4 * i + 3] < 128 ? 1 : 0;
      Y[i] = 0.299 * r + 0.587 * g + 0.114 * b;
      U[i] = -0.168736 * r - 0.331264 * g + 0.5 * b;
      V[i] = 0.5 * r - 0.418688 * g - 0.081312 * b;
    }
    function similar(a, b) {
      if (clear[a] || clear[b]) return clear[a] === clear[b];
      return Math.abs(Y[a] - Y[b]) <= o.yThreshold &&
        Math.abs(U[a] - U[b]) <= o.uThreshold &&
        Math.abs(V[a] - V[b]) <= o.vThreshold;
    }
    const conn = new Uint8Array(n);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        for (let d = 0; d < 4; d++) {
          const nx = x + DX[d], ny = y + DY[d];
          if (nx < 0 || nx >= w || ny >= h) continue;
          const j = ny * w + nx;
          if (similar(i, j)) { conn[i] |= 1 << d; conn[j] |= 1 << (d + 4); }
        }
      }
    }
    return { w, h, conn, clear };
  }

  function neighbor(g, i, d) { return i + DX[d] + DY[d] * g.w; }

  function cut(g, i, d) {
    g.conn[i] &= ~(1 << d);
    g.conn[neighbor(g, i, d)] &= ~(1 << ((d + 4) & 7));
  }

  // Length of the valence-2 chain that contains edge a-b.
  function curveLength(g, a, b) {
    const { conn } = g;
    let len = 1;
    const limit = g.w * g.h;
    for (const [start, from] of [[a, b], [b, a]]) {
      let prev = from, cur = start;
      while (popcount(conn[cur]) === 2 && len < limit) {
        let next = -1;
        for (let d = 0; d < 8; d++) {
          if (!(conn[cur] & (1 << d))) continue;
          const j = neighbor(g, cur, d);
          if (j !== prev) { next = j; break; }
        }
        if (next < 0) break;
        len++; prev = cur; cur = next;
        if (cur === a || cur === b) return len; // closed loop
      }
    }
    return len;
  }

  function componentSize(g, start, x0, y0, x1, y1) {
    const { w, conn } = g;
    const seen = new Set([start]);
    const stack = [start];
    while (stack.length) {
      const i = stack.pop();
      for (let d = 0; d < 8; d++) {
        if (!(conn[i] & (1 << d))) continue;
        const j = neighbor(g, i, d);
        const x = j % w, y = (j / w) | 0;
        if (x < x0 || x > x1 || y < y0 || y > y1 || seen.has(j)) continue;
        seen.add(j); stack.push(j);
      }
    }
    return seen.size;
  }

  function resolveCrossings(g, o) {
    const { w, h, conn } = g;
    const crossings = [];
    let removedFlat = 0, resolved = 0;
    for (let y = 0; y < h - 1; y++) {
      for (let x = 0; x < w - 1; x++) {
        const tl = y * w + x, tr = tl + 1, bl = tl + w;
        if (!(conn[tl] & BIT_SE) || !(conn[tr] & BIT_SW)) continue;
        if ((conn[tl] & BIT_E) && (conn[tl] & BIT_S) && (conn[tr] & BIT_S) && (conn[bl] & BIT_E)) {
          cut(g, tl, 1); cut(g, tr, 3); removedFlat++;
        } else crossings.push([x, y]);
      }
    }
    const half = o.sparseWindow / 2;
    for (const [x, y] of crossings) {
      const tl = y * w + x, tr = tl + 1, bl = tl + w, br = bl + 1;
      if (!(conn[tl] & BIT_SE) || !(conn[tr] & BIT_SW)) continue;
      let w1 = 0, w2 = 0; // votes for keeping tl-br / tr-bl
      const l1 = curveLength(g, tl, br), l2 = curveLength(g, tr, bl);
      if (l1 > l2) w1 += l1 - l2; else w2 += l2 - l1;
      const x0 = x - half + 1, y0 = y - half + 1, x1 = x + half, y1 = y + half;
      const s1 = componentSize(g, tl, x0, y0, x1, y1), s2 = componentSize(g, tr, x0, y0, x1, y1);
      if (s1 < s2) w1 += s2 - s1; else w2 += s1 - s2;
      if (popcount(conn[tl]) === 1 || popcount(conn[br]) === 1) w1 += o.islandWeight;
      if (popcount(conn[tr]) === 1 || popcount(conn[bl]) === 1) w2 += o.islandWeight;
      if (w1 > w2) cut(g, tr, 3);
      else if (w2 > w1) cut(g, tl, 1);
      else { cut(g, tl, 1); cut(g, tr, 3); }
      resolved++;
    }
    return { removedFlat, resolved };
  }

  // ---------------------------------------------------------------- cells
  // Each lattice corner is shared by up to four pixels. When a diagonal
  // survives through the corner, the two connected pixels share a short edge
  // there (offset 1/4 pixel perpendicular to the diagonal), and the two other
  // pixels get their corner cut. Points are integer keys in quarter pixels.

  // CUT[type][k]: points visited at the pixel's corner k (0=TL,1=TR,2=BR,3=BL)
  // in clockwise order. type 1 = TL-BR diagonal, type 2 = TR-BL diagonal.
  const CUT = [null,
    [[[-1, 1], [1, -1]], [[-1, 1]], [[1, -1], [-1, 1]], [[1, -1]]],
    [[[1, 1]], [[-1, -1], [1, 1]], [[-1, -1]], [[1, 1], [-1, -1]]],
  ];
  const CX = [0, 1, 1, 0], CY = [0, 0, 1, 1];

  function buildCells(g) {
    const { w, h, conn } = g;
    const W = 4 * w + 1;
    const types = new Uint8Array((w + 1) * (h + 1));
    for (let cy = 1; cy < h; cy++) {
      for (let cx = 1; cx < w; cx++) {
        const tl = (cy - 1) * w + cx - 1;
        types[cy * (w + 1) + cx] = (conn[tl] & BIT_SE) ? 1 : (conn[tl + 1] & BIT_SW) ? 2 : 0;
      }
    }
    const cells = new Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const pts = [];
        for (let k = 0; k < 4; k++) {
          const cx = x + CX[k], cy = y + CY[k];
          const t = types[cy * (w + 1) + cx];
          if (!t) pts.push(4 * cy * W + 4 * cx);
          else for (const [dx, dy] of CUT[t][k]) pts.push((4 * cy + dy) * W + 4 * cx + dx);
        }
        cells[y * w + x] = pts;
      }
    }
    return { cells, W };
  }

  // -------------------------------------------------------------- regions

  function buildRegions(img, g) {
    const { w, h, conn, clear } = g;
    const { data } = img;
    const n = w * h;
    const parent = new Int32Array(n);
    for (let i = 0; i < n; i++) parent[i] = i;
    const find = (i) => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
    const same = (a, b) => (clear[a] || clear[b]) ? clear[a] === clear[b] :
      data[4 * a] === data[4 * b] && data[4 * a + 1] === data[4 * b + 1] && data[4 * a + 2] === data[4 * b + 2];
    for (let i = 0; i < n; i++) {
      for (let d = 0; d < 4; d++) {
        if (!(conn[i] & (1 << d))) continue;
        const j = neighbor(g, i, d);
        if (same(i, j)) parent[find(i)] = find(j);
      }
    }
    const regionOf = new Int32Array(n);
    const ids = new Map();
    const regions = [];
    for (let i = 0; i < n; i++) {
      const r = find(i);
      if (!ids.has(r)) {
        ids.set(r, regions.length);
        regions.push({
          color: clear[i] ? null : [data[4 * i], data[4 * i + 1], data[4 * i + 2]],
          size: 0,
        });
      }
      const id = ids.get(r);
      regionOf[i] = id;
      regions[id].size++;
    }
    return { regionOf, regions };
  }

  // ---------------------------------------------------------------- curves

  function buildEdges(cells, regionOf, M) {
    const map = new Map();
    const edges = [];
    for (let i = 0; i < cells.length; i++) {
      const pts = cells[i], r = regionOf[i];
      for (let k = 0; k < pts.length; k++) {
        const a = pts[k], b = pts[(k + 1) % pts.length];
        if (a === b) continue;
        const key = a < b ? a * M + b : b * M + a;
        const e = map.get(key);
        if (e === undefined) { map.set(key, edges.length); edges.push({ a, b, right: r, left: -1 }); }
        else edges[e].left = r;
      }
    }
    return edges.filter((e) => e.left !== e.right);
  }

  function traceCurves(edges, isFixedNode) {
    const adj = new Map();
    edges.forEach((e, idx) => {
      for (const v of [e.a, e.b]) {
        if (!adj.has(v)) adj.set(v, []);
        adj.get(v).push(idx);
      }
    });
    const isBreak = (v) => adj.get(v).length !== 2 || isFixedNode(v);
    const used = new Uint8Array(edges.length);
    const curves = [];
    function walk(start, ei) {
      const first = edges[ei];
      const right = first.a === start ? first.right : first.left;
      const left = first.a === start ? first.left : first.right;
      const pts = [start];
      let node = start, e = ei;
      for (;;) {
        used[e] = 1;
        const ed = edges[e];
        node = ed.a === node ? ed.b : ed.a;
        pts.push(node);
        if (node === start || isBreak(node)) break;
        const list = adj.get(node);
        e = list[0] === e ? list[1] : list[0];
        if (used[e]) break;
      }
      const closed = node === start && !isBreak(start);
      if (closed) pts.pop();
      return { pts, right, left, closed };
    }
    for (const [v, list] of adj) {
      if (!isBreak(v)) continue;
      for (const e of list) if (!used[e]) curves.push(walk(v, e));
    }
    for (let e = 0; e < edges.length; e++) if (!used[e]) curves.push(walk(edges[e].a, e));
    return curves;
  }

  function smoothCurve(P, fixed, closed, o, rand) {
    const n = P.length;
    if (n < 3) return;
    const orig = P.map((p) => [p[0], p[1]]);
    const idx = (j) => (closed ? (j + n) % n : j);
    function curv(j) {
      if (!closed && (j <= 0 || j >= n - 1)) return 0;
      j = idx(j);
      if (fixed[j]) return 0;
      // Squared turning angle: scale-invariant, so small shapes don't shrink.
      const a = P[idx(j - 1)], b = P[j], c = P[idx(j + 1)];
      const ux = b[0] - a[0], uy = b[1] - a[1], vx = c[0] - b[0], vy = c[1] - b[1];
      const t = Math.atan2(ux * vy - uy * vx, ux * vx + uy * vy);
      return t * t;
    }
    function local(i) {
      const dx = P[i][0] - orig[i][0], dy = P[i][1] - orig[i][1];
      const d2 = dx * dx + dy * dy;
      return curv(i - 1) + curv(i) + curv(i + 1) + o.positionWeight * d2 * d2;
    }
    const iters = o.smoothIterations;
    for (let it = 0; it < iters; it++) {
      const r = 0.02 + 0.23 * (1 - it / iters);
      for (let i = 0; i < n; i++) {
        if (fixed[i]) continue;
        let best = local(i);
        for (let t = 0; t < 6; t++) {
          const ox = P[i][0], oy = P[i][1];
          P[i][0] += (rand() * 2 - 1) * r;
          P[i][1] += (rand() * 2 - 1) * r;
          const e = local(i);
          if (e < best) best = e; else { P[i][0] = ox; P[i][1] = oy; }
        }
      }
    }
  }

  const fmt = (v) => +v.toFixed(3);
  const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  const pt = (p) => `${fmt(p[0])} ${fmt(p[1])}`;

  // Quadratic B-spline through fixed points; returns start point + commands.
  function splineCommands(P, fixed, closed) {
    const n = P.length;
    if (closed && !fixed.some(Boolean)) {
      const start = mid(P[n - 1], P[0]);
      let cmds = '';
      for (let k = 0; k < n; k++) cmds += `Q${pt(P[k])} ${pt(mid(P[k], P[(k + 1) % n]))}`;
      return { start, cmds };
    }
    let Q = P, F = fixed;
    if (closed) {
      const s = fixed.indexOf(true);
      Q = P.slice(s).concat(P.slice(0, s + 1));
      F = fixed.slice(s).concat(fixed.slice(0, s + 1));
    }
    const m = Q.length;
    let cmds = '';
    let s = 0;
    for (let e = 1; e < m; e++) {
      if (!F[e] && e !== m - 1) continue;
      if (e - s === 1) cmds += `L${pt(Q[e])}`;
      else {
        for (let k = s + 1; k < e; k++) {
          const end = k === e - 1 ? Q[e] : mid(Q[k], Q[k + 1]);
          cmds += `Q${pt(Q[k])} ${pt(end)}`;
        }
      }
      s = e;
    }
    return { start: Q[0], cmds };
  }

  // ------------------------------------------------------------- pipeline

  // Sprites on a transparent background get a transparent margin so their
  // outlines can be smoothed instead of being pinned to the image border.
  function padIfTransparentEdge(img) {
    const { width: w, height: h, data } = img;
    let edge = false;
    for (let x = 0; x < w && !edge; x++) edge = data[4 * x + 3] < 128 || data[4 * ((h - 1) * w + x) + 3] < 128;
    for (let y = 0; y < h && !edge; y++) edge = data[4 * y * w + 3] < 128 || data[4 * (y * w + w - 1) + 3] < 128;
    if (!edge) return { img, pad: 0 };
    const W = w + 2, out = new Uint8ClampedArray(W * (h + 2) * 4);
    for (let y = 0; y < h; y++) out.set(data.subarray(4 * y * w, 4 * (y + 1) * w), 4 * ((y + 1) * W + 1));
    return { img: { width: W, height: h + 2, data: out }, pad: 1 };
  }

  function depixelize(input, options) {
    const o = Object.assign({}, DEFAULTS, options || {});
    const t0 = Date.now();
    const { img, pad } = padIfTransparentEdge(input);
    const g = similarityGraph(img, o);
    const crossingStats = resolveCrossings(g, o);
    const { cells, W } = buildCells(g);
    const { regionOf, regions } = buildRegions(img, g);
    const { w, h } = g;
    const M = W * (4 * h + 1);
    const X = (k) => k % W, Yc = (k) => (k / W) | 0;
    const onBorder = (k) => { const x = X(k), y = Yc(k); return x === 0 || y === 0 || x === 4 * w || y === 4 * h; };

    const edges = buildEdges(cells, regionOf, M);
    const curves = traceCurves(edges, onBorder);
    const rand = mulberry32(o.seed);
    const cosSharp = Math.cos((o.sharpAngle * Math.PI) / 180);

    for (const c of curves) {
      c.P = c.pts.map((k) => [X(k) / 4, Yc(k) / 4]);
      const n = c.P.length;
      c.fixed = c.pts.map((k, i) => onBorder(k) || (!c.closed && (i === 0 || i === n - 1)));
      for (let i = 0; i < n; i++) {
        if (c.fixed[i] || (!c.closed && (i === 0 || i === n - 1))) continue;
        const a = c.P[(i - 1 + n) % n], b = c.P[i], d = c.P[(i + 1) % n];
        const ux = b[0] - a[0], uy = b[1] - a[1], vx = d[0] - b[0], vy = d[1] - b[1];
        const cos = (ux * vx + uy * vy) / Math.hypot(ux, uy) / Math.hypot(vx, vy);
        if (cos < cosSharp) c.fixed[i] = true;
      }
      smoothCurve(c.P, c.fixed, c.closed, o, rand);
      c.fwd = splineCommands(c.P, c.fixed, c.closed);
      c.rev = splineCommands(c.P.slice().reverse(), c.fixed.slice().reverse(), c.closed);
    }

    // Assemble each region's outline from the curves that bound it.
    const byRegion = regions.map(() => []);
    for (const c of curves) {
      const s = c.pts[0], e = c.pts[c.pts.length - 1];
      if (c.right >= 0) byRegion[c.right].push({ geo: c.fwd, s, e, closed: c.closed });
      if (c.left >= 0) byRegion[c.left].push({ geo: c.rev, s: e, e: s, closed: c.closed });
    }
    const paths = [];
    regions.forEach((reg, r) => {
      if (!reg.color) return;
      const pieces = byRegion[r];
      const used = new Array(pieces.length).fill(false);
      const fromNode = new Map();
      pieces.forEach((p, i) => {
        if (p.closed) return;
        if (!fromNode.has(p.s)) fromNode.set(p.s, []);
        fromNode.get(p.s).push(i);
      });
      let d = '';
      for (let i = 0; i < pieces.length; i++) {
        if (used[i]) continue;
        used[i] = true;
        const p = pieces[i];
        d += `M${pt(p.geo.start)}${p.geo.cmds}`;
        if (!p.closed) {
          let cur = p.e;
          while (cur !== p.s) {
            const next = (fromNode.get(cur) || []).find((j) => !used[j]);
            if (next === undefined) break;
            used[next] = true;
            d += pieces[next].geo.cmds;
            cur = pieces[next].e;
          }
        }
        d += 'Z';
      }
      paths.push({ d, color: reg.color, size: reg.size });
    });
    paths.sort((a, b) => b.size - a.size);

    return {
      width: w - 2 * pad, height: h - 2 * pad, pad, paths, cells, W,
      graph: g,
      stats: {
        regions: regions.length, curves: curves.length,
        flatDiagonalsRemoved: crossingStats.removedFlat,
        crossingsResolved: crossingStats.resolved,
        ms: Date.now() - t0,
      },
    };
  }

  function hex(c) { return '#' + c.map((v) => v.toString(16).padStart(2, '0')).join(''); }

  // `extra` is SVG markup drawn on top, in the same (padded) coordinates.
  function toSVG(result, scale, extra) {
    const s = scale || 1;
    const { width: w, height: h, pad } = result;
    const body = result.paths.map((p) => {
      const col = hex(p.color);
      return `<path d="${p.d}" fill="${col}" stroke="${col}" stroke-width="1" vector-effect="non-scaling-stroke" stroke-linejoin="round"/>`;
    }).join('\n');
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${pad} ${pad} ${w} ${h}" width="${w * s}" height="${h * s}">\n${body}${extra || ''}\n</svg>`;
  }

  // Reshaped cells as polygons in pixel units (for visualisation and tests).
  function cellPolygons(result) {
    const { W } = result;
    return result.cells.map((pts) => pts.map((k) => [(k % W) / 4, ((k / W) | 0) / 4]));
  }

  // Detect an integer upscale factor (e.g. a 4x screenshot) and undo it.
  function unscale(img) {
    const { width: w, height: h, data } = img;
    const px = (i) => ((data[4 * i] << 24) | (data[4 * i + 1] << 16) | (data[4 * i + 2] << 8) | data[4 * i + 3]) >>> 0;
    const gcd = (a, b) => { while (b) [a, b] = [b, a % b]; return a; };
    let f = gcd(w, h);
    for (let y = 0; y < h && f > 1; y++) {
      let run = 1;
      for (let x = 1; x < w && f > 1; x++) {
        if (px(y * w + x) === px(y * w + x - 1)) run++; else { f = gcd(f, run); run = 1; }
      }
    }
    for (let x = 0; x < w && f > 1; x++) {
      let run = 1;
      for (let y = 1; y < h && f > 1; y++) {
        if (px(y * w + x) === px((y - 1) * w + x)) run++; else { f = gcd(f, run); run = 1; }
      }
    }
    if (f <= 1) return { image: img, factor: 1 };
    const nw = w / f, nh = h / f, out = new Uint8ClampedArray(nw * nh * 4);
    const off = (f / 2) | 0;
    for (let y = 0; y < nh; y++) {
      for (let x = 0; x < nw; x++) {
        const si = ((y * f + off) * w + x * f + off) * 4, di = (y * nw + x) * 4;
        for (let c = 0; c < 4; c++) out[di + c] = data[si + c];
      }
    }
    return { image: { width: nw, height: nh, data: out }, factor: f };
  }

  // Build an image from rows of palette characters ('.' = transparent).
  function fromRows(rows, palette) {
    const h = rows.length, w = Math.max(...rows.map((r) => r.length));
    const data = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const ch = rows[y][x] || '.';
        if (ch === '.' || !palette[ch]) continue;
        const c = palette[ch];
        const v = parseInt(c.slice(1), 16);
        const i = (y * w + x) * 4;
        data[i] = v >> 16; data[i + 1] = (v >> 8) & 255; data[i + 2] = v & 255; data[i + 3] = 255;
      }
    }
    return { width: w, height: h, data };
  }

  return {
    DEFAULTS, depixelize, toSVG, cellPolygons, unscale, fromRows,
    _internal: { similarityGraph, resolveCrossings, curveLength, buildCells, splineCommands },
  };
});
