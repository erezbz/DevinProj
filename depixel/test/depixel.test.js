const test = require('node:test');
const assert = require('node:assert');
const D = require('../depixel.js');
const { PALETTE, SAMPLES } = require('../samples.js');

const area = (poly) => {
  let a = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i], [x2, y2] = poly[(i + 1) % poly.length];
    a += x1 * y2 - x2 * y1;
  }
  return a / 2;
};

test('samples are rectangular', () => {
  for (const s of SAMPLES) {
    const w = s.rows[0].length;
    for (const r of s.rows) assert.strictEqual(r.length, w, `${s.name}: "${r}"`);
  }
});

test('reshaped cells tile the image exactly', () => {
  for (const s of SAMPLES) {
    const img = D.fromRows(s.rows, PALETTE);
    const res = D.depixelize(img);
    const polys = D.cellPolygons(res);
    let total = 0;
    for (const p of polys) {
      const a = area(p);
      assert.ok(a > 0, `${s.name}: cell with non-positive area`);
      total += a;
    }
    const full = (img.width + 2 * res.pad) * (img.height + 2 * res.pad);
    assert.ok(Math.abs(total - full) < 1e-9, `${s.name}: area ${total}`);
  }
});

test('one-pixel diagonal line keeps its diagonal; background diagonal is cut', () => {
  const rows = ['k....', '.k...', '..k..', '...k.', '....k'];
  const img = D.fromRows(rows, { k: '#000000' });
  const g = D._internal.similarityGraph(img, D.DEFAULTS);
  D._internal.resolveCrossings(g, D.DEFAULTS);
  for (let i = 0; i < 4; i++) {
    const tl = i * 5 + i;
    assert.ok(g.conn[tl] & 2, 'black diagonal kept');
    assert.ok(!(g.conn[tl + 1] & 8), 'white diagonal removed');
  }
});

test('fully similar 2x2 block loses both diagonals', () => {
  const img = D.fromRows(['kk', 'kk'], { k: '#000000' });
  const g = D._internal.similarityGraph(img, D.DEFAULTS);
  D._internal.resolveCrossings(g, D.DEFAULTS);
  assert.strictEqual(g.conn[0] & 2, 0);
  assert.strictEqual(g.conn[1] & 8, 0);
});

test('checkerboard islands are resolved with the island heuristic', () => {
  const img = D.fromRows(['k.k', '.k.', 'k.k'], { k: '#000000' });
  const res = D.depixelize(img);
  assert.ok(res.stats.crossingsResolved >= 1);
});

test('svg: one closed path per opaque region, deterministic', () => {
  for (const s of SAMPLES) {
    const img = D.fromRows(s.rows, PALETTE);
    const a = D.depixelize(img), b = D.depixelize(img);
    const svg = D.toSVG(a, 10);
    assert.strictEqual(svg, D.toSVG(b, 10));
    assert.ok(svg.startsWith('<svg'));
    for (const p of a.paths) {
      assert.ok(p.d.startsWith('M') && p.d.endsWith('Z'));
      assert.ok(!/NaN|Infinity/.test(p.d), `${s.name}: bad number`);
    }
    assert.ok(a.paths.length > 0);
  }
});

test('region outlines are closed loops (pieces chain end to start)', () => {
  // A ring: the hole must come out as its own subpath.
  const rows = ['kkkkk', 'k...k', 'k...k', 'k...k', 'kkkkk'];
  const res = D.depixelize(D.fromRows(rows, { k: '#000000' }));
  assert.strictEqual(res.paths.length, 1);
  assert.strictEqual((res.paths[0].d.match(/M/g) || []).length, 2);
});

test('unscale detects integer upscaling', () => {
  const img = D.fromRows(SAMPLES[0].rows, PALETTE);
  const f = 3, w = img.width * f, h = img.height * f;
  const big = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const si = ((((y / f) | 0) * img.width) + ((x / f) | 0)) * 4, di = (y * w + x) * 4;
    for (let c = 0; c < 4; c++) big[di + c] = img.data[si + c];
  }
  const { image, factor } = D.unscale({ width: w, height: h, data: big });
  assert.strictEqual(factor, 3);
  assert.deepStrictEqual(Array.from(image.data), Array.from(img.data));
});
