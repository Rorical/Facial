// MMD interpolation curves: cubic Bézier from (0,0) to (1,1) with control points
// (x1,y1), (x2,y2) given as bytes 0–127. Given progress x in [0,1], find the curve
// parameter t with Bx(t) = x, return By(t).

function bezier1D(p1: number, p2: number, t: number): number {
  const it = 1 - t;
  return 3 * it * it * t * p1 + 3 * it * t * t * p2 + t * t * t;
}

export function bezierWeight(x1: number, y1: number, x2: number, y2: number, x: number): number {
  if (x1 === y1 && x2 === y2) return x; // linear (MMD default is 20,20,107,107)
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  // Bx is monotonic because control x coordinates lie in [0,1] → bisection is safe.
  let lo = 0;
  let hi = 1;
  let t = x;
  for (let i = 0; i < 24; i++) {
    const bx = bezier1D(x1, x2, t) - x;
    if (Math.abs(bx) < 1e-6) break;
    if (bx > 0) hi = t;
    else lo = t;
    t = (lo + hi) * 0.5;
  }
  return bezier1D(y1, y2, t);
}

/** Evaluate using raw 0–127 bytes. */
export function bezierWeightBytes(x1: number, y1: number, x2: number, y2: number, x: number): number {
  return bezierWeight(x1 / 127, y1 / 127, x2 / 127, y2 / 127, x);
}

/** Curve parameter t with Bx(t) = x (Bx is monotonic for x1, x2 in [0, 1]). */
function solveT(x1: number, x2: number, x: number): number {
  let lo = 0;
  let hi = 1;
  let t = x;
  for (let i = 0; i < 40; i++) {
    const bx = bezier1D(x1, x2, t) - x;
    if (Math.abs(bx) < 1e-9) break;
    if (bx > 0) hi = t;
    else lo = t;
    t = (lo + hi) * 0.5;
  }
  return t;
}

export type Curve = [x1: number, y1: number, x2: number, y2: number];

/**
 * The part of an MMD curve (control points in [0,1]) between progress xa and xb (0 ≤ xa < xb ≤ 1),
 * renormalized to the unit square — i.e. the curve a key needs when a segment is cut.
 * Uses de Casteljau subdivision, so the shape is exact up to byte quantization.
 */
export function subCurve(curve: Curve, xa: number, xb: number): Curve {
  const [x1, y1, x2, y2] = curve;
  if (x1 === y1 && x2 === y2) return [x1, y1, x2, y2]; // linear stays linear
  const ta = solveT(x1, x2, xa);
  const tb = solveT(x1, x2, xb);
  // Cubic Bézier restricted to [ta, tb]: blossom P(u, v, w) of the original control points.
  const px = [0, x1, x2, 1];
  const py = [0, y1, y2, 1];
  const blossom = (p: number[], u: number, v: number, w: number) => {
    // de Casteljau with three (possibly different) parameters
    const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
    const a1 = [lerp(p[0], p[1], u), lerp(p[1], p[2], u), lerp(p[2], p[3], u)];
    const a2 = [lerp(a1[0], a1[1], v), lerp(a1[1], a1[2], v)];
    return lerp(a2[0], a2[1], w);
  };
  const qx = [blossom(px, ta, ta, ta), blossom(px, ta, ta, tb), blossom(px, ta, tb, tb), blossom(px, tb, tb, tb)];
  const qy = [blossom(py, ta, ta, ta), blossom(py, ta, ta, tb), blossom(py, ta, tb, tb), blossom(py, tb, tb, tb)];
  const dx = qx[3] - qx[0];
  const dy = qy[3] - qy[0];
  if (dx < 1e-9 || Math.abs(dy) < 1e-9) return [0.25, 0.25, 0.75, 0.75];
  const nx = (v: number) => (v - qx[0]) / dx;
  const ny = (v: number) => (v - qy[0]) / dy;
  return [nx(qx[1]), ny(qy[1]), nx(qx[2]), ny(qy[2])];
}

/** Curve in [0,1] → MMD bytes (0–127), clamped. */
export function curveToBytes(c: Curve): [number, number, number, number] {
  const b = (v: number) => Math.min(127, Math.max(0, Math.round(v * 127)));
  return [b(c[0]), b(c[1]), b(c[2]), b(c[3])];
}
