// Narrowphase for MMD's three rigid body shapes: sphere, box, capsule (radius, cylinder
// height, along local Y). A pair test returns how far apart the shapes are — a lower bound,
// exact for round shapes — or, when they touch, a contact: a point on either surface, the
// normal from the first shape toward the second, and the penetration depth. Poses are a
// position plus a column-major 3×3 rotation.

export interface Contact {
  p1x: number; p1y: number; p1z: number;
  p2x: number; p2y: number; p2z: number;
  nx: number; ny: number; nz: number;
  depth: number;
}

export const SPHERE = 0;
export const BOX = 1;
export const CAPSULE = 2;

const seg1 = new Float64Array(6);
const seg2 = new Float64Array(6);
const cp = new Float64Array(6);
const pb = new Float64Array(6);
const loc = new Float64Array(9);
const axes = new Float64Array(18);
const GOLDEN = (Math.sqrt(5) - 1) / 2;
const GOLDEN_STEPS = 24;
/** Boxes this far apart along some axis need no closer look. */
const FAR_ENOUGH = 0.5;

/** Conservative bounds of a shape: min xyz, max xyz into out[oo..oo+5]. */
export function shapeAabb(shape: number, size: ArrayLike<number>, x: ArrayLike<number>, xo: number, R: Float64Array, ro: number, out: Float64Array, oo: number): void {
  let ex: number, ey: number, ez: number;
  if (shape === SPHERE) {
    ex = ey = ez = size[0];
  } else if (shape === CAPSULE) {
    const h = size[1] / 2, r = size[0];
    ex = Math.abs(R[ro + 3]) * h + r; ey = Math.abs(R[ro + 4]) * h + r; ez = Math.abs(R[ro + 5]) * h + r;
  } else {
    ex = Math.abs(R[ro]) * size[0] + Math.abs(R[ro + 3]) * size[1] + Math.abs(R[ro + 6]) * size[2];
    ey = Math.abs(R[ro + 1]) * size[0] + Math.abs(R[ro + 4]) * size[1] + Math.abs(R[ro + 7]) * size[2];
    ez = Math.abs(R[ro + 2]) * size[0] + Math.abs(R[ro + 5]) * size[1] + Math.abs(R[ro + 8]) * size[2];
  }
  out[oo] = x[xo] - ex; out[oo + 1] = x[xo + 1] - ey; out[oo + 2] = x[xo + 2] - ez;
  out[oo + 3] = x[xo] + ex; out[oo + 4] = x[xo + 1] + ey; out[oo + 5] = x[xo + 2] + ez;
}

/**
 * Separation of two shapes: positive = at least this far apart; otherwise they touch, `out`
 * holds the contact and the value is minus its depth.
 */
export function collide(
  shape1: number, size1: ArrayLike<number>, x1: ArrayLike<number>, o1: number, R1: Float64Array, r1: number,
  shape2: number, size2: ArrayLike<number>, x2: ArrayLike<number>, o2: number, R2: Float64Array, r2: number,
  out: Contact,
): number {
  // Round shapes first, boxes last, so each pair has one implementation.
  if (shape1 === BOX && shape2 !== BOX) {
    const separation = collide(shape2, size2, x2, o2, R2, r2, shape1, size1, x1, o1, R1, r1, out);
    if (separation > 0) return separation;
    const px = out.p1x, py = out.p1y, pz = out.p1z;
    out.p1x = out.p2x; out.p1y = out.p2y; out.p1z = out.p2z;
    out.p2x = px; out.p2y = py; out.p2z = pz;
    out.nx = -out.nx; out.ny = -out.ny; out.nz = -out.nz;
    return separation;
  }
  if (shape1 !== BOX && shape2 !== BOX) {
    const ra = coreSegment(shape1, size1, x1, o1, R1, r1, seg1);
    const rb = coreSegment(shape2, size2, x2, o2, R2, r2, seg2);
    return roundRound(ra, rb, out);
  }
  if (shape1 !== BOX) {
    const ra = coreSegment(shape1, size1, x1, o1, R1, r1, seg1);
    return roundBox(ra, size2, x2, o2, R2, r2, out);
  }
  return boxBox(size1, x1, o1, R1, r1, size2, x2, o2, R2, r2, out);
}

/** The segment (sphere: a point) a round shape is a radius around; returns the radius. */
function coreSegment(shape: number, size: ArrayLike<number>, x: ArrayLike<number>, xo: number, R: Float64Array, ro: number, out: Float64Array): number {
  const h = shape === CAPSULE ? size[1] / 2 : 0;
  const ax = R[ro + 3] * h, ay = R[ro + 4] * h, az = R[ro + 5] * h;
  out[0] = x[xo] - ax; out[1] = x[xo + 1] - ay; out[2] = x[xo + 2] - az;
  out[3] = x[xo] + ax; out[4] = x[xo + 1] + ay; out[5] = x[xo + 2] + az;
  return size[0];
}

/** Closest points of segments p1q1 and p2q2 (Ericson, Real-Time Collision Detection) into out[0..2], out[3..5]. */
function closestSegmentSegment(
  p1x: number, p1y: number, p1z: number, q1x: number, q1y: number, q1z: number,
  p2x: number, p2y: number, p2z: number, q2x: number, q2y: number, q2z: number, out: Float64Array,
): void {
  const d1x = q1x - p1x, d1y = q1y - p1y, d1z = q1z - p1z;
  const d2x = q2x - p2x, d2y = q2y - p2y, d2z = q2z - p2z;
  const rx = p1x - p2x, ry = p1y - p2y, rz = p1z - p2z;
  const a = d1x * d1x + d1y * d1y + d1z * d1z;
  const e = d2x * d2x + d2y * d2y + d2z * d2z;
  const f = d2x * rx + d2y * ry + d2z * rz;
  const EPS = 1e-12;
  let s: number, t: number;
  if (a <= EPS && e <= EPS) {
    s = t = 0;
  } else if (a <= EPS) {
    s = 0;
    t = clamp01(f / e);
  } else {
    const c = d1x * rx + d1y * ry + d1z * rz;
    if (e <= EPS) {
      t = 0;
      s = clamp01(-c / a);
    } else {
      const b = d1x * d2x + d1y * d2y + d1z * d2z;
      const denom = a * e - b * b;
      s = denom !== 0 ? clamp01((b * f - c * e) / denom) : 0;
      t = (b * s + f) / e;
      if (t < 0) {
        t = 0;
        s = clamp01(-c / a);
      } else if (t > 1) {
        t = 1;
        s = clamp01((b - c) / a);
      }
    }
  }
  out[0] = p1x + d1x * s; out[1] = p1y + d1y * s; out[2] = p1z + d1z * s;
  out[3] = p2x + d2x * t; out[4] = p2y + d2y * t; out[5] = p2z + d2z * t;
}

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

function roundRound(ra: number, rb: number, out: Contact): number {
  closestSegmentSegment(seg1[0], seg1[1], seg1[2], seg1[3], seg1[4], seg1[5], seg2[0], seg2[1], seg2[2], seg2[3], seg2[4], seg2[5], cp);
  const dx = cp[3] - cp[0], dy = cp[4] - cp[1], dz = cp[5] - cp[2];
  const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
  const depth = ra + rb - dist;
  if (depth <= 0) return -depth;
  let nx = 0, ny = 1, nz = 0;
  if (dist > 1e-9) {
    nx = dx / dist; ny = dy / dist; nz = dz / dist;
  }
  out.nx = nx; out.ny = ny; out.nz = nz;
  out.p1x = cp[0] + nx * ra; out.p1y = cp[1] + ny * ra; out.p1z = cp[2] + nz * ra;
  out.p2x = cp[3] - nx * rb; out.p2y = cp[4] - ny * rb; out.p2z = cp[5] - nz * rb;
  out.depth = depth;
  return -depth;
}

/** Box-local coordinates of a world point into out[oo..oo+2]. */
function toBoxLocal(px: number, py: number, pz: number, x: ArrayLike<number>, xo: number, R: Float64Array, ro: number, out: Float64Array, oo: number): void {
  const dx = px - x[xo], dy = py - x[xo + 1], dz = pz - x[xo + 2];
  out[oo] = R[ro] * dx + R[ro + 1] * dy + R[ro + 2] * dz;
  out[oo + 1] = R[ro + 3] * dx + R[ro + 4] * dy + R[ro + 5] * dz;
  out[oo + 2] = R[ro + 6] * dx + R[ro + 7] * dy + R[ro + 8] * dz;
}

/** Distance from a box-local point to the box (0 inside). */
function boxDistance(lx: number, ly: number, lz: number, size: ArrayLike<number>): number {
  const qx = Math.max(Math.abs(lx) - size[0], 0), qy = Math.max(Math.abs(ly) - size[1], 0), qz = Math.max(Math.abs(lz) - size[2], 0);
  return Math.sqrt(qx * qx + qy * qy + qz * qz);
}

/**
 * Signed distance from a box-local point to the box; fills pb with the nearest surface point
 * (0..2) and the outward normal there (3..5), both box-local.
 */
function pointBox(lx: number, ly: number, lz: number, size: ArrayLike<number>): number {
  const ex = size[0], ey = size[1], ez = size[2];
  const cx = lx < -ex ? -ex : lx > ex ? ex : lx;
  const cy = ly < -ey ? -ey : ly > ey ? ey : ly;
  const cz = lz < -ez ? -ez : lz > ez ? ez : lz;
  const dx = lx - cx, dy = ly - cy, dz = lz - cz;
  const d2 = dx * dx + dy * dy + dz * dz;
  if (d2 > 1e-18) {
    const d = Math.sqrt(d2);
    pb[0] = cx; pb[1] = cy; pb[2] = cz;
    pb[3] = dx / d; pb[4] = dy / d; pb[5] = dz / d;
    return d;
  }
  // Inside: push out through the nearest face.
  const px = ex - Math.abs(lx), py = ey - Math.abs(ly), pz = ez - Math.abs(lz);
  pb[0] = lx; pb[1] = ly; pb[2] = lz; pb[3] = pb[4] = pb[5] = 0;
  if (px <= py && px <= pz) {
    const s = lx < 0 ? -1 : 1;
    pb[0] = s * ex; pb[3] = s;
    return -px;
  }
  if (py <= pz) {
    const s = ly < 0 ? -1 : 1;
    pb[1] = s * ey; pb[4] = s;
    return -py;
  }
  const s = lz < 0 ? -1 : 1;
  pb[2] = s * ez; pb[5] = s;
  return -pz;
}

/** A sphere of radius r at box-local point l against the box (shape 1 = sphere, 2 = box). */
function sphereBoxLocal(lx: number, ly: number, lz: number, r: number, size: ArrayLike<number>, x: ArrayLike<number>, xo: number, R: Float64Array, ro: number, out: Contact): number {
  const s = pointBox(lx, ly, lz, size);
  const depth = r - s;
  if (depth <= 0) return -depth;
  // Back to world: p = c + R·l, n = R·n_local (pointing from the box toward the sphere).
  const nx = R[ro] * pb[3] + R[ro + 3] * pb[4] + R[ro + 6] * pb[5];
  const ny = R[ro + 1] * pb[3] + R[ro + 4] * pb[4] + R[ro + 7] * pb[5];
  const nz = R[ro + 2] * pb[3] + R[ro + 5] * pb[4] + R[ro + 8] * pb[5];
  out.p2x = x[xo] + R[ro] * pb[0] + R[ro + 3] * pb[1] + R[ro + 6] * pb[2];
  out.p2y = x[xo + 1] + R[ro + 1] * pb[0] + R[ro + 4] * pb[1] + R[ro + 7] * pb[2];
  out.p2z = x[xo + 2] + R[ro + 2] * pb[0] + R[ro + 5] * pb[1] + R[ro + 8] * pb[2];
  const wx = x[xo] + R[ro] * lx + R[ro + 3] * ly + R[ro + 6] * lz;
  const wy = x[xo + 1] + R[ro + 1] * lx + R[ro + 4] * ly + R[ro + 7] * lz;
  const wz = x[xo + 2] + R[ro + 2] * lx + R[ro + 5] * ly + R[ro + 8] * lz;
  out.p1x = wx - nx * r; out.p1y = wy - ny * r; out.p1z = wz - nz * r;
  out.nx = -nx; out.ny = -ny; out.nz = -nz;
  out.depth = depth;
  return -depth;
}

/** Round shape (segment seg1, radius r) against a box. */
function roundBox(r: number, size: ArrayLike<number>, x: ArrayLike<number>, xo: number, R: Float64Array, ro: number, out: Contact): number {
  toBoxLocal(seg1[0], seg1[1], seg1[2], x, xo, R, ro, loc, 0);
  toBoxLocal(seg1[3], seg1[4], seg1[5], x, xo, R, ro, loc, 3);
  const dx = loc[3] - loc[0], dy = loc[4] - loc[1], dz = loc[5] - loc[2];
  let t = 0;
  if (dx * dx + dy * dy + dz * dz > 1e-18) {
    // The distance from a point sliding along the segment to the box is convex in the
    // parameter: a golden-section search finds the nearest point.
    let lo = 0, hi = 1;
    let t1 = hi - GOLDEN * (hi - lo), t2 = lo + GOLDEN * (hi - lo);
    let f1 = boxDistance(loc[0] + dx * t1, loc[1] + dy * t1, loc[2] + dz * t1, size);
    let f2 = boxDistance(loc[0] + dx * t2, loc[1] + dy * t2, loc[2] + dz * t2, size);
    for (let i = 0; i < GOLDEN_STEPS; i++) {
      if (f1 < f2) {
        hi = t2; t2 = t1; f2 = f1;
        t1 = hi - GOLDEN * (hi - lo);
        f1 = boxDistance(loc[0] + dx * t1, loc[1] + dy * t1, loc[2] + dz * t1, size);
      } else {
        lo = t1; t1 = t2; f1 = f2;
        t2 = lo + GOLDEN * (hi - lo);
        f2 = boxDistance(loc[0] + dx * t2, loc[1] + dy * t2, loc[2] + dz * t2, size);
      }
    }
    t = (lo + hi) / 2;
  }
  const lx = loc[0] + dx * t, ly = loc[1] + dy * t, lz = loc[2] + dz * t;
  const nearest = boxDistance(lx, ly, lz, size);
  if (nearest > r) return nearest - r;
  if (nearest > 0 || dx * dx + dy * dy + dz * dz <= 1e-18) return sphereBoxLocal(lx, ly, lz, r, size, x, xo, R, ro, out);
  return segmentBoxLocal(r, size, x, xo, R, ro, out);
}

/**
 * A segment (box-local, in loc) whose core runs through the box: separating-axis test over
 * the box faces and the segment × box-edge directions gives the shortest way out.
 */
function segmentBoxLocal(r: number, size: ArrayLike<number>, x: ArrayLike<number>, xo: number, R: Float64Array, ro: number, out: Contact): number {
  const dx = loc[3] - loc[0], dy = loc[4] - loc[1], dz = loc[5] - loc[2];
  const mx = (loc[0] + loc[3]) / 2, my = (loc[1] + loc[4]) / 2, mz = (loc[2] + loc[5]) / 2;
  axes.set([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, -dz, dy, dz, 0, -dx, -dy, dx, 0]);
  let best = Infinity, bx = 0, by = 1, bz = 0;
  for (let k = 0; k < 6; k++) {
    let ux = axes[k * 3], uy = axes[k * 3 + 1], uz = axes[k * 3 + 2];
    const len = Math.sqrt(ux * ux + uy * uy + uz * uz);
    if (len < 1e-9) continue;
    ux /= len; uy /= len; uz /= len;
    const hb = size[0] * Math.abs(ux) + size[1] * Math.abs(uy) + size[2] * Math.abs(uz);
    const hs = Math.abs(ux * dx + uy * dy + uz * dz) / 2;
    const dm = ux * mx + uy * my + uz * mz;
    const overlap = hb + hs - Math.abs(dm);
    if (overlap < best) {
      best = overlap;
      const s = dm < 0 ? -1 : 1;
      bx = ux * s; by = uy * s; bz = uz * s;
    }
  }
  // Deepest end of the segment along −u (its middle if both ends are equally deep).
  const da = loc[0] * bx + loc[1] * by + loc[2] * bz, db = loc[3] * bx + loc[4] * by + loc[5] * bz;
  let ex: number, ey: number, ez: number;
  if (Math.abs(da - db) < 1e-6) {
    ex = mx; ey = my; ez = mz;
  } else if (da < db) {
    ex = loc[0]; ey = loc[1]; ez = loc[2];
  } else {
    ex = loc[3]; ey = loc[4]; ez = loc[5];
  }
  const depth = best + r;
  const c1x = ex - bx * r, c1y = ey - by * r, c1z = ez - bz * r;
  const c2x = c1x + bx * depth, c2y = c1y + by * depth, c2z = c1z + bz * depth;
  out.p1x = x[xo] + R[ro] * c1x + R[ro + 3] * c1y + R[ro + 6] * c1z;
  out.p1y = x[xo + 1] + R[ro + 1] * c1x + R[ro + 4] * c1y + R[ro + 7] * c1z;
  out.p1z = x[xo + 2] + R[ro + 2] * c1x + R[ro + 5] * c1y + R[ro + 8] * c1z;
  out.p2x = x[xo] + R[ro] * c2x + R[ro + 3] * c2y + R[ro + 6] * c2z;
  out.p2y = x[xo + 1] + R[ro + 1] * c2x + R[ro + 4] * c2y + R[ro + 7] * c2z;
  out.p2z = x[xo + 2] + R[ro + 2] * c2x + R[ro + 5] * c2y + R[ro + 8] * c2z;
  out.nx = -(R[ro] * bx + R[ro + 3] * by + R[ro + 6] * bz);
  out.ny = -(R[ro + 1] * bx + R[ro + 4] * by + R[ro + 7] * bz);
  out.nz = -(R[ro + 2] * bx + R[ro + 5] * by + R[ro + 8] * bz);
  out.depth = depth;
  return -depth;
}

const C = new Float64Array(9);
const absC = new Float64Array(9);
const AD = new Float64Array(3);
const BD = new Float64Array(3);

/**
 * Boxes by the separating-axis test (15 axes). Apart: the widest gap found along an axis (a
 * lower bound on their distance). Touching: the contact is the deepest vertex or the closest
 * edge pair.
 */
function boxBox(
  e1: ArrayLike<number>, x1: ArrayLike<number>, o1: number, R1: Float64Array, r1: number,
  e2: ArrayLike<number>, x2: ArrayLike<number>, o2: number, R2: Float64Array, r2: number, out: Contact,
): number {
  const Dx = x2[o2] - x1[o1], Dy = x2[o2 + 1] - x1[o1 + 1], Dz = x2[o2 + 2] - x1[o1 + 2];
  for (let i = 0; i < 3; i++) {
    const ax = R1[r1 + i * 3], ay = R1[r1 + i * 3 + 1], az = R1[r1 + i * 3 + 2];
    AD[i] = ax * Dx + ay * Dy + az * Dz;
    for (let j = 0; j < 3; j++) {
      C[i * 3 + j] = ax * R2[r2 + j * 3] + ay * R2[r2 + j * 3 + 1] + az * R2[r2 + j * 3 + 2];
      absC[i * 3 + j] = Math.abs(C[i * 3 + j]) + 1e-9;
    }
  }
  for (let j = 0; j < 3; j++) BD[j] = R2[r2 + j * 3] * Dx + R2[r2 + j * 3 + 1] * Dy + R2[r2 + j * 3 + 2] * Dz;

  let best = Infinity;
  let bestKind = -1; // 0: face of box 1 (axis i), 1: face of box 2 (axis j), 2: edge i × edge j
  let bi = 0, bj = 0;
  let gap = 0; // widest separation seen (0 = none yet)
  for (let i = 0; i < 3; i++) {
    const overlap = e1[i] + e2[0] * absC[i * 3] + e2[1] * absC[i * 3 + 1] + e2[2] * absC[i * 3 + 2] - Math.abs(AD[i]);
    if (overlap <= 0) {
      if (-overlap >= FAR_ENOUGH) return -overlap;
      gap = Math.max(gap, -overlap);
    } else if (overlap < best) {
      best = overlap; bestKind = 0; bi = i;
    }
  }
  for (let j = 0; j < 3; j++) {
    const overlap = e1[0] * absC[j] + e1[1] * absC[3 + j] + e1[2] * absC[6 + j] + e2[j] - Math.abs(BD[j]);
    if (overlap <= 0) {
      if (-overlap >= FAR_ENOUGH) return -overlap;
      gap = Math.max(gap, -overlap);
    } else if (overlap < best) {
      best = overlap; bestKind = 1; bj = j;
    }
  }
  // Edge pairs; a face is kept unless an edge pair is clearly shallower (avoids flip-flopping).
  const faceBest = best;
  for (let i = 0; i < 3; i++) {
    const i1 = (i + 1) % 3, i2 = (i + 2) % 3;
    for (let j = 0; j < 3; j++) {
      const j1 = (j + 1) % 3, j2 = (j + 2) % 3;
      const len2 = 1 - C[i * 3 + j] * C[i * 3 + j];
      if (len2 < 1e-8) continue;
      const ra = e1[i1] * absC[i2 * 3 + j] + e1[i2] * absC[i1 * 3 + j];
      const rb = e2[j1] * absC[i * 3 + j2] + e2[j2] * absC[i * 3 + j1];
      const dist = Math.abs(C[i1 * 3 + j] * AD[i2] - C[i2 * 3 + j] * AD[i1]);
      const overlap = (ra + rb - dist) / Math.sqrt(len2);
      if (overlap <= 0) {
        if (-overlap >= FAR_ENOUGH) return -overlap;
        gap = Math.max(gap, -overlap);
      } else if (overlap < best && overlap < faceBest * 0.95) {
        best = overlap; bestKind = 2; bi = i; bj = j;
      }
    }
  }
  if (gap > 0) return gap;

  // The axis, pointing from box 1 toward box 2.
  let ux: number, uy: number, uz: number;
  if (bestKind === 0) {
    const s = AD[bi] < 0 ? -1 : 1;
    ux = R1[r1 + bi * 3] * s; uy = R1[r1 + bi * 3 + 1] * s; uz = R1[r1 + bi * 3 + 2] * s;
  } else if (bestKind === 1) {
    const s = BD[bj] < 0 ? -1 : 1;
    ux = R2[r2 + bj * 3] * s; uy = R2[r2 + bj * 3 + 1] * s; uz = R2[r2 + bj * 3 + 2] * s;
  } else {
    const ax = R1[r1 + bi * 3], ay = R1[r1 + bi * 3 + 1], az = R1[r1 + bi * 3 + 2];
    const bx = R2[r2 + bj * 3], by = R2[r2 + bj * 3 + 1], bz = R2[r2 + bj * 3 + 2];
    ux = ay * bz - az * by; uy = az * bx - ax * bz; uz = ax * by - ay * bx;
    const len = Math.sqrt(ux * ux + uy * uy + uz * uz);
    const s = (ux * Dx + uy * Dy + uz * Dz < 0 ? -1 : 1) / len;
    ux *= s; uy *= s; uz *= s;
  }
  out.nx = ux; out.ny = uy; out.nz = uz;
  out.depth = best;

  if (bestKind === 2) {
    // Supporting edges: along axis bi of box 1 (toward +u) and axis bj of box 2 (toward −u).
    let p1x = x1[o1], p1y = x1[o1 + 1], p1z = x1[o1 + 2];
    let p2x = x2[o2], p2y = x2[o2 + 1], p2z = x2[o2 + 2];
    for (let k = 0; k < 3; k++) {
      if (k !== bi) {
        const d = R1[r1 + k * 3] * ux + R1[r1 + k * 3 + 1] * uy + R1[r1 + k * 3 + 2] * uz;
        const s = (d < 0 ? -1 : 1) * e1[k];
        p1x += R1[r1 + k * 3] * s; p1y += R1[r1 + k * 3 + 1] * s; p1z += R1[r1 + k * 3 + 2] * s;
      }
      if (k !== bj) {
        const d = R2[r2 + k * 3] * ux + R2[r2 + k * 3 + 1] * uy + R2[r2 + k * 3 + 2] * uz;
        const s = (d < 0 ? 1 : -1) * e2[k];
        p2x += R2[r2 + k * 3] * s; p2y += R2[r2 + k * 3 + 1] * s; p2z += R2[r2 + k * 3 + 2] * s;
      }
    }
    const a1x = R1[r1 + bi * 3] * e1[bi], a1y = R1[r1 + bi * 3 + 1] * e1[bi], a1z = R1[r1 + bi * 3 + 2] * e1[bi];
    const a2x = R2[r2 + bj * 3] * e2[bj], a2y = R2[r2 + bj * 3 + 1] * e2[bj], a2z = R2[r2 + bj * 3 + 2] * e2[bj];
    closestSegmentSegment(p1x - a1x, p1y - a1y, p1z - a1z, p1x + a1x, p1y + a1y, p1z + a1z, p2x - a2x, p2y - a2y, p2z - a2z, p2x + a2x, p2y + a2y, p2z + a2z, cp);
    out.p1x = cp[0]; out.p1y = cp[1]; out.p1z = cp[2];
    out.p2x = cp[3]; out.p2y = cp[4]; out.p2z = cp[5];
    return -best;
  }
  if (bestKind === 0) {
    // Box 2's vertex farthest against u, pushed back onto box 1's face.
    let vx = x2[o2], vy = x2[o2 + 1], vz = x2[o2 + 2];
    for (let k = 0; k < 3; k++) {
      const d = R2[r2 + k * 3] * ux + R2[r2 + k * 3 + 1] * uy + R2[r2 + k * 3 + 2] * uz;
      const s = (d < 0 ? 1 : -1) * e2[k];
      vx += R2[r2 + k * 3] * s; vy += R2[r2 + k * 3 + 1] * s; vz += R2[r2 + k * 3 + 2] * s;
    }
    out.p2x = vx; out.p2y = vy; out.p2z = vz;
    out.p1x = vx + ux * best; out.p1y = vy + uy * best; out.p1z = vz + uz * best;
    return -best;
  }
  let vx = x1[o1], vy = x1[o1 + 1], vz = x1[o1 + 2];
  for (let k = 0; k < 3; k++) {
    const d = R1[r1 + k * 3] * ux + R1[r1 + k * 3 + 1] * uy + R1[r1 + k * 3 + 2] * uz;
    const s = (d < 0 ? -1 : 1) * e1[k];
    vx += R1[r1 + k * 3] * s; vy += R1[r1 + k * 3 + 1] * s; vz += R1[r1 + k * 3 + 2] * s;
  }
  out.p1x = vx; out.p1y = vy; out.p1z = vz;
  out.p2x = vx - ux * best; out.p2y = vy - uy * best; out.p2z = vz - uz * best;
  return -best;
}
