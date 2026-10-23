import type { PhysicsRig } from '../rig';
import type { BodySet } from './body-set';
import { collide, shapeAabb, type Contact } from './collision';
import { mat3FromQuat } from './rotation';

// Rigid body contacts. Once per frame, pairs whose swept bounds overlap are collected
// (sort and sweep along X, MMD's collision groups, and no contact between jointed bodies, as
// Bullet disables it). Every substep the narrowphase runs on those pairs and the contacts are
// solved position-based: penetration is removed rigidly, static friction holds the touching
// points, then at velocity level dynamic friction and restitution are applied — the latter
// also cancels the velocity the position correction itself produced, so a body pushed out of
// a leg is carried along instead of shot out.
//
// Most candidate pairs never touch. Each test also yields how far apart the pair is, and the
// pair is not looked at again until the two bodies have travelled that far between them.

/** Extra room around a body's bounds when collecting pairs (model units). */
const BOUNDS_MARGIN = 0.1;
/** Above this many bodies, remembering every pair's clearance (n² entries) is not worth the memory. */
const MAX_REMEMBERED_BODIES = 1500;

const contact: Contact = { p1x: 0, p1y: 0, p1z: 0, p2x: 0, p2y: 0, p2z: 0, nx: 0, ny: 0, nz: 0, depth: 0 };
const T = new Float64Array(12);

export class ContactSolver {
  private contactCount = 0;
  private readonly n: number;
  private readonly shape: Int32Array;
  private readonly size: ArrayLike<number>[];
  /** Radius of the sphere around the body's center that holds its shape. */
  private readonly reach: Float64Array;
  private readonly friction: Float64Array;
  private readonly restitution: Float64Array;
  /** Takes part in collisions (has volume). */
  private readonly active: Uint8Array;
  /** Pairs allowed to collide, n × n. */
  private readonly allowed: Uint8Array;
  private readonly R: Float64Array;
  private readonly bounds: Float64Array;
  private readonly order: Int32Array;
  private pairs: Int32Array;
  private pairCount = 0;

  /** Per body: how far its surface may have moved in all, and its pose at the last accounting. */
  private readonly travel: Float64Array;
  private readonly lastX: Float64Array;
  private readonly lastQ: Float64Array;
  /** Per pair (n × n): separation at its last test, and the bodies' combined travel then. */
  private readonly clearance: Float64Array | null;
  private readonly clearanceTravel: Float64Array | null;

  private cI = new Int32Array(0);
  private cJ = new Int32Array(0);
  private cLocal = new Float64Array(0); // r1 (3), r2 (3) in body space
  private cNormal = new Float64Array(0);
  private cLambda = new Float64Array(0);
  private cApproach = new Float64Array(0); // normal velocity before the position solve
  private cFriction = new Float64Array(0);
  private cRestitution = new Float64Array(0);

  constructor(rig: PhysicsRig, private readonly bodies: BodySet, enabled: boolean) {
    const n = rig.bodies.length;
    this.n = n;
    this.shape = Int32Array.from(rig.bodies, (b) => b.shape);
    this.size = rig.bodies.map((b) => b.size);
    this.reach = Float64Array.from(rig.bodies, (b) => {
      const s = b.size;
      return b.shape === 0 ? s[0] : b.shape === 2 ? s[0] + s[1] / 2 : Math.sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
    });
    this.friction = Float64Array.from(rig.bodies, (b) => b.source.friction);
    this.restitution = Float64Array.from(rig.bodies, (b) => b.source.restitution);
    this.active = Uint8Array.from(rig.bodies, (b) => {
      const s = b.size;
      const empty = b.shape === 0 ? s[0] <= 0 : b.shape === 1 ? s[0] <= 0 || s[1] <= 0 || s[2] <= 0 : s[0] <= 0;
      return empty || !enabled ? 0 : 1;
    });
    this.allowed = new Uint8Array(n * n);
    for (let i = 0; i < n; i++) {
      const a = rig.bodies[i].source;
      for (let j = i + 1; j < n; j++) {
        const b = rig.bodies[j].source;
        if (!this.active[i] || !this.active[j]) continue;
        if (bodies.invMass[i] === 0 && bodies.invMass[j] === 0) continue;
        if (((1 << a.group) & b.collisionMask) === 0 || ((1 << b.group) & a.collisionMask) === 0) continue;
        this.allowed[i * n + j] = this.allowed[j * n + i] = 1;
      }
    }
    for (const j of rig.joints) this.allowed[j.bodyA * n + j.bodyB] = this.allowed[j.bodyB * n + j.bodyA] = 0;
    this.R = new Float64Array(n * 9);
    this.bounds = new Float64Array(n * 6);
    this.order = Int32Array.from({ length: n }, (_, i) => i);
    this.pairs = new Int32Array(n * 8);
    this.travel = new Float64Array(n);
    this.lastX = new Float64Array(n * 3);
    this.lastQ = new Float64Array(n * 4);
    const remember = n <= MAX_REMEMBERED_BODIES;
    this.clearance = remember ? new Float64Array(n * n) : null;
    this.clearanceTravel = remember ? new Float64Array(n * n) : null;
    this.reserveContacts(n * 2);
    this.reset();
  }

  /** Forget remembered clearances (after bodies were teleported). */
  reset(): void {
    this.travel.fill(0);
    this.lastX.set(this.bodies.x);
    this.lastQ.set(this.bodies.q);
    this.clearance?.fill(0);
    this.clearanceTravel?.fill(0);
  }

  private reserveContacts(capacity: number): void {
    if (this.cI.length >= capacity) return;
    this.cI = new Int32Array(capacity);
    this.cJ = new Int32Array(capacity);
    this.cLocal = new Float64Array(capacity * 6);
    this.cNormal = new Float64Array(capacity * 3);
    this.cLambda = new Float64Array(capacity);
    this.cApproach = new Float64Array(capacity);
    this.cFriction = new Float64Array(capacity);
    this.cRestitution = new Float64Array(capacity);
  }

  /**
   * Collect the pairs that may touch during this frame of `dt` seconds. `kinematicEnd` holds,
   * per body, its pose at the end of the frame (position 3 + quaternion 4), used for the
   * kinematic bodies' sweep.
   */
  findCandidates(dt: number, kinematicEnd: Float64Array): void {
    this.pairCount = 0;
    const bs = this.bodies;
    const { n, R, bounds } = this;
    for (let i = 0; i < n; i++) {
      if (!this.active[i]) continue;
      mat3FromQuat(bs.q, i * 4, T);
      shapeAabb(this.shape[i], this.size[i], bs.x, i * 3, T, 0, bounds, i * 6);
      let grow = BOUNDS_MARGIN;
      if (bs.kinematic[i]) {
        mat3FromQuat(kinematicEnd, i * 7 + 3, T);
        shapeAabb(this.shape[i], this.size[i], kinematicEnd, i * 7, T, 0, R, 0);
        for (let k = 0; k < 3; k++) {
          bounds[i * 6 + k] = Math.min(bounds[i * 6 + k], R[k]);
          bounds[i * 6 + 3 + k] = Math.max(bounds[i * 6 + 3 + k], R[3 + k]);
        }
      } else {
        grow += Math.sqrt((bs.v[i * 3]) * (bs.v[i * 3]) + (bs.v[i * 3 + 1]) * (bs.v[i * 3 + 1]) + (bs.v[i * 3 + 2]) * (bs.v[i * 3 + 2])) * dt;
      }
      for (let k = 0; k < 3; k++) {
        bounds[i * 6 + k] -= grow;
        bounds[i * 6 + 3 + k] += grow;
      }
    }
    // Insertion sort by min x: nearly sorted from the previous frame.
    const order = this.order;
    for (let a = 1; a < n; a++) {
      const i = order[a];
      const key = bounds[i * 6];
      let b = a - 1;
      while (b >= 0 && bounds[order[b] * 6] > key) {
        order[b + 1] = order[b];
        b--;
      }
      order[b + 1] = i;
    }
    for (let a = 0; a < n; a++) {
      const i = order[a];
      if (!this.active[i]) continue;
      const maxX = bounds[i * 6 + 3];
      const row = i * n;
      for (let b = a + 1; b < n; b++) {
        const j = order[b];
        if (bounds[j * 6] > maxX) break;
        if (!this.allowed[row + j]) continue;
        if (bounds[j * 6 + 1] > bounds[i * 6 + 4] || bounds[j * 6 + 4] < bounds[i * 6 + 1]) continue;
        if (bounds[j * 6 + 2] > bounds[i * 6 + 5] || bounds[j * 6 + 5] < bounds[i * 6 + 2]) continue;
        if (this.pairCount * 2 + 2 > this.pairs.length) {
          const bigger = new Int32Array(this.pairs.length * 2);
          bigger.set(this.pairs);
          this.pairs = bigger;
        }
        this.pairs[this.pairCount * 2] = i;
        this.pairs[this.pairCount * 2 + 1] = j;
        this.pairCount++;
      }
    }
  }

  /** Add each body's motion since the last substep to its travel. */
  private accountTravel(): void {
    const { x, q } = this.bodies;
    const { travel, lastX, lastQ } = this;
    for (let i = 0; i < this.n; i++) {
      if (!this.active[i]) continue;
      const o = i * 3, qo = i * 4;
      const moved = Math.sqrt((x[o] - lastX[o]) * (x[o] - lastX[o]) + (x[o + 1] - lastX[o + 1]) * (x[o + 1] - lastX[o + 1]) + (x[o + 2] - lastX[o + 2]) * (x[o + 2] - lastX[o + 2]));
      const dot = Math.abs(q[qo] * lastQ[qo] + q[qo + 1] * lastQ[qo + 1] + q[qo + 2] * lastQ[qo + 2] + q[qo + 3] * lastQ[qo + 3]);
      const turned = 2 * Math.acos(Math.min(1, dot));
      travel[i] += moved + turned * this.reach[i];
      lastX[o] = x[o]; lastX[o + 1] = x[o + 1]; lastX[o + 2] = x[o + 2];
      lastQ[qo] = q[qo]; lastQ[qo + 1] = q[qo + 1]; lastQ[qo + 2] = q[qo + 2]; lastQ[qo + 3] = q[qo + 3];
    }
  }

  /** Narrowphase on the collected pairs at the bodies' current poses. */
  generate(): void {
    this.contactCount = 0;
    if (this.pairCount === 0) return;
    const bs = this.bodies;
    const { n, R, travel, clearance, clearanceTravel } = this;
    this.accountTravel();
    for (let i = 0; i < n; i++) if (this.active[i]) mat3FromQuat(bs.q, i * 4, R.subarray(i * 9, i * 9 + 9));
    this.reserveContacts(this.pairCount);
    for (let p = 0; p < this.pairCount; p++) {
      const i = this.pairs[p * 2], j = this.pairs[p * 2 + 1];
      const key = i * n + j;
      const travelled = travel[i] + travel[j];
      if (clearance && clearance[key] > travelled - clearanceTravel![key]) continue;
      // Bounding spheres first; the shapes only when those overlap.
      const io = i * 3, jo = j * 3;
      let separation = Math.sqrt((bs.x[jo] - bs.x[io]) * (bs.x[jo] - bs.x[io]) + (bs.x[jo + 1] - bs.x[io + 1]) * (bs.x[jo + 1] - bs.x[io + 1]) + (bs.x[jo + 2] - bs.x[io + 2]) * (bs.x[jo + 2] - bs.x[io + 2])) - this.reach[i] - this.reach[j];
      if (separation <= 0) separation = collide(this.shape[i], this.size[i], bs.x, io, R, i * 9, this.shape[j], this.size[j], bs.x, jo, R, j * 9, contact);
      if (clearance) {
        clearance[key] = Math.max(separation, 0);
        clearanceTravel![key] = travelled;
      }
      if (separation > 0) continue;
      const c = this.contactCount++;
      this.cI[c] = i; this.cJ[c] = j;
      bs.localPoint(i, contact.p1x, contact.p1y, contact.p1z, this.cLocal, c * 6);
      bs.localPoint(j, contact.p2x, contact.p2y, contact.p2z, this.cLocal, c * 6 + 3);
      this.cNormal[c * 3] = contact.nx; this.cNormal[c * 3 + 1] = contact.ny; this.cNormal[c * 3 + 2] = contact.nz;
      this.cApproach[c] = this.relativeVelocity(i, j, contact.p1x, contact.p1y, contact.p1z, contact.p2x, contact.p2y, contact.p2z, contact.nx, contact.ny, contact.nz);
      this.cFriction[c] = this.friction[i] * this.friction[j];
      this.cRestitution[c] = this.restitution[i] * this.restitution[j];
    }
  }

  /** Relative velocity of i's point p1 against j's point p2; its normal part is returned, the full vector left in T[6..8]. */
  private relativeVelocity(i: number, j: number, p1x: number, p1y: number, p1z: number, p2x: number, p2y: number, p2z: number, nx: number, ny: number, nz: number): number {
    const { v, w, x } = this.bodies;
    const io = i * 3, jo = j * 3;
    const r1x = p1x - x[io], r1y = p1y - x[io + 1], r1z = p1z - x[io + 2];
    const r2x = p2x - x[jo], r2y = p2y - x[jo + 1], r2z = p2z - x[jo + 2];
    T[6] = v[io] + (w[io + 1] * r1z - w[io + 2] * r1y) - (v[jo] + (w[jo + 1] * r2z - w[jo + 2] * r2y));
    T[7] = v[io + 1] + (w[io + 2] * r1x - w[io] * r1z) - (v[jo + 1] + (w[jo + 2] * r2x - w[jo] * r2z));
    T[8] = v[io + 2] + (w[io] * r1y - w[io + 1] * r1x) - (v[jo + 2] + (w[jo] * r2y - w[jo + 1] * r2x));
    return T[6] * nx + T[7] * ny + T[8] * nz;
  }

  /** Current world points of contact c into T[0..2] (on i) and T[3..5] (on j). */
  private points(c: number): void {
    const o = c * 6;
    this.bodies.worldPoint(this.cI[c], this.cLocal[o], this.cLocal[o + 1], this.cLocal[o + 2], T, 0);
    this.bodies.worldPoint(this.cJ[c], this.cLocal[o + 3], this.cLocal[o + 4], this.cLocal[o + 5], T, 3);
  }

  solvePositions(): void {
    const bs = this.bodies;
    for (let c = 0; c < this.contactCount; c++) {
      const i = this.cI[c], j = this.cJ[c];
      const nx = this.cNormal[c * 3], ny = this.cNormal[c * 3 + 1], nz = this.cNormal[c * 3 + 2];
      this.points(c);
      const d = (T[0] - T[3]) * nx + (T[1] - T[4]) * ny + (T[2] - T[5]) * nz;
      if (d <= 0) {
        this.cLambda[c] = 0;
        continue;
      }
      const lambda = bs.solvePair(i, j, -d * nx, -d * ny, -d * nz, T[0], T[1], T[2], T[3], T[4], T[5], true, 0, false);
      this.cLambda[c] = lambda;
      const mu = this.cFriction[c];
      if (mu <= 0) continue;
      // Static friction: undo this substep's tangential slip of the touching points.
      this.points(c);
      const o = c * 6;
      const { xPrev, qPrev } = bs;
      slidePrev(xPrev, qPrev, i, this.cLocal, o, T, 6);
      slidePrev(xPrev, qPrev, j, this.cLocal, o + 3, T, 9);
      let sx = (T[0] - T[6]) - (T[3] - T[9]);
      let sy = (T[1] - T[7]) - (T[4] - T[10]);
      let sz = (T[2] - T[8]) - (T[5] - T[11]);
      const sn = sx * nx + sy * ny + sz * nz;
      sx -= sn * nx; sy -= sn * ny; sz -= sn * nz;
      const slip = Math.sqrt(sx * sx + sy * sy + sz * sz);
      if (slip < 1e-9) continue;
      const w = bs.inverseMass(i, sx / slip, sy / slip, sz / slip, T[0], T[1], T[2], true) + bs.inverseMass(j, sx / slip, sy / slip, sz / slip, T[3], T[4], T[5], true);
      if (w > 0 && slip / w < mu * lambda) bs.solvePair(i, j, -sx, -sy, -sz, T[0], T[1], T[2], T[3], T[4], T[5], true, 0, false);
    }
  }

  /** Dynamic friction and restitution on the contacts the position solve acted on. */
  solveVelocities(h: number, gravity: number): void {
    const bs = this.bodies;
    for (let c = 0; c < this.contactCount; c++) {
      const lambda = this.cLambda[c];
      if (lambda <= 0) continue;
      const i = this.cI[c], j = this.cJ[c];
      const nx = this.cNormal[c * 3], ny = this.cNormal[c * 3 + 1], nz = this.cNormal[c * 3 + 2];
      this.points(c);
      const p1x = T[0], p1y = T[1], p1z = T[2], p2x = T[3], p2y = T[4], p2z = T[5];
      let vn = this.relativeVelocity(i, j, p1x, p1y, p1z, p2x, p2y, p2z, nx, ny, nz);
      const mu = this.cFriction[c];
      if (mu > 0) {
        const tx = T[6] - vn * nx, ty = T[7] - vn * ny, tz = T[8] - vn * nz;
        const vt = Math.sqrt(tx * tx + ty * ty + tz * tz);
        if (vt > 1e-9) {
          // Coulomb friction against the normal force λ / h², capped at stopping the slide.
          const k = -Math.min((mu * lambda) / h, vt) / vt;
          bs.solvePair(i, j, tx * k, ty * k, tz * k, p1x, p1y, p1z, p2x, p2y, p2z, true, 0, true);
          vn = this.relativeVelocity(i, j, p1x, p1y, p1z, p2x, p2y, p2z, nx, ny, nz);
        }
      }
      // Restitution — none for the slow approach of resting contact — minus the normal velocity
      // the position correction just added.
      const e = Math.abs(vn) <= 2 * gravity * h ? 0 : this.cRestitution[c];
      const dv = -vn + Math.max(-e * this.cApproach[c], 0);
      if (Math.abs(dv) > 1e-12) bs.solvePair(i, j, nx * dv, ny * dv, nz * dv, p1x, p1y, p1z, p2x, p2y, p2z, true, 0, true);
    }
  }
}

/** Where body b's local point was at the start of the substep, into out[oo..oo+2]. */
function slidePrev(xPrev: Float64Array, qPrev: Float64Array, b: number, local: Float64Array, lo: number, out: Float64Array, oo: number): void {
  const qo = b * 4;
  const qx = qPrev[qo], qy = qPrev[qo + 1], qz = qPrev[qo + 2], qw = qPrev[qo + 3];
  const vx = local[lo], vy = local[lo + 1], vz = local[lo + 2];
  const tx = 2 * (qy * vz - qz * vy), ty = 2 * (qz * vx - qx * vz), tz = 2 * (qx * vy - qy * vx);
  out[oo] = xPrev[b * 3] + vx + qw * tx + (qy * tz - qz * ty);
  out[oo + 1] = xPrev[b * 3 + 1] + vy + qw * ty + (qz * tx - qx * tz);
  out[oo + 2] = xPrev[b * 3 + 2] + vz + qw * tz + (qx * ty - qy * tx);
}
