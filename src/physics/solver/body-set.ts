import { FOLLOW_BONE, type PhysicsRig } from '../rig';
import { quatFromMat, rotateByVector, rotateVec, rotateVecInverse } from './rotation';

// Rigid body state and the two primitive operations of position-based dynamics: the
// generalized inverse mass seen along a direction, and applying a correction (a displacement
// or a rotation) weighted by it. Everything lives in flat arrays indexed by rig body.

/** Safety limits on speed (model units / s) and spin (rad / s) against pathological rigs. */
const MAX_SPEED = 500;
const MAX_SPIN = 200;

/** Bullet's inertia for the three shapes (a capsule counts as the box around it). */
function shapeInertia(shape: number, size: ArrayLike<number>, mass: number, out: Float64Array, o: number): void {
  if (shape === 0) {
    const i = 0.4 * mass * size[0] * size[0];
    out[o] = out[o + 1] = out[o + 2] = i;
    return;
  }
  const hx = size[0], hy = shape === 2 ? size[0] + size[1] / 2 : size[1], hz = shape === 2 ? size[0] : size[2];
  const lx = 2 * hx, ly = 2 * hy, lz = 2 * hz;
  out[o] = (mass / 12) * (ly * ly + lz * lz);
  out[o + 1] = (mass / 12) * (lx * lx + lz * lz);
  out[o + 2] = (mass / 12) * (lx * lx + ly * ly);
}

const t = new Float64Array(8);

export class BodySet {
  readonly count: number;
  /** Position (3 per body) and orientation quaternion (4 per body). */
  readonly x: Float64Array;
  readonly q: Float64Array;
  /** Pose at the start of the current substep. */
  readonly xPrev: Float64Array;
  readonly qPrev: Float64Array;
  /** Linear velocity and world-frame angular velocity (3 per body). */
  readonly v: Float64Array;
  readonly w: Float64Array;
  readonly invMass: Float64Array;
  /** Inverse inertia, body-frame diagonal (3 per body). */
  readonly invInertia: Float64Array;
  /** 1 = pose is dictated from outside (follow-bone body); constraints never move it. */
  readonly kinematic: Uint8Array;
  private readonly linearDamping: Float64Array;
  private readonly angularDamping: Float64Array;

  constructor(rig: PhysicsRig) {
    const n = rig.bodies.length;
    this.count = n;
    this.x = new Float64Array(n * 3);
    this.q = new Float64Array(n * 4);
    this.xPrev = new Float64Array(n * 3);
    this.qPrev = new Float64Array(n * 4);
    this.v = new Float64Array(n * 3);
    this.w = new Float64Array(n * 3);
    this.invMass = new Float64Array(n);
    this.invInertia = new Float64Array(n * 3);
    this.kinematic = new Uint8Array(n);
    this.linearDamping = new Float64Array(n);
    this.angularDamping = new Float64Array(n);
    rig.bodies.forEach((b, i) => {
      const kinematic = b.mode === FOLLOW_BONE;
      this.kinematic[i] = kinematic ? 1 : 0;
      // A dynamic body of mass 0 is immovable, as in Bullet.
      const mass = kinematic ? 0 : b.source.mass;
      if (mass > 0) {
        this.invMass[i] = 1 / mass;
        shapeInertia(b.shape, b.size, mass, this.invInertia, i * 3);
        for (let k = 0; k < 3; k++) this.invInertia[i * 3 + k] = this.invInertia[i * 3 + k] > 0 ? 1 / this.invInertia[i * 3 + k] : 0;
      }
      // MMD's damping is the fraction of velocity lost per second; 1 means the body keeps no
      // velocity at all between steps (it then only moves as far as its joints pull it), as in Bullet.
      this.linearDamping[i] = Math.min(1, Math.max(0, b.source.linearDamping));
      this.angularDamping[i] = Math.min(1, Math.max(0, b.source.angularDamping));
      this.setPose(i, b.rest);
    });
    this.xPrev.set(this.x);
    this.qPrev.set(this.q);
  }

  /** Place body i at a column-major 4×4 transform (rotation + translation). */
  setPose(i: number, m: ArrayLike<number>): void {
    this.x[i * 3] = m[12]; this.x[i * 3 + 1] = m[13]; this.x[i * 3 + 2] = m[14];
    quatFromMat(m, 4, this.q, i * 4);
  }

  /** Stop body i where it is, with no memory of earlier motion. */
  freeze(i: number): void {
    for (let k = 0; k < 3; k++) {
      this.v[i * 3 + k] = this.w[i * 3 + k] = 0;
      this.xPrev[i * 3 + k] = this.x[i * 3 + k];
    }
    for (let k = 0; k < 4; k++) this.qPrev[i * 4 + k] = this.q[i * 4 + k];
  }

  beginSubstep(): void {
    this.xPrev.set(this.x);
    this.qPrev.set(this.q);
  }

  /** Explicit step of the free bodies: gravity, then move along the velocities. */
  integrate(h: number, gx: number, gy: number, gz: number): void {
    const { x, q, v, w } = this;
    for (let i = 0; i < this.count; i++) {
      if (this.invMass[i] === 0) continue;
      const o = i * 3;
      v[o] += gx * h; v[o + 1] += gy * h; v[o + 2] += gz * h;
      x[o] += v[o] * h; x[o + 1] += v[o + 1] * h; x[o + 2] += v[o + 2] * h;
      rotateByVector(q, i * 4, w[o] * h, w[o + 1] * h, w[o + 2] * h);
    }
  }

  /** Velocities from the motion over the substep — for every body, so kinematic ones carry theirs into contacts. */
  updateVelocities(h: number): void {
    const { x, q, xPrev, qPrev, v, w } = this;
    const inv = 1 / h;
    for (let i = 0; i < this.count; i++) {
      const o = i * 3, qo = i * 4;
      v[o] = (x[o] - xPrev[o]) * inv; v[o + 1] = (x[o + 1] - xPrev[o + 1]) * inv; v[o + 2] = (x[o + 2] - xPrev[o + 2]) * inv;
      // Δq = q · conj(qPrev); ω = 2 · Δq.xyz / h, taking the short way round.
      const px = -qPrev[qo], py = -qPrev[qo + 1], pz = -qPrev[qo + 2], pw = qPrev[qo + 3];
      const ax = q[qo], ay = q[qo + 1], az = q[qo + 2], aw = q[qo + 3];
      const dx = ax * pw + aw * px + ay * pz - az * py;
      const dy = ay * pw + aw * py + az * px - ax * pz;
      const dz = az * pw + aw * pz + ax * py - ay * px;
      const dw = aw * pw - ax * px - ay * py - az * pz;
      const s = (dw < 0 ? -2 : 2) * inv;
      w[o] = dx * s; w[o + 1] = dy * s; w[o + 2] = dz * s;
    }
  }

  /** Per-body damping and the safety limits, for the free bodies. */
  damp(h: number): void {
    const { v, w } = this;
    for (let i = 0; i < this.count; i++) {
      if (this.invMass[i] === 0) continue;
      const o = i * 3;
      const fl = Math.pow(1 - this.linearDamping[i], h);
      const fa = Math.pow(1 - this.angularDamping[i], h);
      v[o] *= fl; v[o + 1] *= fl; v[o + 2] *= fl;
      w[o] *= fa; w[o + 1] *= fa; w[o + 2] *= fa;
      const speed = Math.sqrt(v[o] * v[o] + (v[o + 1]) * (v[o + 1]) + (v[o + 2]) * (v[o + 2]));
      if (speed > MAX_SPEED) {
        const k = MAX_SPEED / speed;
        v[o] *= k; v[o + 1] *= k; v[o + 2] *= k;
      }
      const spin = Math.sqrt(w[o] * w[o] + (w[o + 1]) * (w[o + 1]) + (w[o + 2]) * (w[o + 2]));
      if (spin > MAX_SPIN) {
        const k = MAX_SPIN / spin;
        w[o] *= k; w[o + 1] *= k; w[o + 2] *= k;
      }
    }
  }

  /** out[oo..oo+2] = world position of body i's local point (lx, ly, lz). */
  worldPoint(i: number, lx: number, ly: number, lz: number, out: Float64Array, oo: number): void {
    rotateVec(this.q, i * 4, lx, ly, lz, out, oo);
    out[oo] += this.x[i * 3]; out[oo + 1] += this.x[i * 3 + 1]; out[oo + 2] += this.x[i * 3 + 2];
  }

  /** out[oo..oo+2] = body i's local coordinates of a world point. */
  localPoint(i: number, wx: number, wy: number, wz: number, out: Float64Array, oo: number): void {
    rotateVecInverse(this.q, i * 4, wx - this.x[i * 3], wy - this.x[i * 3 + 1], wz - this.x[i * 3 + 2], out, oo);
  }

  /**
   * Inverse mass body i presents to a unit direction n applied at world point p — or, when
   * `atPoint` is false, to a rotation about axis n.
   */
  inverseMass(i: number, nx: number, ny: number, nz: number, px: number, py: number, pz: number, atPoint: boolean): number {
    const o = i * 3;
    if (this.invMass[i] === 0 && this.invInertia[o] === 0 && this.invInertia[o + 1] === 0 && this.invInertia[o + 2] === 0) return 0;
    let rx = nx, ry = ny, rz = nz;
    if (atPoint) {
      // r × n, with r the lever arm from the center of mass
      const ax = px - this.x[o], ay = py - this.x[o + 1], az = pz - this.x[o + 2];
      rx = ay * nz - az * ny; ry = az * nx - ax * nz; rz = ax * ny - ay * nx;
    }
    rotateVecInverse(this.q, i * 4, rx, ry, rz, t, 0);
    let w = t[0] * t[0] * this.invInertia[o] + t[1] * t[1] * this.invInertia[o + 1] + t[2] * t[2] * this.invInertia[o + 2];
    if (atPoint) w += this.invMass[i];
    return w;
  }

  /**
   * Move body i by the correction c: a displacement applied at world point p (`atPoint`), or a
   * rotation vector. `velocityLevel` changes the velocities instead of the pose.
   */
  applyCorrection(i: number, cx: number, cy: number, cz: number, px: number, py: number, pz: number, atPoint: boolean, velocityLevel: boolean): void {
    const o = i * 3, qo = i * 4;
    let dx = cx, dy = cy, dz = cz;
    if (atPoint) {
      const m = this.invMass[i];
      const ax = px - this.x[o], ay = py - this.x[o + 1], az = pz - this.x[o + 2];
      const target = velocityLevel ? this.v : this.x;
      target[o] += cx * m; target[o + 1] += cy * m; target[o + 2] += cz * m;
      dx = ay * cz - az * cy; dy = az * cx - ax * cz; dz = ax * cy - ay * cx;
    }
    // Torque-like term through the body-frame inverse inertia, back to world.
    rotateVecInverse(this.q, qo, dx, dy, dz, t, 0);
    rotateVec(this.q, qo, t[0] * this.invInertia[o], t[1] * this.invInertia[o + 1], t[2] * this.invInertia[o + 2], t, 3);
    if (velocityLevel) {
      this.w[o] += t[3]; this.w[o + 1] += t[4]; this.w[o + 2] += t[5];
    } else {
      rotateByVector(this.q, qo, t[3], t[4], t[5]);
    }
  }

  /**
   * One XPBD Gauss–Seidel step on a pair: close the vector c, which points from body a's
   * point pa to body b's point pb — or, when `atPoint` is false, is the rotation body a must
   * make toward body b. Body a moves along +c, body b along −c, each by its share of the
   * generalized inverse mass. Either body may be −1 (the world). `alphaTilde` is the XPBD
   * compliance divided by h² (0 = rigid). Returns the multiplier (the correction's magnitude
   * per unit of inverse mass), which contacts use as the normal force.
   */
  solvePair(
    a: number, b: number, cx: number, cy: number, cz: number,
    pax: number, pay: number, paz: number, pbx: number, pby: number, pbz: number,
    atPoint: boolean, alphaTilde: number, velocityLevel: boolean,
  ): number {
    const c = Math.sqrt(cx * cx + cy * cy + cz * cz);
    if (c < 1e-12) return 0;
    const nx = cx / c, ny = cy / c, nz = cz / c;
    const wa = a >= 0 ? this.inverseMass(a, nx, ny, nz, pax, pay, paz, atPoint) : 0;
    const wb = b >= 0 ? this.inverseMass(b, nx, ny, nz, pbx, pby, pbz, atPoint) : 0;
    const w = wa + wb;
    if (w <= 0) return 0;
    const lambda = c / (w + alphaTilde);
    if (wa > 0) this.applyCorrection(a, nx * lambda, ny * lambda, nz * lambda, pax, pay, paz, atPoint, velocityLevel);
    if (wb > 0) this.applyCorrection(b, -nx * lambda, -ny * lambda, -nz * lambda, pbx, pby, pbz, atPoint, velocityLevel);
    return lambda;
  }
}
