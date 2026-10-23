import { PHYSICS_WITH_BONE, type PhysicsRig } from '../rig';
import type { BodySet } from './body-set';
import { eulerXYZ, mat3FromQuat, quatConjMul, quatFromEulerXYZ, quatFromMat, quatMul, rotateVec, rotationVector } from './rotation';

// MMD's joints are Bullet 6-DOF spring constraints: per axis a translation range in body A's
// joint frame, a rotation range on the Euler angles (R = Rx·Ry·Rz) between the two joint
// frames, and optional springs toward the rest pose. Here each is a position-based
// constraint: ranges are rigid and clamp the pose, springs are compliant (compliance 1/k, so
// a spring with stiffness k pulls with k·x like Bullet's). Also here: the "physics + bone"
// bodies, which MMD keeps pinned at their bone's position while they rotate freely.

const QA = new Float64Array(4);
const QB = new Float64Array(4);
const REL = new Float64Array(4);
const TARGET = new Float64Array(4);
const RA = new Float64Array(9);
const RB = new Float64Array(9);
const E = new Float64Array(3);
const V = new Float64Array(9);

export class JointSolver {
  readonly count: number;
  private readonly a: Int32Array;
  private readonly b: Int32Array;
  /** Joints in solving order: those nearer the body first, so one pass carries its motion down a chain. */
  private readonly order: Int32Array;
  /** Joint frame in each body's local space: anchor (3) and rotation (4). */
  private readonly anchorA: Float64Array;
  private readonly anchorB: Float64Array;
  private readonly frameA: Float64Array;
  private readonly frameB: Float64Array;
  private readonly linMin: Float64Array;
  private readonly linMax: Float64Array;
  private readonly linLocked: Uint8Array;
  private readonly linCompliance: Float64Array;
  private readonly linSprung: Uint8Array;
  private readonly rotMin: Float64Array;
  private readonly rotMax: Float64Array;
  private readonly rotLimited: Uint8Array;
  private readonly rotCompliance: Float64Array;
  private readonly rotSprung: Uint8Array;

  readonly pinCount: number;
  private readonly pinBody: Int32Array;
  private readonly pinLocal: Float64Array;
  private readonly pinTarget: Float64Array;

  constructor(rig: PhysicsRig, private readonly bodies: BodySet) {
    const joints = rig.joints.filter((j) => j.bodyA !== j.bodyB);
    const n = joints.length;
    this.count = n;
    this.a = Int32Array.from(joints, (j) => j.bodyA);
    this.b = Int32Array.from(joints, (j) => j.bodyB);
    this.anchorA = new Float64Array(n * 3);
    this.anchorB = new Float64Array(n * 3);
    this.frameA = new Float64Array(n * 4);
    this.frameB = new Float64Array(n * 4);
    this.linMin = new Float64Array(n * 3);
    this.linMax = new Float64Array(n * 3);
    this.linLocked = new Uint8Array(n);
    this.linCompliance = new Float64Array(n * 3);
    this.linSprung = new Uint8Array(n);
    this.rotMin = new Float64Array(n * 3);
    this.rotMax = new Float64Array(n * 3);
    this.rotLimited = new Uint8Array(n);
    this.rotCompliance = new Float64Array(n * 3);
    this.rotSprung = new Uint8Array(n);

    joints.forEach((j, k) => {
      this.anchorA.set([j.frameA[12], j.frameA[13], j.frameA[14]], k * 3);
      this.anchorB.set([j.frameB[12], j.frameB[13], j.frameB[14]], k * 3);
      quatFromMat(j.frameA, 4, this.frameA, k * 4);
      quatFromMat(j.frameB, 4, this.frameB, k * 4);
      const s = j.source;
      let locked = true, limited = false;
      for (let i = 0; i < 3; i++) {
        // Bullet: a lower limit above the upper one means "free".
        let lo = s.positionMin[i], hi = s.positionMax[i];
        if (lo > hi) {
          lo = -Infinity; hi = Infinity;
        }
        this.linMin[k * 3 + i] = lo; this.linMax[k * 3 + i] = hi;
        if (lo !== 0 || hi !== 0) locked = false;
        let rlo = s.rotationMin[i], rhi = s.rotationMax[i];
        if (rlo > rhi || (rlo <= -Math.PI && rhi >= Math.PI)) {
          rlo = -Infinity; rhi = Infinity;
        } else {
          limited = true;
        }
        this.rotMin[k * 3 + i] = rlo; this.rotMax[k * 3 + i] = rhi;
        if (s.springPosition[i] > 0) {
          this.linCompliance[k * 3 + i] = 1 / s.springPosition[i];
          this.linSprung[k] = 1;
        }
        if (s.springRotation[i] > 0) {
          this.rotCompliance[k * 3 + i] = 1 / s.springRotation[i];
          this.rotSprung[k] = 1;
        }
      }
      this.linLocked[k] = locked ? 1 : 0;
      this.rotLimited[k] = limited ? 1 : 0;
    });

    const depth = new Int32Array(rig.boneParents.length);
    rig.boneParents.forEach((_, bone) => {
      let d = 0;
      for (let p = rig.boneParents[bone]; p >= 0 && d <= depth.length; p = rig.boneParents[p]) d++;
      depth[bone] = d;
    });
    const depthOf = (body: number) => (rig.bodies[body].boneIndex >= 0 ? depth[rig.bodies[body].boneIndex] : 0);
    this.order = Int32Array.from(joints.keys()).sort((x, y) => depthOf(joints[x].bodyB) - depthOf(joints[y].bodyB) || depthOf(joints[x].bodyA) - depthOf(joints[y].bodyA) || x - y);

    const pinned = rig.bodies.map((body, i) => (body.mode === PHYSICS_WITH_BONE && body.boneIndex >= 0 ? i : -1)).filter((i) => i >= 0);
    this.pinCount = pinned.length;
    this.pinBody = Int32Array.from(pinned);
    this.pinLocal = new Float64Array(pinned.length * 3);
    this.pinTarget = new Float64Array(pinned.length * 3);
    pinned.forEach((i, k) => {
      // The bone's origin in the body's own space.
      const inv = rig.bodies[i].offsetInv;
      this.pinLocal.set([inv[12], inv[13], inv[14]], k * 3);
    });
  }

  /** Rig body index of pin k. */
  pinnedBody(k: number): number {
    return this.pinBody[k];
  }

  /** Where pin k's bone origin is this substep. */
  setPinTarget(k: number, x: number, y: number, z: number): void {
    this.pinTarget[k * 3] = x; this.pinTarget[k * 3 + 1] = y; this.pinTarget[k * 3 + 2] = z;
  }

  /**
   * One Gauss–Seidel pass over every joint; `h` is the substep length. Returns the largest
   * anchor separation met along the way — how far from converged the pass started.
   */
  solve(h: number): number {
    const invH2 = 1 / (h * h);
    this.residual = 0;
    for (let k = 0; k < this.count; k++) {
      const j = this.order[k];
      const A = this.a[j], B = this.b[j];
      if (this.rotLimited[j] || this.rotSprung[j]) this.solveRotation(j, A, B, invH2);
      this.solvePosition(j, A, B, invH2);
    }
    return this.residual;
  }

  private residual = 0;

  solvePins(): void {
    const bs = this.bodies;
    for (let k = 0; k < this.pinCount; k++) {
      const i = this.pinBody[k];
      bs.worldPoint(i, this.pinLocal[k * 3], this.pinLocal[k * 3 + 1], this.pinLocal[k * 3 + 2], V, 0);
      bs.solvePair(i, -1, this.pinTarget[k * 3] - V[0], this.pinTarget[k * 3 + 1] - V[1], this.pinTarget[k * 3 + 2] - V[2], V[0], V[1], V[2], 0, 0, 0, true, 0, false);
    }
  }

  /** Relative rotation of joint j's frames as Euler angles (into E), with the frames in QA / QB. */
  private relativeAngles(j: number, A: number, B: number): void {
    const bs = this.bodies;
    quatMul(bs.q, A * 4, this.frameA, j * 4, QA, 0);
    quatMul(bs.q, B * 4, this.frameB, j * 4, QB, 0);
    quatConjMul(QA, 0, QB, 0, REL, 0);
    mat3FromQuat(REL, 0, RA);
    eulerXYZ(RA, E);
  }

  /** World axis the Euler angle `i` turns about: A's x, B's z, or the middle axis between them. */
  private eulerAxis(i: number, out: Float64Array, oo: number): void {
    if (i === 0) {
      mat3FromQuat(QA, 0, RA);
      out[oo] = RA[0]; out[oo + 1] = RA[1]; out[oo + 2] = RA[2];
      return;
    }
    mat3FromQuat(QB, 0, RB);
    if (i === 2) {
      out[oo] = RB[6]; out[oo + 1] = RB[7]; out[oo + 2] = RB[8];
      return;
    }
    mat3FromQuat(QA, 0, RA);
    let x = RB[7] * RA[2] - RB[8] * RA[1], y = RB[8] * RA[0] - RB[6] * RA[2], z = RB[6] * RA[1] - RB[7] * RA[0];
    const len = Math.sqrt(x * x + y * y + z * z);
    if (len < 1e-6) {
      x = RA[3]; y = RA[4]; z = RA[5];
    } else {
      x /= len; y /= len; z /= len;
    }
    out[oo] = x; out[oo + 1] = y; out[oo + 2] = z;
  }

  private solveRotation(j: number, A: number, B: number, invH2: number): void {
    const bs = this.bodies;
    this.relativeAngles(j, A, B);
    if (this.rotSprung[j]) {
      for (let i = 0; i < 3; i++) {
        const compliance = this.rotCompliance[j * 3 + i];
        const angle = E[i];
        if (compliance === 0 || Math.abs(angle) < 1e-9) continue;
        this.eulerAxis(i, V, 0);
        bs.solvePair(A, B, V[0] * angle, V[1] * angle, V[2] * angle, 0, 0, 0, 0, 0, 0, false, compliance * invH2, false);
      }
      if (!this.rotLimited[j]) return;
      this.relativeAngles(j, A, B);
    }
    const o = j * 3;
    let clamped = false;
    for (let i = 0; i < 3; i++) {
      const lo = this.rotMin[o + i], hi = this.rotMax[o + i];
      if (E[i] < lo) {
        E[i] = lo; clamped = true;
      } else if (E[i] > hi) {
        E[i] = hi; clamped = true;
      }
    }
    if (!clamped) return;
    // Turn A so the relative rotation becomes the clamped one: Δ = rel · conj(target), in A's frame.
    quatFromEulerXYZ(E[0], E[1], E[2], TARGET, 0);
    TARGET[0] = -TARGET[0]; TARGET[1] = -TARGET[1]; TARGET[2] = -TARGET[2];
    quatMul(REL, 0, TARGET, 0, REL, 0);
    rotationVector(REL, 0, V, 0);
    rotateVec(QA, 0, V[0], V[1], V[2], V, 3);
    bs.solvePair(A, B, V[3], V[4], V[5], 0, 0, 0, 0, 0, 0, false, 0, false);
  }

  /** Anchors of joint j in world space into V[0..2] (A) and V[3..5] (B). */
  private anchors(j: number, A: number, B: number): void {
    this.bodies.worldPoint(A, this.anchorA[j * 3], this.anchorA[j * 3 + 1], this.anchorA[j * 3 + 2], V, 0);
    this.bodies.worldPoint(B, this.anchorB[j * 3], this.anchorB[j * 3 + 1], this.anchorB[j * 3 + 2], V, 3);
  }

  private solvePosition(j: number, A: number, B: number, invH2: number): void {
    const bs = this.bodies;
    this.anchors(j, A, B);
    if (this.linLocked[j]) {
      this.residual = Math.max(this.residual, Math.sqrt((V[3] - V[0]) * (V[3] - V[0]) + (V[4] - V[1]) * (V[4] - V[1]) + (V[5] - V[2]) * (V[5] - V[2])));
      bs.solvePair(A, B, V[3] - V[0], V[4] - V[1], V[5] - V[2], V[0], V[1], V[2], V[3], V[4], V[5], true, 0, false);
      return;
    }
    const o = j * 3;
    quatMul(bs.q, A * 4, this.frameA, j * 4, QA, 0);
    mat3FromQuat(QA, 0, RA);
    if (this.linSprung[j]) {
      for (let i = 0; i < 3; i++) {
        const compliance = this.linCompliance[o + i];
        if (compliance === 0) continue;
        const ax = RA[i * 3], ay = RA[i * 3 + 1], az = RA[i * 3 + 2];
        const d = (V[3] - V[0]) * ax + (V[4] - V[1]) * ay + (V[5] - V[2]) * az;
        if (Math.abs(d) < 1e-12) continue;
        bs.solvePair(A, B, ax * d, ay * d, az * d, V[0], V[1], V[2], V[3], V[4], V[5], true, compliance * invH2, false);
        this.anchors(j, A, B);
      }
      quatMul(bs.q, A * 4, this.frameA, j * 4, QA, 0);
      mat3FromQuat(QA, 0, RA);
    }
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < 3; i++) {
      const ax = RA[i * 3], ay = RA[i * 3 + 1], az = RA[i * 3 + 2];
      const d = (V[3] - V[0]) * ax + (V[4] - V[1]) * ay + (V[5] - V[2]) * az;
      const lo = this.linMin[o + i], hi = this.linMax[o + i];
      const excess = d < lo ? d - lo : d > hi ? d - hi : 0;
      cx += ax * excess; cy += ay * excess; cz += az * excess;
    }
    if (cx === 0 && cy === 0 && cz === 0) return;
    this.residual = Math.max(this.residual, Math.sqrt(cx * cx + cy * cy + cz * cz));
    bs.solvePair(A, B, cx, cy, cz, V[0], V[1], V[2], V[3], V[4], V[5], true, 0, false);
  }
}
