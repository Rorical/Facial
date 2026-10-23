import { clamp, mat4, quat, vec3, type Mat4, type Quat, type Vec3 } from '../math';
import type { Vec3Tuple } from '../loaders/binary-reader';
import type { RuntimeBone } from './skeleton';

// MMD-style CCD IK (after saba's MMDIkSolver).
//   ikBone     – the IK bone (e.g. 左足ＩＫ): the goal position.
//   target     – the end effector (e.g. 左足首) that should reach the goal.
//   links      – chain bones from the effector side upward (e.g. ひざ, 足).
// Links limited to a single axis (knees) use a signed "plane" solve so they bend the
// right way; other limited links clamp decomposed Euler angles (X→Y→Z order).

interface IkLink {
  bone: RuntimeBone;
  limits: { min: Vec3Tuple; max: Vec3Tuple } | null;
  prevAngle: Vec3;
  planeAngle: number;
  saved: Quat;
}

const tmpInv: Mat4 = mat4.create();
const tmpGoal = vec3.create();
const tmpEff = vec3.create();
const tmpAxis = vec3.create();
const tmpQ = quat.create();
const tmpQ2 = quat.create();
const tmpAnim = quat.create();
const tmpV = vec3.create();

export class IkSolver {
  enabled = true;
  private readonly links: IkLink[];

  constructor(
    readonly ikBone: RuntimeBone,
    readonly target: RuntimeBone,
    links: { bone: RuntimeBone; limits: { min: Vec3Tuple; max: Vec3Tuple } | null }[],
    readonly iterations: number,
    readonly limitAngle: number,
  ) {
    this.links = links.map((l) => ({ ...l, prevAngle: vec3.create(), planeAngle: 0, saved: quat.create() }));
  }

  solve(): void {
    for (const link of this.links) {
      vec3.set(link.prevAngle, 0, 0, 0);
      link.planeAngle = 0;
      quat.identity(link.bone.ikRotation);
      link.bone.updateLocal();
    }
    this.links[this.links.length - 1]?.bone.updateWorldRecursive();

    mat4.getTranslation(tmpGoal, this.ikBone.world);
    mat4.getTranslation(tmpEff, this.target.world);
    // Already at the goal: solving would only push limited links (knees) to their limit.
    if (vec3.distance(tmpGoal, tmpEff) < 1e-5) return;

    let best = Infinity;
    for (let it = 0; it < this.iterations; it++) {
      this.solveIteration(it);
      mat4.getTranslation(tmpGoal, this.ikBone.world);
      mat4.getTranslation(tmpEff, this.target.world);
      const dist = vec3.distance(tmpGoal, tmpEff);
      if (dist < best) {
        best = dist;
        for (const link of this.links) quat.copy(link.saved, link.bone.ikRotation);
      } else {
        // Got worse: restore the best iteration and stop.
        for (const link of this.links) {
          quat.copy(link.bone.ikRotation, link.saved);
          link.bone.updateLocal();
        }
        this.links[this.links.length - 1]?.bone.updateWorldRecursive();
        break;
      }
    }
  }

  private solveIteration(iteration: number): void {
    mat4.getTranslation(tmpGoal, this.ikBone.world);
    for (const link of this.links) {
      const bone = link.bone;
      if (bone === this.target) continue;

      const lim = link.limits;
      if (lim) {
        const axis = singleAxis(lim.min, lim.max);
        if (axis >= 0) {
          this.solvePlane(iteration, link, axis);
          continue;
        }
      }

      // Goal and effector in this link's local space.
      mat4.invert(tmpInv, bone.world);
      mat4.getTranslation(tmpEff, this.target.world);
      const goalLocal = vec3.normalize(tmpV, vec3.transformMat4(tmpV, tmpGoal, tmpInv));
      const effLocal = vec3.normalize(tmpEff, vec3.transformMat4(tmpEff, tmpEff, tmpInv));
      const dot = clamp(vec3.dot(effLocal, goalLocal), -1, 1);
      let angle = Math.acos(dot);
      if (angle < 1e-5) continue;
      angle = Math.min(angle, this.limitAngle);
      vec3.cross(tmpAxis, effLocal, goalLocal);
      if (vec3.length(tmpAxis) < 1e-9) continue;
      vec3.normalize(tmpAxis, tmpAxis);

      // New full rotation = current (ik · anim) · delta.
      bone.animatedRotation(tmpAnim);
      const rot = quat.multiply(tmpQ, bone.ikRotation, tmpAnim);
      quat.multiply(rot, rot, quat.fromAxisAngle(tmpQ2, tmpAxis, angle));

      if (lim) {
        const euler = decomposeXYZ(rot, link.prevAngle);
        for (let c = 0; c < 3; c++) {
          let a = clamp(euler[c], lim.min[c], lim.max[c]);
          a = clamp(a - link.prevAngle[c], -this.limitAngle, this.limitAngle) + link.prevAngle[c];
          link.prevAngle[c] = a;
        }
        quat.fromEulerXYZ(rot, link.prevAngle[0], link.prevAngle[1], link.prevAngle[2]);
      }

      // ikRotation = rot · anim⁻¹
      quat.multiply(bone.ikRotation, rot, quat.conjugate(tmpAnim, tmpAnim));
      bone.updateLocal();
      bone.updateWorldRecursive();
    }
  }

  private solvePlane(iteration: number, link: IkLink, axisIndex: number): void {
    const bone = link.bone;
    const lim = link.limits!;
    const axis = vec3.create(axisIndex === 0 ? 1 : 0, axisIndex === 1 ? 1 : 0, axisIndex === 2 ? 1 : 0);

    mat4.invert(tmpInv, bone.world);
    mat4.getTranslation(tmpEff, this.target.world);
    const goalLocal = vec3.normalize(tmpV, vec3.transformMat4(tmpV, tmpGoal, tmpInv));
    const effLocal = vec3.normalize(tmpEff, vec3.transformMat4(tmpEff, tmpEff, tmpInv));
    const dot = clamp(vec3.dot(effLocal, goalLocal), -1, 1);
    const angle = Math.min(Math.acos(dot), this.limitAngle);

    // Try rotating both ways around the axis; keep the direction that gets closer.
    const r1 = vec3.transformQuat(vec3.create(), effLocal, quat.fromAxisAngle(tmpQ, axis, angle));
    const r2 = vec3.transformQuat(vec3.create(), effLocal, quat.fromAxisAngle(tmpQ, axis, -angle));
    let newAngle = link.planeAngle + (vec3.dot(r1, goalLocal) > vec3.dot(r2, goalLocal) ? angle : -angle);

    const min = lim.min[axisIndex];
    const max = lim.max[axisIndex];
    if (iteration === 0 && (newAngle < min || newAngle > max)) {
      // First iteration may start on the wrong side of the limit range: flip into it.
      if (-newAngle > min && -newAngle < max) {
        newAngle = -newAngle;
      } else {
        const half = (min + max) * 0.5;
        if (Math.abs(half - newAngle) > Math.abs(half + newAngle)) newAngle = -newAngle;
      }
    }
    newAngle = clamp(newAngle, min, max);
    link.planeAngle = newAngle;

    bone.animatedRotation(tmpAnim);
    quat.multiply(bone.ikRotation, quat.fromAxisAngle(tmpQ, axis, newAngle), quat.conjugate(tmpAnim, tmpAnim));
    bone.updateLocal();
    bone.updateWorldRecursive();
  }
}

/** Index of the only axis whose limit range is non-zero (knee-style link), or -1. */
function singleAxis(min: Vec3Tuple, max: Vec3Tuple): number {
  const free = [0, 1, 2].map((i) => min[i] !== 0 || max[i] !== 0);
  const count = free.filter(Boolean).length;
  return count === 1 ? free.indexOf(true) : -1;
}

const TWO_PI = Math.PI * 2;
const normalizeAngle = (a: number) => ((a % TWO_PI) + TWO_PI) % TWO_PI;
const diffAngle = (a: number, b: number) => {
  const d = normalizeAngle(a) - normalizeAngle(b);
  return d > Math.PI ? d - TWO_PI : d < -Math.PI ? d + TWO_PI : d;
};

/**
 * Decompose q into Euler angles for q = qz·qy·qx (matrix Rz·Ry·Rx), choosing among the
 * equivalent solutions the one closest to `prev` so angles stay continuous across iterations.
 */
export function decomposeXYZ(q: Quat, prev: Vec3): [number, number, number] {
  const m = mat4.fromRotationTranslation(mat4.create(), q, vec3.create());
  // Column-major: M(r, c) = m[c * 4 + r]
  const m00 = m[0], m10 = m[1], m20 = m[2];
  const m01 = m[4], m21 = m[6];
  const m12 = m[9], m22 = m[10];

  let x: number, y: number, z: number;
  const sy = -m20;
  if (1 - Math.abs(sy) < 1e-6) {
    // Gimbal lock: pin whichever of X/Z was nearer 0 or π last time.
    y = Math.asin(clamp(sy, -1, 1));
    if (Math.abs(Math.sin(prev[0])) < Math.abs(Math.sin(prev[2]))) {
      if (Math.cos(prev[0]) > 0) { x = 0; z = Math.asin(clamp(-m01, -1, 1)); }
      else { x = Math.PI; z = Math.asin(clamp(m01, -1, 1)); }
    } else {
      if (Math.cos(prev[2]) > 0) { z = 0; x = Math.asin(clamp(-m12, -1, 1)); }
      else { z = Math.PI; x = Math.asin(clamp(m12, -1, 1)); }
    }
  } else {
    x = Math.atan2(m21, m22);
    y = Math.asin(clamp(sy, -1, 1));
    z = Math.atan2(m10, m00);
  }

  const pi = Math.PI;
  const candidates: [number, number, number][] = [[x, y, z]];
  for (const dx of [pi, -pi]) {
    for (const yy of [pi - y, -pi - y]) {
      for (const dz of [pi, -pi]) candidates.push([x + dx, yy, z + dz]);
    }
  }
  let best = candidates[0];
  let bestErr = Infinity;
  for (const c of candidates) {
    const err = Math.abs(diffAngle(c[0], prev[0])) + Math.abs(diffAngle(c[1], prev[1])) + Math.abs(diffAngle(c[2], prev[2]));
    if (err < bestErr) {
      bestErr = err;
      best = c;
    }
  }
  return best;
}
