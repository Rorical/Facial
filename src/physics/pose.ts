import { mat4, quat, vec3, type Mat4, type Quat, type Vec3 } from '../math';

// Blending bone poses: used to sweep kinematic bodies through a frame's substeps and to
// predict the next frame's pose for the pipelined worker.

/** Rigid blend of two matrices (slerp rotation, lerp translation); t > 1 extrapolates. */
export function blendRigid(a: ArrayLike<number>, b: ArrayLike<number>, t: number, out: Mat4, qa: Quat, qb: Quat, v: Vec3): Mat4 {
  quat.fromMat4(qa, a as Mat4);
  quat.fromMat4(qb, b as Mat4);
  quat.slerp(qa, qa, qb, t);
  vec3.set(v, a[12] + (b[12] - a[12]) * t, a[13] + (b[13] - a[13]) * t, a[14] + (b[14] - a[14]) * t);
  return mat4.fromRotationTranslation(out, qa, v);
}

const scratch = { m: mat4.create(), qa: quat.create(), qb: quat.create(), v: vec3.create() };

/** out = the pose one frame after `cur`, assuming the motion from `prev` to `cur` continues. */
export function extrapolatePose(prev: Float64Array, cur: Float64Array, out: Float64Array): void {
  const { m, qa, qb, v } = scratch;
  for (let o = 0; o < cur.length; o += 16) {
    blendRigid(prev.subarray(o, o + 16), cur.subarray(o, o + 16), 2, m, qa, qb, v);
    out.set(m, o);
  }
}
