// Minimal math library.
//
// Coordinate system: we stay in MMD's native LEFT-HANDED space (Y up, model faces -Z)
// for everything, and use a left-handed view/projection. PMX/VMD data is therefore
// used as-is: no Z flips, no quaternion sign juggling.
//
// Matrices are column-major (WGSL layout): element (row r, col c) lives at m[c * 4 + r].
// Quaternions are [x, y, z, w]. CPU-side math uses Float64Array for IK stability.

export type Vec3 = Float64Array;
export type Quat = Float64Array;
export type Mat4 = Float64Array;

export const vec3 = {
  create(x = 0, y = 0, z = 0): Vec3 {
    const out = new Float64Array(3);
    out[0] = x; out[1] = y; out[2] = z;
    return out;
  },
  set(out: Vec3, x: number, y: number, z: number): Vec3 {
    out[0] = x; out[1] = y; out[2] = z;
    return out;
  },
  copy(out: Vec3, a: ArrayLike<number>): Vec3 {
    out[0] = a[0]; out[1] = a[1]; out[2] = a[2];
    return out;
  },
  add(out: Vec3, a: Vec3, b: Vec3): Vec3 {
    out[0] = a[0] + b[0]; out[1] = a[1] + b[1]; out[2] = a[2] + b[2];
    return out;
  },
  sub(out: Vec3, a: Vec3, b: Vec3): Vec3 {
    out[0] = a[0] - b[0]; out[1] = a[1] - b[1]; out[2] = a[2] - b[2];
    return out;
  },
  scale(out: Vec3, a: Vec3, s: number): Vec3 {
    out[0] = a[0] * s; out[1] = a[1] * s; out[2] = a[2] * s;
    return out;
  },
  scaleAndAdd(out: Vec3, a: Vec3, b: ArrayLike<number>, s: number): Vec3 {
    out[0] = a[0] + b[0] * s; out[1] = a[1] + b[1] * s; out[2] = a[2] + b[2] * s;
    return out;
  },
  dot(a: Vec3, b: Vec3): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  },
  cross(out: Vec3, a: Vec3, b: Vec3): Vec3 {
    const ax = a[0], ay = a[1], az = a[2];
    const bx = b[0], by = b[1], bz = b[2];
    out[0] = ay * bz - az * by;
    out[1] = az * bx - ax * bz;
    out[2] = ax * by - ay * bx;
    return out;
  },
  length(a: Vec3): number {
    return Math.hypot(a[0], a[1], a[2]);
  },
  distance(a: Vec3, b: Vec3): number {
    return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  },
  normalize(out: Vec3, a: Vec3): Vec3 {
    const len = Math.hypot(a[0], a[1], a[2]);
    const inv = len > 0 ? 1 / len : 0;
    out[0] = a[0] * inv; out[1] = a[1] * inv; out[2] = a[2] * inv;
    return out;
  },
  /** Transform a point by a 4x4 matrix (w = 1). */
  transformMat4(out: Vec3, a: Vec3, m: Mat4): Vec3 {
    const x = a[0], y = a[1], z = a[2];
    out[0] = m[0] * x + m[4] * y + m[8] * z + m[12];
    out[1] = m[1] * x + m[5] * y + m[9] * z + m[13];
    out[2] = m[2] * x + m[6] * y + m[10] * z + m[14];
    return out;
  },
  transformQuat(out: Vec3, a: Vec3, q: Quat): Vec3 {
    const qx = q[0], qy = q[1], qz = q[2], qw = q[3];
    const x = a[0], y = a[1], z = a[2];
    // t = 2 * cross(q.xyz, v); v' = v + w * t + cross(q.xyz, t)
    const tx = 2 * (qy * z - qz * y);
    const ty = 2 * (qz * x - qx * z);
    const tz = 2 * (qx * y - qy * x);
    out[0] = x + qw * tx + (qy * tz - qz * ty);
    out[1] = y + qw * ty + (qz * tx - qx * tz);
    out[2] = z + qw * tz + (qx * ty - qy * tx);
    return out;
  },
};

export const quat = {
  create(): Quat {
    const out = new Float64Array(4);
    out[3] = 1;
    return out;
  },
  identity(out: Quat): Quat {
    out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 1;
    return out;
  },
  set(out: Quat, x: number, y: number, z: number, w: number): Quat {
    out[0] = x; out[1] = y; out[2] = z; out[3] = w;
    return out;
  },
  copy(out: Quat, a: ArrayLike<number>): Quat {
    out[0] = a[0]; out[1] = a[1]; out[2] = a[2]; out[3] = a[3];
    return out;
  },
  /** out = a * b (rotation b is applied first, then a). */
  multiply(out: Quat, a: Quat, b: Quat): Quat {
    const ax = a[0], ay = a[1], az = a[2], aw = a[3];
    const bx = b[0], by = b[1], bz = b[2], bw = b[3];
    out[0] = ax * bw + aw * bx + ay * bz - az * by;
    out[1] = ay * bw + aw * by + az * bx - ax * bz;
    out[2] = az * bw + aw * bz + ax * by - ay * bx;
    out[3] = aw * bw - ax * bx - ay * by - az * bz;
    return out;
  },
  /** Inverse of a unit quaternion. */
  conjugate(out: Quat, a: Quat): Quat {
    out[0] = -a[0]; out[1] = -a[1]; out[2] = -a[2]; out[3] = a[3];
    return out;
  },
  normalize(out: Quat, a: Quat): Quat {
    const len = Math.hypot(a[0], a[1], a[2], a[3]);
    const inv = len > 0 ? 1 / len : 0;
    out[0] = a[0] * inv; out[1] = a[1] * inv; out[2] = a[2] * inv; out[3] = a[3] * inv;
    return out;
  },
  /** Shortest-path spherical interpolation. t outside [0, 1] extrapolates. */
  slerp(out: Quat, a: Quat, b: ArrayLike<number>, t: number): Quat {
    const ax = a[0], ay = a[1], az = a[2], aw = a[3];
    let bx = b[0], by = b[1], bz = b[2], bw = b[3];
    let cos = ax * bx + ay * by + az * bz + aw * bw;
    if (cos < 0) {
      cos = -cos; bx = -bx; by = -by; bz = -bz; bw = -bw;
    }
    let s0: number, s1: number;
    if (1 - cos > 1e-6) {
      const omega = Math.acos(Math.min(cos, 1));
      const sin = Math.sin(omega);
      s0 = Math.sin((1 - t) * omega) / sin;
      s1 = Math.sin(t * omega) / sin;
    } else {
      s0 = 1 - t;
      s1 = t;
    }
    out[0] = s0 * ax + s1 * bx;
    out[1] = s0 * ay + s1 * by;
    out[2] = s0 * az + s1 * bz;
    out[3] = s0 * aw + s1 * bw;
    return out;
  },
  /** Rotation of `rad` radians around a unit `axis`. */
  fromAxisAngle(out: Quat, axis: ArrayLike<number>, rad: number): Quat {
    const s = Math.sin(rad * 0.5);
    out[0] = axis[0] * s; out[1] = axis[1] * s; out[2] = axis[2] * s; out[3] = Math.cos(rad * 0.5);
    return out;
  },
  /** q = qz * qy * qx, i.e. rotate around X first, then Y, then Z (matrix Rz·Ry·Rx). */
  fromEulerXYZ(out: Quat, x: number, y: number, z: number): Quat {
    const cx = Math.cos(x * 0.5), sx = Math.sin(x * 0.5);
    const cy = Math.cos(y * 0.5), sy = Math.sin(y * 0.5);
    const cz = Math.cos(z * 0.5), sz = Math.sin(z * 0.5);
    out[0] = sx * cy * cz - cx * sy * sz;
    out[1] = cx * sy * cz + sx * cy * sz;
    out[2] = cx * cy * sz - sx * sy * cz;
    out[3] = cx * cy * cz + sx * sy * sz;
    return out;
  },
  /**
   * D3DX yaw-pitch-roll convention used throughout MMD (rigid bodies, joints, camera):
   * q = qy · qx · qz, i.e. matrix Ry·Rx·Rz — roll (Z) first, then pitch (X), then yaw (Y).
   */
  fromYawPitchRoll(out: Quat, x: number, y: number, z: number): Quat {
    const cx = Math.cos(x * 0.5), sx = Math.sin(x * 0.5);
    const cy = Math.cos(y * 0.5), sy = Math.sin(y * 0.5);
    const cz = Math.cos(z * 0.5), sz = Math.sin(z * 0.5);
    // qy·qx = (cy·sx, sy·cx, -sy·sx, cy·cx); then · qz
    const ax = cy * sx, ay = sy * cx, az = -sy * sx, aw = cy * cx;
    out[0] = ax * cz + ay * sz;
    out[1] = ay * cz - ax * sz;
    out[2] = aw * sz + az * cz;
    out[3] = aw * cz - az * sz;
    return out;
  },
  /** Extract the rotation of an (unscaled) matrix. */
  fromMat4(out: Quat, m: Mat4): Quat {
    const m00 = m[0], m01 = m[4], m02 = m[8];
    const m10 = m[1], m11 = m[5], m12 = m[9];
    const m20 = m[2], m21 = m[6], m22 = m[10];
    const trace = m00 + m11 + m22;
    if (trace > 0) {
      const s = 0.5 / Math.sqrt(trace + 1);
      out[3] = 0.25 / s;
      out[0] = (m21 - m12) * s;
      out[1] = (m02 - m20) * s;
      out[2] = (m10 - m01) * s;
    } else if (m00 > m11 && m00 > m22) {
      const s = 2 * Math.sqrt(1 + m00 - m11 - m22);
      out[3] = (m21 - m12) / s;
      out[0] = 0.25 * s;
      out[1] = (m01 + m10) / s;
      out[2] = (m02 + m20) / s;
    } else if (m11 > m22) {
      const s = 2 * Math.sqrt(1 + m11 - m00 - m22);
      out[3] = (m02 - m20) / s;
      out[0] = (m01 + m10) / s;
      out[1] = 0.25 * s;
      out[2] = (m12 + m21) / s;
    } else {
      const s = 2 * Math.sqrt(1 + m22 - m00 - m11);
      out[3] = (m10 - m01) / s;
      out[0] = (m02 + m20) / s;
      out[1] = (m12 + m21) / s;
      out[2] = 0.25 * s;
    }
    return out;
  },
};

export const mat4 = {
  create(): Mat4 {
    const out = new Float64Array(16);
    out[0] = out[5] = out[10] = out[15] = 1;
    return out;
  },
  identity(out: Mat4): Mat4 {
    out.fill(0);
    out[0] = out[5] = out[10] = out[15] = 1;
    return out;
  },
  copy(out: Mat4, a: ArrayLike<number>): Mat4 {
    for (let i = 0; i < 16; i++) out[i] = a[i];
    return out;
  },
  fromRotationTranslation(out: Mat4, q: Quat, v: Vec3): Mat4 {
    const x = q[0], y = q[1], z = q[2], w = q[3];
    const x2 = x + x, y2 = y + y, z2 = z + z;
    const xx = x * x2, xy = x * y2, xz = x * z2;
    const yy = y * y2, yz = y * z2, zz = z * z2;
    const wx = w * x2, wy = w * y2, wz = w * z2;
    out[0] = 1 - (yy + zz); out[1] = xy + wz; out[2] = xz - wy; out[3] = 0;
    out[4] = xy - wz; out[5] = 1 - (xx + zz); out[6] = yz + wx; out[7] = 0;
    out[8] = xz + wy; out[9] = yz - wx; out[10] = 1 - (xx + yy); out[11] = 0;
    out[12] = v[0]; out[13] = v[1]; out[14] = v[2]; out[15] = 1;
    return out;
  },
  multiply(out: Mat4, a: Mat4, b: Mat4): Mat4 {
    const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
    const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
    const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
    const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
    for (let c = 0; c < 4; c++) {
      const b0 = b[c * 4], b1 = b[c * 4 + 1], b2 = b[c * 4 + 2], b3 = b[c * 4 + 3];
      out[c * 4] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
      out[c * 4 + 1] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
      out[c * 4 + 2] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
      out[c * 4 + 3] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    }
    return out;
  },
  invert(out: Mat4, a: Mat4): Mat4 | null {
    const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
    const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
    const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
    const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
    const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10;
    const b02 = a00 * a13 - a03 * a10, b03 = a01 * a12 - a02 * a11;
    const b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
    const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30;
    const b08 = a20 * a33 - a23 * a30, b09 = a21 * a32 - a22 * a31;
    const b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (!det) return null;
    det = 1 / det;
    out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
    out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
    out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
    out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
    out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
    out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
    out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
    out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
    out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
    out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
    out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
    out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
    out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
    out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
    out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
    out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
    return out;
  },
  getTranslation(out: Vec3, m: Mat4): Vec3 {
    out[0] = m[12]; out[1] = m[13]; out[2] = m[14];
    return out;
  },
  /** Left-handed perspective with WebGPU depth range [0, 1]. */
  perspectiveLH(out: Mat4, fovY: number, aspect: number, near: number, far: number): Mat4 {
    const f = 1 / Math.tan(fovY / 2);
    out.fill(0);
    out[0] = f / aspect;
    out[5] = f;
    out[10] = far / (far - near);
    out[11] = 1;
    out[14] = (-near * far) / (far - near);
    return out;
  },
  /** Left-handed orthographic projection with WebGPU depth range [0, 1]. */
  orthoLH(out: Mat4, left: number, right: number, bottom: number, top: number, near: number, far: number): Mat4 {
    out.fill(0);
    out[0] = 2 / (right - left);
    out[5] = 2 / (top - bottom);
    out[10] = 1 / (far - near);
    out[12] = -(right + left) / (right - left);
    out[13] = -(top + bottom) / (top - bottom);
    out[14] = -near / (far - near);
    out[15] = 1;
    return out;
  },
  /** Left-handed look-at view matrix (camera looks down +Z). */
  lookAtLH(out: Mat4, eye: Vec3, target: Vec3, up: Vec3): Mat4 {
    const zx0 = target[0] - eye[0], zy0 = target[1] - eye[1], zz0 = target[2] - eye[2];
    const zl = 1 / Math.hypot(zx0, zy0, zz0);
    const zx = zx0 * zl, zy = zy0 * zl, zz = zz0 * zl;
    // x = normalize(cross(up, z))
    let xx = up[1] * zz - up[2] * zy;
    let xy = up[2] * zx - up[0] * zz;
    let xz = up[0] * zy - up[1] * zx;
    const xl = 1 / (Math.hypot(xx, xy, xz) || 1);
    xx *= xl; xy *= xl; xz *= xl;
    // y = cross(z, x)
    const yx = zy * xz - zz * xy;
    const yy = zz * xx - zx * xz;
    const yz = zx * xy - zy * xx;
    out[0] = xx; out[4] = xy; out[8] = xz; out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
    out[1] = yx; out[5] = yy; out[9] = yz; out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
    out[2] = zx; out[6] = zy; out[10] = zz; out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
    out[3] = 0; out[7] = 0; out[11] = 0; out[15] = 1;
    return out;
  },
};

export const clamp = (v: number, lo: number, hi: number): number => (v < lo ? lo : v > hi ? hi : v);
