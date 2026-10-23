// Quaternion math for the solver on flat arrays (no allocation). Quaternions are (x, y, z, w)
// and q · v · q* rotates a vector, as in math.ts.

/** out (column-major 3×3) = rotation matrix of the unit quaternion at q[o..o+3]. */
export function mat3FromQuat(q: ArrayLike<number>, o: number, out: Float64Array): Float64Array {
  const x = q[o], y = q[o + 1], z = q[o + 2], w = q[o + 3];
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2, yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  out[0] = 1 - (yy + zz); out[1] = xy + wz; out[2] = xz - wy;
  out[3] = xy - wz; out[4] = 1 - (xx + zz); out[5] = yz + wx;
  out[6] = xz + wy; out[7] = yz - wx; out[8] = 1 - (xx + yy);
  return out;
}

/** Unit quaternion of a column-major rotation matrix (16 or 9 floats; `s` is the column stride). */
export function quatFromMat(m: ArrayLike<number>, s: number, out: Float64Array, o: number): void {
  const m00 = m[0], m01 = m[s], m02 = m[2 * s];
  const m10 = m[1], m11 = m[s + 1], m12 = m[2 * s + 1];
  const m20 = m[2], m21 = m[s + 2], m22 = m[2 * s + 2];
  const trace = m00 + m11 + m22;
  if (trace > 0) {
    const k = 0.5 / Math.sqrt(trace + 1);
    out[o + 3] = 0.25 / k;
    out[o] = (m21 - m12) * k; out[o + 1] = (m02 - m20) * k; out[o + 2] = (m10 - m01) * k;
  } else if (m00 > m11 && m00 > m22) {
    const k = 2 * Math.sqrt(1 + m00 - m11 - m22);
    out[o + 3] = (m21 - m12) / k;
    out[o] = 0.25 * k; out[o + 1] = (m01 + m10) / k; out[o + 2] = (m02 + m20) / k;
  } else if (m11 > m22) {
    const k = 2 * Math.sqrt(1 + m11 - m00 - m22);
    out[o + 3] = (m02 - m20) / k;
    out[o] = (m01 + m10) / k; out[o + 1] = 0.25 * k; out[o + 2] = (m12 + m21) / k;
  } else {
    const k = 2 * Math.sqrt(1 + m22 - m00 - m11);
    out[o + 3] = (m10 - m01) / k;
    out[o] = (m02 + m20) / k; out[o + 1] = (m12 + m21) / k; out[o + 2] = 0.25 * k;
  }
}

/** out[oo..] = a[ao..] · b[bo..] (b is applied first). Aliasing is fine. */
export function quatMul(a: ArrayLike<number>, ao: number, b: ArrayLike<number>, bo: number, out: Float64Array, oo: number): void {
  const ax = a[ao], ay = a[ao + 1], az = a[ao + 2], aw = a[ao + 3];
  const bx = b[bo], by = b[bo + 1], bz = b[bo + 2], bw = b[bo + 3];
  out[oo] = ax * bw + aw * bx + ay * bz - az * by;
  out[oo + 1] = ay * bw + aw * by + az * bx - ax * bz;
  out[oo + 2] = az * bw + aw * bz + ax * by - ay * bx;
  out[oo + 3] = aw * bw - ax * bx - ay * by - az * bz;
}

/** out[oo..] = conj(a[ao..]) · b[bo..]: the rotation from a to b, in a's frame. */
export function quatConjMul(a: ArrayLike<number>, ao: number, b: ArrayLike<number>, bo: number, out: Float64Array, oo: number): void {
  const ax = -a[ao], ay = -a[ao + 1], az = -a[ao + 2], aw = a[ao + 3];
  const bx = b[bo], by = b[bo + 1], bz = b[bo + 2], bw = b[bo + 3];
  out[oo] = ax * bw + aw * bx + ay * bz - az * by;
  out[oo + 1] = ay * bw + aw * by + az * bx - ax * bz;
  out[oo + 2] = az * bw + aw * bz + ax * by - ay * bx;
  out[oo + 3] = aw * bw - ax * bx - ay * by - az * bz;
}

/** out[oo..oo+2] = q[o..] applied to (vx, vy, vz). */
export function rotateVec(q: ArrayLike<number>, o: number, vx: number, vy: number, vz: number, out: Float64Array, oo: number): void {
  const qx = q[o], qy = q[o + 1], qz = q[o + 2], qw = q[o + 3];
  const tx = 2 * (qy * vz - qz * vy);
  const ty = 2 * (qz * vx - qx * vz);
  const tz = 2 * (qx * vy - qy * vx);
  out[oo] = vx + qw * tx + (qy * tz - qz * ty);
  out[oo + 1] = vy + qw * ty + (qz * tx - qx * tz);
  out[oo + 2] = vz + qw * tz + (qx * ty - qy * tx);
}

/** out[oo..oo+2] = conj(q[o..]) applied to (vx, vy, vz): world → body frame. */
export function rotateVecInverse(q: ArrayLike<number>, o: number, vx: number, vy: number, vz: number, out: Float64Array, oo: number): void {
  const qx = -q[o], qy = -q[o + 1], qz = -q[o + 2], qw = q[o + 3];
  const tx = 2 * (qy * vz - qz * vy);
  const ty = 2 * (qz * vx - qx * vz);
  const tz = 2 * (qx * vy - qy * vx);
  out[oo] = vx + qw * tx + (qy * tz - qz * ty);
  out[oo + 1] = vy + qw * ty + (qz * tx - qx * tz);
  out[oo + 2] = vz + qw * tz + (qx * ty - qy * tx);
}

/**
 * Euler angles (x, y, z) of a column-major 3×3 rotation with R = Rx · Ry · Rz. This is the
 * decomposition Bullet's 6-DOF constraint applies MMD's joint limits to; y = ±90° is its
 * singularity, resolved the same way Bullet does.
 */
export function eulerXYZ(R: Float64Array, out: Float64Array): void {
  const sy = R[6];
  if (sy < 1 - 1e-9) {
    if (sy > -1 + 1e-9) {
      out[0] = Math.atan2(-R[7], R[8]);
      out[1] = Math.asin(sy);
      out[2] = Math.atan2(-R[3], R[0]);
    } else {
      out[0] = -Math.atan2(R[1], R[4]);
      out[1] = -Math.PI / 2;
      out[2] = 0;
    }
  } else {
    out[0] = Math.atan2(R[1], R[4]);
    out[1] = Math.PI / 2;
    out[2] = 0;
  }
}

/** out[oo..] = quaternion of Rx(x) · Ry(y) · Rz(z). */
export function quatFromEulerXYZ(x: number, y: number, z: number, out: Float64Array, oo: number): void {
  const cx = Math.cos(x / 2), sx = Math.sin(x / 2);
  const cy = Math.cos(y / 2), sy = Math.sin(y / 2);
  const cz = Math.cos(z / 2), sz = Math.sin(z / 2);
  out[oo] = sx * cy * cz + cx * sy * sz;
  out[oo + 1] = cx * sy * cz - sx * cy * sz;
  out[oo + 2] = sx * sy * cz + cx * cy * sz;
  out[oo + 3] = cx * cy * cz - sx * sy * sz;
}

/** out[oo..oo+2] = axis × angle (shortest arc) of the unit quaternion q[o..]. */
export function rotationVector(q: ArrayLike<number>, o: number, out: Float64Array, oo: number): void {
  let x = q[o], y = q[o + 1], z = q[o + 2], w = q[o + 3];
  if (w < 0) {
    x = -x; y = -y; z = -z; w = -w;
  }
  const s = Math.sqrt(x * x + y * y + z * z);
  if (s < 1e-12) {
    out[oo] = out[oo + 1] = out[oo + 2] = 0;
    return;
  }
  const k = (2 * Math.atan2(s, w)) / s;
  out[oo] = x * k; out[oo + 1] = y * k; out[oo + 2] = z * k;
}

/** q[o..] ← exp(r / 2) · q[o..]: rotate by the world-frame rotation vector r (any size). */
export function rotateByVector(q: Float64Array, o: number, rx: number, ry: number, rz: number): void {
  const angle2 = rx * rx + ry * ry + rz * rz;
  if (angle2 < 1e-24) return;
  let ex: number, ey: number, ez: number, ew: number;
  if (angle2 < 0.01) {
    // Small turn: the first-order quaternion, renormalized below, is exact to within 1e-6.
    ex = rx / 2; ey = ry / 2; ez = rz / 2; ew = 1;
  } else {
    const angle = Math.sqrt(angle2);
    const k = Math.sin(angle / 2) / angle;
    ex = rx * k; ey = ry * k; ez = rz * k; ew = Math.cos(angle / 2);
  }
  const qx = q[o], qy = q[o + 1], qz = q[o + 2], qw = q[o + 3];
  const x = ex * qw + ew * qx + ey * qz - ez * qy;
  const y = ey * qw + ew * qy + ez * qx - ex * qz;
  const z = ez * qw + ew * qz + ex * qy - ey * qx;
  const w = ew * qw - ex * qx - ey * qy - ez * qz;
  const inv = 1 / Math.sqrt(x * x + y * y + z * z + w * w);
  q[o] = x * inv; q[o + 1] = y * inv; q[o + 2] = z * inv; q[o + 3] = w * inv;
}
