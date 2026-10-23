import { vec3, type Mat4 } from '../math';
import type { PhysicsRig } from './rig';

// Wireframe line geometry for rigid bodies and joints, built once in each body's local space.
// Each vertex names the body whose world transform moves it, so per frame only body
// transforms are uploaded. Vertex layout (20 bytes): position f32×3, body u32, kind u32.

/** kind: 0 follow bone, 1 physics, 2 physics + bone (= rigid body mode), 3 joint. */
export const WIREFRAME_KIND_JOINT = 3;
export const DEBUG_VERTEX_BYTES = 20;

const SEGMENTS = 24;
const JOINT_SIZE = 0.35;

export interface DebugGeometry {
  data: ArrayBuffer;
  vertexCount: number;
}

export function buildDebugGeometry(rig: PhysicsRig): DebugGeometry {
  const pos: number[] = [];
  const meta: number[] = []; // body, kind per vertex
  const line = (a: ArrayLike<number>, b: ArrayLike<number>, bodyA: number, bodyB: number, kind: number) => {
    pos.push(a[0], a[1], a[2], b[0], b[1], b[2]);
    meta.push(bodyA, kind, bodyB, kind);
  };
  /** Circle (or arc) of radius r around `center` in the plane spanned by axes u, v. */
  const arc = (center: number[], u: number[], v: number[], r: number, from: number, to: number, body: number, kind: number) => {
    const steps = Math.max(2, Math.round((SEGMENTS * (to - from)) / (Math.PI * 2)));
    let prev: number[] | null = null;
    for (let i = 0; i <= steps; i++) {
      const t = from + ((to - from) * i) / steps;
      const c = Math.cos(t) * r, s = Math.sin(t) * r;
      const p = [center[0] + u[0] * c + v[0] * s, center[1] + u[1] * c + v[1] * s, center[2] + u[2] * c + v[2] * s];
      if (prev) line(prev, p, body, body, kind);
      prev = p;
    }
  };
  const X = [1, 0, 0], Y = [0, 1, 0], Z = [0, 0, 1];
  const TAU = Math.PI * 2;

  rig.bodies.forEach((b, i) => {
    const k = b.mode;
    const [sx, sy, sz] = b.size;
    if (b.shape === 0) {
      arc([0, 0, 0], X, Y, sx, 0, TAU, i, k);
      arc([0, 0, 0], Y, Z, sx, 0, TAU, i, k);
      arc([0, 0, 0], Z, X, sx, 0, TAU, i, k);
    } else if (b.shape === 1) {
      const c = (x: number, y: number, z: number) => [x * sx, y * sy, z * sz];
      for (const s of [-1, 1]) {
        for (const t of [-1, 1]) {
          line(c(-1, s, t), c(1, s, t), i, i, k);
          line(c(s, -1, t), c(s, 1, t), i, i, k);
          line(c(s, t, -1), c(s, t, 1), i, i, k);
        }
      }
    } else {
      // Capsule along local Y: radius sx, cylinder height sy.
      const r = sx, h = sy / 2;
      arc([0, h, 0], X, Z, r, 0, TAU, i, k);
      arc([0, -h, 0], X, Z, r, 0, TAU, i, k);
      for (const [x, z] of [[r, 0], [-r, 0], [0, r], [0, -r]]) line([x, -h, z], [x, h, z], i, i, k);
      arc([0, h, 0], X, Y, r, 0, Math.PI, i, k);
      arc([0, -h, 0], X, Y, r, Math.PI, TAU, i, k);
      arc([0, h, 0], Z, Y, r, 0, Math.PI, i, k);
      arc([0, -h, 0], Z, Y, r, Math.PI, TAU, i, k);
    }
  });

  // Joints: an axis cross in body A's space at the joint frame, plus lines to both body centers.
  const p = vec3.create();
  const at = (m: Mat4, x: number, y: number, z: number) => Array.from(vec3.transformMat4(p, vec3.set(p, x, y, z), m));
  for (const j of rig.joints) {
    const f = j.frameA;
    const o = at(f, 0, 0, 0);
    for (const axis of [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) {
      const [x, y, z] = axis.map((v) => v * JOINT_SIZE);
      line(at(f, -x, -y, -z), at(f, x, y, z), j.bodyA, j.bodyA, WIREFRAME_KIND_JOINT);
    }
    line(o, [0, 0, 0], j.bodyA, j.bodyA, WIREFRAME_KIND_JOINT);
    line(o, [0, 0, 0], j.bodyA, j.bodyB, WIREFRAME_KIND_JOINT);
  }

  const vertexCount = pos.length / 3;
  const data = new ArrayBuffer(vertexCount * DEBUG_VERTEX_BYTES);
  const f32 = new Float32Array(data);
  const u32 = new Uint32Array(data);
  for (let v = 0; v < vertexCount; v++) {
    const o = (v * DEBUG_VERTEX_BYTES) / 4;
    f32[o] = pos[v * 3]; f32[o + 1] = pos[v * 3 + 1]; f32[o + 2] = pos[v * 3 + 2];
    u32[o + 3] = meta[v * 2];
    u32[o + 4] = meta[v * 2 + 1];
  }
  return { data, vertexCount };
}
