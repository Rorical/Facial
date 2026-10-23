import type { PmxModel } from '../loaders/pmx';
import { FOLLOW_BONE, type PhysicsRig, type RigBody } from './rig';

// Vertex-level collision for cloth and hair. MMD's physics moves a few dozen rigid bodies and
// the mesh follows their bones, so the mesh itself can still pass through the body between
// those rigid bodies. After skinning, the GPU pushes every cloth vertex back out of the model's
// own "follow bone" rigid bodies (legs, hips, torso, arms, head) — the same shapes the physics
// collides against.
//
// Set up once per model:
//   · cloth vertices: those skinned to bones that physics drives (or their descendants); the
//     share of such weights is how strongly a vertex is pushed, so the rim sewn to the waist stays put
//   · which colliders apply to a vertex: the collision groups of the rigid body that owns it
//   · how close it may come: a vertex that rests inside or against a collider (a waistband
//     inside the hip capsule) keeps that depth, so nothing is deformed in the rest pose

/** Points tested per vertex along its joint-to-vertex line. Must match skinning.wgsl. */
export const CLOTH_SAMPLES = 4;

/** Colliders per model the shader handles (two 32-bit masks per vertex). */
export const MAX_COLLIDERS = 64;

export interface ClothCollision {
  /** Rig body indices of the colliders, in shader order. */
  colliders: number[];
  /** Per collider: shape (0 sphere, 1 box, 2 capsule), size xyz. */
  shapes: Float32Array;
  /** Per vertex: 0 = not cloth … 1 = fully cloth. */
  cloth: Float32Array;
  /**
   * Per vertex, for each of the CLOTH_SAMPLES points tested along its joint-to-vertex line (the
   * last is the vertex itself): the distance to keep from colliders — at most the margin;
   * negative = may stay that deep inside, as it does in the rest pose.
   */
  clearance: Float32Array;
  /** Per vertex: two u32 of collider bits that apply to it. */
  masks: Uint32Array;
  /**
   * Per vertex: the physics-driven bone it mostly follows (-1 if none). The line from that
   * bone's joint to the vertex stands in for the cloth between them, so colliders are also kept
   * out of the inside of long triangles, not just away from their corners.
   */
  anchorBone: Int32Array;
}

/** Signed distance from a point in a body's local space to its shape (negative inside). */
export function shapeDistance(shape: number, size: ArrayLike<number>, x: number, y: number, z: number): number {
  if (shape === 0) return Math.hypot(x, y, z) - size[0];
  if (shape === 2) {
    const half = size[1] / 2;
    return Math.hypot(x, y - Math.min(half, Math.max(-half, y)), z) - size[0];
  }
  const qx = Math.abs(x) - size[0], qy = Math.abs(y) - size[1], qz = Math.abs(z) - size[2];
  return Math.hypot(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0)) + Math.min(Math.max(qx, qy, qz), 0);
}

/** Distance from a model-space point to a body at its rest transform. */
function restDistance(body: RigBody, x: number, y: number, z: number): number {
  const m = body.rest;
  const dx = x - m[12], dy = y - m[13], dz = z - m[14];
  // Rotation only: the inverse is the transpose.
  return shapeDistance(body.shape, body.size,
    m[0] * dx + m[1] * dy + m[2] * dz,
    m[4] * dx + m[5] * dy + m[6] * dz,
    m[8] * dx + m[9] * dy + m[10] * dz);
}

/**
 * `margin`: how far outside a collider's surface vertices are kept, in model units.
 * Returns null if the model has nothing to collide (no physics-driven mesh or no colliders).
 */
export function buildClothCollision(pmx: PmxModel, rig: PhysicsRig, margin: number): ClothCollision | null {
  const colliders = rig.bodies
    .map((b, i) => (b.mode === FOLLOW_BONE && b.boneIndex >= 0 && b.shape >= 0 ? i : -1))
    .filter((i) => i >= 0)
    .slice(0, MAX_COLLIDERS);
  if (colliders.length === 0 || rig.drivenBones.length === 0) return null;

  // The rigid body that moves each bone: its own, or the nearest physics-driven ancestor's.
  const owner = new Int32Array(pmx.bones.length).fill(-2);
  const bodyOfBone = new Map<number, number>();
  rig.drivenBones.forEach((bone, slot) => bodyOfBone.set(bone, rig.drivenBody[slot]));
  const ownerOf = (bone: number): number => {
    if (bone < 0 || bone >= owner.length) return -1;
    if (owner[bone] !== -2) return owner[bone];
    owner[bone] = -1; // guards against parent cycles
    return (owner[bone] = bodyOfBone.get(bone) ?? ownerOf(rig.boneParents[bone]));
  };

  // Per owning body: which colliders its collision groups let it touch.
  const allowed = new Map<number, [number, number]>();
  const allowedFor = (body: number): [number, number] => {
    let bits = allowed.get(body);
    if (!bits) {
      const a = rig.bodies[body].source;
      bits = [0, 0];
      colliders.forEach((c, j) => {
        const b = rig.bodies[c].source;
        if ((1 << b.group) & a.collisionMask && (1 << a.group) & b.collisionMask) bits![j >> 5] |= 1 << (j & 31);
      });
      allowed.set(body, bits);
    }
    return bits;
  };

  const n = pmx.vertexCount;
  const cloth = new Float32Array(n);
  const clearance = new Float32Array(n * CLOTH_SAMPLES).fill(margin);
  const masks = new Uint32Array(n * 2);
  const anchorBone = new Int32Array(n).fill(-1);
  let any = false;
  for (let v = 0; v < n; v++) {
    let weight = 0;
    let best = -1;
    let bestBone = -1;
    let bestWeight = 0;
    for (let k = 0; k < 4; k++) {
      const w = pmx.skinWeights[v * 4 + k];
      if (w <= 0) continue;
      const body = ownerOf(pmx.skinIndices[v * 4 + k]);
      if (body < 0) continue;
      weight += w;
      if (w > bestWeight) {
        bestWeight = w;
        best = body;
        bestBone = pmx.skinIndices[v * 4 + k];
      }
    }
    if (best < 0) continue;
    const bits = allowedFor(best);
    if ((bits[0] | bits[1]) === 0) continue;
    cloth[v] = Math.min(1, weight);
    masks[v * 2] = bits[0];
    masks[v * 2 + 1] = bits[1];
    anchorBone[v] = bestBone;
    const x = pmx.positions[v * 3], y = pmx.positions[v * 3 + 1], z = pmx.positions[v * 3 + 2];
    const joint = pmx.bones[bestBone].position;
    for (let k = 0; k < CLOTH_SAMPLES; k++) {
      const t = (k + 1) / CLOTH_SAMPLES;
      const sx = joint[0] + (x - joint[0]) * t, sy = joint[1] + (y - joint[1]) * t, sz = joint[2] + (z - joint[2]) * t;
      let nearest = margin;
      colliders.forEach((c, j) => {
        if (bits[j >> 5] & (1 << (j & 31))) nearest = Math.min(nearest, restDistance(rig.bodies[c], sx, sy, sz));
      });
      clearance[v * CLOTH_SAMPLES + k] = nearest;
    }
    any = true;
  }
  if (!any) return null;

  const shapes = new Float32Array(colliders.length * 4);
  colliders.forEach((c, j) => shapes.set([rig.bodies[c].shape, ...rig.bodies[c].size], j * 4));
  return { colliders, shapes, cloth, clearance, masks, anchorBone };
}
