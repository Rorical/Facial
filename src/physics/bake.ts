import { mat4, quat, vec3, type Quat, type Vec3 } from '../math';
import type { PmxModel } from '../loaders/pmx';
import type { VmdBoneFrame } from '../loaders/vmd';
import { expandBoneInterpolation } from '../io/vmd-export';
import type { Skeleton } from '../runtime/skeleton';
import type { PhysicsRig } from './rig';
import { buildClothMesh, type ClothMesh } from '../cloth/cloth-mesh';
import { clothVertexData, type ClothSettings } from '../cloth/cloth-sim';
import type { PhysicsRunner } from './runner';

// Physics baking: simulate a frame range once (fixed 30 fps, independent of display rate)
// and store, per frame, every physics bone's transform relative to its parent. Playback then
// replays the bake: deterministic, seekable in both directions, no warm-up, no Bullet cost.
// Bakes can be exported as VMD bone keys (simplified within a tolerance) for other tools.

// This module holds bake data, playback and export and does not load Bullet; the simulation
// itself is in bake-sim.ts (used by the bake worker).

export const BAKE_FLOATS = 7; // position xyz + rotation quaternion xyzw

/** Cloth simulated along with the bake: every particle's position on every frame. */
export interface ClothBake {
  /** Material indices that were simulated as cloth. */
  materials: number[];
  settings: ClothSettings;
  particleCount: number;
  /** Per frame, per particle: position xyz in model space. */
  data: Float32Array;
}
const FLOATS = BAKE_FLOATS;

export interface PhysicsBakeData {
  /** Inclusive frame range. */
  from: number;
  to: number;
  /** Bone indices baked (the rig's physics-driven bones, parents first). */
  bones: Int32Array;
  /** Per frame, per bone: parent-relative position (3) + rotation quaternion (4). */
  data: Float32Array;
  cloth?: ClothBake | null;
}

export class PhysicsBake implements PhysicsBakeData {
  constructor(
    readonly from: number, readonly to: number, readonly bones: Int32Array, readonly data: Float32Array,
    readonly cloth: ClothBake | null = null,
  ) {}

  static fromData(d: PhysicsBakeData): PhysicsBake {
    return new PhysicsBake(d.from, d.to, d.bones, d.data, d.cloth ?? null);
  }

  get frameCount(): number {
    return this.to - this.from + 1;
  }

  /** Parent-relative transform of bone `slot` at a (fractional) frame, clamped to the range. */
  sample(frame: number, slot: number, outPos: Vec3, outRot: Quat, tmp: Quat): void {
    const f = Math.min(this.to, Math.max(this.from, frame)) - this.from;
    const f0 = Math.floor(f);
    const f1 = Math.min(f0 + 1, this.frameCount - 1);
    const t = f - f0;
    const n = this.bones.length;
    const a = (f0 * n + slot) * FLOATS;
    const b = (f1 * n + slot) * FLOATS;
    const d = this.data;
    for (let c = 0; c < 3; c++) outPos[c] = d[a + c] + (d[b + c] - d[a + c]) * t;
    quat.set(outRot, d[a + 3], d[a + 4], d[a + 5], d[a + 6]);
    if (t > 0) quat.slerp(outRot, outRot, quat.set(tmp, d[b + 3], d[b + 4], d[b + 5], d[b + 6]), t);
  }
}

/** Replays a bake as the model's physics. */
export class BakedPhysics implements PhysicsRunner {
  readonly kind = 'baked';
  readonly bodyCount: number;
  readonly jointCount: number;
  enabled = true;
  wantDebug = false;
  readonly stepMs = 0;
  /** Wireframes fall back to bone-derived body transforms. */
  readonly bodyTransforms = null;
  readonly freshRatio = 1;
  private readonly pos = vec3.create();
  private readonly rot = quat.create();
  private readonly tmp = quat.create();
  /** The cloth's topology, rebuilt from the model (null if the bake has no cloth or it no longer fits). */
  readonly clothMesh: ClothMesh | null = null;
  clothFrame: Float32Array | null = null;
  private readonly clothPositions: Float32Array | null = null;
  private readonly normalSigns: Float32Array | null = null;

  constructor(readonly bake: PhysicsBake, readonly rig: PhysicsRig, pmx?: PmxModel) {
    this.bodyCount = rig.bodies.length;
    this.jointCount = rig.joints.length;
    const cloth = bake.cloth;
    if (cloth && pmx) {
      const mesh = buildClothMesh(pmx, rig, cloth.materials);
      if (mesh && mesh.particleCount === cloth.particleCount) {
        this.clothMesh = mesh;
        this.clothPositions = new Float32Array(mesh.particleCount * 3);
        this.clothFrame = new Float32Array(mesh.vertices.length * 8);
        // Keep each vertex's normal on the side the model had it (inner faces, flipped shells).
        const rest = clothVertexData(mesh, mesh.rest, new Float32Array(mesh.vertices.length * 8));
        this.normalSigns = Float32Array.from(mesh.vertices, (v, i) =>
          rest[i * 8 + 4] * pmx.normals[v * 3] + rest[i * 8 + 5] * pmx.normals[v * 3 + 1] + rest[i * 8 + 6] * pmx.normals[v * 3 + 2] < 0 ? -1 : 1);
      }
    }
  }

  /** This frame's cloth: particle positions blended between the two baked frames around it. */
  private sampleCloth(frame: number): void {
    const cloth = this.bake.cloth!;
    const mesh = this.clothMesh!;
    const out = this.clothPositions!;
    const { bake } = this;
    const f = Math.min(bake.to, Math.max(bake.from, frame)) - bake.from;
    const f0 = Math.floor(f);
    const f1 = Math.min(f0 + 1, bake.frameCount - 1);
    const t = f - f0;
    const stride = mesh.particleCount * 3;
    const a = cloth.data.subarray(f0 * stride, f0 * stride + stride);
    const b = cloth.data.subarray(f1 * stride, f1 * stride + stride);
    for (let i = 0; i < stride; i++) out[i] = a[i] + (b[i] - a[i]) * t;
    const data = clothVertexData(mesh, out, this.clothFrame!);
    const signs = this.normalSigns!;
    for (let v = 0; v < signs.length; v++) {
      if (signs[v] < 0) {
        data[v * 8 + 4] = -data[v * 8 + 4]; data[v * 8 + 5] = -data[v * 8 + 5]; data[v * 8 + 6] = -data[v * 8 + 6];
      }
    }
  }

  update(skeleton: Skeleton, _dt: number, frame: number): boolean {
    const { bake } = this;
    for (let slot = 0; slot < bake.bones.length; slot++) {
      const bone = skeleton.bones[bake.bones[slot]];
      if (!bone) continue;
      bake.sample(frame, slot, this.pos, this.rot, this.tmp);
      mat4.fromRotationTranslation(bone.physicsLocal, this.rot, this.pos);
      bone.physicsKeepTranslation = false; // the bake already holds the final transform
      bone.physicsActive = true;
    }
    if (this.clothMesh) this.sampleCloth(frame);
    return true;
  }

  requestReset(): void {}
  dispose(): void {}
}

// ---------------------------------------------------------------------------------------
// Export as VMD bone keys

export interface BakeVmdOptions {
  /** Only this range (inclusive), optionally rebased so `from` becomes 0. */
  range?: { from: number; to: number; rebase: boolean };
  /**
   * Max visible error in model units (1 ≈ 8 cm): how far the bone's origin *or tip* may move
   * away from the baked pose. Default 0.02 ≈ 1.6 mm.
   */
  tolerance?: number;
  /** Hard cap on rotation error regardless of bone length (default 3°). */
  maxAngle?: number;
}

/** Simplification presets offered in the UI (tolerance in model units). */
export const BAKE_EXPORT_PRESETS = {
  exact: { label: '精确（误差 ≤ 0.4 mm）', tolerance: 0.005 },
  standard: { label: '标准（误差 ≤ 1.6 mm）', tolerance: 0.02 },
  compact: { label: '紧凑（误差 ≤ 4 mm，文件更小）', tolerance: 0.05 },
} as const;
export type BakeExportPreset = keyof typeof BAKE_EXPORT_PRESETS;

/**
 * Length used to turn a rotation error into a tip displacement: the tail if the bone has one,
 * else its farthest child, else a typical hair / cloth segment.
 */
function boneLength(pmx: PmxModel, index: number, childReach: Float64Array): number {
  const b = pmx.bones[index];
  let len = 0;
  if (b.tailBoneIndex >= 0 && b.tailBoneIndex < pmx.bones.length) {
    const t = pmx.bones[b.tailBoneIndex].position;
    len = Math.hypot(t[0] - b.position[0], t[1] - b.position[1], t[2] - b.position[2]);
  } else {
    len = Math.hypot(...b.tailOffset);
  }
  len = Math.max(len, childReach[index]);
  return len > 1e-3 ? len : 0.5;
}

const LINEAR_64 = expandBoneInterpolation([20, 20, 20, 20, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107]);

function childReaches(pmx: PmxModel): Float64Array {
  const reach = new Float64Array(pmx.bones.length);
  pmx.bones.forEach((c) => {
    if (c.parentIndex < 0 || c.parentIndex >= pmx.bones.length) return;
    const p = pmx.bones[c.parentIndex].position;
    const d = Math.hypot(c.position[0] - p[0], c.position[1] - p[1], c.position[2] - p[2]);
    reach[c.parentIndex] = Math.max(reach[c.parentIndex], d);
  });
  return reach;
}

/**
 * Bake → VMD keys with linear interpolation, dropping keys that linear / slerp interpolation
 * between their neighbours reproduces within tolerance (Douglas–Peucker). The tolerance is a
 * visible distance: position error at the bone's origin, rotation error at its tip.
 * A VMD key stores the offset from the bone's rest position relative to its parent, hence
 * `- localOffset`. Moving cloth / hair changes every frame, so expect many keys — MMD tools
 * write baked physics one key per frame.
 */
export function bakeToVmdBoneFrames(bake: PhysicsBake, pmx: PmxModel, opts: BakeVmdOptions = {}): VmdBoneFrame[] {
  const posTol = opts.tolerance ?? 0.02;
  const maxAngle = opts.maxAngle ?? (3 * Math.PI) / 180;
  const reach = childReaches(pmx);
  const first = Math.max(bake.from, opts.range?.from ?? bake.from);
  const last = Math.min(bake.to, opts.range?.to ?? bake.to);
  if (last < first) return [];
  const shift = opts.range?.rebase ? opts.range.from : 0;
  const n = bake.bones.length;
  const count = last - first + 1;
  const out: VmdBoneFrame[] = [];
  const qa = quat.create(), qb = quat.create(), qi = quat.create(), qt = quat.create();

  for (let slot = 0; slot < n; slot++) {
    const def = pmx.bones[bake.bones[slot]];
    // Rotation error allowed so the tip moves at most posTol (capped).
    const rotTol = Math.min(maxAngle, posTol / boneLength(pmx, bake.bones[slot], reach));
    const parent = def.parentIndex >= 0 ? pmx.bones[def.parentIndex] : null;
    const offset = parent ? def.position.map((v, c) => v - parent.position[c]) : def.position;
    const at = (i: number) => ((first - bake.from + i) * n + slot) * FLOATS;
    const d = bake.data;

    const error = (i: number, a: number, b: number): number => {
      const t = (i - a) / (b - a);
      const oa = at(a), ob = at(b), oi = at(i);
      let pe = 0;
      for (let c = 0; c < 3; c++) pe = Math.max(pe, Math.abs(d[oa + c] + (d[ob + c] - d[oa + c]) * t - d[oi + c]));
      quat.set(qa, d[oa + 3], d[oa + 4], d[oa + 5], d[oa + 6]);
      quat.set(qb, d[ob + 3], d[ob + 4], d[ob + 5], d[ob + 6]);
      quat.set(qi, d[oi + 3], d[oi + 4], d[oi + 5], d[oi + 6]);
      quat.slerp(qt, qa, qb, t);
      const dot = Math.min(1, Math.abs(qt[0] * qi[0] + qt[1] * qi[1] + qt[2] * qi[2] + qt[3] * qi[3]));
      return Math.max(pe / posTol, (2 * Math.acos(dot)) / rotTol);
    };

    const keep = new Uint8Array(count);
    keep[0] = 1;
    keep[count - 1] = 1;
    const stack: [number, number][] = [[0, count - 1]];
    while (stack.length) {
      const [a, b] = stack.pop()!;
      let worst = -1;
      let worstErr = 1;
      for (let i = a + 1; i < b; i++) {
        const e = error(i, a, b);
        if (e > worstErr) {
          worstErr = e;
          worst = i;
        }
      }
      if (worst >= 0) {
        keep[worst] = 1;
        stack.push([a, worst], [worst, b]);
      }
    }

    for (let i = 0; i < count; i++) {
      if (!keep[i]) continue;
      const o = at(i);
      out.push({
        boneName: def.name,
        frame: first + i - shift,
        position: [d[o] - offset[0], d[o + 1] - offset[1], d[o + 2] - offset[2]],
        rotation: [d[o + 3], d[o + 4], d[o + 5], d[o + 6]],
        interpolation: LINEAR_64.slice(),
      });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Binary form for project files: small JSON header + raw data

export interface BakeMeta {
  from: number;
  to: number;
  bones: number[];
  /** Cloth baked along with it (its positions are in a second file). */
  cloth?: { materials: number[]; settings: ClothSettings; particleCount: number } | null;
}

const bytesOf = (a: Float32Array) => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);

export function serializeBake(bake: PhysicsBake): { meta: BakeMeta; data: Uint8Array; cloth: Uint8Array | null } {
  const c = bake.cloth;
  return {
    meta: {
      from: bake.from, to: bake.to, bones: Array.from(bake.bones),
      cloth: c ? { materials: c.materials, settings: c.settings, particleCount: c.particleCount } : null,
    },
    data: bytesOf(bake.data),
    cloth: c ? bytesOf(c.data) : null,
  };
}

export function deserializeBake(meta: BakeMeta, bytes: ArrayBuffer, clothBytes?: ArrayBuffer | null): PhysicsBake {
  const data = new Float32Array(bytes);
  const frames = meta.to - meta.from + 1;
  if (data.length !== frames * meta.bones.length * FLOATS) throw new Error('物理烘焙数据损坏');
  let cloth: ClothBake | null = null;
  if (meta.cloth && clothBytes) {
    const positions = new Float32Array(clothBytes);
    if (positions.length === frames * meta.cloth.particleCount * 3) cloth = { ...meta.cloth, data: positions };
  }
  return new PhysicsBake(meta.from, meta.to, Int32Array.from(meta.bones), data, cloth);
}
