import { quat, type Quat, type Vec3 } from '../math';
import type { VmdLightFrame, VmdMotion, VmdSelfShadowFrame } from '../loaders/vmd';
import { bezierWeightBytes } from './bezier';

// VMD keyframes regrouped into per-name tracks with typed arrays, sampled by frame
// number (MMD runs at 30 fps; fractional frames interpolate).

export const MMD_FPS = 30;

/** Largest index i with frames[i] <= f, or -1. */
function findKey(frames: Uint32Array, f: number): number {
  let lo = 0;
  let hi = frames.length - 1;
  if (hi < 0 || f < frames[0]) return -1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (frames[mid] <= f) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

export class BoneTrack {
  constructor(
    readonly frames: Uint32Array,
    readonly positions: Float32Array, // 3 per key
    readonly rotations: Float32Array, // 4 per key
    /**
     * 64 raw bytes per key, kept verbatim so VMDs round-trip losslessly. MMD evaluates the
     * first 16: [x1 X Y Z R][y1 …][x2 …][y2 …].
     */
    readonly interpolation: Uint8Array,
  ) {}

  private static readonly tmpA = quat.create();

  sample(f: number, outPos: Vec3, outRot: Quat): void {
    const n = this.frames.length;
    let k = findKey(this.frames, f);
    if (k < 0) k = 0;
    if (k >= n - 1 || f <= this.frames[k]) {
      outPos[0] = this.positions[k * 3]; outPos[1] = this.positions[k * 3 + 1]; outPos[2] = this.positions[k * 3 + 2];
      quat.copy(outRot, this.rotations.subarray(k * 4, k * 4 + 4));
      return;
    }
    const f0 = this.frames[k];
    const f1 = this.frames[k + 1];
    const x = (f - f0) / (f1 - f0);
    const ip = this.interpolation;
    const b = (k + 1) * 64; // the curve is stored on the later key
    for (let c = 0; c < 3; c++) {
      const w = bezierWeightBytes(ip[b + c], ip[b + 4 + c], ip[b + 8 + c], ip[b + 12 + c], x);
      const p0 = this.positions[k * 3 + c];
      outPos[c] = p0 + (this.positions[(k + 1) * 3 + c] - p0) * w;
    }
    const wr = bezierWeightBytes(ip[b + 3], ip[b + 7], ip[b + 11], ip[b + 15], x);
    const q0 = quat.copy(BoneTrack.tmpA, this.rotations.subarray(k * 4, k * 4 + 4));
    quat.slerp(outRot, q0, this.rotations.subarray((k + 1) * 4, (k + 1) * 4 + 4), wr);
  }
}

export class MorphTrack {
  constructor(readonly frames: Uint32Array, readonly weights: Float32Array) {}

  sample(f: number): number {
    const n = this.frames.length;
    let k = findKey(this.frames, f);
    if (k < 0) k = 0;
    if (k >= n - 1 || f <= this.frames[k]) return this.weights[k];
    const x = (f - this.frames[k]) / (this.frames[k + 1] - this.frames[k]);
    return this.weights[k] + (this.weights[k + 1] - this.weights[k]) * x;
  }
}

export interface CameraPose {
  target: Vec3;
  /** Euler angles (radians) as stored in the VMD. */
  rotation: Vec3;
  /** Signed; MMD's default is -45 (camera in front of the model). */
  distance: number;
  /** Vertical field of view, degrees. */
  fov: number;
  /** false = MMD's "perspective off" (parallel projection). Stepped, never blended. */
  perspective?: boolean;
}

export class CameraTrack {
  constructor(
    readonly frames: Uint32Array,
    readonly targets: Float32Array, // 3 per key
    readonly rotations: Float32Array, // 3 per key
    readonly distances: Float32Array,
    readonly fovs: Float32Array,
    readonly interpolation: Uint8Array, // 24 per key: 6 channels (X, Y, Z, rotation, distance, fov) × [x1, x2, y1, y2]
    readonly perspective: Uint8Array, // 1 = perspective, 0 = orthographic
  ) {}

  sample(f: number, out: CameraPose): void {
    const n = this.frames.length;
    let k = findKey(this.frames, f);
    if (k < 0) k = 0;
    out.perspective = this.perspective[k] !== 0;
    // Keys one frame apart are camera cuts: hold, never blend across them.
    if (k >= n - 1 || f <= this.frames[k] || this.frames[k + 1] - this.frames[k] <= 1) {
      out.target[0] = this.targets[k * 3]; out.target[1] = this.targets[k * 3 + 1]; out.target[2] = this.targets[k * 3 + 2];
      out.rotation[0] = this.rotations[k * 3]; out.rotation[1] = this.rotations[k * 3 + 1]; out.rotation[2] = this.rotations[k * 3 + 2];
      out.distance = this.distances[k];
      out.fov = this.fovs[k];
      return;
    }
    const x = (f - this.frames[k]) / (this.frames[k + 1] - this.frames[k]);
    const ip = this.interpolation;
    const b = (k + 1) * 24;
    const weight = (channel: number) =>
      bezierWeightBytes(ip[b + channel * 4], ip[b + channel * 4 + 2], ip[b + channel * 4 + 1], ip[b + channel * 4 + 3], x);
    const lerp = (arr: Float32Array, i: number, w: number) => arr[k * 3 + i] + (arr[(k + 1) * 3 + i] - arr[k * 3 + i]) * w;
    for (let c = 0; c < 3; c++) out.target[c] = lerp(this.targets, c, weight(c));
    const wr = weight(3);
    for (let c = 0; c < 3; c++) out.rotation[c] = lerp(this.rotations, c, wr); // Euler lerp, like MMD
    const wd = weight(4);
    out.distance = this.distances[k] + (this.distances[k + 1] - this.distances[k]) * wd;
    const wf = weight(5);
    out.fov = this.fovs[k] + (this.fovs[k + 1] - this.fovs[k]) * wf;
  }
}

export interface PropertyKey {
  frame: number;
  visible: boolean;
  ik: Map<string, boolean>;
}

export class MmdMotion {
  readonly boneTracks = new Map<string, BoneTrack>();
  readonly morphTracks = new Map<string, MorphTrack>();
  /** Camera motion, if any VMD had camera keys (the last such VMD wins). */
  cameraTrack: CameraTrack | null = null;
  /** Model visibility / IK on-off keys, sorted by frame. Stepped (no interpolation). */
  readonly propertyKeys: PropertyKey[] = [];
  /** Light keys (colour + direction, blended linearly) and self-shadow keys, sorted by frame. */
  lightFrames: VmdLightFrame[] = [];
  selfShadowFrames: VmdSelfShadowFrame[] = [];
  /** Last keyframe number across all tracks. */
  lastFrame = 0;

  static fromVmd(vmd: VmdMotion): MmdMotion {
    const motion = new MmdMotion();
    motion.merge(vmd);
    return motion;
  }

  /**
   * Add a VMD's keys, the way MMD loads a motion onto a model that already has one: keys are
   * added to the existing tracks and replace only keys on the same frame. (Body motions usually
   * carry one zero key per morph at frame 0; replacing whole tracks would let those wipe out a
   * facial motion loaded earlier.)
   */
  merge(vmd: VmdMotion): void {
    const bones = groupBy(vmd.boneFrames, (k) => k.boneName);
    for (const [name, keys] of bones) {
      const prev = this.boneTracks.get(name);
      const earlier = prev
        ? Array.from(prev.frames, (frame, i) => ({
            frame,
            position: prev.positions.subarray(i * 3, i * 3 + 3),
            rotation: prev.rotations.subarray(i * 4, i * 4 + 4),
            interpolation: prev.interpolation.subarray(i * 64, i * 64 + 64),
          }))
        : [];
      const sorted = dedupeByFrame([...earlier, ...keys]);
      const n = sorted.length;
      const frames = new Uint32Array(n);
      const positions = new Float32Array(n * 3);
      const rotations = new Float32Array(n * 4);
      const interpolation = new Uint8Array(n * 64);
      sorted.forEach((k, i) => {
        frames[i] = k.frame;
        positions.set(k.position, i * 3);
        const q = quat.normalize(quat.create(), Float64Array.from(k.rotation));
        rotations.set(q, i * 4);
        interpolation.set(k.interpolation.subarray(0, 64), i * 64);
      });
      this.boneTracks.set(name, new BoneTrack(frames, positions, rotations, interpolation));
      this.lastFrame = Math.max(this.lastFrame, frames[n - 1]);
    }

    const morphs = groupBy(vmd.morphFrames, (k) => k.morphName);
    for (const [name, keys] of morphs) {
      const prev = this.morphTracks.get(name);
      const earlier = prev ? Array.from(prev.frames, (frame, i) => ({ frame, weight: prev.weights[i] })) : [];
      const sorted = dedupeByFrame([...earlier, ...keys]);
      const frames = Uint32Array.from(sorted, (k) => k.frame);
      const weights = Float32Array.from(sorted, (k) => k.weight);
      this.morphTracks.set(name, new MorphTrack(frames, weights));
      this.lastFrame = Math.max(this.lastFrame, frames[frames.length - 1]);
    }

    if (vmd.cameraFrames.length > 0) {
      const keys = dedupeByFrame(vmd.cameraFrames);
      const n = keys.length;
      const track = new CameraTrack(
        new Uint32Array(n), new Float32Array(n * 3), new Float32Array(n * 3),
        new Float32Array(n), new Float32Array(n), new Uint8Array(n * 24), new Uint8Array(n),
      );
      keys.forEach((k, i) => {
        track.frames[i] = k.frame;
        track.targets.set(k.position, i * 3);
        track.rotations.set(k.rotation, i * 3);
        track.distances[i] = k.distance;
        track.fovs[i] = k.fov;
        track.interpolation.set(k.interpolation, i * 24);
        track.perspective[i] = k.perspective ? 1 : 0;
      });
      this.cameraTrack = track;
      this.lastFrame = Math.max(this.lastFrame, track.frames[n - 1]);
    }

    for (const p of vmd.propertyFrames) {
      this.propertyKeys.push({ frame: p.frame, visible: p.visible, ik: new Map(p.ikStates.map((s) => [s.name, s.enabled])) });
      this.lastFrame = Math.max(this.lastFrame, p.frame);
    }
    this.propertyKeys.sort((a, b) => a.frame - b.frame);

    if (vmd.lightFrames.length) this.lightFrames = dedupeByFrame([...this.lightFrames, ...vmd.lightFrames]);
    if (vmd.selfShadowFrames.length) this.selfShadowFrames = dedupeByFrame([...this.selfShadowFrames, ...vmd.selfShadowFrames]);
    for (const k of [...vmd.lightFrames, ...vmd.selfShadowFrames]) this.lastFrame = Math.max(this.lastFrame, k.frame);
  }

  /** Light at frame f. Returns false (and leaves `out` alone) if the motion has no light keys. */
  sampleLight(f: number, out: { direction: number[]; color: number[] }): boolean {
    const keys = this.lightFrames;
    if (keys.length === 0) return false;
    const k = lastKeyAt(keys, f);
    const a = keys[Math.max(k, 0)];
    const b = keys[Math.min(k + 1, keys.length - 1)];
    const t = k < 0 || b.frame <= a.frame ? 0 : (f - a.frame) / (b.frame - a.frame);
    for (let c = 0; c < 3; c++) {
      out.direction[c] = a.direction[c] + (b.direction[c] - a.direction[c]) * t;
      out.color[c] = a.color[c] + (b.color[c] - a.color[c]) * t;
    }
    return true;
  }

  /**
   * Self-shadow setting at frame f, or null if the motion has none. `mode` 0 = off (1 and 2
   * are MMD's two shadow-map layouts); `range` is how far from the camera shadows reach.
   */
  selfShadowAt(f: number): { mode: number; range: number } | null {
    const keys = this.selfShadowFrames;
    if (keys.length === 0) return null;
    const k = lastKeyAt(keys, f);
    const a = keys[Math.max(k, 0)];
    const b = keys[Math.min(k + 1, keys.length - 1)];
    const t = k < 0 || b.frame <= a.frame ? 0 : (f - a.frame) / (b.frame - a.frame);
    return { mode: a.mode, range: selfShadowRange(a.distance + (b.distance - a.distance) * t) };
  }

  get isEmpty(): boolean {
    return this.boneTracks.size === 0 && this.morphTracks.size === 0 && !this.cameraTrack &&
      this.propertyKeys.length === 0 && this.lightFrames.length === 0 && this.selfShadowFrames.length === 0;
  }

  /** IK enable states in effect at frame f, or null if the motion has none. */
  ikStatesAt(f: number): Map<string, boolean> | null {
    let found: Map<string, boolean> | null = null;
    for (const key of this.propertyKeys) {
      if (key.frame > f) break;
      found = key.ik;
    }
    return found ?? this.propertyKeys[0]?.ik ?? null;
  }
}

/** Index of the last key with frame <= f, or -1. */
function lastKeyAt(keys: { frame: number }[], f: number): number {
  let k = -1;
  while (k + 1 < keys.length && keys[k + 1].frame <= f) k++;
  return k;
}

/**
 * A VMD stores MMD's shadow "distance" setting v (0–9999, default 8875) as 0.1 − v·10⁻⁵: larger v
 * means a tighter, sharper shadow area around the camera. MMD doesn't document the exact extent;
 * (10000 − v) / 10 world units (112 at the default) matches it closely enough in practice.
 */
export function selfShadowRange(stored: number): number {
  return Math.max(stored * 10000, 1);
}

function groupBy<T>(items: T[], key: (t: T) => string): Map<string, T[]> {
  const map = new Map<string, T[]>();
  for (const item of items) {
    const k = key(item);
    const list = map.get(k);
    if (list) list.push(item);
    else map.set(k, [item]);
  }
  return map;
}

function dedupeByFrame<T extends { frame: number }>(keys: T[]): T[] {
  const byFrame = new Map<number, T>();
  for (const k of keys) byFrame.set(k.frame, k); // last one wins
  return Array.from(byFrame.values()).sort((a, b) => a.frame - b.frame);
}

/**
 * Split a VMD into what belongs to a model (bones, morphs, visibility/IK) and what belongs to
 * the scene (camera, light, self-shadow). MMD itself saves these as separate files, but many
 * distributed VMDs combine them.
 */
export function splitVmd(vmd: VmdMotion): { model: VmdMotion; scene: VmdMotion } {
  const empty = { boneFrames: [], morphFrames: [], cameraFrames: [], lightFrames: [], selfShadowFrames: [], propertyFrames: [] };
  return {
    model: { ...empty, modelName: vmd.modelName, boneFrames: vmd.boneFrames, morphFrames: vmd.morphFrames, propertyFrames: vmd.propertyFrames },
    scene: { ...empty, modelName: vmd.modelName, cameraFrames: vmd.cameraFrames, lightFrames: vmd.lightFrames, selfShadowFrames: vmd.selfShadowFrames },
  };
}

/** Sorted unique keyframe numbers across tracks (for timelines). */
export function keyframeUnion(tracks: Iterable<{ frames: Uint32Array }>): Uint32Array {
  const set = new Set<number>();
  for (const t of tracks) for (const f of t.frames) set.add(f);
  return Uint32Array.from([...set].sort((a, b) => a - b));
}
