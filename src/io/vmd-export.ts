import { quat, vec3 } from '../math';
import { curveToBytes, subCurve, type Curve } from '../animation/bezier';
import type { BoneTrack, CameraTrack, MmdMotion, MorphTrack } from '../animation/motion';
import type { VmdBoneFrame, VmdCameraFrame, VmdMorphFrame, VmdMotion, VmdPropertyFrame } from '../loaders/vmd';

// MmdMotion → VmdMotion, optionally cut to a frame range. Untouched keys are copied verbatim
// (including all 64 interpolation bytes). Where the range starts or ends inside a curve
// segment, a key is inserted with the sampled value and the neighbouring curves are cut
// exactly (de Casteljau), so the trimmed clip moves the same as the original.

export interface ExportRange {
  from: number;
  to: number;
  /** Shift frames so `from` becomes frame 0. */
  rebase: boolean;
}

export interface ExportParts {
  /** Bones, morphs, visibility / IK. */
  model: boolean;
  /** Camera, light, self-shadow. */
  scene: boolean;
}

const LINEAR: Curve = [20 / 127, 20 / 127, 107 / 127, 107 / 127];

/** Expand 16 evaluated bytes into MMD's 64-byte layout (rows shifted by one, padded 01 00 00). */
export function expandBoneInterpolation(base: ArrayLike<number>): Uint8Array {
  const out = new Uint8Array(64);
  for (let row = 0; row < 4; row++) {
    for (let i = 0; i < 16; i++) {
      const src = i + row;
      out[row * 16 + i] = src < 16 ? base[src] : src === 16 ? 1 : 0;
    }
  }
  return out;
}

/** Emitted key list for one track: original key indices, or boundary keys (index -1). */
interface Plan {
  frames: number[];
  /** Original key index, or -1 for an inserted boundary key. */
  source: number[];
}

function plan(frames: Uint32Array, range: ExportRange | undefined): Plan {
  const out: Plan = { frames: [], source: [] };
  if (!range) {
    frames.forEach((f, i) => { out.frames.push(f); out.source.push(i); });
    return out;
  }
  const { from, to } = range;
  if (!frames.includes(from)) { out.frames.push(from); out.source.push(-1); }
  frames.forEach((f, i) => {
    if (f >= from && f <= to) { out.frames.push(f); out.source.push(i); }
  });
  if (!frames.includes(to) && to > from) { out.frames.push(to); out.source.push(-1); }
  return out;
}

/**
 * Curve for an emitted segment [a, b] that is only part of an original segment: the matching
 * piece of that segment's curve (callers copy original segments verbatim instead).
 */
function segmentCurve(frames: Uint32Array, a: number, b: number, curveOf: (key: number) => Curve): Curve {
  // Original segment [f_k, f_k+1] that contains [a, b].
  let k = -1;
  for (let i = 0; i + 1 < frames.length; i++) if (frames[i] <= a && b <= frames[i + 1]) { k = i; break; }
  if (k < 0) return LINEAR; // before the first / after the last key: value is held
  const span = frames[k + 1] - frames[k];
  return subCurve(curveOf(k + 1), (a - frames[k]) / span, (b - frames[k]) / span);
}

function exportBones(name: string, t: BoneTrack, range: ExportRange | undefined): VmdBoneFrame[] {
  const p = plan(t.frames, range);
  const pos = vec3.create();
  const rot = quat.create();
  const curve = (key: number, c: number): Curve => {
    const b = key * 64;
    const ip = t.interpolation;
    return [ip[b + c] / 127, ip[b + 4 + c] / 127, ip[b + 8 + c] / 127, ip[b + 12 + c] / 127];
  };
  return p.frames.map((frame, i) => {
    const src = p.source[i];
    let position: [number, number, number];
    let rotation: [number, number, number, number];
    if (src >= 0) {
      position = [t.positions[src * 3], t.positions[src * 3 + 1], t.positions[src * 3 + 2]];
      rotation = [t.rotations[src * 4], t.rotations[src * 4 + 1], t.rotations[src * 4 + 2], t.rotations[src * 4 + 3]];
    } else {
      t.sample(frame, pos, rot);
      position = [pos[0], pos[1], pos[2]];
      rotation = [rot[0], rot[1], rot[2], rot[3]];
    }
    let interpolation: Uint8Array;
    // An original key whose previous key was also emitted keeps its bytes verbatim.
    const verbatim = src > 0 && t.frames[src - 1] === p.frames[i - 1];
    if (i === 0 || verbatim) {
      interpolation = src >= 0 ? t.interpolation.slice(src * 64, src * 64 + 64) : expandBoneInterpolation(linearBase());
    } else {
      const base = new Array<number>(16);
      for (let c = 0; c < 4; c++) {
        const cut = segmentCurve(t.frames, p.frames[i - 1], frame, (key) => curve(key, c));
        const [x1, y1, x2, y2] = curveToBytes(cut);
        base[c] = x1; base[4 + c] = y1; base[8 + c] = x2; base[12 + c] = y2;
      }
      interpolation = expandBoneInterpolation(base);
    }
    return { boneName: name, frame: rebase(frame, range), position, rotation, interpolation };
  });
}

function linearBase(): number[] {
  return [20, 20, 20, 20, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107];
}

function exportMorphs(name: string, t: MorphTrack, range: ExportRange | undefined): VmdMorphFrame[] {
  const p = plan(t.frames, range);
  return p.frames.map((frame, i) => ({
    morphName: name,
    frame: rebase(frame, range),
    weight: p.source[i] >= 0 ? t.weights[p.source[i]] : t.sample(frame),
  }));
}

function exportCamera(t: CameraTrack, range: ExportRange | undefined): VmdCameraFrame[] {
  const p = plan(t.frames, range);
  const pose = { target: vec3.create(), rotation: vec3.create(), distance: 0, fov: 0 };
  const curve = (key: number, c: number): Curve => {
    const b = key * 24 + c * 4;
    const ip = t.interpolation;
    return [ip[b] / 127, ip[b + 2] / 127, ip[b + 1] / 127, ip[b + 3] / 127]; // stored x1, x2, y1, y2
  };
  return p.frames.map((frame, i) => {
    const src = p.source[i];
    if (src >= 0) {
      pose.target.set(t.targets.subarray(src * 3, src * 3 + 3));
      pose.rotation.set(t.rotations.subarray(src * 3, src * 3 + 3));
      pose.distance = t.distances[src];
      pose.fov = t.fovs[src];
    } else {
      t.sample(frame, pose);
    }
    let interpolation: Uint8Array;
    const verbatim = src > 0 && t.frames[src - 1] === p.frames[i - 1];
    if (src >= 0 && (i === 0 || verbatim)) {
      interpolation = t.interpolation.slice(src * 24, src * 24 + 24);
    } else {
      interpolation = new Uint8Array(24);
      for (let c = 0; c < 6; c++) {
        const cut = i === 0 ? LINEAR : segmentCurve(t.frames, p.frames[i - 1], frame, (key) => curve(key, c));
        const [x1, y1, x2, y2] = curveToBytes(cut);
        interpolation.set([x1, x2, y1, y2], c * 4);
      }
    }
    return {
      frame: rebase(frame, range),
      distance: pose.distance,
      position: [pose.target[0], pose.target[1], pose.target[2]],
      rotation: [pose.rotation[0], pose.rotation[1], pose.rotation[2]],
      interpolation,
      fov: Math.round(pose.fov),
      perspective: src >= 0 ? t.perspective[src] === 1 : true,
    };
  });
}

function rebase(frame: number, range: ExportRange | undefined): number {
  return range?.rebase ? frame - range.from : frame;
}

/** Stepped keys (visibility / IK, light, self-shadow): keys inside the range plus the one in effect at its start. */
function steppedKeys<T extends { frame: number }>(keys: T[], range: ExportRange | undefined): T[] {
  if (!range) return keys.map((k) => ({ ...k }));
  const out: T[] = [];
  let active: T | null = null;
  for (const k of keys) if (k.frame <= range.from) active = k;
  if (active && active.frame !== range.from) out.push({ ...active, frame: range.from });
  for (const k of keys) if (k.frame >= range.from && k.frame <= range.to) out.push({ ...k });
  return out.map((k) => ({ ...k, frame: rebase(k.frame, range) }));
}

export function motionToVmd(motion: MmdMotion, modelName: string, parts: ExportParts, range?: ExportRange): VmdMotion {
  const vmd: VmdMotion = {
    modelName, boneFrames: [], morphFrames: [], cameraFrames: [], lightFrames: [], selfShadowFrames: [], propertyFrames: [],
  };
  if (range && !(range.to > range.from)) throw new Error('导出区间无效');
  if (parts.model) {
    for (const [name, t] of motion.boneTracks) vmd.boneFrames.push(...exportBones(name, t, range));
    for (const [name, t] of motion.morphTracks) vmd.morphFrames.push(...exportMorphs(name, t, range));
    vmd.propertyFrames = steppedKeys(motion.propertyKeys, range).map((k): VmdPropertyFrame => ({
      frame: k.frame,
      visible: k.visible,
      ikStates: [...k.ik].map(([name, enabled]) => ({ name, enabled })),
    }));
  }
  if (parts.scene) {
    if (motion.cameraTrack) vmd.cameraFrames = exportCamera(motion.cameraTrack, range);
    vmd.lightFrames = steppedKeys(motion.lightFrames, range);
    vmd.selfShadowFrames = steppedKeys(motion.selfShadowFrames, range);
  }
  return vmd;
}
