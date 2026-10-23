import type { VmdMotion } from '../loaders/vmd';
import { encodeSjis } from './sjis';

// VMD serializer (inverse of loaders/vmd.ts). Names are Shift-JIS in fixed-width fields;
// characters Shift-JIS can't represent (e.g. many simplified-Chinese hanzi) become "?".

export interface VmdWriteResult {
  buffer: ArrayBuffer;
  warnings: string[];
}

/** VMD model name MMD uses for camera / light motion files. */
export const CAMERA_VMD_MODEL_NAME = 'カメラ・照明';

export function writeVmd(vmd: VmdMotion): VmdWriteResult {
  const warnings = new Set<string>();
  const size =
    30 + 20 +
    4 + vmd.boneFrames.length * 111 +
    4 + vmd.morphFrames.length * 23 +
    4 + vmd.cameraFrames.length * 61 +
    4 + vmd.lightFrames.length * 28 +
    4 + vmd.selfShadowFrames.length * 9 +
    4 + vmd.propertyFrames.reduce((s, p) => s + 9 + p.ikStates.length * 21, 0);
  const buffer = new ArrayBuffer(size);
  const u8 = new Uint8Array(buffer);
  const dv = new DataView(buffer);
  let p = 0;

  const name = (text: string, len: number, what: string | null) => {
    const r = encodeSjis(text, len);
    if (!what) {
      // Informational field: write what fits, no warning.
      u8.set(r.bytes, p);
      p += len;
      return;
    }
    if (r.unmappable.length) warnings.add(`${what}「${text}」含 Shift-JIS 无法表示的字符（${[...new Set(r.unmappable)].join('')}），已写为 ?`);
    if (r.truncated) warnings.add(`${what}「${text}」超过 ${len} 字节，已截断`);
    u8.set(r.bytes, p);
    p += len; // remaining bytes stay 0 (NUL padding)
  };
  const u32 = (v: number) => { dv.setUint32(p, v, true); p += 4; };
  const f32 = (...vs: number[]) => { for (const v of vs) { dv.setFloat32(p, v, true); p += 4; } };
  const bytes = (b: Uint8Array, len: number) => { u8.set(b.subarray(0, len), p); p += len; };

  const sig = 'Vocaloid Motion Data 0002';
  for (let i = 0; i < sig.length; i++) u8[p + i] = sig.charCodeAt(i);
  p += 30;
  // MMD only uses the model name for a "made for another model" notice: no warning for it.
  name(vmd.modelName, 20, null);

  u32(vmd.boneFrames.length);
  for (const k of vmd.boneFrames) {
    name(k.boneName, 15, '骨骼名');
    u32(k.frame);
    f32(...k.position, ...k.rotation);
    bytes(k.interpolation, 64);
  }

  u32(vmd.morphFrames.length);
  for (const k of vmd.morphFrames) {
    name(k.morphName, 15, '表情名');
    u32(k.frame);
    f32(k.weight);
  }

  u32(vmd.cameraFrames.length);
  for (const k of vmd.cameraFrames) {
    u32(k.frame);
    f32(k.distance, ...k.position, ...k.rotation);
    bytes(k.interpolation, 24);
    u32(k.fov);
    u8[p++] = k.perspective ? 0 : 1;
  }

  u32(vmd.lightFrames.length);
  for (const k of vmd.lightFrames) {
    u32(k.frame);
    f32(...k.color, ...k.direction);
  }

  u32(vmd.selfShadowFrames.length);
  for (const k of vmd.selfShadowFrames) {
    u32(k.frame);
    u8[p++] = k.mode;
    f32(k.distance);
  }

  u32(vmd.propertyFrames.length);
  for (const k of vmd.propertyFrames) {
    u32(k.frame);
    u8[p++] = k.visible ? 1 : 0;
    u32(k.ikStates.length);
    for (const s of k.ikStates) {
      name(s.name, 20, 'IK 名');
      u8[p++] = s.enabled ? 1 : 0;
    }
  }

  return { buffer, warnings: [...warnings] };
}
