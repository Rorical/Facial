import { BinaryReader, decodeFixedString, decodeSjisName, type Vec3Tuple, type Vec4Tuple } from './binary-reader';

// VMD (Vocaloid Motion Data) parser. Names are Shift-JIS, fixed-length, NUL-terminated.
// Sections after bone frames are optional: old files simply end early.

export interface VmdBoneFrame {
  boneName: string;
  frame: number;
  position: Vec3Tuple;
  rotation: Vec4Tuple;
  /**
   * 64 raw bytes. The first 16 are what MMD uses, laid out per channel (X, Y, Z, Rotation):
   * [x1 ×4][y1 ×4][x2 ×4][y2 ×4], each 0–127. They describe the curve from the
   * previous keyframe up to this one.
   */
  interpolation: Uint8Array;
}

export interface VmdMorphFrame {
  morphName: string;
  frame: number;
  weight: number;
}

export interface VmdCameraFrame {
  frame: number;
  distance: number;
  position: Vec3Tuple;
  rotation: Vec3Tuple;
  /** 24 bytes: per channel (X, Y, Z, Rotation, Distance, FOV) → x1, x2, y1, y2. */
  interpolation: Uint8Array;
  fov: number;
  perspective: boolean;
}

export interface VmdLightFrame {
  frame: number;
  color: Vec3Tuple;
  direction: Vec3Tuple;
}

export interface VmdSelfShadowFrame {
  frame: number;
  mode: number;
  distance: number;
}

export interface VmdPropertyFrame {
  frame: number;
  visible: boolean;
  ikStates: { name: string; enabled: boolean }[];
}

export interface VmdMotion {
  modelName: string;
  boneFrames: VmdBoneFrame[];
  morphFrames: VmdMorphFrame[];
  cameraFrames: VmdCameraFrame[];
  lightFrames: VmdLightFrame[];
  selfShadowFrames: VmdSelfShadowFrame[];
  propertyFrames: VmdPropertyFrame[];
}

export function parseVmd(buffer: ArrayBuffer): VmdMotion {
  const r = new BinaryReader(buffer);
  const sjis = new TextDecoder('shift-jis');
  const ascii = new TextDecoder('ascii');

  const signature = decodeFixedString(r.bytes(30), ascii);
  if (!signature.startsWith('Vocaloid Motion Data')) {
    throw new Error(`Not a VMD file (signature "${signature}")`);
  }
  const modelNameLength = signature === 'Vocaloid Motion Data file' ? 10 : 20;
  const modelName = decodeFixedString(r.bytes(modelNameLength), sjis);

  const motion: VmdMotion = {
    modelName,
    boneFrames: [],
    morphFrames: [],
    cameraFrames: [],
    lightFrames: [],
    selfShadowFrames: [],
    propertyFrames: [],
  };

  const count = (): number => (r.remaining >= 4 ? r.u32() : 0);

  const boneCount = count();
  for (let i = 0; i < boneCount; i++) {
    motion.boneFrames.push({
      boneName: decodeSjisName(r.bytes(15), sjis),
      frame: r.u32(),
      position: r.vec3(),
      rotation: r.vec4(),
      interpolation: r.bytes(64).slice(),
    });
  }

  const morphCount = count();
  for (let i = 0; i < morphCount; i++) {
    motion.morphFrames.push({
      morphName: decodeSjisName(r.bytes(15), sjis),
      frame: r.u32(),
      weight: r.f32(),
    });
  }

  const cameraCount = count();
  for (let i = 0; i < cameraCount; i++) {
    motion.cameraFrames.push({
      frame: r.u32(),
      distance: r.f32(),
      position: r.vec3(),
      rotation: r.vec3(),
      interpolation: r.bytes(24).slice(),
      fov: r.u32(),
      perspective: r.u8() === 0,
    });
  }

  const lightCount = count();
  for (let i = 0; i < lightCount; i++) {
    motion.lightFrames.push({ frame: r.u32(), color: r.vec3(), direction: r.vec3() });
  }

  const shadowCount = count();
  for (let i = 0; i < shadowCount; i++) {
    motion.selfShadowFrames.push({ frame: r.u32(), mode: r.u8(), distance: r.f32() });
  }

  const propertyCount = count();
  for (let i = 0; i < propertyCount; i++) {
    const frame = r.u32();
    const visible = r.u8() === 1;
    const ikCount = r.u32();
    const ikStates = [];
    for (let k = 0; k < ikCount; k++) {
      ikStates.push({ name: decodeSjisName(r.bytes(20), sjis), enabled: r.u8() === 1 });
    }
    motion.propertyFrames.push({ frame, visible, ikStates });
  }

  return motion;
}
