import { BinaryReader, type Vec3Tuple, type Vec4Tuple } from './binary-reader';

// PMX 2.0 / 2.1 parser. Reference: PMX仕様 (PmxEditor bundled spec).
// Soft bodies (2.1 only, very rare) are not parsed.

export enum SkinType {
  BDEF1 = 0,
  BDEF2 = 1,
  BDEF4 = 2,
  SDEF = 3,
  QDEF = 4,
}

export const MaterialFlag = {
  DoubleSided: 0x01,
  GroundShadow: 0x02,
  CastSelfShadow: 0x04,
  ReceiveSelfShadow: 0x08,
  Edge: 0x10,
  VertexColor: 0x20,
  PointDraw: 0x40,
  LineDraw: 0x80,
} as const;

export const BoneFlag = {
  TailIsBone: 0x0001,
  Rotatable: 0x0002,
  Translatable: 0x0004,
  Visible: 0x0008,
  Enabled: 0x0010,
  IK: 0x0020,
  InheritLocal: 0x0080,
  InheritRotation: 0x0100,
  InheritTranslation: 0x0200,
  FixedAxis: 0x0400,
  LocalAxes: 0x0800,
  AfterPhysics: 0x1000,
  ExternalParent: 0x2000,
} as const;

export enum SphereMode {
  None = 0,
  Multiply = 1,
  Add = 2,
  SubTexture = 3,
}

export enum MorphKind {
  Group = 0,
  Vertex = 1,
  Bone = 2,
  UV = 3,
  AdditionalUV1 = 4,
  AdditionalUV2 = 5,
  AdditionalUV3 = 6,
  AdditionalUV4 = 7,
  Material = 8,
  Flip = 9,
  Impulse = 10,
}

export interface PmxHeader {
  version: number;
  encoding: 'utf-16le' | 'utf-8';
  additionalUvCount: number;
  vertexIndexSize: number;
  textureIndexSize: number;
  materialIndexSize: number;
  boneIndexSize: number;
  morphIndexSize: number;
  rigidBodyIndexSize: number;
}

export interface PmxMaterial {
  name: string;
  nameEn: string;
  diffuse: Vec4Tuple;
  specular: Vec3Tuple;
  specularPower: number;
  ambient: Vec3Tuple;
  flags: number;
  edgeColor: Vec4Tuple;
  edgeSize: number;
  textureIndex: number;
  sphereTextureIndex: number;
  sphereMode: SphereMode;
  /** true: toonIndex is a shared toon (0-9 → toon01.bmp … toon10.bmp); false: a texture index. */
  sharedToon: boolean;
  toonIndex: number;
  memo: string;
  /** Number of vertex indices (not triangles) drawn by this material. */
  indexCount: number;
}

export interface PmxIkLink {
  boneIndex: number;
  limits: { min: Vec3Tuple; max: Vec3Tuple } | null;
}

export interface PmxIk {
  targetIndex: number;
  loopCount: number;
  /** Max rotation per iteration, radians. */
  limitAngle: number;
  links: PmxIkLink[];
}

export interface PmxBone {
  name: string;
  nameEn: string;
  position: Vec3Tuple;
  parentIndex: number;
  layer: number;
  flags: number;
  tailBoneIndex: number;
  tailOffset: Vec3Tuple;
  inherit: { parentIndex: number; weight: number } | null;
  fixedAxis: Vec3Tuple | null;
  localAxes: { x: Vec3Tuple; z: Vec3Tuple } | null;
  externalParentKey: number;
  ik: PmxIk | null;
}

export interface PmxMaterialMorphOffset {
  materialIndex: number;
  /** 0 = multiply, 1 = add */
  operation: number;
  diffuse: Vec4Tuple;
  specular: Vec3Tuple;
  specularPower: number;
  ambient: Vec3Tuple;
  edgeColor: Vec4Tuple;
  edgeSize: number;
  textureTint: Vec4Tuple;
  sphereTint: Vec4Tuple;
  toonTint: Vec4Tuple;
}

export type PmxMorph = {
  name: string;
  nameEn: string;
  panel: number;
} & (
  | { kind: MorphKind.Group | MorphKind.Flip; children: { morphIndex: number; weight: number }[] }
  | { kind: MorphKind.Vertex; vertexIndices: Int32Array; offsets: Float32Array /* xyz */ }
  | {
      kind: MorphKind.Bone;
      bones: { boneIndex: number; translation: Vec3Tuple; rotation: Vec4Tuple }[];
    }
  | {
      kind: MorphKind.UV | MorphKind.AdditionalUV1 | MorphKind.AdditionalUV2 | MorphKind.AdditionalUV3 | MorphKind.AdditionalUV4;
      vertexIndices: Int32Array;
      offsets: Float32Array /* xyzw */;
    }
  | { kind: MorphKind.Material; materials: PmxMaterialMorphOffset[] }
  | {
      kind: MorphKind.Impulse;
      impulses: { rigidBodyIndex: number; local: boolean; velocity: Vec3Tuple; torque: Vec3Tuple }[];
    }
);

export interface PmxDisplayFrame {
  name: string;
  nameEn: string;
  special: boolean;
  items: { type: 'bone' | 'morph'; index: number }[];
}

export interface PmxRigidBody {
  name: string;
  nameEn: string;
  boneIndex: number;
  group: number;
  collisionMask: number;
  /** 0 = sphere, 1 = box, 2 = capsule */
  shape: number;
  size: Vec3Tuple;
  position: Vec3Tuple;
  rotation: Vec3Tuple;
  mass: number;
  linearDamping: number;
  angularDamping: number;
  restitution: number;
  friction: number;
  /** 0 = follow bone, 1 = physics, 2 = physics + bone position */
  mode: number;
}

export interface PmxJoint {
  name: string;
  nameEn: string;
  type: number;
  rigidBodyA: number;
  rigidBodyB: number;
  position: Vec3Tuple;
  rotation: Vec3Tuple;
  positionMin: Vec3Tuple;
  positionMax: Vec3Tuple;
  rotationMin: Vec3Tuple;
  rotationMax: Vec3Tuple;
  springPosition: Vec3Tuple;
  springRotation: Vec3Tuple;
}

export interface PmxModel {
  /** Source file format; PMD files are converted to this structure by parsePmd. */
  format: 'PMX' | 'PMD' | 'X';
  /** A DirectX .x accessory: lit like MMD lights accessories (plain diffuse, no toon). */
  accessory?: boolean;
  header: PmxHeader;
  name: string;
  nameEn: string;
  comment: string;
  commentEn: string;

  vertexCount: number;
  positions: Float32Array; // 3 per vertex
  normals: Float32Array; // 3 per vertex
  uvs: Float32Array; // 2 per vertex
  additionalUvs: Float32Array[]; // additionalUvCount arrays, 4 per vertex
  skinTypes: Uint8Array;
  /** 4 per vertex; unused slots are 0 with weight 0. */
  skinIndices: Uint32Array;
  skinWeights: Float32Array; // 4 per vertex
  /** SDEF parameters for SDEF vertices only. */
  sdef: { vertexIndices: Uint32Array; c: Float32Array; r0: Float32Array; r1: Float32Array };
  edgeScales: Float32Array;

  indices: Uint32Array;
  textures: string[];
  materials: PmxMaterial[];
  bones: PmxBone[];
  morphs: PmxMorph[];
  displayFrames: PmxDisplayFrame[];
  rigidBodies: PmxRigidBody[];
  joints: PmxJoint[];
}

export function parsePmx(buffer: ArrayBuffer): PmxModel {
  const r = new BinaryReader(buffer);

  const magic = String.fromCharCode(r.u8(), r.u8(), r.u8(), r.u8());
  if (magic !== 'PMX ') throw new Error(`Not a PMX file (magic "${magic}")`);
  const version = r.f32();
  const globalsCount = r.u8();
  const globals = r.bytes(globalsCount);
  if (globalsCount < 8) throw new Error(`Invalid PMX globals count ${globalsCount}`);

  const header: PmxHeader = {
    version,
    encoding: globals[0] === 0 ? 'utf-16le' : 'utf-8',
    additionalUvCount: globals[1],
    vertexIndexSize: globals[2],
    textureIndexSize: globals[3],
    materialIndexSize: globals[4],
    boneIndexSize: globals[5],
    morphIndexSize: globals[6],
    rigidBodyIndexSize: globals[7],
  };

  const decoder = new TextDecoder(header.encoding);
  const text = (): string => {
    const len = r.i32();
    return len > 0 ? decoder.decode(r.bytes(len)) : '';
  };
  const boneIdx = () => r.signedIndex(header.boneIndexSize);
  const texIdx = () => r.signedIndex(header.textureIndexSize);
  const matIdx = () => r.signedIndex(header.materialIndexSize);
  const morphIdx = () => r.signedIndex(header.morphIndexSize);
  const rigidIdx = () => r.signedIndex(header.rigidBodyIndexSize);
  const vertIdx = () => r.vertexIndex(header.vertexIndexSize);

  const name = text();
  const nameEn = text();
  const comment = text();
  const commentEn = text();

  // ---- Vertices ----
  const vertexCount = r.i32();
  const positions = new Float32Array(vertexCount * 3);
  const normals = new Float32Array(vertexCount * 3);
  const uvs = new Float32Array(vertexCount * 2);
  const additionalUvs = Array.from({ length: header.additionalUvCount }, () => new Float32Array(vertexCount * 4));
  const skinTypes = new Uint8Array(vertexCount);
  const skinIndices = new Uint32Array(vertexCount * 4);
  const skinWeights = new Float32Array(vertexCount * 4);
  const edgeScales = new Float32Array(vertexCount);
  const sdefVerts: number[] = [];
  const sdefC: number[] = [];
  const sdefR0: number[] = [];
  const sdefR1: number[] = [];

  const setBone = (v: number, slot: number, index: number, weight: number) => {
    // Invalid (-1) bone slots become bone 0 with zero weight.
    skinIndices[v * 4 + slot] = index >= 0 ? index : 0;
    skinWeights[v * 4 + slot] = index >= 0 ? weight : 0;
  };

  for (let v = 0; v < vertexCount; v++) {
    positions[v * 3] = r.f32(); positions[v * 3 + 1] = r.f32(); positions[v * 3 + 2] = r.f32();
    normals[v * 3] = r.f32(); normals[v * 3 + 1] = r.f32(); normals[v * 3 + 2] = r.f32();
    uvs[v * 2] = r.f32(); uvs[v * 2 + 1] = r.f32();
    for (let a = 0; a < header.additionalUvCount; a++) {
      const dst = additionalUvs[a];
      dst[v * 4] = r.f32(); dst[v * 4 + 1] = r.f32(); dst[v * 4 + 2] = r.f32(); dst[v * 4 + 3] = r.f32();
    }
    const type = r.u8() as SkinType;
    skinTypes[v] = type;
    switch (type) {
      case SkinType.BDEF1:
        setBone(v, 0, boneIdx(), 1);
        break;
      case SkinType.BDEF2: {
        const b0 = boneIdx(), b1 = boneIdx(), w = r.f32();
        setBone(v, 0, b0, w);
        setBone(v, 1, b1, 1 - w);
        break;
      }
      case SkinType.BDEF4:
      case SkinType.QDEF: {
        const b = [boneIdx(), boneIdx(), boneIdx(), boneIdx()];
        const w = [r.f32(), r.f32(), r.f32(), r.f32()];
        for (let i = 0; i < 4; i++) setBone(v, i, b[i], w[i]);
        break;
      }
      case SkinType.SDEF: {
        const b0 = boneIdx(), b1 = boneIdx(), w = r.f32();
        setBone(v, 0, b0, w);
        setBone(v, 1, b1, 1 - w);
        sdefVerts.push(v);
        sdefC.push(...r.vec3());
        sdefR0.push(...r.vec3());
        sdefR1.push(...r.vec3());
        break;
      }
      default:
        throw new Error(`Unknown skin type ${type} at vertex ${v}`);
    }
    // Normalize weights so malformed exports don't scale the mesh.
    const o = v * 4;
    const sum = skinWeights[o] + skinWeights[o + 1] + skinWeights[o + 2] + skinWeights[o + 3];
    if (sum > 0 && Math.abs(sum - 1) > 1e-4) {
      for (let i = 0; i < 4; i++) skinWeights[o + i] /= sum;
    } else if (sum <= 0) {
      skinWeights[o] = 1;
    }
    edgeScales[v] = r.f32();
  }

  // ---- Faces ----
  const indexCount = r.i32();
  const indices = new Uint32Array(indexCount);
  for (let i = 0; i < indexCount; i++) indices[i] = vertIdx();

  // ---- Textures ----
  const textureCount = r.i32();
  const textures: string[] = [];
  for (let i = 0; i < textureCount; i++) textures.push(text());

  // ---- Materials ----
  const materialCount = r.i32();
  const materials: PmxMaterial[] = [];
  for (let i = 0; i < materialCount; i++) {
    const mName = text();
    const mNameEn = text();
    const diffuse = r.vec4();
    const specular = r.vec3();
    const specularPower = r.f32();
    const ambient = r.vec3();
    const flags = r.u8();
    const edgeColor = r.vec4();
    const edgeSize = r.f32();
    const textureIndex = texIdx();
    const sphereTextureIndex = texIdx();
    const sphereMode = r.u8() as SphereMode;
    const sharedToon = r.u8() === 1;
    const toonIndex = sharedToon ? r.u8() : texIdx();
    const memo = text();
    const count = r.i32();
    materials.push({
      name: mName, nameEn: mNameEn, diffuse, specular, specularPower, ambient, flags, edgeColor, edgeSize,
      textureIndex, sphereTextureIndex, sphereMode, sharedToon, toonIndex, memo, indexCount: count,
    });
  }

  // ---- Bones ----
  const boneCount = r.i32();
  const bones: PmxBone[] = [];
  for (let i = 0; i < boneCount; i++) {
    const bName = text();
    const bNameEn = text();
    const position = r.vec3();
    const parentIndex = boneIdx();
    const layer = r.i32();
    const flags = r.u16();
    let tailBoneIndex = -1;
    let tailOffset: Vec3Tuple = [0, 0, 0];
    if (flags & BoneFlag.TailIsBone) tailBoneIndex = boneIdx();
    else tailOffset = r.vec3();
    let inherit: PmxBone['inherit'] = null;
    if (flags & (BoneFlag.InheritRotation | BoneFlag.InheritTranslation)) {
      inherit = { parentIndex: boneIdx(), weight: r.f32() };
    }
    const fixedAxis = flags & BoneFlag.FixedAxis ? r.vec3() : null;
    const localAxes = flags & BoneFlag.LocalAxes ? { x: r.vec3(), z: r.vec3() } : null;
    const externalParentKey = flags & BoneFlag.ExternalParent ? r.i32() : -1;
    let ik: PmxIk | null = null;
    if (flags & BoneFlag.IK) {
      const targetIndex = boneIdx();
      const loopCount = r.i32();
      const limitAngle = r.f32();
      const linkCount = r.i32();
      const links: PmxIkLink[] = [];
      for (let l = 0; l < linkCount; l++) {
        const boneIndex = boneIdx();
        const hasLimits = r.u8() === 1;
        links.push({ boneIndex, limits: hasLimits ? { min: r.vec3(), max: r.vec3() } : null });
      }
      ik = { targetIndex, loopCount, limitAngle, links };
    }
    bones.push({
      name: bName, nameEn: bNameEn, position, parentIndex, layer, flags, tailBoneIndex, tailOffset,
      inherit, fixedAxis, localAxes, externalParentKey, ik,
    });
  }

  // ---- Morphs ----
  const morphCount = r.i32();
  const morphs: PmxMorph[] = [];
  for (let i = 0; i < morphCount; i++) {
    const mName = text();
    const mNameEn = text();
    const panel = r.u8();
    const kind = r.u8() as MorphKind;
    const n = r.i32();
    const base = { name: mName, nameEn: mNameEn, panel };
    switch (kind) {
      case MorphKind.Group:
      case MorphKind.Flip: {
        const children = [];
        for (let k = 0; k < n; k++) children.push({ morphIndex: morphIdx(), weight: r.f32() });
        morphs.push({ ...base, kind, children });
        break;
      }
      case MorphKind.Vertex: {
        const vertexIndices = new Int32Array(n);
        const offsets = new Float32Array(n * 3);
        for (let k = 0; k < n; k++) {
          vertexIndices[k] = vertIdx();
          offsets[k * 3] = r.f32(); offsets[k * 3 + 1] = r.f32(); offsets[k * 3 + 2] = r.f32();
        }
        morphs.push({ ...base, kind, vertexIndices, offsets });
        break;
      }
      case MorphKind.Bone: {
        const list = [];
        for (let k = 0; k < n; k++) list.push({ boneIndex: boneIdx(), translation: r.vec3(), rotation: r.vec4() });
        morphs.push({ ...base, kind, bones: list });
        break;
      }
      case MorphKind.UV:
      case MorphKind.AdditionalUV1:
      case MorphKind.AdditionalUV2:
      case MorphKind.AdditionalUV3:
      case MorphKind.AdditionalUV4: {
        const vertexIndices = new Int32Array(n);
        const offsets = new Float32Array(n * 4);
        for (let k = 0; k < n; k++) {
          vertexIndices[k] = vertIdx();
          offsets[k * 4] = r.f32(); offsets[k * 4 + 1] = r.f32(); offsets[k * 4 + 2] = r.f32(); offsets[k * 4 + 3] = r.f32();
        }
        morphs.push({ ...base, kind, vertexIndices, offsets });
        break;
      }
      case MorphKind.Material: {
        const list: PmxMaterialMorphOffset[] = [];
        for (let k = 0; k < n; k++) {
          list.push({
            materialIndex: matIdx(),
            operation: r.u8(),
            diffuse: r.vec4(),
            specular: r.vec3(),
            specularPower: r.f32(),
            ambient: r.vec3(),
            edgeColor: r.vec4(),
            edgeSize: r.f32(),
            textureTint: r.vec4(),
            sphereTint: r.vec4(),
            toonTint: r.vec4(),
          });
        }
        morphs.push({ ...base, kind, materials: list });
        break;
      }
      case MorphKind.Impulse: {
        const impulses = [];
        for (let k = 0; k < n; k++) {
          impulses.push({ rigidBodyIndex: rigidIdx(), local: r.u8() === 1, velocity: r.vec3(), torque: r.vec3() });
        }
        morphs.push({ ...base, kind, impulses });
        break;
      }
      default:
        throw new Error(`Unknown morph type ${kind} in morph "${mName}"`);
    }
  }

  // ---- Display frames ----
  const frameCount = r.i32();
  const displayFrames: PmxDisplayFrame[] = [];
  for (let i = 0; i < frameCount; i++) {
    const fName = text();
    const fNameEn = text();
    const special = r.u8() === 1;
    const n = r.i32();
    const items: PmxDisplayFrame['items'] = [];
    for (let k = 0; k < n; k++) {
      const type = r.u8();
      items.push(type === 0 ? { type: 'bone', index: boneIdx() } : { type: 'morph', index: morphIdx() });
    }
    displayFrames.push({ name: fName, nameEn: fNameEn, special, items });
  }

  // ---- Rigid bodies ----
  const rigidBodies: PmxRigidBody[] = [];
  const rigidCount = r.remaining >= 4 ? r.i32() : 0;
  for (let i = 0; i < rigidCount; i++) {
    rigidBodies.push({
      name: text(),
      nameEn: text(),
      boneIndex: boneIdx(),
      group: r.u8(),
      collisionMask: r.u16(),
      shape: r.u8(),
      size: r.vec3(),
      position: r.vec3(),
      rotation: r.vec3(),
      mass: r.f32(),
      linearDamping: r.f32(),
      angularDamping: r.f32(),
      restitution: r.f32(),
      friction: r.f32(),
      mode: r.u8(),
    });
  }

  // ---- Joints ----
  const joints: PmxJoint[] = [];
  const jointCount = r.remaining >= 4 ? r.i32() : 0;
  for (let i = 0; i < jointCount; i++) {
    joints.push({
      name: text(),
      nameEn: text(),
      type: r.u8(),
      rigidBodyA: rigidIdx(),
      rigidBodyB: rigidIdx(),
      position: r.vec3(),
      rotation: r.vec3(),
      positionMin: r.vec3(),
      positionMax: r.vec3(),
      rotationMin: r.vec3(),
      rotationMax: r.vec3(),
      springPosition: r.vec3(),
      springRotation: r.vec3(),
    });
  }

  return {
    format: 'PMX', header, name, nameEn, comment, commentEn,
    vertexCount, positions, normals, uvs, additionalUvs, skinTypes, skinIndices, skinWeights,
    sdef: {
      vertexIndices: Uint32Array.from(sdefVerts),
      c: Float32Array.from(sdefC),
      r0: Float32Array.from(sdefR0),
      r1: Float32Array.from(sdefR1),
    },
    edgeScales, indices, textures, materials, bones, morphs, displayFrames, rigidBodies, joints,
  };
}
