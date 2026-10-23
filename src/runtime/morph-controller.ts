import { quat, vec3 } from '../math';
import { MaterialFlag, MorphKind, SphereMode, type PmxMaterialMorphOffset, type PmxModel } from '../loaders/pmx';
import type { Skeleton } from './skeleton';

// Resolves morph weights each frame:
//   * group morphs are expanded into their children (one level, as in MMD);
//   * vertex / UV morph weights go to the GPU (applied in the skinning compute pass);
//   * bone morphs feed the skeleton; material morphs produce per-material uniforms.

/** Floats per material uniform block. Must match `Material` in mmd.wgsl. */
export const MATERIAL_UNIFORM_FLOATS = 64;
/** Bytes between material blocks (minUniformBufferOffsetAlignment). */
export const MATERIAL_UNIFORM_STRIDE = 256;

// Offsets (in floats) into a material block.
const DIFFUSE = 0, SPECULAR = 4, AMBIENT = 8, EDGE_COLOR = 12, EDGE_PARAMS = 16;
const TEX_MUL = 20, TEX_ADD = 24, SPHERE_MUL = 28, SPHERE_ADD = 32, TOON_MUL = 36, TOON_ADD = 40, MODES = 44, INFO = 48, HEAD_POS = 52, HEAD_FORWARD = 56, HEAD_UP = 60;

// Per-material accumulators: 4 diffuse, 3 specular, 1 power, 3 ambient, 4 edge color,
// 1 edge size, 4 texture tint, 4 sphere tint, 4 toon tint = 28 values.
const FACTOR_COUNT = 28;
const F_DIFFUSE = 0, F_SPECULAR = 4, F_POWER = 7, F_AMBIENT = 8, F_EDGE = 11, F_EDGE_SIZE = 15;
const F_TEX = 16, F_SPHERE = 20, F_TOON = 24;

/** What part of a character a material is, guessed from its name (for custom shaders). */
export const MaterialPart = { Other: 0, Face: 1, FaceDetail: 2, Hair: 3, Skin: 4 } as const;

/**
 * Face: the face skin. FaceDetail: eyes, brows, lashes, mouth, teeth, expression decals.
 * Names are Japanese / Chinese / English as found in MMD models; unknown names are Other.
 */
export function classifyMaterial(name: string, nameEn = ''): number {
  const n = `${name} ${nameEn}`.toLowerCase();
  if (/袖口|领口|領口|襟|眼鏡|眼镜|めがね|メガネ|glass|surface|蛋|宠物|寵物|egg|pet/.test(n)) return MaterialPart.Other;
  if (/目|眼|瞳|眉|睫|まつ|まゆ|口|舌|歯|齒|齿|牙|eye|brow|lash|mouth|tongue|teeth|tooth|iris|pupil|表情|biaoq/.test(n)) return MaterialPart.FaceDetail;
  if (/顏|顔|颜|脸|臉|かお|face/.test(n)) return MaterialPart.Face;
  if (/髪|髮|头发|頭髮|发|hair|ヘア|刘海|前髪/.test(n)) return MaterialPart.Hair;
  if (/肌|肤|膚|skin|素体|body/.test(n)) return MaterialPart.Skin;
  return MaterialPart.Other;
}

/**
 * Face details that anime shading may draw over the hair in front of them: eyes, brows and
 * lashes. Not the mouth, and not expression overlays that sit hidden behind the face.
 */
export function showsThroughHair(name: string, nameEn = ''): boolean {
  const n = `${name} ${nameEn}`.toLowerCase();
  if (classifyMaterial(name, nameEn) !== MaterialPart.FaceDetail) return false;
  if (/影|表情|biaoq|口|舌|歯|齒|齿|牙|mouth|tongue|teeth|tooth/.test(n)) return false;
  return /目|眼|瞳|眉|睫|まつ|まゆ|eye|brow|lash|iris|pupil/.test(n);
}

export interface MaterialTextureFlags {
  hasTexture: boolean;
  hasSphere: boolean;
  hasToon: boolean;
}

export class MorphController {
  /** Input weights by morph index (from animation / UI). */
  readonly weights: Float32Array;
  /** Effective weights after group expansion, uploaded to the GPU. */
  readonly effective: Float32Array;
  /** Packed material uniform blocks, MATERIAL_UNIFORM_STRIDE bytes apart. */
  readonly materialUniforms: Float32Array;

  private readonly base: Float32Array;
  private readonly mul: Float32Array;
  private readonly add: Float32Array;
  private readonly strideFloats = MATERIAL_UNIFORM_STRIDE / 4;
  private readonly tq = quat.create();
  private readonly tq2 = quat.create();

  constructor(private readonly pmx: PmxModel, textureFlags: MaterialTextureFlags[]) {
    const n = pmx.morphs.length;
    this.weights = new Float32Array(n);
    this.effective = new Float32Array(Math.max(n, 1));
    const matCount = pmx.materials.length;
    this.materialUniforms = new Float32Array(Math.max(matCount, 1) * this.strideFloats);
    this.mul = new Float32Array(matCount * FACTOR_COUNT);
    this.add = new Float32Array(matCount * FACTOR_COUNT);
    this.base = new Float32Array(matCount * FACTOR_COUNT);
    pmx.materials.forEach((m, i) => {
      this.base.set([...m.diffuse, ...m.specular, m.specularPower, ...m.ambient, ...m.edgeColor, m.edgeSize], i * FACTOR_COUNT);
    });

    // Static parts of the uniform block: texture presence, sphere mode, self-shadow receiving.
    const u32 = new Uint32Array(this.materialUniforms.buffer);
    pmx.materials.forEach((m, i) => {
      const o = i * this.strideFloats + MODES;
      const flags = textureFlags[i];
      // A sub-texture is looked up with additional UV 1; without that channel there is nothing to map it by.
      const usable = m.sphereMode !== SphereMode.SubTexture || pmx.additionalUvs.length > 0;
      const sphereMode = flags.hasSphere && usable ? m.sphereMode : SphereMode.None;
      u32[o] = flags.hasTexture ? 1 : 0;
      u32[o + 1] = sphereMode;
      u32[o + 2] = pmx.accessory ? 2 : flags.hasToon ? 1 : 0;
      u32[o + 3] = m.flags & MaterialFlag.ReceiveSelfShadow ? 1 : 0;
      this.materialUniforms[i * this.strideFloats + INFO] = i;
      this.materialUniforms[i * this.strideFloats + INFO + 1] = classifyMaterial(m.name, m.nameEn);
    });
  }

  /** Expand groups, apply bone morphs to the skeleton and rebuild material uniforms. */
  update(skeleton: Skeleton): void {
    const { morphs } = this.pmx;
    const eff = this.effective;
    eff.set(this.weights);
    for (let i = 0; i < morphs.length; i++) {
      const m = morphs[i];
      const w = this.weights[i];
      if (w === 0 || m.kind !== MorphKind.Group) continue;
      for (const c of m.children) {
        if (c.morphIndex >= 0 && c.morphIndex < morphs.length && c.morphIndex !== i) eff[c.morphIndex] += w * c.weight;
      }
    }

    this.mul.fill(1);
    this.add.fill(0);
    for (let i = 0; i < morphs.length; i++) {
      const m = morphs[i];
      const w = eff[i];
      if (w === 0) continue;
      if (m.kind === MorphKind.Bone) {
        for (const b of m.bones) {
          const bone = skeleton.bones[b.boneIndex];
          if (!bone) continue;
          vec3.scaleAndAdd(bone.morphPosition, bone.morphPosition, b.translation, w);
          quat.slerp(this.tq, quat.identity(this.tq2), b.rotation, w);
          quat.multiply(bone.morphRotation, bone.morphRotation, this.tq);
        }
      } else if (m.kind === MorphKind.Material) {
        for (const off of m.materials) {
          if (off.materialIndex < 0) {
            for (let mi = 0; mi < this.pmx.materials.length; mi++) this.accumulateMaterial(mi, off, w);
          } else if (off.materialIndex < this.pmx.materials.length) {
            this.accumulateMaterial(off.materialIndex, off, w);
          }
        }
      }
    }
    this.writeMaterials();
  }

  /** Which auxiliary maps each material has (bit 0 ramp, 1 light map, 2 face map), for shaders. */
  setAuxFlags(flags: number[] | null): void {
    for (let i = 0; i < this.pmx.materials.length; i++) this.materialUniforms[i * this.strideFloats + INFO + 2] = flags?.[i] ?? 0;
  }

  /**
   * Where the head is and which way it faces, for custom shaders (face shading that follows the
   * head rather than the bumpy face normals). Call after the skeleton is posed.
   */
  writeHead(skeleton: Skeleton): void {
    const head = skeleton.byName.get('頭');
    if (!head) return;
    const w = head.world;
    const u = this.materialUniforms;
    for (let i = 0; i < this.pmx.materials.length; i++) {
      const o = i * this.strideFloats;
      u[o + HEAD_POS] = w[12]; u[o + HEAD_POS + 1] = w[13]; u[o + HEAD_POS + 2] = w[14]; u[o + HEAD_POS + 3] = 1;
      // MMD models face -Z.
      u[o + HEAD_FORWARD] = -w[8]; u[o + HEAD_FORWARD + 1] = -w[9]; u[o + HEAD_FORWARD + 2] = -w[10];
      u[o + HEAD_UP] = w[4]; u[o + HEAD_UP + 1] = w[5]; u[o + HEAD_UP + 2] = w[6];
    }
  }

  private accumulateMaterial(mi: number, off: PmxMaterialMorphOffset, w: number): void {
    const values = [
      ...off.diffuse, ...off.specular, off.specularPower, ...off.ambient, ...off.edgeColor, off.edgeSize,
      ...off.textureTint, ...off.sphereTint, ...off.toonTint,
    ];
    const o = mi * FACTOR_COUNT;
    if (off.operation === 0) {
      // Multiply: acc = lerp(acc, acc * v, w)
      for (let k = 0; k < FACTOR_COUNT; k++) this.mul[o + k] += this.mul[o + k] * (values[k] - 1) * w;
    } else {
      for (let k = 0; k < FACTOR_COUNT; k++) this.add[o + k] += values[k] * w;
    }
  }

  private writeMaterials(): void {
    const u = this.materialUniforms;
    const { base, mul, add } = this;
    for (let i = 0; i < this.pmx.materials.length; i++) {
      const o = i * this.strideFloats;
      const f = i * FACTOR_COUNT;
      const v = (k: number) => base[f + k] * mul[f + k] + add[f + k];
      u[o + DIFFUSE] = v(F_DIFFUSE); u[o + DIFFUSE + 1] = v(F_DIFFUSE + 1);
      u[o + DIFFUSE + 2] = v(F_DIFFUSE + 2); u[o + DIFFUSE + 3] = v(F_DIFFUSE + 3);
      u[o + SPECULAR] = v(F_SPECULAR); u[o + SPECULAR + 1] = v(F_SPECULAR + 1);
      u[o + SPECULAR + 2] = v(F_SPECULAR + 2); u[o + SPECULAR + 3] = v(F_POWER);
      u[o + AMBIENT] = v(F_AMBIENT); u[o + AMBIENT + 1] = v(F_AMBIENT + 1);
      u[o + AMBIENT + 2] = v(F_AMBIENT + 2); u[o + AMBIENT + 3] = 0;
      u[o + EDGE_COLOR] = v(F_EDGE); u[o + EDGE_COLOR + 1] = v(F_EDGE + 1);
      u[o + EDGE_COLOR + 2] = v(F_EDGE + 2); u[o + EDGE_COLOR + 3] = v(F_EDGE + 3);
      u[o + EDGE_PARAMS] = v(F_EDGE_SIZE); u[o + EDGE_PARAMS + 1] = 0;
      u[o + EDGE_PARAMS + 2] = 0; u[o + EDGE_PARAMS + 3] = 0;
      // Texture tints have no base value: mul starts at 1, add at 0.
      for (let k = 0; k < 4; k++) {
        u[o + TEX_MUL + k] = mul[f + F_TEX + k];
        u[o + TEX_ADD + k] = add[f + F_TEX + k];
        u[o + SPHERE_MUL + k] = mul[f + F_SPHERE + k];
        u[o + SPHERE_ADD + k] = add[f + F_SPHERE + k];
        u[o + TOON_MUL + k] = mul[f + F_TOON + k];
        u[o + TOON_ADD + k] = add[f + F_TOON + k];
      }
    }
  }
}
