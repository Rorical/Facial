import { mat4, quat, vec3, type Mat4, type Quat, type Vec3 } from '../math';
import { BoneFlag, type PmxBone } from '../loaders/pmx';
import { IkSolver } from './ik-solver';

// Bone evaluation follows MMD's rules (cross-checked against saba / babylon-mmd):
//   * bones are processed in (after-physics, deform layer, index) order;
//   * local = T(offset + anim + morph + append) · R(ik · anim · morph · append);
//   * IK is solved at the IK bone's turn in that order;
//   * "after physics" bones run in a second phase; in between, `physicsHook` may overwrite
//     world matrices of physics-driven bones (see physics/mmd-physics.ts).

const IDENTITY_Q = quat.create();

export class RuntimeBone {
  readonly name: string;
  readonly index: number;
  parent: RuntimeBone | null = null;
  /** The model's placement in the scene (shared by all bones; used by those without a parent). */
  root: Mat4 = IDENTITY;
  readonly children: RuntimeBone[] = [];
  readonly layer: number;
  readonly afterPhysics: boolean;

  /** Rest position in model space. */
  readonly restPosition: Vec3;
  /** Rest offset from the parent. */
  readonly localOffset: Vec3;

  // Per-frame inputs.
  readonly animPosition = vec3.create();
  readonly animRotation = quat.create();
  readonly morphPosition = vec3.create();
  readonly morphRotation = quat.create();
  readonly ikRotation = quat.create();

  // Inherit ("付与") bones.
  appendParent: RuntimeBone | null = null;
  appendWeight = 0;
  readonly appendRotate: boolean;
  readonly appendTranslate: boolean;
  readonly appendLocal: boolean;
  readonly appendRotation = quat.create();
  readonly appendTranslation = vec3.create();

  ikSolver: IkSolver | null = null;

  // Physics output (see physics/runners.ts): transform relative to the parent bone.
  readonly physicsLocal = mat4.create();
  physicsActive = false;
  /** "Physics + bone position" bodies keep the animated world translation. */
  physicsKeepTranslation = false;

  readonly local = mat4.create();
  readonly world = mat4.create();

  constructor(index: number, def: PmxBone, parentDef: PmxBone | null) {
    this.index = index;
    this.name = def.name;
    this.layer = def.layer;
    this.afterPhysics = (def.flags & BoneFlag.AfterPhysics) !== 0;
    this.restPosition = vec3.create(...def.position);
    this.localOffset = parentDef
      ? vec3.create(def.position[0] - parentDef.position[0], def.position[1] - parentDef.position[1], def.position[2] - parentDef.position[2])
      : vec3.create(...def.position);
    this.appendRotate = (def.flags & BoneFlag.InheritRotation) !== 0;
    this.appendTranslate = (def.flags & BoneFlag.InheritTranslation) !== 0;
    this.appendLocal = (def.flags & BoneFlag.InheritLocal) !== 0;
  }

  /** anim · morph rotation (the "user" rotation, without IK or inherit). */
  animatedRotation(out: Quat): Quat {
    return quat.multiply(out, this.animRotation, this.morphRotation);
  }

  animatedTranslation(out: Vec3): Vec3 {
    return vec3.add(out, this.animPosition, this.morphPosition);
  }

  private static readonly tq = quat.create();
  private static readonly tq2 = quat.create();
  private static readonly tv = vec3.create();

  updateAppend(): void {
    const p = this.appendParent;
    if (!p) return;
    if (this.appendRotate) {
      const src = RuntimeBone.tq;
      if (!this.appendLocal && p.appendParent && p.appendRotate) quat.copy(src, p.appendRotation);
      else p.animatedRotation(src);
      quat.multiply(src, p.ikRotation, src);
      quat.slerp(this.appendRotation, IDENTITY_Q, src, this.appendWeight);
    }
    if (this.appendTranslate) {
      const src = RuntimeBone.tv;
      if (!this.appendLocal && p.appendParent && p.appendTranslate) vec3.copy(src, p.appendTranslation);
      else p.animatedTranslation(src);
      vec3.scale(this.appendTranslation, src, this.appendWeight);
    }
  }

  updateLocal(): void {
    const t = RuntimeBone.tv;
    vec3.add(t, this.localOffset, this.animPosition);
    vec3.add(t, t, this.morphPosition);
    if (this.appendTranslate) vec3.add(t, t, this.appendTranslation);

    const r = RuntimeBone.tq2;
    this.animatedRotation(r);
    quat.multiply(r, this.ikRotation, r);
    if (this.appendRotate) quat.multiply(r, r, this.appendRotation);
    mat4.fromRotationTranslation(this.local, r, t);
  }

  updateWorld(): void {
    if (this.parent) mat4.multiply(this.world, this.parent.world, this.local);
    else mat4.multiply(this.world, this.root, this.local);
  }

  /** Update this bone's world matrix and every descendant's (after IK changes). */
  updateWorldRecursive(): void {
    this.updateWorld();
    for (const c of this.children) c.updateWorldRecursive();
  }

  resetInputs(): void {
    vec3.set(this.animPosition, 0, 0, 0);
    quat.identity(this.animRotation);
    vec3.set(this.morphPosition, 0, 0, 0);
    quat.identity(this.morphRotation);
  }
}

const IDENTITY = mat4.create();

export class Skeleton {
  /** Where the whole model stands in the scene. Edit in place. */
  readonly root = mat4.create();
  readonly bones: RuntimeBone[];
  readonly byName = new Map<string, RuntimeBone>();
  readonly ikSolvers: IkSolver[] = [];
  /**
   * Called between the before- and after-physics phases. Marks physics-driven bones
   * (`physicsActive` + `physicsLocal`) and returns true when there is something to apply.
   */
  physicsHook: (() => boolean) | null = null;
  private readonly beforePhysics: RuntimeBone[];
  private readonly afterPhysics: RuntimeBone[];
  /** Parents before children; used for the pre-pass. */
  private readonly hierarchyOrder: RuntimeBone[];

  constructor(defs: PmxBone[]) {
    this.bones = defs.map((def, i) => {
      const parentDef = def.parentIndex >= 0 && def.parentIndex < defs.length ? defs[def.parentIndex] : null;
      const bone = new RuntimeBone(i, def, parentDef);
      bone.root = this.root;
      return bone;
    });
    defs.forEach((def, i) => {
      const bone = this.bones[i];
      if (!this.byName.has(bone.name)) this.byName.set(bone.name, bone);
      if (def.parentIndex >= 0 && def.parentIndex < defs.length && def.parentIndex !== i) {
        bone.parent = this.bones[def.parentIndex];
        bone.parent.children.push(bone);
      }
      if (def.inherit && def.inherit.parentIndex >= 0 && def.inherit.parentIndex < defs.length && def.inherit.parentIndex !== i) {
        bone.appendParent = this.bones[def.inherit.parentIndex];
        bone.appendWeight = def.inherit.weight;
      }
      if (def.ik && def.ik.targetIndex >= 0 && def.ik.targetIndex < defs.length) {
        const links = def.ik.links
          .filter((l) => l.boneIndex >= 0 && l.boneIndex < defs.length)
          .map((l) => ({ bone: this.bones[l.boneIndex], limits: l.limits }));
        bone.ikSolver = new IkSolver(bone, this.bones[def.ik.targetIndex], links, def.ik.loopCount, def.ik.limitAngle);
        this.ikSolvers.push(bone.ikSolver);
      }
    });

    const sorted = [...this.bones].sort((a, b) => a.layer - b.layer || a.index - b.index);
    this.beforePhysics = sorted.filter((b) => !b.afterPhysics);
    this.afterPhysics = sorted.filter((b) => b.afterPhysics);

    this.hierarchyOrder = [];
    const visit = (b: RuntimeBone) => {
      this.hierarchyOrder.push(b);
      b.children.forEach(visit);
    };
    this.bones.filter((b) => !b.parent).forEach(visit);
    // Parent cycles (malformed models) would leave bones unvisited; append them anyway.
    if (this.hierarchyOrder.length !== this.bones.length) {
      const seen = new Set(this.hierarchyOrder);
      for (const b of this.bones) if (!seen.has(b)) this.hierarchyOrder.push(b);
    }
  }

  resetInputs(): void {
    for (const b of this.bones) b.resetInputs();
  }

  update(): void {
    for (const b of this.bones) quat.identity(b.ikRotation);
    // Pre-pass so IK sees current (not last-frame) matrices for bones later in deform order.
    for (const b of this.hierarchyOrder) {
      b.updateLocal();
      b.updateWorld();
    }
    this.updatePhase(this.beforePhysics);
    if (this.physicsHook) {
      for (const b of this.bones) b.physicsActive = false;
      if (this.physicsHook()) this.applyPhysics();
    }
    this.updatePhase(this.afterPhysics);
  }

  /**
   * Parents first: physics bones take parent.world · physicsLocal; everything else is
   * recomputed so children of physics bones (e.g. hair tips without bodies) follow them.
   */
  private applyPhysics(): void {
    for (const b of this.hierarchyOrder) {
      if (!b.physicsActive) {
        b.updateWorld();
        continue;
      }
      const w = b.world;
      const x = w[12], y = w[13], z = w[14];
      if (b.parent) mat4.multiply(w, b.parent.world, b.physicsLocal);
      else mat4.copy(w, b.physicsLocal); // no parent: physics reports the world transform
      if (b.physicsKeepTranslation) {
        w[12] = x; w[13] = y; w[14] = z;
      }
    }
  }

  private updatePhase(bones: RuntimeBone[]): void {
    for (const b of bones) {
      if (b.appendParent) b.updateAppend();
      b.updateLocal();
      b.updateWorld();
      if (b.ikSolver?.enabled) b.ikSolver.solve();
    }
  }

  private static readonly tq = quat.create();

  /**
   * Write GPU skinning data: per bone a mat4 skin matrix (world · inverse bind) followed by
   * the world rotation quaternion (needed for SDEF). 20 floats per bone.
   */
  writeSkinning(out: Float32Array): void {
    const q = Skeleton.tq;
    for (let i = 0; i < this.bones.length; i++) {
      const b = this.bones[i];
      const w = b.world;
      const o = i * 20;
      for (let k = 0; k < 12; k++) out[o + k] = w[k];
      // Inverse bind is a pure translation by -rest: col3 = world · (-rest, 1).
      const rx = b.restPosition[0], ry = b.restPosition[1], rz = b.restPosition[2];
      out[o + 12] = w[12] - (w[0] * rx + w[4] * ry + w[8] * rz);
      out[o + 13] = w[13] - (w[1] * rx + w[5] * ry + w[9] * rz);
      out[o + 14] = w[14] - (w[2] * rx + w[6] * ry + w[10] * rz);
      out[o + 15] = 1;
      quat.fromMat4(q, w);
      out[o + 16] = q[0]; out[o + 17] = q[1]; out[o + 18] = q[2]; out[o + 19] = q[3];
    }
  }
}
