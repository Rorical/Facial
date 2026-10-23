import { mat4, quat, vec3, type Mat4 } from '../math';
import type { PmxJoint, PmxModel, PmxRigidBody } from '../loaders/pmx';

// Static physics setup derived from a PMX, shared by the solver (main thread or
// worker) and the wireframe debug view. Pure data + math: no solver state, no DOM.

export const FOLLOW_BONE = 0;
export const PHYSICS = 1;
export const PHYSICS_WITH_BONE = 2;

/** The subset of a PmxModel physics needs (structured-clone friendly, sent to the worker). */
export type PhysicsModelData = Pick<PmxModel, 'rigidBodies' | 'joints'> & {
  bones: { parentIndex: number; position: [number, number, number] }[];
};

export function physicsModelData(pmx: PmxModel): PhysicsModelData {
  return {
    rigidBodies: pmx.rigidBodies,
    joints: pmx.joints,
    bones: pmx.bones.map((b) => ({ parentIndex: b.parentIndex, position: b.position })),
  };
}

export interface RigBody {
  /** Index into PmxModel.rigidBodies, and the record itself (mass, damping, groups…). */
  index: number;
  source: PmxRigidBody;
  boneIndex: number; // -1 = none
  /** Effective mode after MMD's joint-based adjustment. */
  mode: number;
  /** 0 sphere, 1 box, 2 capsule; -1 = unsupported (skipped). */
  shape: number;
  size: [number, number, number];
  /** Rest transform in model space. */
  rest: Mat4;
  /** body = boneWorld · offset */
  offset: Mat4;
  offsetInv: Mat4;
}

export interface RigJoint {
  source: PmxJoint;
  /** Indices into PhysicsRig.bodies. */
  bodyA: number;
  bodyB: number;
  /** Joint frame in body A's / body B's local space. */
  frameA: Mat4;
  frameB: Mat4;
}

export interface PhysicsRig {
  bodies: RigBody[];
  joints: RigJoint[];
  /** Bones written by physics, parents before children. */
  drivenBones: Int32Array;
  /** Per driven bone: the body driving it (index into bodies). */
  drivenBody: Int32Array;
  /** Per driven bone: 1 = keep the animated translation (mode 2). */
  drivenKeepTranslation: Uint8Array;
  boneParents: Int32Array;
}

const yprMatrix = (rot: ArrayLike<number>, pos: ArrayLike<number>): Mat4 =>
  mat4.fromRotationTranslation(mat4.create(), quat.fromYawPitchRoll(quat.create(), rot[0], rot[1], rot[2]), vec3.copy(vec3.create(), pos));

export function buildPhysicsRig(model: PhysicsModelData): PhysicsRig {
  const boneCount = model.bones.length;
  const boneParents = Int32Array.from(model.bones, (b, i) => (b.parentIndex >= 0 && b.parentIndex < boneCount && b.parentIndex !== i ? b.parentIndex : -1));

  const bodies: RigBody[] = [];
  const bodyOfRigid = new Int32Array(model.rigidBodies.length).fill(-1);
  model.rigidBodies.forEach((rb, index) => {
    if (rb.shape < 0 || rb.shape > 2) {
      console.warn(`Unknown rigid body shape ${rb.shape} (${rb.name})`);
      return;
    }
    const boneIndex = rb.boneIndex >= 0 && rb.boneIndex < boneCount ? rb.boneIndex : -1;
    const rest = yprMatrix(rb.rotation, rb.position);
    // Bones have no rest rotation, so the offset is the rest transform shifted by -bonePos.
    const offset = mat4.copy(mat4.create(), rest);
    if (boneIndex >= 0) {
      const p = model.bones[boneIndex].position;
      offset[12] -= p[0]; offset[13] -= p[1]; offset[14] -= p[2];
    }
    bodyOfRigid[index] = bodies.length;
    bodies.push({
      index, source: rb, boneIndex, mode: rb.mode, shape: rb.shape, size: [...rb.size],
      rest, offset, offsetInv: mat4.invert(mat4.create(), offset) ?? mat4.create(),
    });
  });

  const joints: RigJoint[] = [];
  const inv = mat4.create();
  for (const j of model.joints) {
    const ia = j.rigidBodyA >= 0 && j.rigidBodyA < bodyOfRigid.length ? bodyOfRigid[j.rigidBodyA] : -1;
    const ib = j.rigidBodyB >= 0 && j.rigidBodyB < bodyOfRigid.length ? bodyOfRigid[j.rigidBodyB] : -1;
    if (ia < 0 || ib < 0) continue;
    const a = bodies[ia];
    const b = bodies[ib];
    const jointRest = yprMatrix(j.rotation, j.position);
    joints.push({
      source: j,
      bodyA: ia,
      bodyB: ib,
      frameA: mat4.multiply(mat4.create(), mat4.invert(inv, a.rest)!, jointRest),
      frameB: mat4.multiply(mat4.create(), mat4.invert(inv, b.rest)!, jointRest),
    });
    // A "physics + bone" body hanging off a physics body behaves as plain physics in MMD.
    const parentOf = (x: RigBody) => (x.boneIndex >= 0 ? boneParents[x.boneIndex] : -2);
    if (a.mode !== FOLLOW_BONE && b.mode === PHYSICS_WITH_BONE && b.boneIndex >= 0 && parentOf(b) === a.boneIndex) b.mode = PHYSICS;
    else if (b.mode !== FOLLOW_BONE && a.mode === PHYSICS_WITH_BONE && a.boneIndex >= 0 && parentOf(a) === b.boneIndex) a.mode = PHYSICS;
  }

  // Driven bones: the first dynamic body on each bone wins; order parents before children.
  const bodyOfBone = new Map<number, number>();
  bodies.forEach((b, i) => {
    if (b.boneIndex >= 0 && b.mode !== FOLLOW_BONE && !bodyOfBone.has(b.boneIndex)) bodyOfBone.set(b.boneIndex, i);
  });
  const depth = (bone: number) => {
    let d = 0;
    for (let p = boneParents[bone]; p >= 0 && d <= boneCount; p = boneParents[p]) d++;
    return d;
  };
  const driven = [...bodyOfBone.keys()].sort((x, y) => depth(x) - depth(y) || x - y);

  return {
    bodies,
    joints,
    drivenBones: Int32Array.from(driven),
    drivenBody: Int32Array.from(driven, (bone) => bodyOfBone.get(bone)!),
    drivenKeepTranslation: Uint8Array.from(driven, (bone) => (bodies[bodyOfBone.get(bone)!].mode === PHYSICS_WITH_BONE ? 1 : 0)),
    boneParents,
  };
}
