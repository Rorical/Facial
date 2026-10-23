import { mat4, quat, vec3 } from '../math';
import type { MmdMotion } from '../animation/motion';
import type { PmxModel } from '../loaders/pmx';
import { MmdModel } from '../runtime/mmd-model';
import { BAKE_FLOATS, PhysicsBake } from './bake';
import { PhysicsWorld, type PhysicsOptions } from './physics-world';
import { buildPhysicsRig, physicsModelData } from './rig';
import { InlinePhysics } from './runner';
import { buildClothMesh } from '../cloth/cloth-mesh';
import { ClothSim, clothTargets, type ClothSettings } from '../cloth/cloth-sim';
import type { ClothBake } from './bake';

// Runs the physics simulation for a bake (used by the bake worker).

export interface BakeOptions {
  from: number;
  to: number;
  /** Settle steps at the first frame before recording (hair / skirts fall into place). */
  warmupSteps?: number;
  physics?: Partial<PhysicsOptions>;
  /** The model's motion translation scale (see MmdModel.motionScale). */
  motionScale?: number;
  /** Simulate these materials as cloth along with the bake. */
  cloth?: { materials: number[]; settings: ClothSettings } | null;
  onProgress?: (done: number, total: number) => void;
}

export const BAKE_FPS = 30;
/** Before recording, the cloth follows the body from the rest pose into the first frame's pose… */
const CLOTH_POSE_IN_FRAMES = 30;
/** …and then settles there, this many frames in all (2 s). */
const CLOTH_WARMUP_FRAMES = 60;

/**
 * Simulate `motion` on `pmx` over [from, to] and record the physics bones. Synchronous: run
 * it in a worker (see bake-worker.ts) for long ranges.
 */
export function bakePhysics(pmx: PmxModel, motion: MmdMotion | null, opts: BakeOptions): PhysicsBake {
  const { from, to } = opts;
  if (!(to >= from)) throw new Error('烘焙区间无效');
  const model = new MmdModel(pmx, pmx.materials.map(() => ({ hasTexture: false, hasSphere: false, hasToon: false })));
  model.setMotion(motion);
  model.motionScale = opts.motionScale ?? 1;
  const rig = buildPhysicsRig(physicsModelData(pmx));
  const runner = new InlinePhysics(new PhysicsWorld(rig, opts.physics), rig, pmx.bones.length);
  runner.requestReset(opts.warmupSteps ?? 60);
  model.attachPhysics(runner);

  const bones = Int32Array.from(rig.drivenBones);
  const n = bones.length;
  const total = to - from + 1;
  const data = new Float32Array(total * n * BAKE_FLOATS);
  const inv = mat4.create();
  const local = mat4.create();
  const q = quat.create();
  const skeleton = model.skeleton;

  // Cloth: simulated on top of the posed skeleton, one frame at a time.
  const mesh = opts.cloth?.materials.length ? buildClothMesh(pmx, rig, opts.cloth.materials) : null;
  const sim = mesh ? new ClothSim(mesh, opts.cloth!.settings) : null;
  const clothData = mesh ? new Float32Array(total * mesh.particleCount * 3) : null;
  const targets = mesh ? new Float32Array(mesh.particleCount * 3) : null;
  const colliderCount = mesh?.collision?.colliders.length ?? 0;
  const colliders = new Float32Array(colliderCount * 16);
  const body = mat4.create();
  const readPose = (): void => {
    clothTargets(mesh!, (b) => skeleton.bones[b].world, (b) => pmx.bones[b].position, targets!);
    mesh!.collision?.colliders.forEach((index, j) => {
      const rb = rig.bodies[index];
      colliders.set(mat4.multiply(body, skeleton.bones[rb.boneIndex].world, rb.offset), j * 16);
    });
  };
  /**
   * Bring the cloth into the first frame's pose: the body is moved there from the rest pose
   * over a second, with the cloth simulated along the way, then left to settle. Dropping the
   * cloth straight into a pose with, say, the legs apart would start it inside the legs, and
   * it could be pushed out on the wrong side.
   */
  const warmUpCloth = (): void => {
    const endTargets = targets!.slice();
    const endColliders = colliders.slice();
    const qa = quat.create(), qb = quat.create(), qt = quat.create();
    const from = mat4.create(), to = mat4.create();
    const at = vec3.create();
    const rests = mesh!.collision?.colliders.map((index) => rig.bodies[index].rest) ?? [];
    for (let k = 1; k <= CLOTH_WARMUP_FRAMES; k++) {
      const t = Math.min(1, k / CLOTH_POSE_IN_FRAMES);
      for (let i = 0; i < targets!.length; i++) targets![i] = mesh!.rest[i] + (endTargets[i] - mesh!.rest[i]) * t;
      rests.forEach((rest, j) => {
        mat4.copy(from, rest);
        mat4.copy(to, endColliders.subarray(j * 16, j * 16 + 16));
        quat.slerp(qt, quat.normalize(qa, quat.fromMat4(qa, from)), quat.normalize(qb, quat.fromMat4(qb, to)), t);
        mat4.fromRotationTranslation(body, qt, vec3.set(at, from[12] + (to[12] - from[12]) * t, from[13] + (to[13] - from[13]) * t, from[14] + (to[14] - from[14]) * t));
        colliders.set(body, j * 16);
      });
      sim!.step(targets!, colliders, 1 / BAKE_FPS);
    }
    targets!.set(endTargets);
    colliders.set(endColliders);
  };

  try {
    for (let i = 0; i < total; i++) {
      model.update(from + i, 1 / BAKE_FPS);
      if (sim && clothData) {
        readPose();
        if (i === 0) warmUpCloth();
        sim.step(targets!, colliders, 1 / BAKE_FPS);
        clothData.set(sim.positions, i * mesh!.particleCount * 3);
      }
      for (let slot = 0; slot < n; slot++) {
        const bone = skeleton.bones[bones[slot]];
        if (bone.parent) mat4.multiply(local, mat4.invert(inv, bone.parent.world)!, bone.world);
        else mat4.copy(local, bone.world);
        quat.normalize(q, quat.fromMat4(q, local));
        const o = (i * n + slot) * BAKE_FLOATS;
        data[o] = local[12]; data[o + 1] = local[13]; data[o + 2] = local[14];
        data[o + 3] = q[0]; data[o + 4] = q[1]; data[o + 5] = q[2]; data[o + 6] = q[3];
      }
      if ((i + 1) % 15 === 0 || i + 1 === total) opts.onProgress?.(i + 1, total);
    }
  } finally {
    model.attachPhysics(null);
  }
  const cloth: ClothBake | null = mesh && clothData
    ? { materials: mesh.materials, settings: opts.cloth!.settings, particleCount: mesh.particleCount, data: clothData }
    : null;
  return new PhysicsBake(from, to, bones, data, cloth);
}

