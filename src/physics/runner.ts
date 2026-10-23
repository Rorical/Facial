import { mat4 } from '../math';
import type { Skeleton } from '../runtime/skeleton';
import type { PhysicsWorld } from './physics-world';
import type { PhysicsRig } from './rig';

// The physics runner interface and the main-thread runner. Kept free of Worker / DOM code
// so it can also run inside workers (physics baking).

/** worker / main: live simulation; baked: replays a precomputed bake. */
export type PhysicsKind = 'worker' | 'main' | 'baked';

export interface PhysicsRunner {
  readonly kind: PhysicsKind;
  readonly rig: PhysicsRig;
  readonly bodyCount: number;
  readonly jointCount: number;
  enabled: boolean;
  /** Also produce body transforms for wireframe drawing. */
  wantDebug: boolean;
  /** Smoothed simulation time per step, ms (worker: measured on the worker thread). */
  readonly stepMs: number;
  /** Newest body world transforms (16 floats per rig body), or null if none yet. */
  readonly bodyTransforms: Float32Array | null;
  /** Share of recent frames that got a new physics result (1 = every frame). */
  readonly freshRatio: number;
  /**
   * Runs between the before- and after-physics bone phases. Marks driven bones with their
   * physics transforms; returns false when there is nothing to apply.
   * `frame` is the animation frame being evaluated (used by baked playback).
   */
  update(skeleton: Skeleton, dt: number, frame: number): boolean;
  /** Baked cloth for the frame last updated (see MmdModel.clothFrame); only baked physics has it. */
  readonly clothFrame?: Float32Array | null;
  /** Snap bodies to the current pose (seek / loop / enable), optionally settling first. */
  requestReset(warmupSteps?: number): void;
  dispose(): void;
}

export const MAX_DT = 0.1;

export function packBoneWorlds(skeleton: Skeleton, out: Float64Array): void {
  const bones = skeleton.bones;
  for (let i = 0; i < bones.length; i++) out.set(bones[i].world, i * 16);
}

export function applyLocals(skeleton: Skeleton, rig: PhysicsRig, locals: Float64Array): void {
  for (let slot = 0; slot < rig.drivenBones.length; slot++) {
    const bone = skeleton.bones[rig.drivenBones[slot]];
    if (!bone) continue;
    mat4.copy(bone.physicsLocal, locals.subarray(slot * 16, slot * 16 + 16));
    bone.physicsKeepTranslation = rig.drivenKeepTranslation[slot] === 1;
    bone.physicsActive = true;
  }
}

/** Body transforms implied by the current bones (for wireframes while physics is off / baked). */
export function bodyTransformsFromBones(rig: PhysicsRig, skeleton: Skeleton, out: Float32Array): Float32Array {
  const m = mat4.create();
  rig.bodies.forEach((b, i) => {
    const bone = b.boneIndex >= 0 ? skeleton.bones[b.boneIndex] : null;
    if (bone) mat4.multiply(m, bone.world, b.offset);
    else mat4.copy(m, b.rest);
    out.set(m, i * 16);
  });
  return out;
}

export class InlinePhysics implements PhysicsRunner {
  readonly kind = 'main';
  readonly bodyCount: number;
  readonly jointCount: number;
  enabled = true;
  wantDebug = false;
  stepMs = 0;
  bodyTransforms: Float32Array | null = null;
  readonly freshRatio = 1;
  private readonly worlds: Float64Array;
  private readonly locals: Float64Array;
  private readonly debug: Float32Array;
  private reset = true;
  private warmup = 0;

  constructor(private readonly world: PhysicsWorld, readonly rig: PhysicsRig, boneCount: number) {
    this.bodyCount = world.bodyCount;
    this.jointCount = world.jointCount;
    this.worlds = new Float64Array(boneCount * 16);
    this.locals = new Float64Array(rig.drivenBones.length * 16);
    this.debug = new Float32Array(rig.bodies.length * 16);
  }

  update(skeleton: Skeleton, dt: number): boolean {
    const t0 = performance.now();
    const warmup = this.warmup;
    packBoneWorlds(skeleton, this.worlds);
    this.world.step({ boneWorlds: this.worlds, dt: Math.min(dt, MAX_DT), reset: this.reset, warmup }, this.locals);
    this.reset = false;
    this.warmup = 0;
    applyLocals(skeleton, this.rig, this.locals);
    if (this.wantDebug) {
      this.world.readBodyTransforms(this.debug);
      this.bodyTransforms = this.debug;
    }
    if (warmup === 0) this.stepMs = this.stepMs * 0.9 + (performance.now() - t0) * 0.1;
    return true;
  }

  requestReset(warmupSteps = 0): void {
    this.reset = true;
    this.warmup = Math.max(this.warmup, warmupSteps);
  }

  dispose(): void {
    // Nothing to release: the solver is plain JavaScript.
  }
}
