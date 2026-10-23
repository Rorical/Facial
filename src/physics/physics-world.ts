import { mat4, quat, vec3, type Mat4 } from '../math';
import { FOLLOW_BONE, type PhysicsRig } from './rig';
import { blendRigid, extrapolatePose } from './pose';
import { BodySet } from './solver/body-set';
import { ContactSolver } from './solver/contact-solver';
import { JointSolver } from './solver/joint-solver';

// MMD rigid body physics as position-based dynamics (XPBD with substeps, after Müller et al.
// 2020, "Detailed Rigid Body Simulation with Extended Position Based Dynamics"). Rules follow
// MMD:
//   * mode 0 "follow bone": kinematic, driven by its bone, swept smoothly through the substeps;
//   * mode 1 "physics": dynamic, the bone follows the body;
//   * mode 2 "physics + bone position": dynamic, pinned at its bone's position; the bone takes
//     only the body's rotation;
//   * joints: 6-DOF limits (rigid) and springs (compliant), gravity (0, −98, 0) in MMD units.
// Every constraint is a projection, so stiff joints, tiny masses and deep penetrations never
// blow up; small substeps replace solver iterations. Pure TypeScript, so one source runs on
// the main thread, in the physics worker and in the bake worker, and gives identical results.
//
// This class only sees arrays (bone world matrices in, bone transforms out). Output transforms
// are relative to the parent bone, so a result computed for an earlier pose re-attaches
// cleanly to the current one.

export interface PhysicsOptions {
  gravity: [number, number, number];
  /** Longest substep (s): each frame is split into equal substeps no longer than this. */
  substep: number;
  maxSubsteps: number;
  /**
   * Constraint passes per substep: one when joints start the pass within `tolerance` (model
   * units) of closed, as they do at rest and in ordinary motion; up to `maxIterations` when a
   * fast swing has stretched a chain.
   */
  maxIterations: number;
  tolerance: number;
  /** Rigid-body collisions on/off (joints still act). */
  collisions: boolean;
  /**
   * Simulate toward the pose one frame ahead (extrapolated from the last two frames).
   * For pipelined (worker) use, where results are shown a frame late: kinematic bodies
   * such as legs are then where they'll actually be, instead of a frame behind.
   */
  predict: boolean;
}

export const DEFAULT_PHYSICS_OPTIONS: PhysicsOptions = {
  gravity: [0, -98, 0],
  substep: 1 / 240,
  maxSubsteps: 48,
  // Contacts keep some joints a few millimetres open at all times; the tolerance sits above
  // that so a model at rest costs one pass.
  maxIterations: 3,
  tolerance: 0.05,
  collisions: true,
  predict: false,
};

/** Length of one warm-up step (settling after a reset). */
export const WARMUP_STEP = 1 / 60;

export interface StepRequest {
  /** 16 floats per bone: current bone world matrices (after the before-physics phase). */
  boneWorlds: Float64Array;
  dt: number;
  /** Teleport bodies to the bones first (seek / loop / (re)enable). */
  reset: boolean;
  /** Extra settle steps after a reset (e.g. 60 on load). */
  warmup: number;
}

export class PhysicsWorld {
  readonly bodyCount: number;
  readonly jointCount: number;

  private readonly bodies: BodySet;
  private readonly joints: JointSolver;
  private readonly contacts: ContactSolver;
  private readonly options: PhysicsOptions;
  private readonly physWorlds: Float64Array; // scratch: physics world matrix per driven bone slot
  /** Bone worlds of the previous step (for substep sweeps and prediction). */
  private readonly prevWorlds: Float64Array;
  private readonly predicted: Float64Array;
  private hasPrevWorlds = false;
  /** Kinematic bodies' poses at the end of the frame (3 + 4 per body), for the broadphase. */
  private readonly kinematicEnd: Float64Array;
  private readonly slotOfBone = new Map<number, number>();
  private readonly m = mat4.create();
  private readonly m2 = mat4.create();
  private readonly q = quat.create();
  private readonly q2 = quat.create();
  private readonly v = vec3.create();

  constructor(private readonly rig: PhysicsRig, options: Partial<PhysicsOptions> = {}) {
    this.options = { ...DEFAULT_PHYSICS_OPTIONS, ...options };
    this.bodies = new BodySet(rig);
    this.joints = new JointSolver(rig, this.bodies);
    this.contacts = new ContactSolver(rig, this.bodies, this.options.collisions);
    rig.drivenBones.forEach((bone, slot) => this.slotOfBone.set(bone, slot));
    this.physWorlds = new Float64Array(rig.drivenBones.length * 16);
    this.prevWorlds = new Float64Array(rig.boneParents.length * 16);
    this.predicted = new Float64Array(rig.boneParents.length * 16);
    this.kinematicEnd = new Float64Array(rig.bodies.length * 7);
    this.bodyCount = rig.bodies.length;
    this.jointCount = this.joints.count;
  }

  /**
   * Advance the simulation and write, per driven bone (rig.drivenBones order), its physics
   * transform relative to its parent bone (16 floats each) into `outLocals`.
   */
  step(req: StepRequest, outLocals: Float64Array): void {
    if (req.reset || req.warmup > 0) {
      this.hasPrevWorlds = false; // never sweep or extrapolate across a jump
      this.resetBodies(req.boneWorlds);
    }
    // `from` → `to` is the pose change the substeps sweep through.
    let from = this.prevWorlds;
    let to = req.boneWorlds;
    if (this.options.predict && this.hasPrevWorlds) {
      extrapolatePose(this.prevWorlds, req.boneWorlds, this.predicted);
      from = req.boneWorlds;
      to = this.predicted;
    }
    const sweep = this.hasPrevWorlds;

    for (let i = 0; i < req.warmup; i++) this.advance(WARMUP_STEP, to, null);
    if (req.dt > 0) this.advance(req.dt, to, sweep ? from : null);
    else this.syncKinematic(to, null, 1);
    this.computeLocals(to, outLocals);
    // Update history last: `from` may alias prevWorlds during the substeps above.
    this.prevWorlds.set(req.boneWorlds);
    this.hasPrevWorlds = true;
  }

  /** Current world transform of every body (16 floats each), for debug drawing. */
  readBodyTransforms(out: Float32Array): void {
    for (let i = 0; i < this.bodyCount; i++) out.set(this.bodyWorld(i), i * 16);
  }

  private bodyWorld(i: number): Mat4 {
    const b = this.bodies;
    return mat4.fromRotationTranslation(this.m, b.q.subarray(i * 4, i * 4 + 4), b.x.subarray(i * 3, i * 3 + 3));
  }

  /** Simulate `dt` seconds toward bone pose `to`, sweeping kinematic bodies from `from` if given. */
  private advance(dt: number, to: Float64Array, from: Float64Array | null): void {
    const { bodies, joints, contacts, options } = this;
    const n = Math.min(options.maxSubsteps, Math.max(1, Math.ceil(dt / options.substep - 1e-6)));
    const h = dt / n;
    const [gx, gy, gz] = options.gravity;
    const g = Math.hypot(gx, gy, gz);
    this.kinematicEndPoses(to);
    contacts.findCandidates(dt, this.kinematicEnd);
    for (let s = 1; s <= n; s++) {
      bodies.beginSubstep();
      this.syncKinematic(to, from, s / n);
      bodies.integrate(h, gx, gy, gz);
      contacts.generate();
      for (let it = 0; it < options.maxIterations; it++) {
        const residual = joints.solve(h);
        joints.solvePins();
        contacts.solvePositions();
        if (residual < options.tolerance) break;
      }
      bodies.updateVelocities(h);
      contacts.solveVelocities(h, g);
      bodies.damp(h);
    }
  }

  private computeLocals(boneWorlds: Float64Array, out: Float64Array): void {
    const { rig, m2 } = this;
    for (let slot = 0; slot < rig.drivenBones.length; slot++) {
      const bone = rig.drivenBones[slot];
      const body = rig.bodies[rig.drivenBody[slot]];
      const phys = this.physWorlds.subarray(slot * 16, slot * 16 + 16) as unknown as Mat4;
      mat4.multiply(phys, this.bodyWorld(rig.drivenBody[slot]), body.offsetInv);
      if (rig.drivenKeepTranslation[slot]) {
        phys[12] = boneWorlds[bone * 16 + 12]; phys[13] = boneWorlds[bone * 16 + 13]; phys[14] = boneWorlds[bone * 16 + 14];
      }
      const parent = rig.boneParents[bone];
      const local = out.subarray(slot * 16, slot * 16 + 16) as unknown as Mat4;
      if (parent < 0) {
        mat4.copy(local, phys);
        continue;
      }
      const parentSlot = this.slotOfBone.get(parent);
      const parentWorld = parentSlot !== undefined
        ? this.physWorlds.subarray(parentSlot * 16, parentSlot * 16 + 16)
        : boneWorlds.subarray(parent * 16, parent * 16 + 16);
      mat4.invert(m2, parentWorld as unknown as Mat4);
      mat4.multiply(local, m2, phys);
    }
  }

  /** Kinematic bodies' poses at bone pose `to`, for the frame's broadphase sweep. */
  private kinematicEndPoses(to: Float64Array): void {
    const { rig, bodies, kinematicEnd } = this;
    for (let i = 0; i < rig.bodies.length; i++) {
      const rb = rig.bodies[i];
      if (!bodies.kinematic[i]) continue;
      if (rb.boneIndex >= 0) mat4.multiply(this.m, to.subarray(rb.boneIndex * 16, rb.boneIndex * 16 + 16) as unknown as Mat4, rb.offset);
      else mat4.copy(this.m, rb.rest);
      kinematicEnd[i * 7] = this.m[12]; kinematicEnd[i * 7 + 1] = this.m[13]; kinematicEnd[i * 7 + 2] = this.m[14];
      kinematicEnd.set(quat.fromMat4(this.q, this.m), i * 7 + 3);
    }
  }

  /**
   * Move kinematic bodies and pin targets to their bones. With `from` given, the pose is
   * blended from → to by t, so a fast limb sweeps through the substeps instead of teleporting
   * into cloth.
   */
  private syncKinematic(to: Float64Array, from: Float64Array | null, t: number): void {
    const { rig, bodies, joints } = this;
    const boneAt = (bone: number): Mat4 => {
      const o = bone * 16;
      const target = to.subarray(o, o + 16) as unknown as Mat4;
      if (!from || t >= 1) return target;
      return blendRigid(from.subarray(o, o + 16), target, t, this.m2, this.q, this.q2, this.v);
    };
    for (let i = 0; i < rig.bodies.length; i++) {
      const rb = rig.bodies[i];
      if (rb.mode !== FOLLOW_BONE || rb.boneIndex < 0) continue;
      bodies.setPose(i, mat4.multiply(this.m, boneAt(rb.boneIndex), rb.offset));
    }
    for (let k = 0; k < joints.pinCount; k++) {
      const o = rig.bodies[joints.pinnedBody(k)].boneIndex * 16 + 12;
      const s = from && t < 1 ? t : 1;
      const f = from ?? to;
      joints.setPinTarget(k, f[o] + (to[o] - f[o]) * s, f[o + 1] + (to[o + 1] - f[o + 1]) * s, f[o + 2] + (to[o + 2] - f[o + 2]) * s);
    }
  }

  private resetBodies(boneWorlds: Float64Array): void {
    const { rig, bodies } = this;
    for (let i = 0; i < rig.bodies.length; i++) {
      const rb = rig.bodies[i];
      if (rb.boneIndex >= 0) mat4.multiply(this.m, boneWorlds.subarray(rb.boneIndex * 16, rb.boneIndex * 16 + 16) as unknown as Mat4, rb.offset);
      else mat4.copy(this.m, rb.rest);
      bodies.setPose(i, this.m);
      bodies.freeze(i);
    }
    this.contacts.reset();
  }
}
