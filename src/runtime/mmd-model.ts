import { vec3, type Vec3 } from '../math';
import type { PmxModel } from '../loaders/pmx';
import type { BoneTrack, MmdMotion, MorphTrack } from '../animation/motion';
import type { PhysicsRunner } from '../physics/runner';
import { vmdName } from '../io/sjis';
import { MorphController, type MaterialTextureFlags } from './morph-controller';
import { Skeleton } from './skeleton';

/** CPU side of a model: samples the motion and produces per-frame GPU inputs. */
export class MmdModel {
  readonly skeleton: Skeleton;
  readonly morphs: MorphController;
  /** 20 floats per bone: skin matrix + world rotation quaternion. */
  readonly boneData: Float32Array;

  /** Approximate bounding sphere of the posed model (for the shadow frustum). */
  readonly bounds: { center: Vec3; radius: number } = { center: vec3.create(), radius: 1 };
  private readonly restCenter: Vec3;
  private readonly restRadius: number;

  /**
   * Scales every bone translation in the motion. A motion's steps and hip height fit the legs of
   * the model it was made on; on a model with longer or shorter legs this keeps the feet planted.
   */
  motionScale = 1;

  /**
   * This frame's baked cloth, if the physics in use plays one back: 8 floats (position, normal)
   * per cloth vertex, in model space. null otherwise.
   */
  clothFrame: Float32Array | null = null;

  physics: PhysicsRunner | null = null;
  private physicsDt = 1 / 60;
  private physicsFrame = 0;
  private motion: MmdMotion | null = null;
  private boneTracks: (BoneTrack | undefined)[] = [];
  private morphTracks: (MorphTrack | undefined)[] = [];

  constructor(readonly pmx: PmxModel, textureFlags: MaterialTextureFlags[]) {
    this.skeleton = new Skeleton(pmx.bones);
    this.morphs = new MorphController(pmx, textureFlags);
    this.boneData = new Float32Array(Math.max(pmx.bones.length, 1) * 20);

    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < pmx.vertexCount * 3; i++) {
      const c = i % 3;
      const v = pmx.positions[i];
      if (v < min[c]) min[c] = v;
      if (v > max[c]) max[c] = v;
    }
    const empty = pmx.vertexCount === 0;
    this.restCenter = empty ? vec3.create(0, 10, 0) : vec3.create((min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2);
    this.restRadius = empty ? 10 : Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2;
  }

  /** Rest bounds shifted by how far センター (the body's root) has moved. */
  private updateBounds(): void {
    const anchor = this.skeleton.byName.get('センター') ?? this.skeleton.bones[0];
    const c = this.bounds.center;
    vec3.copy(c, this.restCenter);
    if (anchor) {
      c[0] += anchor.world[12] - anchor.restPosition[0];
      c[1] += anchor.world[13] - anchor.restPosition[1];
      c[2] += anchor.world[14] - anchor.restPosition[2];
    }
    this.bounds.radius = this.restRadius * 1.1;
  }

  /** Replaces (and disposes) the current physics runner. */
  attachPhysics(physics: PhysicsRunner | null): void {
    this.physics?.dispose();
    this.physics = physics;
    this.skeleton.physicsHook = physics ? () => physics.enabled && physics.update(this.skeleton, this.physicsDt, this.physicsFrame) : null;
  }

  setMotion(motion: MmdMotion | null): { boundBones: number; boundMorphs: number } {
    this.motion = motion;
    // VMD names are cut at 15 bytes: longer model names match by their cut form.
    this.boneTracks = this.skeleton.bones.map((b) => motion?.boneTracks.get(b.name) ?? motion?.boneTracks.get(vmdName(b.name)));
    this.morphTracks = this.pmx.morphs.map((m) => motion?.morphTracks.get(m.name) ?? motion?.morphTracks.get(vmdName(m.name)));
    return {
      boundBones: this.boneTracks.filter(Boolean).length,
      boundMorphs: this.morphTracks.filter(Boolean).length,
    };
  }

  /**
   * Evaluate the pose at `frame` (30 fps units, fractional allowed).
   * `dt` (seconds) is how far physics advances.
   */
  update(frame: number, dt = 1 / 60): void {
    const { skeleton, morphs } = this;
    this.physicsDt = dt;
    this.physicsFrame = frame;
    skeleton.resetInputs();

    for (let i = 0; i < this.boneTracks.length; i++) {
      const track = this.boneTracks[i];
      if (track) {
        const bone = skeleton.bones[i];
        track.sample(frame, bone.animPosition, bone.animRotation);
        if (this.motionScale !== 1) vec3.scale(bone.animPosition, bone.animPosition, this.motionScale);
      }
    }

    for (let i = 0; i < this.morphTracks.length; i++) {
      morphs.weights[i] = this.morphTracks[i]?.sample(frame) ?? 0;
    }

    const ikStates = this.motion?.ikStatesAt(frame);
    for (const solver of skeleton.ikSolvers) {
      solver.enabled = ikStates?.get(solver.ikBone.name) ?? true;
    }

    morphs.update(skeleton);
    skeleton.update();
    this.clothFrame = (this.physics?.enabled && this.physics.clothFrame) || null;
    morphs.writeHead(skeleton);
    skeleton.writeSkinning(this.boneData);
    this.updateBounds();
  }
}
