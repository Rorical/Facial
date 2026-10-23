import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import type { Vec3Tuple } from '../src/loaders/binary-reader';
import { BoneFlag, parsePmx, type PmxBone, type PmxJoint, type PmxModel, type PmxRigidBody } from '../src/loaders/pmx';
import { mat4, quat, vec3 } from '../src/math';
import { MmdMotion } from '../src/animation/motion';
import { buildDebugGeometry, DEBUG_VERTEX_BYTES } from '../src/physics/debug-geometry';
import { PhysicsWorld, type PhysicsOptions } from '../src/physics/physics-world';
import { extrapolatePose } from '../src/physics/pose';
import { buildPhysicsRig, physicsModelData, type PhysicsRig } from '../src/physics/rig';
import { InlinePhysics } from '../src/physics/runners';
import { BakedPhysics, bakeToVmdBoneFrames, deserializeBake, serializeBake } from '../src/physics/bake';
import { bakePhysics } from '../src/physics/bake-sim';
import { MmdModel } from '../src/runtime/mmd-model';
import type { Skeleton } from '../src/runtime/skeleton';

function bone(name: string, position: Vec3Tuple, parentIndex: number): PmxBone {
  return {
    name, nameEn: '', position, parentIndex, layer: 0, flags: BoneFlag.Rotatable | BoneFlag.Visible | BoneFlag.Enabled,
    tailBoneIndex: -1, tailOffset: [0, 0, 0], inherit: null, fixedAxis: null, localAxes: null, externalParentKey: -1, ik: null,
  };
}

function body(name: string, boneIndex: number, position: Vec3Tuple, mode: number, shape = 0, size: Vec3Tuple = [0.5, 0, 0], extra: Partial<PmxRigidBody> = {}): PmxRigidBody {
  return {
    name, nameEn: '', boneIndex, group: 0, collisionMask: 0, shape, size, position, rotation: [0, 0, 0],
    mass: 1, linearDamping: 0.5, angularDamping: 0.5, restitution: 0, friction: 0.5, mode, ...extra,
  };
}

function joint(a: number, b: number, position: Vec3Tuple, extra: Partial<PmxJoint> = {}): PmxJoint {
  return {
    name: 'j', nameEn: '', type: 0, rigidBodyA: a, rigidBodyB: b, position, rotation: [0, 0, 0],
    positionMin: [0, 0, 0], positionMax: [0, 0, 0], rotationMin: [-Math.PI, -Math.PI, -Math.PI], rotationMax: [Math.PI, Math.PI, Math.PI],
    springPosition: [0, 0, 0], springRotation: [0, 0, 0], ...extra,
  };
}

function model(bones: PmxBone[], rigidBodies: PmxRigidBody[], joints: PmxJoint[]): PmxModel {
  return {
    format: 'PMX',
    header: { version: 2, encoding: 'utf-8', additionalUvCount: 0, vertexIndexSize: 1, textureIndexSize: 1, materialIndexSize: 1, boneIndexSize: 1, morphIndexSize: 1, rigidBodyIndexSize: 1 },
    name: 'test', nameEn: '', comment: '', commentEn: '',
    vertexCount: 0, positions: new Float32Array(0), normals: new Float32Array(0), uvs: new Float32Array(0), additionalUvs: [],
    skinTypes: new Uint8Array(0), skinIndices: new Uint32Array(0), skinWeights: new Float32Array(0),
    sdef: { vertexIndices: new Uint32Array(0), c: new Float32Array(0), r0: new Float32Array(0), r1: new Float32Array(0) },
    edgeScales: new Float32Array(0), indices: new Uint32Array(0), textures: [], materials: [],
    bones, morphs: [], displayFrames: [], rigidBodies, joints,
  };
}

/**
 * root (kinematic body) ── joint ── arm bone (physics body, sticking out along +X) ── tip bone (no body)
 * Gravity should swing the arm down, carrying the tip with it.
 */
function pendulum(mode = 1, jointExtra: Partial<PmxJoint> = {}): PmxModel {
  return model(
    [bone('root', [0, 10, 0], -1), bone('arm', [0, 10, 0], 0), bone('tip', [4, 10, 0], 1)],
    [body('rootBody', 0, [0, 10, 0], 0), body('armBody', 1, [2, 10, 0], mode, 2, [0.3, 3, 0], { rotation: [0, 0, -Math.PI / 2] })],
    [joint(0, 1, [0, 10, 0], jointExtra)],
  );
}

function inline(pmx: PmxModel, options: Partial<PhysicsOptions> = {}): InlinePhysics {
  const rig = buildPhysicsRig(physicsModelData(pmx));
  return new InlinePhysics(new PhysicsWorld(rig, options), rig, pmx.bones.length);
}

/** Emulates the worker pipeline: every frame applies the result computed one frame earlier. */
class LaggedPhysics extends InlinePhysics {
  private previous: Float64Array | null = null;
  constructor(pmx: PmxModel, rig: PhysicsRig) {
    super(new PhysicsWorld(rig), rig, pmx.bones.length);
  }
  override update(skeleton: Skeleton, dt: number): boolean {
    super.update(skeleton, dt); // real step: writes this frame's result into the bones
    const current = new Float64Array(this.rig.drivenBones.length * 16);
    this.rig.drivenBones.forEach((b, slot) => current.set(skeleton.bones[b].physicsLocal, slot * 16));
    const stale = this.previous;
    this.previous = current;
    if (!stale) return false;
    this.rig.drivenBones.forEach((b, slot) => skeleton.bones[b].physicsLocal.set(stale.subarray(slot * 16, slot * 16 + 16)));
    return true;
  }
}

/** Motion that moves `root` along +X by `speed` units per frame. */
function rootMotion(speed: number): MmdMotion {
  const interpolation = new Uint8Array(64).fill(20).fill(107, 8, 16); // linear
  return MmdMotion.fromVmd({
    modelName: '', morphFrames: [], cameraFrames: [], lightFrames: [], selfShadowFrames: [], propertyFrames: [],
    boneFrames: [0, 60].map((frame) => ({ boneName: 'root', frame, position: [frame * speed, 0, 0], rotation: [0, 0, 0, 1], interpolation })),
  });
}

const world = (m: MmdModel, name: string) => mat4.getTranslation(vec3.create(), m.skeleton.byName.get(name)!.world);

describe('rigid body physics', () => {
  it('swings a physics bone down under gravity and drags its child along', () => {
    const pmx = pendulum();
    const model = new MmdModel(pmx, []);
    const physics = inline(pmx);
    model.attachPhysics(physics);
    expect(physics.bodyCount).toBe(2);
    expect(physics.jointCount).toBe(1);
    for (let i = 0; i < 180; i++) model.update(0, 1 / 60);

    const arm = world(model, 'arm');
    const tip = world(model, 'tip');
    // Pivot stays at the joint; the tip hangs roughly 4 units below it.
    expect(vec3.distance(arm, vec3.create(0, 10, 0))).toBeLessThan(0.2);
    expect(tip[1]).toBeLessThan(7);
    expect(vec3.distance(arm, tip)).toBeCloseTo(4, 1);
    model.attachPhysics(null);
  });

  it('kinematic bodies follow their bone', () => {
    const pmx = pendulum();
    const model = new MmdModel(pmx, []);
    model.attachPhysics(inline(pmx));
    model.setMotion(rootMotion(0.1));
    for (let i = 0; i <= 60; i++) model.update(i, 1 / 60);
    // The arm is dragged along with the moving root (6 units to the right).
    expect(world(model, 'arm')[0]).toBeGreaterThan(5);
    model.attachPhysics(null);
  });

  it('"physics + bone" keeps the bone position from animation', () => {
    const pmx = pendulum(2);
    // Detach the body from its parent's body so the mode isn't downgraded to plain physics.
    pmx.bones[1].parentIndex = -1;
    const model = new MmdModel(pmx, []);
    model.attachPhysics(inline(pmx));
    for (let i = 0; i < 120; i++) model.update(0, 1 / 60);
    expect(vec3.distance(world(model, 'arm'), vec3.create(0, 10, 0))).toBeLessThan(1e-4);
    // …but still takes its rotation: the tip swings down.
    expect(world(model, 'tip')[1]).toBeLessThan(8);
    model.attachPhysics(null);
  });

  it('one-frame-late (worker) results stay attached to a fast-moving parent', () => {
    // Root moves 1 unit per frame: a world-space result from the previous frame would sit 1 unit behind.
    const pivotError = (lagged: boolean) => {
      const pmx = pendulum();
      const model = new MmdModel(pmx, []);
      model.attachPhysics(lagged ? new LaggedPhysics(pmx, buildPhysicsRig(physicsModelData(pmx))) : inline(pmx));
      model.setMotion(rootMotion(1));
      for (let i = 0; i <= 30; i++) model.update(i, 1 / 60);
      const d = vec3.distance(world(model, 'arm'), world(model, 'root'));
      model.attachPhysics(null);
      return d;
    };
    const sameFrame = pivotError(false); // joint softness alone at 60 units/s
    const oneFrameLate = pivotError(true);
    expect(oneFrameLate).toBeLessThan(sameFrame + 0.15);
    expect(oneFrameLate).toBeLessThan(0.75);
  });

  it('keeps a limited joint inside its range', () => {
    // The arm may only drop 30° about Z (its capsule points along +X at rest).
    const limit = (30 * Math.PI) / 180;
    const pmx = pendulum(1, { rotationMin: [0, 0, -limit], rotationMax: [0, 0, 0] });
    const m = new MmdModel(pmx, []);
    m.attachPhysics(inline(pmx));
    for (let i = 0; i < 180; i++) m.update(0, 1 / 60);
    const tip = world(m, 'tip');
    const angle = Math.atan2(10 - tip[1], tip[0]);
    expect(angle).toBeGreaterThan(limit - 0.03);
    expect(angle).toBeLessThan(limit + 0.03);
    m.attachPhysics(null);
  });

  it('holds a rotation spring part way against gravity', () => {
    // A torsion spring about Z balances the arm's weight at k·θ = m·g·L/2·cos θ.
    const k = 400;
    const pmx = pendulum(1, { springRotation: [0, 0, k] });
    const m = new MmdModel(pmx, []);
    m.attachPhysics(inline(pmx));
    for (let i = 0; i < 300; i++) m.update(0, 1 / 60);
    const tip = world(m, 'tip');
    const angle = Math.atan2(10 - tip[1], tip[0]);
    const expected = (() => {
      let t = 0.5;
      for (let i = 0; i < 50; i++) t = (98 * 2 * Math.cos(t)) / k;
      return t;
    })();
    expect(angle).toBeCloseTo(expected, 1);
    m.attachPhysics(null);
  });

  it('carries a long chain through a fast swing without stretching', () => {
    // 12 capsules hanging from a root that whips 8 units sideways in 10 frames.
    const bones = [bone('root', [0, 20, 0], -1)];
    const bodies = [body('rootBody', 0, [0, 20, 0], 0)];
    const joints: PmxJoint[] = [];
    for (let i = 1; i <= 12; i++) {
      bones.push(bone(`h${i}`, [0, 20 - (i - 1), 0], i - 1));
      bodies.push(body(`b${i}`, i, [0, 20 - (i - 1) - 0.5, 0], 1, 2, [0.1, 0.8, 0], { mass: 0.1, linearDamping: 0.9, angularDamping: 0.9 }));
      joints.push(joint(i - 1, i, [0, 20 - (i - 1), 0], { rotationMin: [-0.4, -0.4, -0.4], rotationMax: [0.4, 0.4, 0.4] }));
    }
    const pmx = model(bones, bodies, joints);
    const m = new MmdModel(pmx, []);
    const interpolation = new Uint8Array(64).fill(20).fill(107, 8, 16);
    m.setMotion(MmdMotion.fromVmd({
      modelName: '', morphFrames: [], cameraFrames: [], lightFrames: [], selfShadowFrames: [], propertyFrames: [],
      boneFrames: [0, 10, 20].map((frame, k) => ({ boneName: 'root', frame, position: [k === 1 ? 8 : 0, 0, 0], rotation: [0, 0, 0, 1], interpolation })),
    }));
    const physics = inline(pmx);
    physics.requestReset(30);
    m.attachPhysics(physics);
    let worstGap = 0;
    for (let f = 0; f <= 40; f += 0.5) {
      m.update(f, 1 / 60);
      for (let i = 1; i <= 12; i++) {
        const joint = world(m, `h${i}`);
        const parentTip = i === 1 ? world(m, 'root') : vec3.transformMat4(vec3.create(), vec3.create(0, -1, 0), m.skeleton.byName.get(`h${i - 1}`)!.world);
        worstGap = Math.max(worstGap, vec3.distance(joint, parentTip));
      }
      for (const b of m.skeleton.bones) for (const v of b.world) expect(Number.isFinite(v)).toBe(true);
    }
    expect(worstGap).toBeLessThan(0.15);
    m.attachPhysics(null);
  });

  it('rests a falling body on a kinematic one and stops it at the surface', () => {
    const pmx = model(
      [bone('floor', [0, 0, 0], -1), bone('ball', [0, 6, 0], -1)],
      [
        body('floorBody', 0, [0, 0, 0], 0, 1, [4, 0.5, 4], { collisionMask: 0xffff }),
        body('ballBody', 1, [0, 6, 0], 1, 0, [1, 0, 0], { collisionMask: 0xffff, linearDamping: 0, angularDamping: 0 }),
      ],
      [],
    );
    const m = new MmdModel(pmx, []);
    const physics = inline(pmx);
    m.attachPhysics(physics);
    for (let i = 0; i < 120; i++) m.update(0, 1 / 60);
    expect(world(m, 'ball')[1]).toBeCloseTo(1.5, 1);
    // Still at rest a second later (no bounce, no sinking).
    for (let i = 0; i < 60; i++) m.update(0, 1 / 60);
    expect(world(m, 'ball')[1]).toBeCloseTo(1.5, 1);
    m.attachPhysics(null);
  });

  it('builds wireframe geometry for bodies and joints', () => {
    const geo = buildDebugGeometry(buildPhysicsRig(physicsModelData(pendulum())));
    expect(geo.vertexCount % 2).toBe(0);
    expect(geo.data.byteLength).toBe(geo.vertexCount * DEBUG_VERTEX_BYTES);
    const u32 = new Uint32Array(geo.data);
    const kinds = new Set<number>();
    const bodies = new Set<number>();
    for (let v = 0; v < geo.vertexCount; v++) {
      bodies.add(u32[v * 5 + 3]);
      kinds.add(u32[v * 5 + 4]);
    }
    expect([...bodies].sort()).toEqual([0, 1]);
    expect([...kinds].sort()).toEqual([0, 1, 3]); // follow-bone body, physics body, joint
  });

  it('predicts the next pose exactly for constant linear and angular velocity', () => {
    const pose = (k: number) => {
      const m = mat4.fromRotationTranslation(mat4.create(), quat.fromAxisAngle(quat.create(), [0, 1, 0], 0.2 * k), vec3.create(k, 2 * k, 0));
      return Float64Array.from(m);
    };
    const out = new Float64Array(16);
    extrapolatePose(pose(1), pose(2), out);
    const expected = pose(3);
    for (let i = 0; i < 16; i++) expect(out[i]).toBeCloseTo(expected[i], 10);
  });

  it('yaw-pitch-roll quaternion matches Ry·Rx·Rz', () => {
    const [x, y, z] = [0.3, -1.1, 0.7];
    const expected = quat.create();
    quat.multiply(expected, quat.fromAxisAngle(quat.create(), [0, 1, 0], y), quat.fromAxisAngle(quat.create(), [1, 0, 0], x));
    quat.multiply(expected, expected, quat.fromAxisAngle(quat.create(), [0, 0, 1], z));
    const q = quat.fromYawPitchRoll(quat.create(), x, y, z);
    for (let i = 0; i < 4; i++) expect(q[i]).toBeCloseTo(expected[i], 10);
  });
});

describe('physics baking', () => {
  const pmx = pendulum();
  const motion = () => rootMotion(0.2);
  const armTip = (m: MmdModel) => [...world(m, 'arm'), ...world(m, 'tip')];

  it('replays exactly what a live simulation at 30 fps produced', () => {
    // Live reference: same fixed 30 fps stepping and warm-up as the bake.
    const live = new MmdModel(pmx, []);
    live.setMotion(motion());
    const runner = inline(pmx);
    runner.requestReset(60);
    live.attachPhysics(runner);
    const reference: number[][] = [];
    for (let f = 0; f <= 60; f++) {
      live.update(f, 1 / 30);
      reference.push(armTip(live));
    }
    live.attachPhysics(null);

    const bake = bakePhysics(pmx, motion(), { from: 0, to: 60, warmupSteps: 60 });
    expect(bake.frameCount).toBe(61);
    const played = new MmdModel(pmx, []);
    played.setMotion(motion());
    played.attachPhysics(new BakedPhysics(bake, buildPhysicsRig(physicsModelData(pmx))));
    // Out of order on purpose: baked playback is seekable.
    for (const f of [60, 0, 37, 12, 59]) {
      played.update(f, 0);
      armTip(played).forEach((v, i) => expect(v).toBeCloseTo(reference[f][i], 4));
    }
  });

  it('is deterministic', () => {
    const a = bakePhysics(pmx, motion(), { from: 0, to: 40 });
    const b = bakePhysics(pmx, motion(), { from: 0, to: 40 });
    expect(b.data).toEqual(a.data);
  });

  it('exports VMD keys that reproduce the bake as plain animation', () => {
    const bake = bakePhysics(pmx, motion(), { from: 0, to: 60 });
    const frames = bakeToVmdBoneFrames(bake, pmx);
    expect(frames.length).toBeGreaterThan(1);
    expect(frames.length).toBeLessThan(61); // simplified
    // Play the root motion + baked arm keys with no physics at all.
    const vmd = { modelName: '', boneFrames: [...motionToVmdBones(), ...frames], morphFrames: [], cameraFrames: [], lightFrames: [], selfShadowFrames: [], propertyFrames: [] };
    const anim = new MmdModel(pmx, []);
    anim.setMotion(MmdMotion.fromVmd(vmd));
    const baked = new MmdModel(pmx, []);
    baked.setMotion(motion());
    baked.attachPhysics(new BakedPhysics(bake, buildPhysicsRig(physicsModelData(pmx))));
    for (let f = 0; f <= 60; f += 3) {
      anim.update(f, 0);
      baked.update(f, 0);
      expect(vec3.distance(world(anim, 'tip'), world(baked, 'tip'))).toBeLessThan(0.05);
    }
  });

  it('round-trips through the project binary form', () => {
    const bake = bakePhysics(pmx, motion(), { from: 5, to: 25 });
    const { meta, data } = serializeBake(bake);
    const again = deserializeBake(meta, data.slice().buffer);
    expect(again.from).toBe(5);
    expect(again.data).toEqual(bake.data);
    expect(Array.from(again.bones)).toEqual(Array.from(bake.bones));
  });

  /** The root keys of rootMotion(0.2), as VMD frames. */
  function motionToVmdBones() {
    const interpolation = new Uint8Array(64).fill(20).fill(107, 8, 16);
    return [0, 60].map((frame) => ({ boneName: 'root', frame, position: [frame * 0.2, 0, 0] as [number, number, number], rotation: [0, 0, 0, 1] as [number, number, number, number], interpolation }));
  }
});

const SAMPLE_DIR = process.env.MMD_SAMPLE_DIR ?? '';
const pmxName = SAMPLE_DIR && existsSync(SAMPLE_DIR) ? readdirSync(SAMPLE_DIR).find((n) => /\.pmx$/i.test(n)) : undefined;

describe.skipIf(!pmxName)('physics on the sample model', () => {
  it('stays stable (no NaN, no explosion) and reports timing', () => {
    const b = readFileSync(join(SAMPLE_DIR, pmxName!));
    const pmx = parsePmx(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer);
    const model = new MmdModel(pmx, pmx.materials.map(() => ({ hasTexture: false, hasSphere: false, hasToon: false })));
    const physics = inline(pmx);
    physics.requestReset(60);
    model.attachPhysics(physics);

    const t0 = performance.now();
    const frames = 240;
    for (let i = 0; i < frames; i++) model.update(0, 1 / 60);
    const ms = (performance.now() - t0) / frames;

    let maxDrift = 0;
    const driven = Array.from(physics.rig.drivenBones, (i) => model.skeleton.bones[i]);
    const { rig } = physics;
    const held = new Set(rig.joints.flatMap((j) => [j.bodyA, j.bodyB]));
    driven.forEach((bone, slot) => {
      for (const v of bone.world) expect(Number.isFinite(v)).toBe(true);
      // A dynamic body with no joint at all falls freely (some models ship such strays).
      const body = rig.bodies[rig.drivenBody[slot]];
      if (!held.has(rig.drivenBody[slot]) && body.mode !== 2) return;
      const p = mat4.getTranslation(vec3.create(), bone.world);
      maxDrift = Math.max(maxDrift, vec3.distance(p, bone.restPosition));
    });
    console.info(
      `${physics.bodyCount} bodies, ${physics.jointCount} joints, ${driven.length} driven bones; ` +
        `${ms.toFixed(2)} ms/frame; max drift from rest ${maxDrift.toFixed(2)}`,
    );
    // At rest, hair/skirt settle but shouldn't fly away (model is ~20 units tall).
    expect(maxDrift).toBeLessThan(8);
    model.attachPhysics(null);
  });

  it('bakes 120 frames and simplifies the VMD export', () => {
    const b = readFileSync(join(SAMPLE_DIR, pmxName!));
    const pmx = parsePmx(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer);
    const t0 = performance.now();
    const bake = bakePhysics(pmx, rootMotion(0), { from: 0, to: 119 });
    const ms = performance.now() - t0;
    const keys = bakeToVmdBoneFrames(bake, pmx);
    const full = bake.bones.length * bake.frameCount;
    console.info(`bake: ${bake.bones.length} bones × ${bake.frameCount} frames in ${ms.toFixed(0)} ms (${(ms / bake.frameCount).toFixed(1)} ms/frame); VMD keys ${keys.length} of ${full} (${((100 * keys.length) / full).toFixed(1)}%)`);
    for (const v of bake.data) expect(Number.isFinite(v)).toBe(true);
    expect(keys.length).toBeLessThan(full);
  });
});

describe('vertex cloth collision setup', () => {
  it('measures distances to the three rigid body shapes', async () => {
    const { shapeDistance } = await import('../src/physics/cloth-collision');
    expect(shapeDistance(0, [2, 0, 0], 3, 0, 0)).toBeCloseTo(1); // sphere
    expect(shapeDistance(0, [2, 0, 0], 0, 0, 0)).toBeCloseTo(-2);
    expect(shapeDistance(2, [1, 4, 0], 0, 5, 0)).toBeCloseTo(2); // capsule: above its top cap
    expect(shapeDistance(2, [1, 4, 0], 3, 1, 0)).toBeCloseTo(2); // beside its axis
    expect(shapeDistance(1, [1, 2, 3], 2, 0, 0)).toBeCloseTo(1); // box
    expect(shapeDistance(1, [1, 2, 3], 0, 0, 0)).toBeCloseTo(-1);
  });

  it('marks vertices on physics bones as cloth and keeps their rest depth', async () => {
    const { buildClothCollision } = await import('../src/physics/cloth-collision');
    const { buildPhysicsRig, physicsModelData } = await import('../src/physics/rig');
    const body = (name: string, boneIndex: number, mode: number, position: [number, number, number], group: number, collisionMask: number) => ({
      name, nameEn: '', boneIndex, group, collisionMask, shape: 0, size: [1, 0, 0] as [number, number, number], position,
      rotation: [0, 0, 0] as [number, number, number], mass: 1, linearDamping: 0, angularDamping: 0, restitution: 0, friction: 0, mode,
    });
    const bone = (position: [number, number, number], parentIndex: number) => ({ position, parentIndex });
    const pmx = {
      bones: [bone([0, 0, 0], -1), bone([0, -1, 0], 0), bone([0, -2, 0], 1)],
      // Body 0 follows bone 0 (a collider); body 1 is physics on bone 1; bone 2 has no body of its own.
      rigidBodies: [body('hip', 0, 0, [0, 0, 0], 0, 0xffff), body('skirt', 1, 1, [0, -1.5, 0], 1, 0xffff)],
      joints: [],
      vertexCount: 3,
      // v0 on the hip bone; v1 on the skirt bone, resting inside the hip sphere; v2 on the skirt's child, well away.
      positions: Float32Array.from([0, 0, 0, 0, -0.5, 0, 0, -3, 0]),
      skinIndices: Uint32Array.from([0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0]),
      skinWeights: Float32Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]),
    } as unknown as import('../src/loaders/pmx').PmxModel;
    const rig = buildPhysicsRig(physicsModelData(pmx));
    const cloth = buildClothCollision(pmx, rig, 0.05)!;
    expect(cloth.colliders).toEqual([0]);
    expect(Array.from(cloth.cloth)).toEqual([0, 1, 1]);
    // Four points per vertex along its joint-to-vertex line; the last is the vertex itself.
    expect(cloth.clearance[1 * 4 + 3]).toBeCloseTo(-0.5); // may stay as deep as it rests
    expect(cloth.clearance[2 * 4 + 3]).toBeCloseTo(0.05); // elsewhere: the margin
    expect(cloth.anchorBone[2]).toBe(2);
    expect(cloth.masks[2]).toBe(1);

    // Collision groups that exclude each other: nothing to do.
    pmx.rigidBodies[1].collisionMask = 0xffff & ~1;
    expect(buildClothCollision(pmx, buildPhysicsRig(physicsModelData(pmx)), 0.05)).toBeNull();
  });
});
