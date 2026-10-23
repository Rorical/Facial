import { describe, expect, it } from 'vitest';
import type { Vec3Tuple } from '../src/loaders/binary-reader';
import { BoneFlag, type PmxBone, type PmxIk } from '../src/loaders/pmx';
import { mat4, quat, vec3 } from '../src/math';
import { decomposeXYZ } from '../src/runtime/ik-solver';
import { Skeleton } from '../src/runtime/skeleton';

function bone(name: string, position: Vec3Tuple, parentIndex: number, extra: Partial<PmxBone> = {}): PmxBone {
  return {
    name, nameEn: '', position, parentIndex, layer: 0,
    flags: BoneFlag.Rotatable | BoneFlag.Visible | BoneFlag.Enabled,
    tailBoneIndex: -1, tailOffset: [0, 0, 0], inherit: null, fixedAxis: null, localAxes: null,
    externalParentKey: -1, ik: null,
    ...extra,
  };
}

function worldPos(s: Skeleton, name: string) {
  return mat4.getTranslation(vec3.create(), s.byName.get(name)!.world);
}

/** A single MMD-style leg: thigh → knee → ankle, driven by a leg IK bone. */
function makeLeg(): Skeleton {
  const ik: PmxIk = {
    targetIndex: 3,
    loopCount: 40,
    limitAngle: 2,
    links: [
      { boneIndex: 2, limits: { min: [-Math.PI, 0, 0], max: [-0.008726646, 0, 0] } }, // knee: X only, bends one way
      { boneIndex: 1, limits: null },
    ],
  };
  return new Skeleton([
    bone('root', [0, 0, 0], -1),
    bone('thigh', [0, 10, 0], 0),
    bone('knee', [0, 5, 0], 1),
    bone('ankle', [0, 0, 0], 2),
    bone('legIK', [0, 0, 0], 0, { flags: BoneFlag.Rotatable | BoneFlag.Translatable | BoneFlag.IK, ik }),
  ]);
}

describe('IK', () => {
  it('rest pose stays put', () => {
    const s = makeLeg();
    s.update();
    const a = worldPos(s, 'ankle');
    expect(vec3.length(a)).toBeLessThan(1e-4);
  });

  it('moves the ankle to the IK goal with the knee bending forward (-Z)', () => {
    const s = makeLeg();
    vec3.set(s.byName.get('legIK')!.animPosition, 0, 3, -1);
    s.update();
    const ankle = worldPos(s, 'ankle');
    expect(vec3.distance(ankle, vec3.create(0, 3, -1))).toBeLessThan(0.05);
    // Model faces -Z, so a correctly bent knee points toward -Z.
    expect(worldPos(s, 'knee')[2]).toBeLessThan(0);
    // Bone lengths are preserved.
    expect(vec3.distance(worldPos(s, 'thigh'), worldPos(s, 'knee'))).toBeCloseTo(5, 4);
    expect(vec3.distance(worldPos(s, 'knee'), ankle)).toBeCloseTo(5, 4);
  });

  it('does nothing when disabled', () => {
    const s = makeLeg();
    vec3.set(s.byName.get('legIK')!.animPosition, 0, 3, -1);
    s.ikSolvers[0].enabled = false;
    s.update();
    expect(vec3.length(worldPos(s, 'ankle'))).toBeLessThan(1e-4);
  });
});

describe('inherit (付与) bones', () => {
  it('inherits a weighted rotation', () => {
    const s = new Skeleton([
      bone('root', [0, 0, 0], -1),
      bone('src', [0, 1, 0], 0),
      bone('dst', [1, 1, 0], 0, {
        flags: BoneFlag.Rotatable | BoneFlag.InheritRotation,
        inherit: { parentIndex: 1, weight: 0.5 },
      }),
    ]);
    quat.fromAxisAngle(s.byName.get('src')!.animRotation, [0, 0, 1], Math.PI / 2);
    s.update();
    const q = quat.fromMat4(quat.create(), s.byName.get('dst')!.world);
    expect(q[2]).toBeCloseTo(Math.sin(Math.PI / 8), 5); // 45° about Z
    expect(q[3]).toBeCloseTo(Math.cos(Math.PI / 8), 5);
  });
});

describe('skinning matrices', () => {
  it('are identity in the rest pose', () => {
    const s = makeLeg();
    s.update();
    const out = new Float32Array(s.bones.length * 20);
    s.writeSkinning(out);
    for (let b = 0; b < s.bones.length; b++) {
      const m = out.subarray(b * 20, b * 20 + 16);
      const identity = mat4.create();
      for (let k = 0; k < 16; k++) expect(m[k]).toBeCloseTo(identity[k], 5);
      expect(out[b * 20 + 19]).toBeCloseTo(1, 5); // identity quaternion w
    }
  });
});

describe('decomposeXYZ', () => {
  it('round-trips Euler angles', () => {
    for (const [x, y, z] of [[0.3, -0.2, 1.1], [-1.2, 0.7, -0.4], [0, 0, 0], [1.5, -1.4, 0.2]]) {
      const q = quat.fromEulerXYZ(quat.create(), x, y, z);
      const e = decomposeXYZ(q, vec3.create(x, y, z));
      expect(e[0]).toBeCloseTo(x, 5);
      expect(e[1]).toBeCloseTo(y, 5);
      expect(e[2]).toBeCloseTo(z, 5);
    }
  });
});

describe('MMD camera pose', () => {
  it('places the default camera in front of the model (-Z)', async () => {
    const { OrbitCamera } = await import('../src/renderer/camera');
    const cam = new OrbitCamera();
    cam.pose = { target: vec3.create(0, 10, 0), rotation: vec3.create(0, 0, 0), distance: -45, fov: 30 };
    cam.update(1);
    expect(Array.from(cam.eye).map((v) => Math.round(v * 1e4) / 1e4)).toEqual([0, 10, -45]);
    // Target is straight ahead: view-space position (0, 0, 45).
    const t = vec3.transformMat4(vec3.create(), vec3.create(0, 10, 0), cam.view);
    expect(t[0]).toBeCloseTo(0, 5);
    expect(t[1]).toBeCloseTo(0, 5);
    expect(t[2]).toBeCloseTo(45, 5);
  });
});
