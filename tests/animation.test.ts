import { describe, expect, it } from 'vitest';
import { bezierWeight, bezierWeightBytes } from '../src/animation/bezier';
import { MmdMotion } from '../src/animation/motion';
import { parseVmd } from '../src/loaders/vmd';
import { quat, vec3 } from '../src/math';

describe('bezier', () => {
  it('is linear for the default MMD curve', () => {
    for (const x of [0, 0.1, 0.5, 0.77, 1]) expect(bezierWeightBytes(20, 20, 107, 107, x)).toBeCloseTo(x, 6);
  });

  it('hits the endpoints and is monotonic for ease-in-out', () => {
    expect(bezierWeight(0.5, 0, 0.5, 1, 0)).toBe(0);
    expect(bezierWeight(0.5, 0, 0.5, 1, 1)).toBe(1);
    expect(bezierWeight(0.5, 0, 0.5, 1, 0.5)).toBeCloseTo(0.5, 4);
    let prev = 0;
    for (let x = 0.05; x < 1; x += 0.05) {
      const y = bezierWeight(0.5, 0, 0.5, 1, x);
      expect(y).toBeGreaterThanOrEqual(prev);
      prev = y;
    }
    expect(bezierWeight(0.5, 0, 0.5, 1, 0.1)).toBeLessThan(0.1); // eases in
  });
});

/** Build a minimal VMD with ASCII names (ASCII is valid Shift-JIS). */
function buildVmd(
  bones: { name: string; frame: number; pos: number[]; rot: number[]; interp?: number[] }[],
  morphs: { name: string; frame: number; weight: number }[],
): ArrayBuffer {
  const size = 30 + 20 + 4 + bones.length * 111 + 4 + morphs.length * 23 + 4 * 4;
  const buf = new ArrayBuffer(size);
  const u8 = new Uint8Array(buf);
  const dv = new DataView(buf);
  let p = 0;
  const str = (s: string, len: number) => {
    for (let i = 0; i < len; i++) u8[p + i] = i < s.length ? s.charCodeAt(i) : 0;
    p += len;
  };
  str('Vocaloid Motion Data 0002', 30);
  str('test', 20);
  dv.setUint32(p, bones.length, true); p += 4;
  for (const b of bones) {
    str(b.name, 15);
    dv.setUint32(p, b.frame, true); p += 4;
    for (const v of [...b.pos, ...b.rot]) { dv.setFloat32(p, v, true); p += 4; }
    const ip = b.interp ?? [20, 20, 20, 20, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107];
    for (let i = 0; i < 64; i++) u8[p + i] = ip[i % 16];
    p += 64;
  }
  dv.setUint32(p, morphs.length, true); p += 4;
  for (const m of morphs) {
    str(m.name, 15);
    dv.setUint32(p, m.frame, true); p += 4;
    dv.setFloat32(p, m.weight, true); p += 4;
  }
  // camera, light, self shadow, property: 0 each
  return buf;
}

describe('VMD + motion sampling', () => {
  const s = Math.SQRT1_2;
  const vmd = parseVmd(
    buildVmd(
      [
        { name: 'center', frame: 10, pos: [10, 0, 0], rot: [0, s, 0, s] }, // 90° about Y
        { name: 'center', frame: 0, pos: [0, 0, 0], rot: [0, 0, 0, 1] }, // out of order on purpose
      ],
      [
        { name: 'smile', frame: 0, weight: 0 },
        { name: 'smile', frame: 30, weight: 1 },
      ],
    ),
  );

  it('parses frames', () => {
    expect(vmd.modelName).toBe('test');
    expect(vmd.boneFrames).toHaveLength(2);
    expect(vmd.morphFrames).toHaveLength(2);
    expect(vmd.boneFrames[0].boneName).toBe('center');
    expect(vmd.cameraFrames).toHaveLength(0);
  });

  it('interpolates bones and morphs', () => {
    const motion = MmdMotion.fromVmd(vmd);
    expect(motion.lastFrame).toBe(30);
    const track = motion.boneTracks.get('center')!;
    const pos = vec3.create();
    const rot = quat.create();

    track.sample(5, pos, rot);
    expect(pos[0]).toBeCloseTo(5, 5);
    // Halfway slerp between identity and 90° about Y is 45° about Y.
    expect(rot[1]).toBeCloseTo(Math.sin(Math.PI / 8), 5);
    expect(rot[3]).toBeCloseTo(Math.cos(Math.PI / 8), 5);

    track.sample(-3, pos, rot);
    expect(pos[0]).toBe(0);
    track.sample(99, pos, rot);
    expect(pos[0]).toBe(10);

    expect(motion.morphTracks.get('smile')!.sample(15)).toBeCloseTo(0.5, 6);
  });
});

describe('camera motion', () => {
  const linear = new Uint8Array(Array.from({ length: 6 }, () => [20, 107, 20, 107]).flat());
  const key = (frame: number, x: number, distance: number) => ({
    frame, distance, position: [x, 10, 0] as [number, number, number], rotation: [0, 0, 0] as [number, number, number],
    interpolation: linear, fov: 30, perspective: true,
  });
  const motion = new MmdMotion();
  motion.merge({
    modelName: '', boneFrames: [], morphFrames: [], lightFrames: [], selfShadowFrames: [], propertyFrames: [],
    cameraFrames: [key(0, 0, -45), key(10, 10, -25), key(20, 10, -25), key(21, -10, -60)],
  });
  const pose = { target: vec3.create(), rotation: vec3.create(), distance: 0, fov: 0 };

  it('interpolates between keys', () => {
    const track = motion.cameraTrack!;
    expect(motion.lastFrame).toBe(21);
    track.sample(5, pose);
    expect(pose.target[0]).toBeCloseTo(5, 5);
    expect(pose.distance).toBeCloseTo(-35, 5);
  });

  it('treats keys one frame apart as a cut', () => {
    motion.cameraTrack!.sample(20.5, pose);
    expect(pose.target[0]).toBe(10);
    expect(pose.distance).toBe(-25);
    motion.cameraTrack!.sample(21, pose);
    expect(pose.target[0]).toBe(-10);
  });
});
