import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { subCurve, bezierWeight } from '../src/animation/bezier';
import { MmdMotion } from '../src/animation/motion';
import { FileSet } from '../src/loaders/asset-source';
import { parsePmx } from '../src/loaders/pmx';
import { parseVmd, type VmdMotion } from '../src/loaders/vmd';
import { writePmx } from '../src/io/pmx-writer';
import { encodeSjis } from '../src/io/sjis';
import { motionToVmd } from '../src/io/vmd-export';
import { CAMERA_VMD_MODEL_NAME, writeVmd } from '../src/io/vmd-writer';
import { readZip, writeZip } from '../src/io/zip';
import { quat, vec3 } from '../src/math';

const SAMPLE_DIR = process.env.MMD_SAMPLE_DIR ?? '';
const pmxName = SAMPLE_DIR && existsSync(SAMPLE_DIR) ? readdirSync(SAMPLE_DIR).find((n) => /\.pmx$/i.test(n)) : undefined;
const toArrayBuffer = (b: Buffer) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;

describe('Shift-JIS encoder', () => {
  it('round-trips Japanese names through the platform decoder', () => {
    for (const s of ['センター', '左足ＩＫ', 'まばたき', 'ｳｨﾝｸ２右', 'カメラ・照明', 'abc_123']) {
      const r = encodeSjis(s);
      expect(r.unmappable).toEqual([]);
      expect(new TextDecoder('shift-jis').decode(r.bytes)).toBe(s);
    }
  });

  it('never splits a double-byte character when truncating', () => {
    const r = encodeSjis('左足首左足首左足首', 15); // 2 bytes each → 7 chars = 14 bytes
    expect(r.bytes.length).toBe(14);
    expect(r.truncated).toBe(true);
  });

  it('reports characters without a Shift-JIS code', () => {
    expect(encodeSjis('头发').unmappable.length).toBeGreaterThan(0);
  });

  it('substitutes look-alikes (em dash → ―)', () => {
    const r = encodeSjis('星—');
    expect(r.unmappable).toEqual([]);
    expect(new TextDecoder('shift-jis').decode(r.bytes)).toBe('星―');
  });
});

describe('zip', () => {
  it('round-trips stored and deflated entries with Unicode names', async () => {
    const text = new TextEncoder().encode('ほげ'.repeat(500));
    const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 1, 2, 3]);
    const blob = await writeZip([
      { path: 'モデル/model.pmx', data: text },
      { path: 'モデル/tex/face.png', data: png },
    ]);
    const entries = await readZip(blob);
    expect(entries.map((e) => e.path)).toEqual(['モデル/model.pmx', 'モデル/tex/face.png']);
    expect(await entries[0].read()).toEqual(text);
    expect(await entries[1].read()).toEqual(png);
    // The text compresses, so the archive is smaller than its contents.
    expect(blob.size).toBeLessThan(text.length);
  });

  it('exposes a zip as a FileSet with case-insensitive texture lookup', async () => {
    const blob = await writeZip([{ path: 'Miku/Tex/Face.PNG', data: new Uint8Array([1, 2, 3]) }]);
    const set = await FileSet.fromZip(blob);
    const found = await set.resolve('miku', 'tex\\face.png');
    expect(found).not.toBeNull();
    expect(new Uint8Array(await found!.arrayBuffer())).toEqual(new Uint8Array([1, 2, 3]));
  });
});

// ---------------------------------------------------------------------------------------

const ease = (): Uint8Array => {
  const ip = new Uint8Array(64);
  // X: ease-in-out, Y: linear, Z: fast-in, R: ease-out; rows 1–3 filled with junk to check verbatim copy
  const base = [64, 20, 0, 20, 0, 20, 0, 0, 64, 107, 127, 107, 127, 107, 127, 127];
  ip.set(base, 0);
  for (let i = 16; i < 64; i++) ip[i] = i;
  return ip;
};

function sampleVmd(): VmdMotion {
  const q = (deg: number) => Array.from(quat.fromAxisAngle(quat.create(), [0, 1, 0], (deg * Math.PI) / 180)) as [number, number, number, number];
  const cam = (frame: number, x: number, dist: number) => ({
    frame, distance: dist, position: [x, 10, 0] as [number, number, number], rotation: [0.1, 0.2 * x, 0] as [number, number, number],
    interpolation: Uint8Array.from({ length: 24 }, (_, i) => [20, 107, 20, 107, 64, 64, 0, 127][i % 8]), fov: 30, perspective: true,
  });
  return {
    modelName: 'テスト',
    boneFrames: [
      { boneName: 'センター', frame: 0, position: [0, 0, 0], rotation: q(0), interpolation: ease() },
      { boneName: 'センター', frame: 30, position: [10, 5, -3], rotation: q(90), interpolation: ease() },
      { boneName: 'センター', frame: 60, position: [0, 0, 0], rotation: q(0), interpolation: ease() },
      { boneName: '左足ＩＫ', frame: 10, position: [0, 1, 0], rotation: q(0), interpolation: ease() },
    ],
    morphFrames: [
      { morphName: 'あ', frame: 0, weight: 0 },
      { morphName: 'あ', frame: 40, weight: 1 },
    ],
    cameraFrames: [cam(0, 0, -45), cam(50, 4, -30), cam(51, -4, -20)],
    lightFrames: [{ frame: 0, color: [0.6, 0.6, 0.6], direction: [-0.5, -1, 0.5] }],
    selfShadowFrames: [{ frame: 0, mode: 1, distance: 0.0875 }],
    propertyFrames: [{ frame: 0, visible: true, ikStates: [{ name: '左足ＩＫ', enabled: true }] }, { frame: 20, visible: true, ikStates: [{ name: '左足ＩＫ', enabled: false }] }],
  };
}

describe('VMD writer', () => {
  it('round-trips a VMD byte-exactly through parse → write', () => {
    const first = writeVmd(sampleVmd());
    expect(first.warnings).toEqual([]);
    const reparsed = parseVmd(first.buffer);
    expect(reparsed.modelName).toBe('テスト');
    expect(reparsed.boneFrames[1].interpolation).toEqual(ease());
    const second = writeVmd(reparsed);
    expect(new Uint8Array(second.buffer)).toEqual(new Uint8Array(first.buffer));
  });

  it('exports a motion losslessly and splits model vs camera parts', () => {
    const motion = MmdMotion.fromVmd(sampleVmd());
    const body = motionToVmd(motion, 'モデル', { model: true, scene: false });
    const camera = motionToVmd(motion, CAMERA_VMD_MODEL_NAME, { model: false, scene: true });
    expect(body.boneFrames).toHaveLength(4);
    expect(body.cameraFrames).toHaveLength(0);
    expect(camera.boneFrames).toHaveLength(0);
    expect(camera.cameraFrames).toHaveLength(3);
    expect(camera.lightFrames).toHaveLength(1);
    // Interpolation bytes survive verbatim, all 64 of them.
    expect(body.boneFrames.find((k) => k.frame === 30)!.interpolation).toEqual(ease());
    // Re-imported motion samples identically.
    const again = MmdMotion.fromVmd(parseVmd(writeVmd(body).buffer));
    const [p1, r1, p2, r2] = [vec3.create(), quat.create(), vec3.create(), quat.create()];
    for (const f of [0, 7.5, 15, 29, 45.25]) {
      motion.boneTracks.get('センター')!.sample(f, p1, r1);
      again.boneTracks.get('センター')!.sample(f, p2, r2);
      for (let c = 0; c < 3; c++) expect(p2[c]).toBeCloseTo(p1[c], 4);
    }
  });

  it('trims to a range: sampled pose is preserved (curves are cut exactly) and frames rebase', () => {
    const motion = MmdMotion.fromVmd(sampleVmd());
    const range = { from: 12, to: 47, rebase: true };
    const cut = MmdMotion.fromVmd(parseVmd(writeVmd(motionToVmd(motion, 'm', { model: true, scene: true }, range)).buffer));
    expect(cut.lastFrame).toBe(35);
    const [p1, r1, p2, r2] = [vec3.create(), quat.create(), vec3.create(), quat.create()];
    for (let f = 12; f <= 47; f += 0.5) {
      motion.boneTracks.get('センター')!.sample(f, p1, r1);
      cut.boneTracks.get('センター')!.sample(f - 12, p2, r2);
      // Byte quantization of the cut curves: well under 1% of the 10-unit move.
      for (let c = 0; c < 3; c++) expect(Math.abs(p2[c] - p1[c])).toBeLessThan(0.08);
      expect(Math.abs(quat.multiply(quat.create(), r1, quat.conjugate(quat.create(), r2))[3])).toBeGreaterThan(0.9995);
      expect(cut.morphTracks.get('あ')!.sample(f - 12)).toBeCloseTo(motion.morphTracks.get('あ')!.sample(f), 5);
    }
    // IK state in effect at the start of the range is carried to frame 0.
    expect(cut.ikStatesAt(0)?.get('左足ＩＫ')).toBe(true);
    expect(cut.ikStatesAt(10)?.get('左足ＩＫ')).toBe(false);
  });

  it('warns about names Shift-JIS cannot hold', () => {
    const vmd = sampleVmd();
    vmd.boneFrames[0].boneName = '头';
    expect(writeVmd(vmd).warnings.length).toBe(1);
  });
});

describe('Bézier sub-curves', () => {
  it('match the original curve on the cut interval', () => {
    const curve: [number, number, number, number] = [0.6, 0.05, 0.3, 0.95];
    const [xa, xb] = [0.2, 0.7];
    const sub = subCurve(curve, xa, xb);
    const ya = bezierWeight(...curve, xa);
    const yb = bezierWeight(...curve, xb);
    for (let x = xa; x <= xb; x += 0.05) {
      const expected = bezierWeight(...curve, x);
      const got = ya + (yb - ya) * bezierWeight(...sub, (x - xa) / (xb - xa));
      expect(got).toBeCloseTo(expected, 5);
    }
  });
});

describe('PMX writer', () => {
  it.skipIf(!pmxName)('round-trips the sample model', () => {
    const original = parsePmx(toArrayBuffer(readFileSync(join(SAMPLE_DIR, pmxName!))));
    const again = parsePmx(writePmx(original));
    expect(again.vertexCount).toBe(original.vertexCount);
    expect(again.positions).toEqual(original.positions);
    expect(again.skinWeights).toEqual(original.skinWeights);
    expect(again.indices).toEqual(original.indices);
    expect(again.textures).toEqual(original.textures);
    expect(again.materials).toEqual(original.materials);
    expect(again.bones).toEqual(original.bones);
    expect(again.morphs).toEqual(original.morphs);
    expect(again.rigidBodies).toEqual(original.rigidBodies);
    expect(again.joints).toEqual(original.joints);
  });
});

describe('names cut by the 15-byte VMD field', () => {
  it('keeps half a character and writes it back byte-for-byte', async () => {
    const { decodeSjisName } = await import('../src/loaders/binary-reader');
    const { encodeSjis, vmdName } = await import('../src/io/sjis');
    const decoder = new TextDecoder('shift-jis');
    // "Bone_Piao020_R" + the first byte of a double-byte character: exactly 15 bytes.
    const bytes = Uint8Array.from([...new TextEncoder().encode('Bone_Piao020_R'), 0x89]);
    const name = decodeSjisName(bytes, decoder);
    expect(name.startsWith('Bone_Piao020_R')).toBe(true);
    expect(name).not.toContain('\ufffd');
    const back = encodeSjis(name, 15);
    expect(Array.from(back.bytes)).toEqual(Array.from(bytes));
    expect(back.unmappable).toEqual([]);

    // A model name longer than 15 bytes matches the track by its cut form.
    const full = 'Bone_Piao020_R右'; // 右 = 0x89 0x45 in Shift-JIS
    expect(vmdName(full)).toBe(name);
    expect(vmdName('センター')).toBe('センター');
  });
});
