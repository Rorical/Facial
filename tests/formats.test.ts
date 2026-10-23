import { describe, expect, it } from 'vitest';
import { decodeDdsToRgba, parseDds } from '../src/loaders/image-decoders';
import { parsePmd } from '../src/loaders/pmd';
import { writePmx } from '../src/io/pmx-writer';
import { parsePmx } from '../src/loaders/pmx';
import { BoneFlag, MaterialFlag, MorphKind, SphereMode } from '../src/loaders/pmx';
import { mat4, vec3 } from '../src/math';
import { Skeleton } from '../src/runtime/skeleton';

/** Little-endian byte writer for building test files. */
class Writer {
  private bytes: number[] = [];
  private scratch = new DataView(new ArrayBuffer(4));
  u8(v: number) { this.bytes.push(v & 255); return this; }
  u16(v: number) { this.scratch.setUint16(0, v, true); this.bytes.push(...new Uint8Array(this.scratch.buffer, 0, 2)); return this; }
  u32(v: number) { this.scratch.setUint32(0, v, true); this.bytes.push(...new Uint8Array(this.scratch.buffer, 0, 4)); return this; }
  f32(...vs: number[]) { for (const v of vs) { this.scratch.setFloat32(0, v, true); this.bytes.push(...new Uint8Array(this.scratch.buffer, 0, 4)); } return this; }
  /** Fixed-length field from raw (Shift-JIS) bytes or an ASCII string. */
  str(s: string | number[], len: number) {
    const b = typeof s === 'string' ? Array.from(s, (c) => c.charCodeAt(0)) : s;
    for (let i = 0; i < len; i++) this.bytes.push(i < b.length ? b[i] : 0);
    return this;
  }
  buffer(): ArrayBuffer { return new Uint8Array(this.bytes).buffer; }
}

const SJIS = {
  center: [131, 90, 131, 147, 131, 94, 129, 91],
  leg: [141, 182, 145, 171],
  knee: [141, 182, 130, 208, 130, 180],
  ankle: [141, 182, 145, 171, 142, 241],
  legIK: [141, 182, 145, 171, 130, 104, 130, 106],
  a: [130, 160],
};

function buildPmd(): ArrayBuffer {
  const w = new Writer();
  w.str('Pmd', 3).f32(1).str('test model', 20).str('comment', 256);
  // 3 vertices
  w.u32(3);
  const verts = [[0, 0, 0], [1, 0, 0], [0, 1, 0]];
  verts.forEach((p, i) => w.f32(...p, 0, 0, -1, 0, 0).u16(1).u16(2).u8(i === 0 ? 100 : 40).u8(i === 2 ? 1 : 0));
  // faces
  w.u32(3).u16(0).u16(1).u16(2);
  // 1 material: diffuse rgba, power, specular, ambient, toon 0 (toon01), edge 1, 3 indices, textures
  w.u32(1).f32(1, 1, 1, 0.5).f32(5).f32(0.1, 0.1, 0.1).f32(0.5, 0.5, 0.5).u8(0).u8(1).u32(3).str('tex\\a.bmp*b.spa', 20);
  // bones: センター(1), 左足(0), 左ひざ(0), 左足首(0), 左足ＩＫ(2)
  const bones: [number[], number, number, number[]][] = [
    [SJIS.center, 0xffff, 1, [0, 8, 0]],
    [SJIS.leg, 0, 0, [0, 10, 0]],
    [SJIS.knee, 1, 0, [0, 5, 0]],
    [SJIS.ankle, 2, 0, [0, 0, 0]],
    [SJIS.legIK, 0xffff, 2, [0, 0, 0]],
  ];
  w.u16(bones.length);
  for (const [name, parent, type, pos] of bones) w.str(name, 20).u16(parent).u16(0).u8(type).u16(0).f32(...pos);
  // IK: bone 4 → target 3, chain [2, 1], 40 iterations, limit 0.5 (×4 = 2 rad)
  w.u16(1).u16(4).u16(3).u8(2).u16(40).f32(0.5).u16(2).u16(1);
  // morphs: base (vertices 1, 2) + "あ" moving base entry 1 (= vertex 2)
  w.u16(2);
  w.str('base', 20).u32(2).u8(0).u32(1).f32(1, 0, 0).u32(2).f32(0, 1, 0);
  w.str(SJIS.a, 20).u32(1).u8(3).u32(1).f32(0, 0.5, 0);
  // display lists: none
  w.u8(0).u8(0).u32(0);
  // no English names
  w.u8(0);
  // toon list
  for (let i = 0; i < 10; i++) w.str(`toon${String(i + 1).padStart(2, '0')}.bmp`, 100);
  // 1 rigid body on 左足, relative position (0, 1, 0)
  w.u32(1).str('rb', 20).u16(1).u8(0).u16(0xffff).u8(0).f32(1, 1, 1).f32(0, 1, 0).f32(0, 0, 0).f32(1, 0.5, 0.5, 0, 0.5).u8(1);
  w.u32(0); // joints
  return w.buffer();
}

describe('PMD', () => {
  const pmd = parsePmd(buildPmd());

  it('converts vertices, faces and materials', () => {
    expect(pmd.format).toBe('PMD');
    expect(pmd.name).toBe('test model');
    expect(pmd.vertexCount).toBe(3);
    expect(Array.from(pmd.indices)).toEqual([0, 1, 2]);
    expect(Array.from(pmd.skinWeights.subarray(4, 6))).toEqual([expect.closeTo(0.4), expect.closeTo(0.6)]);
    expect(Array.from(pmd.edgeScales)).toEqual([1, 1, 0]);
    const m = pmd.materials[0];
    expect(pmd.textures[m.textureIndex]).toBe('tex\\a.bmp');
    expect(pmd.textures[m.sphereTextureIndex]).toBe('b.spa');
    expect(m.sphereMode).toBe(SphereMode.Add);
    expect(m.sharedToon).toBe(true);
    expect(m.toonIndex).toBe(0);
    expect(m.flags & MaterialFlag.DoubleSided).toBeTruthy(); // alpha 0.5
    expect(m.flags & MaterialFlag.Edge).toBeTruthy();
  });

  it('converts bones and IK with MMD rules', () => {
    expect(pmd.bones.map((b) => b.name)).toEqual(['センター', '左足', '左ひざ', '左足首', '左足ＩＫ']);
    expect(pmd.bones[0].flags & BoneFlag.Translatable).toBeTruthy();
    const ik = pmd.bones[4].ik!;
    expect(ik.targetIndex).toBe(3);
    expect(ik.limitAngle).toBeCloseTo(2);
    expect(ik.links[0].limits).not.toBeNull(); // knee gets the implicit limit
    expect(ik.links[1].limits).toBeNull();

    // The converted rig actually solves.
    const s = new Skeleton(pmd.bones);
    vec3.set(s.byName.get('左足ＩＫ')!.animPosition, 0, 3, -1);
    s.update();
    const ankle = mat4.getTranslation(vec3.create(), s.byName.get('左足首')!.world);
    expect(vec3.distance(ankle, vec3.create(0, 3, -1))).toBeLessThan(0.05);
  });

  it('converts to a valid PMX (PMD → PMX export)', () => {
    const pmx = parsePmx(writePmx(pmd));
    expect(pmx.format).toBe('PMX');
    expect(pmx.positions).toEqual(pmd.positions);
    expect(pmx.materials).toEqual(pmd.materials);
    expect(pmx.bones).toEqual(pmd.bones);
    expect(pmx.morphs).toEqual(pmd.morphs);
    expect(pmx.rigidBodies).toEqual(pmd.rigidBodies);
    // MMD's two special display frames are added for PMD sources.
    expect(pmx.displayFrames.slice(0, 2).map((f) => [f.name, f.special])).toEqual([['Root', true], ['表情', true]]);
    expect(pmx.displayFrames[1].items).toEqual([{ type: 'morph', index: 0 }]);
  });

  it('maps morphs through the base morph and places rigid bodies', () => {
    expect(pmd.morphs).toHaveLength(1);
    const m = pmd.morphs[0];
    expect(m.name).toBe('あ');
    expect(m.kind).toBe(MorphKind.Vertex);
    if (m.kind === MorphKind.Vertex) expect(Array.from(m.vertexIndices)).toEqual([2]);
    expect(pmd.rigidBodies[0].position).toEqual([0, 11, 0]); // bone (0,10,0) + (0,1,0)
  });
});

function ddsHeader(w: Writer, width: number, height: number, mips: number, fourCC: string | null, rgb?: { bits: number; masks: number[] }) {
  w.str('DDS ', 4).u32(124).u32(0x1007 | (mips > 1 ? 0x20000 : 0)).u32(height).u32(width).u32(0).u32(0).u32(mips);
  for (let i = 0; i < 11; i++) w.u32(0);
  w.u32(32).u32(fourCC ? 0x4 : 0x41).str(fourCC ?? '', 4).u32(rgb?.bits ?? 0);
  for (const m of rgb?.masks ?? [0, 0, 0, 0]) w.u32(m);
  w.u32(0x1000).u32(0).u32(0).u32(0).u32(0);
}

describe('DDS', () => {
  it('decodes DXT1 (4-color and 1-bit alpha blocks)', () => {
    const w = new Writer();
    ddsHeader(w, 8, 4, 1, 'DXT1');
    // Block 1: c0 = red (0xF800) > c1 = blue (0x001F); all pixels index 0 → red
    w.u16(0xf800).u16(0x001f).u32(0);
    // Block 2: c0 = blue < c1 = red → 3-color mode; all pixels index 3 → transparent
    w.u16(0x001f).u16(0xf800).u32(0xffffffff);
    const dds = parseDds(w.buffer());
    expect(dds.format).toBe('bc1');
    const img = decodeDdsToRgba(dds);
    expect(Array.from(img.data.subarray(0, 4))).toEqual([255, 0, 0, 255]);
    expect(img.data[(0 * 8 + 4) * 4 + 3]).toBe(0);
  });

  it('decodes DXT5 alpha', () => {
    const w = new Writer();
    ddsHeader(w, 4, 4, 1, 'DXT5');
    // alpha: a0 = 255, a1 = 0, all indices 1 → 0; color: white
    const alphaBits = [0b01001001, 0b10010010, 0b00100100, 0b01001001, 0b10010010, 0b00100100]; // index 1 everywhere
    w.u8(255).u8(0);
    alphaBits.forEach((b) => w.u8(b));
    w.u16(0xffff).u16(0xffff).u32(0);
    const img = decodeDdsToRgba(parseDds(w.buffer()));
    expect(Array.from(img.data.subarray(0, 4))).toEqual([255, 255, 255, 0]);
    expect(img.data[15 * 4 + 3]).toBe(0);
  });

  it('reads an uncompressed BGRA mip chain', () => {
    const w = new Writer();
    ddsHeader(w, 2, 2, 2, null, { bits: 32, masks: [0xff0000, 0xff00, 0xff, 0xff000000] });
    for (let i = 0; i < 4; i++) w.u8(0x30).u8(0x20).u8(0x10).u8(0x80); // B G R A
    w.u8(1).u8(2).u8(3).u8(4); // 1×1 mip
    const dds = parseDds(w.buffer());
    expect(dds.format).toBe('rgba');
    expect(dds.levels.map((l) => [l.width, l.height])).toEqual([[2, 2], [1, 1]]);
    expect(Array.from(dds.levels[0].data.subarray(0, 4))).toEqual([0x10, 0x20, 0x30, 0x80]);
  });
});
