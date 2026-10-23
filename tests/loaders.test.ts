import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { normalizePath } from '../src/loaders/asset-source';
import { decodeBmp, decodeTga } from '../src/loaders/image-decoders';
import { MorphKind, parsePmx } from '../src/loaders/pmx';

// Optional real-model checks: point MMD_SAMPLE_DIR at a folder containing a .pmx
// (and its textures). Skipped when absent.
const SAMPLE_DIR = process.env.MMD_SAMPLE_DIR ?? '';
const hasSample = SAMPLE_DIR !== '' && existsSync(SAMPLE_DIR);
const sampleFiles = hasSample ? readdirSync(SAMPLE_DIR) : [];
const toArrayBuffer = (b: Buffer) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;

describe('normalizePath', () => {
  it('handles backslashes, dot segments, case and NFD', () => {
    expect(normalizePath('Model\\Tex\\..\\Face.PNG')).toBe('model/face.png');
    expect(normalizePath('./a//b/./c.bmp')).toBe('a/b/c.bmp');
    expect(normalizePath('テクスチャ/ト'.normalize('NFD'))).toBe(normalizePath('テクスチャ/ト'));
  });
});

describe('image decoders', () => {
  it('decodes an uncompressed bottom-up 24-bit TGA', () => {
    // 2×1 image: header (18 bytes) + BGR pixels, bottom-up (descriptor 0)
    const bytes = new Uint8Array([0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 24, 0, 255, 0, 0, 0, 0, 255]);
    const img = decodeTga(bytes.buffer);
    expect([img.width, img.height]).toEqual([2, 1]);
    expect(Array.from(img.data)).toEqual([0, 0, 255, 255, 255, 0, 0, 255]);
  });

  it('decodes an RLE TGA', () => {
    // 3×1, one run packet of 3 red pixels (32-bit, 8 alpha bits, top-down)
    const bytes = new Uint8Array([0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 1, 0, 32, 0x28, 0x82, 0, 0, 255, 128]);
    const img = decodeTga(bytes.buffer);
    expect(Array.from(img.data)).toEqual([255, 0, 0, 128, 255, 0, 0, 128, 255, 0, 0, 128]);
  });

  it.skipIf(!hasSample)('decodes sample TGA/BMP files', () => {
    for (const f of sampleFiles.filter((n) => /\.(tga|bmp)$/i.test(n))) {
      const buf = toArrayBuffer(readFileSync(join(SAMPLE_DIR, f)));
      const img = /\.tga$/i.test(f) ? decodeTga(buf) : decodeBmp(buf);
      expect(img.width, f).toBeGreaterThan(0);
      expect(img.data.length, f).toBe(img.width * img.height * 4);
    }
  });
});

describe.skipIf(!hasSample)('PMX sample', () => {
  const pmxName = sampleFiles.find((n) => /\.pmx$/i.test(n));
  const pmx = pmxName ? parsePmx(toArrayBuffer(readFileSync(join(SAMPLE_DIR, pmxName)))) : null;

  it('parses a consistent model', () => {
    expect(pmx).not.toBeNull();
    const m = pmx!;
    expect(m.vertexCount).toBeGreaterThan(0);
    expect(m.indices.length % 3).toBe(0);
    expect(m.indices.reduce((a, b) => Math.max(a, b), 0)).toBeLessThan(m.vertexCount);
    expect(m.materials.reduce((s, mat) => s + mat.indexCount, 0)).toBe(m.indices.length);
    for (const b of m.bones) expect(b.parentIndex).toBeLessThan(m.bones.length);
    expect(m.skinIndices.reduce((a, b) => Math.max(a, b), 0)).toBeLessThan(m.bones.length);
    for (const morph of m.morphs) {
      if (morph.kind === MorphKind.Vertex) for (const v of morph.vertexIndices) expect(v).toBeLessThan(m.vertexCount);
    }
    console.info(
      `${m.name}: ${m.vertexCount} verts, ${m.indices.length / 3} tris, ${m.materials.length} materials, ` +
        `${m.bones.length} bones (${m.bones.filter((b) => b.ik).length} IK), ${m.morphs.length} morphs, ` +
        `${m.rigidBodies.length} rigid bodies, ${m.joints.length} joints, ${m.sdef.vertexIndices.length} SDEF verts`,
    );
    console.info('textures:', m.textures);
  });
});
