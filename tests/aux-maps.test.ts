import { describe, expect, it } from 'vitest';
import { bakeVertexOcclusion, generateRamp, occlusionToLightMaps, TriangleBvh } from '../src/bake/aux-maps';
import type { PmxMaterial, PmxModel } from '../src/loaders/pmx';

/** Two unit quads: a floor at y = 0 and, optionally, a lid just above it. */
function quads(withLid: boolean): PmxModel {
  const quad = (y: number, ny: number) => ({
    positions: [-1, y, -1, 1, y, -1, 1, y, 1, -1, y, 1],
    normals: [0, ny, 0, 0, ny, 0, 0, ny, 0, 0, ny, 0],
    uvs: [0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9],
  });
  const parts = withLid ? [quad(0, 1), quad(0.3, -1)] : [quad(0, 1)];
  const material = (name: string): PmxMaterial => ({
    name, nameEn: '', diffuse: [1, 1, 1, 1], specular: [0, 0, 0], specularPower: 0, ambient: [0, 0, 0], flags: 0,
    edgeColor: [0, 0, 0, 1], edgeSize: 0, textureIndex: 0, sphereTextureIndex: -1, sphereMode: 0, sharedToon: false, toonIndex: -1,
    memo: '', indexCount: 6,
  });
  return {
    vertexCount: parts.length * 4,
    positions: Float32Array.from(parts.flatMap((p) => p.positions)),
    normals: Float32Array.from(parts.flatMap((p) => p.normals)),
    uvs: Float32Array.from(parts.flatMap((p) => p.uvs)),
    indices: Uint32Array.from(parts.flatMap((_, i) => [0, 1, 2, 0, 2, 3].map((k) => k + i * 4))),
    textures: ['a.png'],
    materials: parts.map((_, i) => material(`m${i}`)),
    bones: [],
  } as unknown as PmxModel;
}

const bvh2 = (m: PmxModel) => new TriangleBvh(m.positions, m.indices);

describe('auxiliary map baking', () => {
  it('finds ray hits with the BVH', () => {
    const m = quads(false);
    const bvh = new TriangleBvh(m.positions, m.indices);
    expect(bvh.occluded(0, 1, 0, 0, -1, 0, 5)).toBe(true);
    expect(bvh.occluded(0, 1, 0, 0, -1, 0, 0.5)).toBe(false); // too short
    expect(bvh.occluded(3, 1, 0, 0, -1, 0, 5)).toBe(false); // beside the quad
    expect(bvh.occluded(0, 1, 0, 0, 1, 0, 5)).toBe(false); // pointing away
  });

  it('bakes open surfaces bright and covered ones dark', async () => {
    const open = await bakeVertexOcclusion(quads(false));
    const openness = (a: Float32Array) => [0, 1, 2, 3].map((v) => a[v * 2]);
    expect(Math.min(...openness(open))).toBeGreaterThan(0.95);
    const covered = await bakeVertexOcclusion(quads(true));
    // Corners of the floor under a lid 0.3 above it: much of the sky is blocked.
    expect(Math.max(...openness(covered))).toBeLessThan(0.8);
    // Thinness: nothing behind the lone floor (thick by convention); the lid's underside has the floor 0.3 behind it… in front, not behind.
    expect(open[1]).toBeCloseTo(0);
    expect(bvh2(quads(true)).nearest(0, 1, 0, 0, -1, 0, 5)).toBeCloseTo(0.7);
  });

  it('draws vertex values into the UV layout, open where nothing is mapped', () => {
    const m = quads(false);
    const map = occlusionToLightMaps(m, Float32Array.from([0, 1, 0, 1, 0, 1, 0, 1])).get(0)!;
    const at = (u: number, v: number) => map.data[(Math.floor(v * map.height) * map.width + Math.floor(u * map.width)) * 4];
    expect(at(0.5, 0.5)).toBe(0);
    expect(map.data[(Math.floor(0.5 * map.height) * map.width + Math.floor(0.5 * map.width)) * 4 + 1]).toBe(255); // thinness in green
    expect(at(0.02, 0.02)).toBe(255);
    expect(map.data[3]).toBe(255);
  });

  it('generates a ramp from shadow colour to white', () => {
    const ramp = generateRamp();
    expect(ramp.height).toBe(4);
    const px = (row: number, x: number) => Array.from(ramp.data.subarray((row * ramp.width + x) * 4, (row * ramp.width + x) * 4 + 3));
    expect(px(0, 255)).toEqual([255, 255, 255]);
    expect(px(0, 0)).toEqual([179, 173, 217]); // cloth shadow 0.70, 0.68, 0.85
    expect(px(1, 0)[0]).toBe(255); // skin shadow keeps its red
  });
});
