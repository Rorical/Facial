import { describe, expect, it } from 'vitest';
import type { ClothMesh } from '../src/cloth/cloth-mesh';
import { ClothSim, clothVertexData } from '../src/cloth/cloth-sim';

/** A square sheet of n × n particles in the XZ plane at y = 0, its first row (z = 0) held. */
function sheet(n: number, size: number, collider: { y: number; radius: number } | null): ClothMesh {
  const at = (i: number, j: number) => j * n + i;
  const rest = new Float32Array(n * n * 3);
  const free = new Float32Array(n * n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      rest.set([(i / (n - 1) - 0.5) * size, 0, (j / (n - 1)) * size], at(i, j) * 3);
      free[at(i, j)] = j === 0 ? 0 : 1;
    }
  }
  const edges: number[] = [], bends: number[] = [], tris: number[] = [];
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      if (i + 1 < n) edges.push(at(i, j), at(i + 1, j));
      if (j + 1 < n) edges.push(at(i, j), at(i, j + 1));
      if (i + 1 < n && j + 1 < n) {
        edges.push(at(i, j), at(i + 1, j + 1));
        tris.push(at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j), at(i + 1, j + 1), at(i, j + 1));
      }
      if (i + 2 < n) bends.push(at(i, j), at(i + 2, j));
      if (j + 2 < n) bends.push(at(i, j), at(i, j + 2));
    }
  }
  const dist = (a: number, b: number) => Math.hypot(rest[a * 3] - rest[b * 3], rest[a * 3 + 1] - rest[b * 3 + 1], rest[a * 3 + 2] - rest[b * 3 + 2]);
  const lengths = (list: number[]) => Float32Array.from({ length: list.length / 2 }, (_, k) => dist(list[k * 2], list[k * 2 + 1]));
  const count = n * n;
  return {
    materials: [0], vertices: Int32Array.from({ length: count }, (_, i) => i), vertexParticle: Int32Array.from({ length: count }, (_, i) => i),
    particleCount: count, rest, free, pinBones: new Int32Array(count * 4).fill(-1), pinWeights: new Float32Array(count * 4),
    edges: Int32Array.from(edges), edgeRest: lengths(edges), bends: Int32Array.from(bends), bendRest: lengths(bends),
    collision: collider
      ? { colliders: [0], shapes: Float32Array.from([0, collider.radius, 0, 0]), masks: new Uint32Array(count * 2).fill(1), clearance: new Float32Array(count).fill(0.05) }
      : null,
    triangles: Int32Array.from(tris), followers: new Int32Array(0), followerCoords: new Float32Array(0),
  };
}

const sphereAt = (y: number) => Float32Array.from([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, y, 2, 1]);

describe('cloth solver', () => {
  it('hangs from its held edge without stretching', () => {
    const mesh = sheet(9, 4, null);
    const sim = new ClothSim(mesh, { stiffness: 0.3, damping: 0.3, shape: 0, substeps: 12 });
    for (let f = 0; f < 120; f++) sim.step(mesh.rest, new Float32Array(0), 1 / 30);
    const x = sim.positions;
    // The held row has not moved.
    for (let i = 0; i < 9; i++) expect(Math.hypot(x[i * 3] - mesh.rest[i * 3], x[i * 3 + 1], x[i * 3 + 2])).toBeLessThan(1e-5);
    // The far row has swung down to hang below the held edge…
    const far = (8 * 9 + 4) * 3;
    expect(x[far + 1]).toBeLessThan(-3.5);
    // …and no edge is more than a few percent longer than at rest.
    let worst = 0;
    for (let k = 0; k < mesh.edgeRest.length; k++) {
      const a = mesh.edges[k * 2] * 3, b = mesh.edges[k * 2 + 1] * 3;
      worst = Math.max(worst, Math.hypot(x[a] - x[b], x[a + 1] - x[b + 1], x[a + 2] - x[b + 2]) / mesh.edgeRest[k]);
    }
    expect(worst).toBeLessThan(1.05);
    expect(x.every(Number.isFinite)).toBe(true);
  });

  it('drapes over a sphere instead of passing through it', () => {
    const radius = 1.2, center = -1.6;
    const mesh = sheet(13, 4, { y: center, radius });
    const sim = new ClothSim(mesh, { stiffness: 0.3, damping: 0.4, shape: 0, substeps: 16 });
    for (let f = 0; f < 150; f++) sim.step(mesh.rest, sphereAt(center), 1 / 30);
    const x = sim.positions;
    let nearest = Infinity;
    for (let p = 0; p < mesh.particleCount; p++) nearest = Math.min(nearest, Math.hypot(x[p * 3], x[p * 3 + 1] - center, x[p * 3 + 2] - 2));
    expect(nearest).toBeGreaterThan(radius);
    // Something rests on it: without the sphere the sheet's middle would hang far lower.
    const middle = (6 * 13 + 6) * 3;
    expect(x[middle + 1]).toBeGreaterThan(center);
  });

  it('keeps its modelled shape when asked to', () => {
    const mesh = sheet(9, 4, null);
    const sag = (shape: number) => {
      const sim = new ClothSim(mesh, { stiffness: 0.3, damping: 0.3, shape, substeps: 12 });
      for (let f = 0; f < 120; f++) sim.step(mesh.rest, new Float32Array(0), 1 / 30);
      return -sim.positions[(8 * 9 + 4) * 3 + 1];
    };
    expect(sag(1)).toBeLessThan(0.4); // stays nearly horizontal, as modelled
    expect(sag(0.5)).toBeLessThan(sag(0));
  });

  it('gives each vertex a unit normal from its faces', () => {
    const mesh = sheet(3, 2, null);
    const data = clothVertexData(mesh, mesh.rest, new Float32Array(mesh.vertices.length * 8));
    for (let v = 0; v < 9; v++) {
      expect(Math.hypot(data[v * 8 + 4], data[v * 8 + 5], data[v * 8 + 6])).toBeCloseTo(1);
      expect(Math.abs(data[v * 8 + 5])).toBeCloseTo(1); // a flat sheet in XZ: normals along Y
    }
  });
});
