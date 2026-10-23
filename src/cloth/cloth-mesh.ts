import type { PmxModel } from '../loaders/pmx';
import { buildClothCollision, CLOTH_SAMPLES } from '../physics/cloth-collision';
import type { PhysicsRig } from '../physics/rig';
import { classifyMaterial, MaterialPart } from '../runtime/morph-controller';

// Turns chosen materials of a model into a cloth: particles, the constraints between them,
// where the cloth is held, and what it collides with. Everything is derived from the model:
//   · particles: the materials' vertices, with vertices at the same place (UV seams, split
//     normals) welded into one
//   · held (pinned) particles: those skinned to the body rather than to physics bones — a
//     skirt's waist row. In between, a particle is held in proportion
//   · where a held particle should be each frame: skinned as usual, except that physics bones
//     count as rigidly attached to the body bone above them
//   · stretch constraints along mesh edges, bend constraints across each pair of triangles
//     sharing an edge (rest lengths from the rest pose, so pleats are remembered)
//   · collision with the model's follow-bone rigid bodies, as in cloth-collision.ts
//   · pieces with nothing holding them (trims, ornaments modelled as separate shells) ride on
//     the nearest triangle of the cloth instead of being simulated

const WELD = 1e-4;
const COLLISION_MARGIN = 0.06;

export interface ClothMesh {
  /** Material indices simulated. */
  materials: number[];
  /** Model vertices of those materials, in ascending order. */
  vertices: Int32Array;
  /** Per entry of `vertices`: its particle. */
  vertexParticle: Int32Array;
  particleCount: number;
  /** Rest positions, 3 per particle. */
  rest: Float32Array;
  /** Per particle: 0 = held to the body … 1 = free. */
  free: Float32Array;
  /** Up to 4 (bone, weight) per particle for the held position; physics bones replaced by their body ancestor. */
  pinBones: Int32Array;
  pinWeights: Float32Array;
  /** Pairs of particles with a rest length: [a, b] in `edges`, length in `edgeRest`. */
  edges: Int32Array;
  edgeRest: Float32Array;
  bends: Int32Array;
  bendRest: Float32Array;
  /** Collision (null if the model has no colliders): rig body per collider, shapes, per-particle masks and clearance. */
  collision: { colliders: number[]; shapes: Float32Array; masks: Uint32Array; clearance: Float32Array } | null;
  /** Triangles as indices into `vertices` (for normals). */
  triangles: Int32Array;
  /**
   * Particles that ride on a triangle instead of being simulated: particle, the three host
   * particles, then (u, v, h): position = a + u·(b−a) + v·(c−a) + h·unit normal.
   */
  followers: Int32Array;
  followerCoords: Float32Array;
}

/** Materials that look like cloth worth simulating: mostly on physics bones, and not hair. */
export function suggestClothMaterials(pmx: PmxModel, rig: PhysicsRig): { index: number; name: string; suggested: boolean; share: number }[] {
  const collision = buildClothCollision(pmx, rig, COLLISION_MARGIN);
  const physicsBones = new Set<number>();
  const driven = new Set(rig.drivenBones);
  pmx.bones.forEach((_, i) => {
    for (let b = i, guard = 0; b >= 0 && guard < pmx.bones.length; b = rig.boneParents[b], guard++) {
      if (driven.has(b)) {
        physicsBones.add(i);
        break;
      }
    }
  });
  const out: { index: number; name: string; suggested: boolean; share: number }[] = [];
  let first = 0;
  pmx.materials.forEach((m, index) => {
    const verts = new Set(pmx.indices.subarray(first, first + m.indexCount));
    first += m.indexCount;
    if (verts.size === 0) return;
    let onPhysics = 0;
    let held = 0;
    for (const v of verts) {
      let w = 0;
      for (let k = 0; k < 4; k++) if (physicsBones.has(pmx.skinIndices[v * 4 + k])) w += pmx.skinWeights[v * 4 + k];
      if (w > 0.5) onPhysics++;
      else held++;
    }
    const share = onPhysics / verts.size;
    if (share < 0.2) return;
    const part = classifyMaterial(m.name, m.nameEn);
    const clothName = /裙|スカート|skirt|袖|sleeve|披|マント|cape|coat|コート|裾|スカーフ|scarf|围巾|圍巾|マフラー|リボン|飘带|飄帶/i.test(`${m.name} ${m.nameEn}`);
    // Needs something to hang from, and should not be hair or an accessory.
    out.push({ index, name: m.name, share, suggested: clothName && part !== MaterialPart.Hair && held > 0 && !!collision });
  });
  return out;
}

export function buildClothMesh(pmx: PmxModel, rig: PhysicsRig, materials: number[]): ClothMesh | null {
  const chosen = new Set(materials);
  const tris: number[] = [];
  let first = 0;
  pmx.materials.forEach((m, index) => {
    if (chosen.has(index)) for (let i = first; i + 2 < first + m.indexCount; i += 3) tris.push(pmx.indices[i], pmx.indices[i + 1], pmx.indices[i + 2]);
    first += m.indexCount;
  });
  if (tris.length === 0) return null;

  const vertices = Int32Array.from([...new Set(tris)].sort((a, b) => a - b));
  const slot = new Map<number, number>();
  vertices.forEach((v, i) => slot.set(v, i));

  // Weld vertices at the same position into particles.
  const key = (v: number) => `${Math.round(pmx.positions[v * 3] / WELD)},${Math.round(pmx.positions[v * 3 + 1] / WELD)},${Math.round(pmx.positions[v * 3 + 2] / WELD)}`;
  const particleOf = new Map<string, number>();
  const vertexParticle = new Int32Array(vertices.length);
  const firstVertex: number[] = [];
  vertices.forEach((v, i) => {
    const k = key(v);
    let p = particleOf.get(k);
    if (p === undefined) {
      p = firstVertex.length;
      particleOf.set(k, p);
      firstVertex.push(v);
    }
    vertexParticle[i] = p;
  });
  const particleCount = firstVertex.length;
  const rest = new Float32Array(particleCount * 3);
  firstVertex.forEach((v, p) => rest.set(pmx.positions.subarray(v * 3, v * 3 + 3), p * 3));

  // The body bone a physics bone hangs from, and how free each particle is.
  const driven = new Set(rig.drivenBones);
  const anchorOf = new Int32Array(pmx.bones.length).fill(-2);
  const anchor = (bone: number): number => {
    if (bone < 0 || bone >= anchorOf.length) return -1;
    if (anchorOf[bone] !== -2) return anchorOf[bone];
    anchorOf[bone] = -1;
    // A bone is "physics" if it or an ancestor is driven; its anchor is the first ancestor that is not.
    let b = bone;
    const chain: number[] = [];
    for (let guard = 0; b >= 0 && guard < pmx.bones.length; guard++) {
      chain.push(b);
      b = rig.boneParents[b];
    }
    let lastDriven = -1;
    chain.forEach((c, i) => {
      if (driven.has(c)) lastDriven = i;
    });
    return (anchorOf[bone] = lastDriven < 0 ? bone : (chain[lastDriven + 1] ?? -1));
  };
  const free = new Float32Array(particleCount);
  const pinBones = new Int32Array(particleCount * 4).fill(-1);
  const pinWeights = new Float32Array(particleCount * 4);
  firstVertex.forEach((v, p) => {
    let physics = 0;
    for (let k = 0; k < 4; k++) {
      const w = pmx.skinWeights[v * 4 + k];
      if (w <= 0) continue;
      const bone = pmx.skinIndices[v * 4 + k];
      const a = anchor(bone);
      if (a !== bone) physics += w;
      pinBones[p * 4 + k] = a;
      pinWeights[p * 4 + k] = w;
    }
    free[p] = Math.min(1, physics);
  });

  // Edges and the triangles on each side of them.
  const particleTris = new Int32Array(tris.length);
  tris.forEach((v, i) => (particleTris[i] = vertexParticle[slot.get(v)!]));
  const edgeKey = (a: number, b: number) => (a < b ? a * particleCount + b : b * particleCount + a);
  const edgeOpposites = new Map<number, number[]>();
  for (let i = 0; i < particleTris.length; i += 3) {
    const t = [particleTris[i], particleTris[i + 1], particleTris[i + 2]];
    if (t[0] === t[1] || t[1] === t[2] || t[0] === t[2]) continue;
    for (let k = 0; k < 3; k++) {
      const e = edgeKey(t[k], t[(k + 1) % 3]);
      const list = edgeOpposites.get(e);
      if (list) list.push(t[(k + 2) % 3]);
      else edgeOpposites.set(e, [t[(k + 2) % 3]]);
    }
  }
  const dist = (a: number, b: number) => Math.hypot(rest[a * 3] - rest[b * 3], rest[a * 3 + 1] - rest[b * 3 + 1], rest[a * 3 + 2] - rest[b * 3 + 2]);
  const edges: number[] = [];
  const bends: number[] = [];
  const bendSeen = new Set<number>();
  for (const [e, opposites] of edgeOpposites) {
    const a = Math.floor(e / particleCount), b = e % particleCount;
    edges.push(a, b);
    if (opposites.length >= 2 && opposites[0] !== opposites[1]) {
      const k = edgeKey(opposites[0], opposites[1]);
      if (!bendSeen.has(k) && !edgeOpposites.has(k)) {
        bendSeen.add(k);
        bends.push(opposites[0], opposites[1]);
      }
    }
  }

  // Pieces nothing holds (no held particle in their connected component) ride on the cloth.
  const parent = Int32Array.from({ length: particleCount }, (_, i) => i);
  const find = (x: number): number => {
    while (parent[x] !== x) x = parent[x] = parent[parent[x]];
    return x;
  };
  for (let i = 0; i < edges.length; i += 2) parent[find(edges[i])] = find(edges[i + 1]);
  const held = new Set<number>();
  for (let p = 0; p < particleCount; p++) if (free[p] < 0.5) held.add(find(p));
  const loose: number[] = [];
  for (let p = 0; p < particleCount; p++) if (!held.has(find(p))) loose.push(p);
  const followers: number[] = [];
  const followerCoords: number[] = [];
  if (loose.length > 0 && loose.length < particleCount) {
    // Host triangles: those of held pieces.
    const hosts: number[] = [];
    for (let i = 0; i < particleTris.length; i += 3) if (held.has(find(particleTris[i]))) hosts.push(i);
    for (const p of loose) {
      const px = rest[p * 3], py = rest[p * 3 + 1], pz = rest[p * 3 + 2];
      let best = -1;
      let bestDistance = Infinity;
      for (const i of hosts) {
        let cx = 0, cy = 0, cz = 0;
        for (let k = 0; k < 3; k++) {
          cx += rest[particleTris[i + k] * 3]; cy += rest[particleTris[i + k] * 3 + 1]; cz += rest[particleTris[i + k] * 3 + 2];
        }
        const d = Math.hypot(px - cx / 3, py - cy / 3, pz - cz / 3);
        if (d < bestDistance) {
          bestDistance = d;
          best = i;
        }
      }
      if (best < 0) continue;
      const [a, b, c] = [particleTris[best], particleTris[best + 1], particleTris[best + 2]];
      const e1 = [0, 1, 2].map((k) => rest[b * 3 + k] - rest[a * 3 + k]);
      const e2 = [0, 1, 2].map((k) => rest[c * 3 + k] - rest[a * 3 + k]);
      const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
      const nl = Math.hypot(n[0], n[1], n[2]) || 1;
      const d = [px - rest[a * 3], py - rest[a * 3 + 1], pz - rest[a * 3 + 2]];
      const h = (d[0] * n[0] + d[1] * n[1] + d[2] * n[2]) / nl;
      // Solve d − h·n̂ = u·e1 + v·e2 in the triangle's plane.
      const q = d.map((x, k) => x - (h * n[k]) / nl);
      const d11 = e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2], d12 = e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2], d22 = e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2];
      const q1 = q[0] * e1[0] + q[1] * e1[1] + q[2] * e1[2], q2 = q[0] * e2[0] + q[1] * e2[1] + q[2] * e2[2];
      const det = d11 * d22 - d12 * d12 || 1;
      followers.push(p, a, b, c);
      followerCoords.push((q1 * d22 - q2 * d12) / det, (q2 * d11 - q1 * d12) / det, h);
    }
  }
  const isFollower = new Uint8Array(particleCount);
  for (let i = 0; i < followers.length; i += 4) isFollower[followers[i]] = 1;
  // Followers take no part in the simulation.
  const keep = (list: number[]) => {
    const out: number[] = [];
    for (let i = 0; i < list.length; i += 2) if (!isFollower[list[i]] && !isFollower[list[i + 1]]) out.push(list[i], list[i + 1]);
    return out;
  };
  const simEdges = keep(edges);
  const simBends = keep(bends);

  // Collision: reuse the per-vertex setup, read at each particle's first vertex.
  const perVertex = buildClothCollision(pmx, rig, COLLISION_MARGIN);
  let collision: ClothMesh['collision'] = null;
  if (perVertex) {
    const masks = new Uint32Array(particleCount * 2);
    const clearance = new Float32Array(particleCount).fill(COLLISION_MARGIN);
    firstVertex.forEach((v, p) => {
      masks[p * 2] = perVertex.masks[v * 2];
      masks[p * 2 + 1] = perVertex.masks[v * 2 + 1];
      clearance[p] = perVertex.clearance[v * CLOTH_SAMPLES + CLOTH_SAMPLES - 1];
    });
    collision = { colliders: perVertex.colliders, shapes: perVertex.shapes, masks, clearance };
  }

  return {
    materials: [...chosen].sort((a, b) => a - b),
    vertices, vertexParticle, particleCount, rest, free, pinBones, pinWeights,
    edges: Int32Array.from(simEdges), edgeRest: Float32Array.from({ length: simEdges.length / 2 }, (_, i) => dist(simEdges[i * 2], simEdges[i * 2 + 1])),
    bends: Int32Array.from(simBends), bendRest: Float32Array.from({ length: simBends.length / 2 }, (_, i) => dist(simBends[i * 2], simBends[i * 2 + 1])),
    collision,
    triangles: Int32Array.from(tris, (v) => slot.get(v)!),
    followers: Int32Array.from(followers), followerCoords: Float32Array.from(followerCoords),
  };
}
