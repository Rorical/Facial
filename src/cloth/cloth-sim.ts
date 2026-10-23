import type { ClothMesh } from './cloth-mesh';

// Cloth solver: position-based dynamics with small substeps (XPBD). Each substep moves the
// particles under gravity, then enforces, in order: edge lengths (the cloth does not stretch),
// bend distances (it resists folding, and keeps pleats), where it is held, and staying outside
// the body's colliders. Runs on the CPU inside the physics bake.

export interface ClothSettings {
  /** 0 = limp … 1 = stiff: how strongly the cloth keeps its rest shape against bending. */
  stiffness: number;
  /** 0 = swings freely … 1 = heavy air drag. */
  damping: number;
  /**
   * 0 = hangs purely by its own weight … 1 = keeps the silhouette it has on the model. A flared
   * or pleated skirt is modelled standing away from the legs; as plain cloth it would collapse
   * onto them (and its pleats would give it enough slack to fall between them). This pulls each
   * particle gently towards where it sits on the model, like a petticoat would.
   */
  shape: number;
  /** Solver steps per frame (30 fps): more = less stretch and steadier collisions, slower. */
  substeps: number;
}

export const DEFAULT_CLOTH: ClothSettings = { stiffness: 0.6, damping: 0.3, shape: 0.5, substeps: 12 };

/** MMD's gravity, in model units per second². */
const GRAVITY = -98;
const FRICTION = 0.35;

export class ClothSim {
  /** Current positions, 3 per particle (model space). */
  readonly positions: Float32Array;
  private readonly previous: Float32Array;
  private readonly velocities: Float32Array;
  /** Held positions at the start and end of the frame being stepped. */
  private readonly targetFrom: Float32Array;
  private readonly targetTo: Float32Array;
  /** Collider transforms (16 floats each) at the start and end of the frame. */
  private readonly colliderFrom: Float32Array;
  private readonly colliderTo: Float32Array;
  private readonly collider = new Float32Array(16);
  private readonly simulated: Uint8Array;
  private started = false;

  constructor(readonly mesh: ClothMesh, readonly settings: ClothSettings) {
    const n = mesh.particleCount;
    this.positions = new Float32Array(n * 3);
    this.previous = new Float32Array(n * 3);
    this.velocities = new Float32Array(n * 3);
    this.targetFrom = new Float32Array(n * 3);
    this.targetTo = new Float32Array(n * 3);
    const colliders = mesh.collision?.colliders.length ?? 0;
    this.colliderFrom = new Float32Array(colliders * 16);
    this.colliderTo = new Float32Array(colliders * 16);
    this.simulated = new Uint8Array(n).fill(1);
    for (let i = 0; i < mesh.followers.length; i += 4) this.simulated[mesh.followers[i]] = 0;
  }

  /**
   * Advance one frame of `dt` seconds. `targets`: where each particle would be if held to the
   * body (3 per particle); `colliders`: this frame's collider transforms (16 floats each).
   * The first call places the cloth at its targets.
   */
  step(targets: Float32Array, colliders: Float32Array, dt: number): void {
    const { mesh, positions: x, previous: prev, velocities: vel } = this;
    const n = mesh.particleCount;
    if (!this.started) {
      this.started = true;
      x.set(targets);
      this.targetTo.set(targets);
      this.colliderTo.set(colliders);
    }
    this.targetFrom.set(this.targetTo);
    this.targetTo.set(targets);
    this.colliderFrom.set(this.colliderTo);
    this.colliderTo.set(colliders);

    const substeps = Math.max(1, Math.round(this.settings.substeps));
    const h = dt / substeps;
    // XPBD compliance for bending: stiff cloth ≈ 0, limp cloth lets bend distances give way.
    const bendCompliance = Math.pow(1 - Math.min(1, Math.max(0, this.settings.stiffness)), 3) * 2e-3 / (h * h);
    const drag = Math.max(0, 1 - (0.2 + this.settings.damping * 6) * h);
    // Shape keeping as a spring towards the held position, sized by how far the cloth may sag
    // under gravity before the spring holds it: 4 units at the soft end, 0.16 at the firm end.
    const shape = Math.min(1, Math.max(0, this.settings.shape ?? 0));
    const pull = shape > 0 ? Math.min(1, (-GRAVITY / (4 * Math.pow(0.04, shape))) * h * h) : 0;
    const { edges, edgeRest, bends, bendRest, free } = mesh;

    for (let s = 1; s <= substeps; s++) {
      const t = s / substeps;
      // Predict
      for (let p = 0; p < n; p++) {
        if (!this.simulated[p]) continue;
        const o = p * 3;
        vel[o + 1] += GRAVITY * h;
        for (let c = 0; c < 3; c++) {
          vel[o + c] *= drag;
          prev[o + c] = x[o + c];
          x[o + c] += vel[o + c] * h;
        }
      }
      // Stretch: rigid (zero compliance)
      for (let i = 0; i < edgeRest.length; i++) this.solveDistance(edges[i * 2], edges[i * 2 + 1], edgeRest[i], 0);
      for (let i = 0; i < bendRest.length; i++) this.solveDistance(bends[i * 2], bends[i * 2 + 1], bendRest[i], bendCompliance);
      // Held particles follow the body; partly held ones are pulled towards it.
      for (let p = 0; p < n; p++) {
        if (!this.simulated[p]) continue;
        // Fully free particles only feel the shape-keeping pull.
        const keep = free[p] >= 1 ? 1 - pull : free[p];
        if (keep >= 1) continue;
        const o = p * 3;
        for (let c = 0; c < 3; c++) {
          const target = this.targetFrom[o + c] + (this.targetTo[o + c] - this.targetFrom[o + c]) * t;
          x[o + c] = target + (x[o + c] - target) * keep;
        }
      }
      this.collide(t);
      // Velocities from the positions reached
      for (let p = 0; p < n; p++) {
        if (!this.simulated[p]) continue;
        const o = p * 3;
        for (let c = 0; c < 3; c++) vel[o + c] = (x[o + c] - prev[o + c]) / h;
      }
    }
    this.placeFollowers();
  }

  /** Inverse mass: held particles don't give way to constraints. */
  private weight(p: number): number {
    return this.mesh.free[p];
  }

  private solveDistance(a: number, b: number, rest: number, compliance: number): void {
    const x = this.positions;
    const wa = this.weight(a), wb = this.weight(b);
    const w = wa + wb;
    if (w <= 0) return;
    const ao = a * 3, bo = b * 3;
    const dx = x[bo] - x[ao], dy = x[bo + 1] - x[ao + 1], dz = x[bo + 2] - x[ao + 2];
    const len = Math.hypot(dx, dy, dz);
    if (len < 1e-9) return;
    const k = (len - rest) / (len * (w + compliance));
    x[ao] += dx * k * wa; x[ao + 1] += dy * k * wa; x[ao + 2] += dz * k * wa;
    x[bo] -= dx * k * wb; x[bo + 1] -= dy * k * wb; x[bo + 2] -= dz * k * wb;
  }

  private collide(t: number): void {
    const c = this.mesh.collision;
    if (!c) return;
    const x = this.positions, prev = this.previous;
    const m = this.collider;
    for (let j = 0; j < c.colliders.length; j++) {
      const word = j >> 5, bit = 1 << (j & 31);
      // Linear blend of the transform across the frame: fine for the small turn within one frame.
      for (let k = 0; k < 16; k++) m[k] = this.colliderFrom[j * 16 + k] + (this.colliderTo[j * 16 + k] - this.colliderFrom[j * 16 + k]) * t;
      const shape = c.shapes[j * 4], sx = c.shapes[j * 4 + 1], sy = c.shapes[j * 4 + 2], sz = c.shapes[j * 4 + 3];
      for (let p = 0; p < this.mesh.particleCount; p++) {
        if (!(c.masks[p * 2 + word] & bit) || !this.simulated[p] || this.mesh.free[p] <= 0) continue;
        const o = p * 3;
        const dx = x[o] - m[12], dy = x[o + 1] - m[13], dz = x[o + 2] - m[14];
        const lx = m[0] * dx + m[1] * dy + m[2] * dz, ly = m[4] * dx + m[5] * dy + m[6] * dz, lz = m[8] * dx + m[9] * dy + m[10] * dz;
        let dist: number, nx: number, ny: number, nz: number;
        if (shape === 1) {
          const qx = Math.abs(lx) - sx, qy = Math.abs(ly) - sy, qz = Math.abs(lz) - sz;
          const ox = Math.max(qx, 0), oy = Math.max(qy, 0), oz = Math.max(qz, 0);
          const outside = Math.hypot(ox, oy, oz);
          dist = outside + Math.min(Math.max(qx, qy, qz), 0);
          if (outside > 0) {
            nx = (ox / outside) * Math.sign(lx); ny = (oy / outside) * Math.sign(ly); nz = (oz / outside) * Math.sign(lz);
          } else if (qx > qy && qx > qz) {
            nx = Math.sign(lx) || 1; ny = 0; nz = 0;
          } else if (qy > qz) {
            nx = 0; ny = Math.sign(ly) || 1; nz = 0;
          } else {
            nx = 0; ny = 0; nz = Math.sign(lz) || 1;
          }
        } else {
          const ay = shape === 2 ? ly - Math.min(sy / 2, Math.max(-sy / 2, ly)) : ly;
          const len = Math.hypot(lx, ay, lz);
          dist = len - sx;
          if (len > 1e-9) {
            nx = lx / len; ny = ay / len; nz = lz / len;
          } else {
            nx = 1; ny = 0; nz = 0;
          }
        }
        const push = c.clearance[p] - dist;
        if (push <= 0) continue;
        // Back to model space: n_world = R · n_local
        const wx = m[0] * nx + m[4] * ny + m[8] * nz, wy = m[1] * nx + m[5] * ny + m[9] * nz, wz = m[2] * nx + m[6] * ny + m[10] * nz;
        x[o] += wx * push; x[o + 1] += wy * push; x[o + 2] += wz * push;
        // Friction: lose part of this substep's sliding along the surface.
        const vx = x[o] - prev[o], vy = x[o + 1] - prev[o + 1], vz = x[o + 2] - prev[o + 2];
        const vn = vx * wx + vy * wy + vz * wz;
        x[o] -= (vx - wx * vn) * FRICTION; x[o + 1] -= (vy - wy * vn) * FRICTION; x[o + 2] -= (vz - wz * vn) * FRICTION;
      }
    }
  }

  /** Trims and ornaments ride on their host triangle. */
  private placeFollowers(): void {
    const { followers, followerCoords } = this.mesh;
    const x = this.positions;
    for (let i = 0; i < followers.length; i += 4) {
      const p = followers[i] * 3, a = followers[i + 1] * 3, b = followers[i + 2] * 3, c = followers[i + 3] * 3;
      const u = followerCoords[(i / 4) * 3], v = followerCoords[(i / 4) * 3 + 1], h = followerCoords[(i / 4) * 3 + 2];
      const e1x = x[b] - x[a], e1y = x[b + 1] - x[a + 1], e1z = x[b + 2] - x[a + 2];
      const e2x = x[c] - x[a], e2y = x[c + 1] - x[a + 1], e2z = x[c + 2] - x[a + 2];
      let nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
      const nl = Math.hypot(nx, ny, nz) || 1;
      nx /= nl; ny /= nl; nz /= nl;
      x[p] = x[a] + u * e1x + v * e2x + h * nx;
      x[p + 1] = x[a + 1] + u * e1y + v * e2y + h * ny;
      x[p + 2] = x[a + 2] + u * e1z + v * e2z + h * nz;
    }
  }
}

/**
 * Where each particle would be if held to the body: linear-blend skinned with its pin bones.
 * `boneWorlds(bone)` returns that bone's world matrix; `restOf(bone)` its rest position.
 */
export function clothTargets(
  mesh: ClothMesh, boneWorld: (bone: number) => ArrayLike<number>, boneRest: (bone: number) => ArrayLike<number>, out: Float32Array,
): Float32Array {
  const { rest, pinBones, pinWeights } = mesh;
  for (let p = 0; p < mesh.particleCount; p++) {
    let x = 0, y = 0, z = 0, total = 0;
    for (let k = 0; k < 4; k++) {
      const bone = pinBones[p * 4 + k];
      const w = pinWeights[p * 4 + k];
      if (bone < 0 || w <= 0) continue;
      const m = boneWorld(bone);
      const r = boneRest(bone);
      // world · (rest position − bone rest position)
      const dx = rest[p * 3] - r[0], dy = rest[p * 3 + 1] - r[1], dz = rest[p * 3 + 2] - r[2];
      x += (m[0] * dx + m[4] * dy + m[8] * dz + m[12]) * w;
      y += (m[1] * dx + m[5] * dy + m[9] * dz + m[13]) * w;
      z += (m[2] * dx + m[6] * dy + m[10] * dz + m[14]) * w;
      total += w;
    }
    if (total > 0) {
      out[p * 3] = x / total; out[p * 3 + 1] = y / total; out[p * 3 + 2] = z / total;
    } else {
      out[p * 3] = rest[p * 3]; out[p * 3 + 1] = rest[p * 3 + 1]; out[p * 3 + 2] = rest[p * 3 + 2];
    }
  }
  return out;
}

/**
 * Per cloth vertex: position and a normal from the faces around that vertex (so hard edges the
 * model had at split vertices stay hard). 8 floats each: position xyz, 0, normal xyz, 0.
 */
export function clothVertexData(mesh: ClothMesh, positions: Float32Array, out: Float32Array): Float32Array {
  const { vertexParticle, triangles } = mesh;
  const count = vertexParticle.length;
  for (let v = 0; v < count; v++) {
    const p = vertexParticle[v] * 3, o = v * 8;
    out[o] = positions[p]; out[o + 1] = positions[p + 1]; out[o + 2] = positions[p + 2];
    out[o + 4] = out[o + 5] = out[o + 6] = 0;
  }
  for (let i = 0; i < triangles.length; i += 3) {
    const a = triangles[i] * 8, b = triangles[i + 1] * 8, c = triangles[i + 2] * 8;
    const e1x = out[b] - out[a], e1y = out[b + 1] - out[a + 1], e1z = out[b + 2] - out[a + 2];
    const e2x = out[c] - out[a], e2y = out[c + 1] - out[a + 1], e2z = out[c + 2] - out[a + 2];
    // Clockwise triangles face the viewer in MMD's left-handed space; area-weighted.
    const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
    for (const k of [a, b, c]) {
      out[k + 4] += nx; out[k + 5] += ny; out[k + 6] += nz;
    }
  }
  for (let v = 0; v < count; v++) {
    const o = v * 8 + 4;
    const len = Math.hypot(out[o], out[o + 1], out[o + 2]) || 1;
    out[o] /= len; out[o + 1] /= len; out[o + 2] /= len;
  }
  return out;
}
