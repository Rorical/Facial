import type { PmxModel } from '../loaders/pmx';
import { classifyMaterial, MaterialPart } from '../runtime/morph-controller';

// Auxiliary maps for toon shaders, generated from the model itself when the hand-painted
// originals are not available:
//   · ambient occlusion, baked per vertex by ray casting and drawn into each texture's UV layout
//   · a face shadow map: for every point of the face, how far the light may turn to either side
//     before that point falls into shadow (what games paint by hand as an "SDF face map")
//   · a colour ramp (lit → shadow colour per kind of material)
// Everything is computed on the rest pose.

export interface RgbaImage {
  width: number;
  height: number;
  data: Uint8Array;
}

// ---------------------------------------------------------------------------------------
// BVH over triangles, for "is anything in the way" queries

const LEAF_SIZE = 4;

export class TriangleBvh {
  /** Per node: min xyz, max xyz. */
  private readonly bounds: Float32Array;
  /** Per node: [left child, right child] or [-(first + 1), count] for a leaf. */
  private readonly links: Int32Array;
  /** Triangle corner positions in node order: 9 floats each. */
  private readonly tris: Float32Array;
  private nodeCount = 0;

  constructor(positions: Float32Array, indices: ArrayLike<number>) {
    const n = Math.floor(indices.length / 3);
    const order = new Uint32Array(n);
    const centroids = new Float32Array(n * 3);
    const triBounds = new Float32Array(n * 6);
    for (let t = 0; t < n; t++) {
      order[t] = t;
      for (let c = 0; c < 3; c++) {
        const a = positions[indices[t * 3] * 3 + c], b = positions[indices[t * 3 + 1] * 3 + c], d = positions[indices[t * 3 + 2] * 3 + c];
        centroids[t * 3 + c] = (a + b + d) / 3;
        triBounds[t * 6 + c] = Math.min(a, b, d);
        triBounds[t * 6 + 3 + c] = Math.max(a, b, d);
      }
    }
    this.bounds = new Float32Array(Math.max(1, n * 2) * 6);
    this.links = new Int32Array(Math.max(1, n * 2) * 2);
    if (n > 0) this.build(order, centroids, triBounds, 0, n);
    this.tris = new Float32Array(n * 9);
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < 3; k++) {
        const v = indices[order[i] * 3 + k] * 3;
        this.tris[i * 9 + k * 3] = positions[v];
        this.tris[i * 9 + k * 3 + 1] = positions[v + 1];
        this.tris[i * 9 + k * 3 + 2] = positions[v + 2];
      }
    }
  }

  private build(order: Uint32Array, centroids: Float32Array, triBounds: Float32Array, start: number, end: number): number {
    const node = this.nodeCount++;
    const b = this.bounds;
    const o = node * 6;
    b[o] = b[o + 1] = b[o + 2] = Infinity;
    b[o + 3] = b[o + 4] = b[o + 5] = -Infinity;
    const cmin = [Infinity, Infinity, Infinity];
    const cmax = [-Infinity, -Infinity, -Infinity];
    for (let i = start; i < end; i++) {
      const t = order[i];
      for (let c = 0; c < 3; c++) {
        b[o + c] = Math.min(b[o + c], triBounds[t * 6 + c]);
        b[o + 3 + c] = Math.max(b[o + 3 + c], triBounds[t * 6 + 3 + c]);
        cmin[c] = Math.min(cmin[c], centroids[t * 3 + c]);
        cmax[c] = Math.max(cmax[c], centroids[t * 3 + c]);
      }
    }
    const axis = cmax[0] - cmin[0] > cmax[1] - cmin[1] ? (cmax[0] - cmin[0] > cmax[2] - cmin[2] ? 0 : 2) : cmax[1] - cmin[1] > cmax[2] - cmin[2] ? 1 : 2;
    if (end - start <= LEAF_SIZE || cmax[axis] - cmin[axis] < 1e-9) {
      this.links[node * 2] = -(start + 1);
      this.links[node * 2 + 1] = end - start;
      return node;
    }
    // Partition around the middle of the centroids' extent.
    const split = (cmin[axis] + cmax[axis]) / 2;
    let i = start;
    let j = end - 1;
    while (i <= j) {
      if (centroids[order[i] * 3 + axis] < split) i++;
      else {
        const tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
        j--;
      }
    }
    if (i === start || i === end) i = (start + end) >> 1;
    this.links[node * 2] = this.build(order, centroids, triBounds, start, i);
    this.links[node * 2 + 1] = this.build(order, centroids, triBounds, i, end);
    return node;
  }

  private readonly stack = new Int32Array(128);

  /** True if the ray from (ox, oy, oz) along the unit direction hits a triangle before tMax. */
  occluded(ox: number, oy: number, oz: number, dx: number, dy: number, dz: number, tMax: number): boolean {
    return this.trace(ox, oy, oz, dx, dy, dz, tMax, true) < tMax;
  }

  /** Distance to the nearest triangle along the ray, or tMax if there is none that close. */
  nearest(ox: number, oy: number, oz: number, dx: number, dy: number, dz: number, tMax: number): number {
    return this.trace(ox, oy, oz, dx, dy, dz, tMax, false);
  }

  private trace(ox: number, oy: number, oz: number, dx: number, dy: number, dz: number, limit: number, any: boolean): number {
    let tMax = limit;
    if (this.nodeCount === 0) return tMax;
    const { bounds, links, tris, stack } = this;
    // Axis-parallel rays: a tiny component instead of dividing by zero.
    const inv = (d: number) => 1 / (Math.abs(d) < 1e-9 ? 1e-9 : d);
    const ix = inv(dx), iy = inv(dy), iz = inv(dz);
    let sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
      const node = stack[--sp];
      const o = node * 6;
      // Slab test
      let t0 = (bounds[o] - ox) * ix, t1 = (bounds[o + 3] - ox) * ix;
      let tmin = Math.min(t0, t1), tmax = Math.max(t0, t1);
      t0 = (bounds[o + 1] - oy) * iy; t1 = (bounds[o + 4] - oy) * iy;
      tmin = Math.max(tmin, Math.min(t0, t1)); tmax = Math.min(tmax, Math.max(t0, t1));
      t0 = (bounds[o + 2] - oz) * iz; t1 = (bounds[o + 5] - oz) * iz;
      tmin = Math.max(tmin, Math.min(t0, t1)); tmax = Math.min(tmax, Math.max(t0, t1));
      if (tmax < Math.max(tmin, 0) || tmin > tMax) continue;

      const a = links[node * 2];
      if (a >= 0) {
        if (sp + 2 <= stack.length) {
          stack[sp++] = a;
          stack[sp++] = links[node * 2 + 1];
        }
        continue;
      }
      const first = -a - 1;
      const count = links[node * 2 + 1];
      for (let k = first; k < first + count; k++) {
        // Möller–Trumbore, either side
        const p = k * 9;
        const e1x = tris[p + 3] - tris[p], e1y = tris[p + 4] - tris[p + 1], e1z = tris[p + 5] - tris[p + 2];
        const e2x = tris[p + 6] - tris[p], e2y = tris[p + 7] - tris[p + 1], e2z = tris[p + 8] - tris[p + 2];
        const px = dy * e2z - dz * e2y, py = dz * e2x - dx * e2z, pz = dx * e2y - dy * e2x;
        const det = e1x * px + e1y * py + e1z * pz;
        if (det > -1e-12 && det < 1e-12) continue;
        const inv = 1 / det;
        const sx = ox - tris[p], sy = oy - tris[p + 1], sz = oz - tris[p + 2];
        const u = (sx * px + sy * py + sz * pz) * inv;
        if (u < 0 || u > 1) continue;
        const qx = sy * e1z - sz * e1y, qy = sz * e1x - sx * e1z, qz = sx * e1y - sy * e1x;
        const v = (dx * qx + dy * qy + dz * qz) * inv;
        if (v < 0 || u + v > 1) continue;
        const t = (e2x * qx + e2y * qy + e2z * qz) * inv;
        if (t > 1e-4 && t < tMax) {
          if (any) return t;
          tMax = t;
        }
      }
    }
    return tMax;
  }
}

// ---------------------------------------------------------------------------------------
// Geometry helpers

interface MaterialRange {
  index: number;
  first: number;
  count: number;
  part: number;
}

function materialRanges(pmx: PmxModel): MaterialRange[] {
  let first = 0;
  return pmx.materials.map((m, index) => {
    const r = { index, first, count: m.indexCount, part: classifyMaterial(m.name, m.nameEn) };
    first += m.indexCount;
    return r;
  });
}

function collectIndices(pmx: PmxModel, ranges: MaterialRange[]): Uint32Array {
  const out = new Uint32Array(ranges.reduce((n, r) => n + r.count, 0));
  let o = 0;
  for (const r of ranges) {
    out.set(pmx.indices.subarray(r.first, r.first + r.count), o);
    o += r.count;
  }
  return out;
}

/** Evenly spread, cosine-weighted directions over the hemisphere around +Z. */
function hemisphere(count: number): Float32Array {
  const out = new Float32Array(count * 3);
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < count; i++) {
    const r = Math.sqrt((i + 0.5) / count);
    const a = i * golden;
    out[i * 3] = r * Math.cos(a);
    out[i * 3 + 1] = r * Math.sin(a);
    out[i * 3 + 2] = Math.sqrt(Math.max(0, 1 - r * r));
  }
  return out;
}

const pause = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

// ---------------------------------------------------------------------------------------
// Ambient occlusion

export interface BakeProgress {
  (message: string): void;
}

const AO_RAYS = 48;
/** How far occluders count, in model units (about 14 cm): creases and overlaps, not the whole body. */
const AO_DISTANCE = 1.8;

const THICKNESS_RAYS = 12;
/** Thickness beyond which no light gets through, in model units (about 12 cm). */
const THICKNESS_MAX = 1.5;

/**
 * Two values per vertex. [0] openness: 1 = nothing nearby, 0 = enclosed (ambient occlusion).
 * [1] thinness: 1 = paper thin … 0 = THICKNESS_MAX or more of material behind the surface
 * (ears, fingers and the rim of the nose come out thin; the torso thick) — what light shining
 * through skin needs.
 */
export async function bakeVertexOcclusion(pmx: PmxModel, onProgress?: BakeProgress): Promise<Float32Array> {
  const n = pmx.vertexCount;
  const ao = new Float32Array(n * 2);
  for (let v = 0; v < n; v++) ao[v * 2] = 1;
  // Decals and see-through layers (alpha below 0.9) don't block light.
  const solid = materialRanges(pmx).filter((r) => pmx.materials[r.index].diffuse[3] >= 0.9);
  const bvh = new TriangleBvh(pmx.positions, collectIndices(pmx, solid));
  const dirs = hemisphere(AO_RAYS);
  const used = new Uint8Array(n);
  for (const i of pmx.indices) used[i] = 1;
  const { positions: P, normals: N } = pmx;
  let last = performance.now();
  for (let v = 0; v < n; v++) {
    if (!used[v]) continue;
    const nx = N[v * 3], ny = N[v * 3 + 1], nz = N[v * 3 + 2];
    // Tangent frame around the normal
    const ax = Math.abs(nx) > 0.9 ? 0 : 1, ay = Math.abs(nx) > 0.9 ? 1 : 0;
    let tx = ay * nz, ty = -ax * nz, tz = ax * ny - ay * nx;
    const tl = Math.hypot(tx, ty, tz) || 1;
    tx /= tl; ty /= tl; tz /= tl;
    const bx = ny * tz - nz * ty, by = nz * tx - nx * tz, bz = nx * ty - ny * tx;
    const ox = P[v * 3] + nx * 0.03, oy = P[v * 3 + 1] + ny * 0.03, oz = P[v * 3 + 2] + nz * 0.03;
    let hits = 0;
    for (let k = 0; k < AO_RAYS; k++) {
      const x = dirs[k * 3], y = dirs[k * 3 + 1], z = dirs[k * 3 + 2];
      if (bvh.occluded(ox, oy, oz, tx * x + bx * y + nx * z, ty * x + by * y + ny * z, tz * x + bz * y + nz * z, AO_DISTANCE)) hits++;
    }
    ao[v * 2] = 1 - hits / AO_RAYS;
    // Thickness: how far it is to the other side, looking inwards through a cone of directions.
    const ix = P[v * 3] - nx * 0.03, iy = P[v * 3 + 1] - ny * 0.03, iz = P[v * 3 + 2] - nz * 0.03;
    let depth = 0;
    for (let k = 0; k < THICKNESS_RAYS; k++) {
      // Every fourth AO direction, narrowed towards the inward normal.
      const d = k * Math.floor(AO_RAYS / THICKNESS_RAYS);
      const x = dirs[d * 3] * 0.6, y = dirs[d * 3 + 1] * 0.6, z = Math.sqrt(Math.max(0, 1 - 0.36 * (dirs[d * 3] ** 2 + dirs[d * 3 + 1] ** 2)));
      depth += bvh.nearest(ix, iy, iz, tx * x + bx * y - nx * z, ty * x + by * y - ny * z, tz * x + bz * y - nz * z, THICKNESS_MAX);
    }
    ao[v * 2 + 1] = 1 - depth / (THICKNESS_RAYS * THICKNESS_MAX);
    if ((v & 511) === 0 && performance.now() - last > 60) {
      onProgress?.(`计算环境遮蔽… ${Math.round((v / n) * 100)}%`);
      await pause();
      last = performance.now();
    }
  }
  return ao;
}

// ---------------------------------------------------------------------------------------
// Drawing per-vertex values into a texture's UV layout

/** Up to 4 channels per vertex (0–1), drawn with `channels` values per pixel; returns value + coverage. */
function rasterize(
  pmx: PmxModel, ranges: MaterialRange[], values: Float32Array, channels: number, size: number,
): { pixels: Float32Array; covered: Uint8Array } {
  const pixels = new Float32Array(size * size * channels);
  const covered = new Uint8Array(size * size);
  const { uvs, indices } = pmx;
  for (const r of ranges) {
    for (let i = r.first; i + 2 < r.first + r.count; i += 3) {
      const a = indices[i], b = indices[i + 1], c = indices[i + 2];
      // Tiled UVs: bring the triangle into the 0–1 tile its first corner lies in.
      const ou = Math.floor(uvs[a * 2]), ov = Math.floor(uvs[a * 2 + 1]);
      const ax = (uvs[a * 2] - ou) * size, ay = (uvs[a * 2 + 1] - ov) * size;
      const bx = (uvs[b * 2] - ou) * size, by = (uvs[b * 2 + 1] - ov) * size;
      const cx = (uvs[c * 2] - ou) * size, cy = (uvs[c * 2 + 1] - ov) * size;
      const area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
      if (Math.abs(area) < 1e-9) continue;
      const x0 = Math.max(0, Math.floor(Math.min(ax, bx, cx) - 0.5)), x1 = Math.min(size - 1, Math.ceil(Math.max(ax, bx, cx) + 0.5));
      const y0 = Math.max(0, Math.floor(Math.min(ay, by, cy) - 0.5)), y1 = Math.min(size - 1, Math.ceil(Math.max(ay, by, cy) + 0.5));
      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          const px = x + 0.5, py = y + 0.5;
          let w0 = ((bx - px) * (cy - py) - (cx - px) * (by - py)) / area;
          let w1 = ((cx - px) * (ay - py) - (ax - px) * (cy - py)) / area;
          let w2 = 1 - w0 - w1;
          // A little slack so thin triangles and seams are not missed.
          if (w0 < -0.02 || w1 < -0.02 || w2 < -0.02) continue;
          w0 = Math.max(w0, 0); w1 = Math.max(w1, 0); w2 = Math.max(w2, 0);
          const sum = w0 + w1 + w2;
          const o = (y * size + x) * channels;
          for (let ch = 0; ch < channels; ch++) {
            pixels[o + ch] = (values[a * channels + ch] * w0 + values[b * channels + ch] * w1 + values[c * channels + ch] * w2) / sum;
          }
          covered[y * size + x] = 1;
        }
      }
    }
  }
  return { pixels, covered };
}

/** Average covered neighbours; `grow` also fills uncovered pixels next to covered ones (seam padding). */
function blur(pixels: Float32Array, covered: Uint8Array, channels: number, size: number, passes: number, grow: boolean): void {
  const tmp = new Float32Array(pixels.length);
  const next = new Uint8Array(covered.length);
  for (let pass = 0; pass < passes; pass++) {
    next.set(covered);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const i = y * size + x;
        if (!covered[i] && !grow) continue;
        let count = 0;
        for (let ch = 0; ch < channels; ch++) tmp[i * channels + ch] = 0;
        for (let dy = -1; dy <= 1; dy++) {
          const yy = y + dy;
          if (yy < 0 || yy >= size) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx;
            if (xx < 0 || xx >= size) continue;
            const j = yy * size + xx;
            if (!covered[j]) continue;
            count++;
            for (let ch = 0; ch < channels; ch++) tmp[i * channels + ch] += pixels[j * channels + ch];
          }
        }
        if (count === 0) continue;
        for (let ch = 0; ch < channels; ch++) tmp[i * channels + ch] /= count;
        next[i] = 1;
      }
    }
    for (let i = 0; i < covered.length; i++) {
      if (!next[i] || (!covered[i] && !grow)) continue;
      for (let ch = 0; ch < channels; ch++) pixels[i * channels + ch] = tmp[i * channels + ch];
    }
    covered.set(next);
  }
}

const LIGHT_MAP_SIZE = 1024;
const FACE_MAP_SIZE = 512;

/**
 * One light map per base texture (keyed by texture index), from bakeVertexOcclusion's values:
 * r = ambient occlusion (1 = open), g = thinness (1 = thin), b unused (0), a = 1.
 * Pixels no triangle covers are fully open and thick.
 */
export function occlusionToLightMaps(pmx: PmxModel, occlusion: Float32Array): Map<number, RgbaImage> {
  const byTexture = new Map<number, MaterialRange[]>();
  for (const r of materialRanges(pmx)) {
    const t = pmx.materials[r.index].textureIndex;
    if (t < 0 || t >= pmx.textures.length || r.count === 0) continue;
    const list = byTexture.get(t);
    if (list) list.push(r);
    else byTexture.set(t, [r]);
  }
  const out = new Map<number, RgbaImage>();
  for (const [t, ranges] of byTexture) {
    const size = LIGHT_MAP_SIZE;
    const { pixels, covered } = rasterize(pmx, ranges, occlusion, 2, size);
    blur(pixels, covered, 2, size, 1, false); // vertex values are coarse: soften the facets
    blur(pixels, covered, 2, size, 4, true); // pad across UV seams
    const data = new Uint8Array(size * size * 4);
    for (let i = 0; i < size * size; i++) {
      data[i * 4] = covered[i] ? Math.round(Math.min(1, Math.max(0, pixels[i * 2])) * 255) : 255;
      data[i * 4 + 1] = covered[i] ? Math.round(Math.min(1, Math.max(0, pixels[i * 2 + 1])) * 255) : 0;
      data[i * 4 + 3] = 255;
    }
    out.set(t, { width: size, height: size, data });
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Face shadow map

const FACE_STEPS = 45;
/** An RGBA image as a PNG file (for saving with a project). */
export async function imageToPng(image: RgbaImage): Promise<Blob> {
  const canvas = new OffscreenCanvas(image.width, image.height);
  const pixels = new ImageData(new Uint8ClampedArray(image.data), image.width, image.height);
  canvas.getContext('2d')!.putImageData(pixels, 0, 0);
  return canvas.convertToBlob({ type: 'image/png' });
}


/**
 * For each base texture used by the face: r = how far (0–1 = 0–180°) the light may turn from
 * straight ahead towards the head's right (cross(up, forward)) before the point goes dark,
 * g = the same towards the left, b = 1 where the face is, a = 1.
 * The turn is within the horizontal plane of the head; occlusion by the nose and cheeks is
 * ray cast against the face itself. Returns an empty map if the model has no 頭 bone or no
 * material recognised as the face.
 */
export async function bakeFaceMaps(pmx: PmxModel, onProgress?: BakeProgress): Promise<Map<number, RgbaImage>> {
  const out = new Map<number, RgbaImage>();
  const head = pmx.bones.find((b) => b.name === '頭');
  const ranges = materialRanges(pmx);
  const skin = ranges.filter((r) => r.part === MaterialPart.Face && pmx.materials[r.index].diffuse[3] >= 0.9);
  const all = ranges.filter((r) => r.part === MaterialPart.Face || r.part === MaterialPart.FaceDetail);
  if (!head || skin.length === 0) return out;

  const bvh = new TriangleBvh(pmx.positions, collectIndices(pmx, skin));
  const n = pmx.vertexCount;
  const values = new Float32Array(n * 2).fill(1);
  // Painted-on features, kept sharper than the broad shadow sweep: see faceDetail().
  const detail = new Float32Array(n * 2);
  const landmarks = faceLandmarks(pmx, skin);
  const done = new Uint8Array(n);
  const { positions: P, normals: N } = pmx;
  const cx = head.position[0], cz = head.position[2];
  let last = performance.now();
  let processed = 0;
  for (const r of all) {
    for (let i = r.first; i < r.first + r.count; i++) {
      const v = pmx.indices[i];
      if (done[v]) continue;
      done[v] = 1;
      // The shading normal: mostly a cylinder around the head, with a little of the real
      // (flattened) normal so the bridge of the nose and the jaw still turn.
      let ax = P[v * 3] - cx, az = P[v * 3 + 2] - cz;
      const al = Math.hypot(ax, az) || 1;
      ax /= al; az /= al;
      let nx = N[v * 3], ny = N[v * 3 + 1] * 0.2, nz = N[v * 3 + 2];
      const nl = Math.hypot(nx, ny, nz) || 1;
      nx /= nl; ny /= nl; nz /= nl;
      let sx = ax * 0.75 + nx * 0.25, sy = ny * 0.25, sz = az * 0.75 + nz * 0.25;
      const sl = Math.hypot(sx, sy, sz) || 1;
      sx /= sl; sy /= sl; sz /= sl;
      const ox = P[v * 3] + N[v * 3] * 0.03, oy = P[v * 3 + 1] + N[v * 3 + 1] * 0.03, oz = P[v * 3 + 2] + N[v * 3 + 2] * 0.03;
      for (let side = 0; side < 2; side++) {
        let limit = 1;
        for (let k = 0; k <= FACE_STEPS; k++) {
          const angle = (k / FACE_STEPS) * Math.PI;
          // forward = -Z, right = cross(up, forward) = -X
          const lx = (side === 0 ? -1 : 1) * Math.sin(angle), lz = -Math.cos(angle);
          const lit = sx * lx + sz * lz > 0 && !bvh.occluded(ox, oy, oz, lx, 0, lz, 4);
          if (!lit) {
            limit = k / FACE_STEPS;
            break;
          }
        }
        values[v * 2 + side] = limit;
        if (landmarks) detail[v * 2 + side] = faceDetail(landmarks, P[v * 3], P[v * 3 + 1], side);
      }
      if ((++processed & 255) === 0 && performance.now() - last > 60) {
        onProgress?.('计算脸部阴影图…');
        await pause();
        last = performance.now();
      }
    }
  }

  const byTexture = new Map<number, MaterialRange[]>();
  for (const r of all) {
    const t = pmx.materials[r.index].textureIndex;
    if (t < 0 || t >= pmx.textures.length || r.count === 0) continue;
    const list = byTexture.get(t);
    if (list) list.push(r);
    else byTexture.set(t, [r]);
  }
  for (const [t, list] of byTexture) {
    const size = FACE_MAP_SIZE;
    // Skin first so eyes and brows (drawn over it in UV space only where they have their own area) don't override it.
    const { pixels, covered } = rasterize(pmx, [...list].sort((a, b) => b.part - a.part), values, 2, size);
    const mask = covered.slice();
    blur(pixels, covered, 2, size, 10, false); // the shadow edge should sweep smoothly
    blur(pixels, covered, 2, size, 4, true);
    // The details get only a light blur, so the cheek triangle and the nose keep their shape.
    const fine = rasterize(pmx, [...list].sort((a, b) => b.part - a.part), detail, 2, size);
    blur(fine.pixels, fine.covered, 2, size, 2, false);
    blur(fine.pixels, fine.covered, 2, size, 4, true);
    for (let i = 0; i < pixels.length; i++) pixels[i] = Math.min(1, Math.max(0, pixels[i] + fine.pixels[i]));
    const data = new Uint8Array(size * size * 4);
    for (let i = 0; i < size * size; i++) {
      data[i * 4] = covered[i] ? Math.round(pixels[i * 2] * 255) : 255;
      data[i * 4 + 1] = covered[i] ? Math.round(pixels[i * 2 + 1] * 255) : 255;
      data[i * 4 + 2] = mask[i] ? 255 : 0;
      data[i * 4 + 3] = 255;
    }
    out.set(t, { width: size, height: size, data });
  }
  return out;
}

interface FaceLandmarks {
  /** Height of the eyes, and half the distance between them (the unit everything is sized by). */
  eyeY: number;
  eye: number;
  /** The tip of the nose (x, y). */
  noseX: number;
  noseY: number;
}

/** Eye and nose positions from the eye bones and the face mesh; null if the model has no eye bones. */
function faceLandmarks(pmx: PmxModel, skin: MaterialRange[]): FaceLandmarks | null {
  const left = pmx.bones.find((b) => b.name === '左目');
  const right = pmx.bones.find((b) => b.name === '右目');
  if (!left || !right) return null;
  const eye = Math.abs(left.position[0] - right.position[0]) / 2;
  const eyeY = (left.position[1] + right.position[1]) / 2;
  if (eye < 1e-3) return null;
  // The nose tip: the point of the face furthest forward (-Z) between and below the eyes.
  let noseX = 0, noseY = eyeY - eye * 1.2, front = Infinity;
  for (const r of skin) {
    for (let i = r.first; i < r.first + r.count; i++) {
      const v = pmx.indices[i] * 3;
      const x = pmx.positions[v], y = pmx.positions[v + 1], z = pmx.positions[v + 2];
      if (Math.abs(x) < eye * 0.5 && y < eyeY - eye * 0.3 && y > eyeY - eye * 2.2 && z < front) {
        front = z;
        noseX = x;
        noseY = y;
      }
    }
  }
  return { eyeY, eye, noseX, noseY };
}

const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/**
 * What artists paint into a face shadow map on top of the plain sweep, as a change to how far
 * the light may turn (side 0: towards -X, side 1: towards +X) before the point at (x, y) goes dark:
 *   · the cheek triangle: a patch under the eye on the side away from the light stays lit a while
 *     longer, so a side-lit face shows a triangle of light on the shadowed cheek
 *   · the nose: its far flank goes dark early (a shadow line beside the bridge), and the tip
 *     keeps a small highlight
 */
function faceDetail(f: FaceLandmarks, x: number, y: number, side: number): number {
  const e = f.eye;
  // +1 on the half of the face away from the light.
  const away = side === 0 ? 1 : -1;
  let change = 0;

  // Cheek triangle: from just under the eye, narrowing to a point further down.
  const top = f.eyeY - e * 0.7, bottom = f.eyeY - e * 2.1;
  const down = (top - y) / (top - bottom); // 0 at the top edge, 1 at the point
  if (down > -0.15 && down < 1.1) {
    const halfWidth = e * 0.62 * (1 - Math.min(1, Math.max(0, down)));
    const fromCenter = Math.abs(x * away - e * 1.02);
    const inside = smooth(halfWidth + e * 0.1, halfWidth - e * 0.06, fromCenter) * smooth(-0.12, 0.05, down) * smooth(1.08, 0.85, down);
    change += 0.17 * inside;
  }

  // Nose: distance from the tip in the face plane.
  const dx = (x - f.noseX) * away, dy = y - f.noseY;
  const near = Math.hypot(dx, dy * 0.6);
  // The flank away from the light, along the bridge above the tip.
  change -= 0.2 * smooth(e * 0.04, e * 0.16, dx) * smooth(e * 0.42, e * 0.22, dx) * smooth(-e * 0.25, e * 0.1, dy) * smooth(e * 1.25, e * 0.7, dy);
  // The tip itself catches the light a little longer.
  change += 0.14 * smooth(e * 0.2, e * 0.07, near);
  return change;
}

// ---------------------------------------------------------------------------------------
// Colour ramp

export interface RampColors {
  /** Shadow colour multipliers. */
  cloth: [number, number, number];
  skin: [number, number, number];
  hair: [number, number, number];
  /** Tint of the skin's light / shadow boundary. */
  blush: [number, number, number];
}

export const DEFAULT_RAMP: RampColors = {
  cloth: [0.7, 0.68, 0.85],
  skin: [1.0, 0.87, 0.88],
  hair: [0.8, 0.69, 0.8],
  blush: [1.0, 0.72, 0.68],
};

/**
 * Rows (top to bottom): cloth, skin, hair, face detail. Along a row, u = 0 is full shadow and
 * u = 1 fully lit; the value multiplies the material's colour. The step sits in the middle
 * with a narrow soft edge.
 */
export function generateRamp(colors: RampColors = DEFAULT_RAMP): RgbaImage {
  const width = 256;
  const rows: [number, number, number][] = [
    colors.cloth, colors.skin, colors.hair,
    // Eyes, brows and mouth keep most of their brightness in shadow.
    [0.4 + colors.skin[0] * 0.6, 0.4 + colors.skin[1] * 0.6, 0.4 + colors.skin[2] * 0.6],
  ];
  const data = new Uint8Array(width * rows.length * 4);
  rows.forEach((shadow, row) => {
    for (let x = 0; x < width; x++) {
      const u = (x + 0.5) / width;
      const t = Math.min(1, Math.max(0, (u - 0.46) / 0.08));
      const lit = t * t * (3 - 2 * t);
      const edge = row === 1 ? lit * (1 - lit) * 4 * 0.5 : 0;
      for (let c = 0; c < 3; c++) {
        const base = shadow[c] + (1 - shadow[c]) * lit;
        data[(row * width + x) * 4 + c] = Math.round(255 * (base + (base * colors.blush[c] - base) * edge));
      }
      data[(row * width + x) * 4 + 3] = 255;
    }
  });
  return { width, height: rows.length, data };
}
