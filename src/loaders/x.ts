import type { Vec3Tuple } from './binary-reader';
import { BoneFlag, MaterialFlag, SphereMode, type PmxMaterial, type PmxModel } from './pmx';

// DirectX .x (text format) → the PMX structure, for MMD accessories and stages. An accessory
// has no skeleton, so it gets one root bone that every vertex follows; MMD shows .x files ten
// times their stored size.
//
// Read: Frame hierarchy (with FrameTransformMatrix), Mesh, MeshNormals, MeshTextureCoords,
// MeshMaterialList with inline or referenced Materials. Binary and compressed .x files are
// not supported.

const ACCESSORY_SCALE = 10;

type Token = { t: 'num'; v: number } | { t: 'str'; v: string } | { t: 'id'; v: string } | { t: '{' } | { t: '}' };

/** `;` and `,` only separate values in .x text, so they are dropped like whitespace. */
function tokenize(text: string): Token[] {
  const tokens: Token[] = [];
  const re = /\/\/[^\n]*|#[^\n]*|"((?:[^"\\]|\\.)*)"|([{}])|<[^>]*>|([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)(?![\w.])|([A-Za-z_][\w.\-]*)|[;,\s]+|./g;
  for (let m = re.exec(text); m; m = re.exec(text)) {
    if (m[1] !== undefined) tokens.push({ t: 'str', v: m[1] });
    else if (m[2]) tokens.push({ t: m[2] as '{' | '}' });
    else if (m[3] !== undefined) tokens.push({ t: 'num', v: Number(m[3]) });
    else if (m[4]) tokens.push({ t: 'id', v: m[4] });
  }
  return tokens;
}

interface XMaterial {
  diffuse: [number, number, number, number];
  power: number;
  specular: Vec3Tuple;
  emissive: Vec3Tuple;
  texture: string | null;
}

interface XMesh {
  positions: number[];
  /** Vertex indices per face (3 or more). */
  faces: number[][];
  normals: number[] | null;
  normalFaces: number[][] | null;
  uvs: number[] | null;
  faceMaterials: number[];
  materials: XMaterial[];
  /** Column-major world transform from the enclosing frames. */
  transform: Float64Array;
}

const IDENTITY = Float64Array.from([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

function multiply(a: Float64Array, b: Float64Array): Float64Array {
  const out = new Float64Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      out[c * 4 + r] = s;
    }
  }
  return out;
}

class Parser {
  private i = 0;
  readonly meshes: XMesh[] = [];
  private readonly namedMaterials = new Map<string, XMaterial>();

  constructor(private readonly tokens: Token[]) {}

  private peek(): Token | undefined {
    return this.tokens[this.i];
  }

  private num(): number {
    const t = this.tokens[this.i++];
    if (!t || t.t !== 'num') throw new Error('.x 文件格式错误：这里应该是数字');
    return t.v;
  }

  private nums(n: number): number[] {
    const out = new Array<number>(n);
    for (let k = 0; k < n; k++) out[k] = this.num();
    return out;
  }

  /** Skip to the `}` matching an already consumed `{`. */
  private skipBlock(): void {
    for (let depth = 1; depth > 0 && this.i < this.tokens.length; ) {
      const t = this.tokens[this.i++];
      if (t.t === '{') depth++;
      else if (t.t === '}') depth--;
    }
  }

  /** Reads `Type [name] {`; returns null at a `}` or the end. */
  private open(): { type: string; name: string } | null {
    const t = this.peek();
    if (!t || t.t === '}') return null;
    this.i++;
    if (t.t === '{') {
      // `{ Name }`: a reference to an earlier object.
      const ref = this.tokens[this.i];
      this.skipBlock();
      return { type: '@ref', name: ref?.t === 'id' ? ref.v : '' };
    }
    if (t.t !== 'id') return this.open();
    let name = '';
    const next = this.peek();
    if (next?.t === 'id') {
      name = next.v;
      this.i++;
    }
    if (this.peek()?.t !== '{') return this.open(); // stray identifier
    this.i++;
    return { type: t.v, name };
  }

  parse(transform: Float64Array = IDENTITY): void {
    for (let o = this.open(); o; o = this.open()) {
      switch (o.type) {
        case 'template':
        case '@ref':
          if (o.type === 'template') this.skipBlock();
          break;
        case 'Material':
          this.namedMaterials.set(o.name, this.material());
          break;
        case 'Frame':
          this.frame(transform);
          break;
        case 'Mesh':
          this.mesh(transform);
          break;
        default:
          this.skipBlock();
      }
    }
  }

  private frame(parent: Float64Array): void {
    let transform = parent;
    // The transform precedes the children it applies to in every exporter's output.
    for (let o = this.open(); o; o = this.open()) {
      if (o.type === 'FrameTransformMatrix') {
        // Stored row-major for row vectors, which is column-major for column vectors.
        transform = multiply(parent, Float64Array.from(this.nums(16)));
        this.skipBlock();
      } else if (o.type === 'Frame') this.frame(transform);
      else if (o.type === 'Mesh') this.mesh(transform);
      else if (o.type !== '@ref') this.skipBlock();
    }
    this.i++; // }
  }

  private faces(): number[][] {
    const n = this.num();
    const out = new Array<number[]>(n);
    for (let f = 0; f < n; f++) out[f] = this.nums(this.num());
    return out;
  }

  private material(): XMaterial {
    const diffuse = this.nums(4) as XMaterial['diffuse'];
    const power = this.num();
    const specular = this.nums(3) as Vec3Tuple;
    const emissive = this.nums(3) as Vec3Tuple;
    let texture: string | null = null;
    for (let o = this.open(); o; o = this.open()) {
      if (o.type === 'TextureFilename') {
        const t = this.peek();
        if (t?.t === 'str') texture = t.v.replace(/\\\\/g, '\\');
      }
      if (o.type !== '@ref') this.skipBlock();
    }
    this.i++;
    return { diffuse, power, specular, emissive, texture };
  }

  private mesh(transform: Float64Array): void {
    const mesh: XMesh = {
      positions: this.nums(this.num() * 3),
      faces: this.faces(),
      normals: null, normalFaces: null, uvs: null, faceMaterials: [], materials: [], transform,
    };
    for (let o = this.open(); o; o = this.open()) {
      switch (o.type) {
        case 'MeshNormals':
          mesh.normals = this.nums(this.num() * 3);
          mesh.normalFaces = this.faces();
          this.skipBlock();
          break;
        case 'MeshTextureCoords':
          mesh.uvs = this.nums(this.num() * 2);
          this.skipBlock();
          break;
        case 'MeshMaterialList': {
          this.num(); // material count
          mesh.faceMaterials = this.nums(this.num());
          for (let m = this.open(); m; m = this.open()) {
            if (m.type === 'Material') mesh.materials.push(this.material());
            else if (m.type === '@ref') {
              const named = this.namedMaterials.get(m.name);
              if (named) mesh.materials.push(named);
            } else this.skipBlock();
          }
          this.i++;
          break;
        }
        case '@ref':
          break;
        default:
          this.skipBlock();
      }
    }
    this.i++;
    this.meshes.push(mesh);
  }
}

function decodeText(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(bytes);
  } catch {
    return new TextDecoder('shift-jis').decode(bytes); // most MMD accessories
  }
}

export function parseX(buffer: ArrayBuffer, name: string): PmxModel {
  const magic = new TextDecoder('ascii').decode(new Uint8Array(buffer, 0, Math.min(16, buffer.byteLength)));
  if (!magic.startsWith('xof ')) throw new Error(`${name} 不是 DirectX .x 文件`);
  const kind = magic.slice(8, 12);
  if (kind !== 'txt ') {
    throw new Error(`${name} 是${kind.includes('zip') ? '压缩' : '二进制'}格式的 .x 文件，目前只支持文本格式（可用 PMX Editor 或 Blender 另存）`);
  }
  const parser = new Parser(tokenize(decodeText(buffer).slice(16)));
  parser.parse();

  const positions: number[] = [];
  const normals: number[] = [];
  const uvs: number[] = [];
  const textures: string[] = [];
  const textureIndex = (path: string | null): number => {
    if (!path) return -1;
    const i = textures.indexOf(path);
    return i >= 0 ? i : textures.push(path) - 1;
  };
  const materials: PmxMaterial[] = [];
  const indexGroups: number[][] = [];
  const fallback: XMaterial = { diffuse: [0.8, 0.8, 0.8, 1], power: 0, specular: [0, 0, 0], emissive: [0.2, 0.2, 0.2], texture: null };

  for (const mesh of parser.meshes) {
    const t = mesh.transform;
    const base = materials.length;
    const meshMaterials = mesh.materials.length ? mesh.materials : [fallback];
    for (const [k, m] of meshMaterials.entries()) {
      // "color.png*shine.sph" names a texture and a sphere map; a lone .sph / .spa is a sphere map.
      const [first, second] = (m.texture ?? '').split('*');
      const isSphere = (p?: string) => !!p && /\.sp[ha]$/i.test(p);
      const sphere = isSphere(second) ? second : isSphere(first) ? first : null;
      const tex = first && !isSphere(first) ? first : null;
      materials.push({
        name: `材質${base + k + 1}`, nameEn: `Material${base + k + 1}`,
        diffuse: m.diffuse, specular: m.specular, specularPower: m.power, ambient: m.emissive,
        flags: MaterialFlag.GroundShadow | MaterialFlag.CastSelfShadow | MaterialFlag.ReceiveSelfShadow,
        edgeColor: [0, 0, 0, 1], edgeSize: 0,
        textureIndex: textureIndex(tex), sphereTextureIndex: textureIndex(sphere),
        sphereMode: sphere ? (/\.spa$/i.test(sphere) ? SphereMode.Add : SphereMode.Multiply) : SphereMode.None,
        sharedToon: false, toonIndex: -1, memo: '', indexCount: 0,
      });
      indexGroups.push([]);
    }

    // Normals are indexed separately: one output vertex per (position, normal) pair used.
    const smooth = mesh.normals ? null : smoothNormals(mesh);
    const seen = new Map<number, number>();
    const normalCount = mesh.normals ? mesh.normals.length / 3 : 0;
    const vertex = (p: number, n: number): number => {
      const key = p * (normalCount + 1) + n;
      let v = seen.get(key);
      if (v !== undefined) return v;
      v = positions.length / 3;
      seen.set(key, v);
      const x = mesh.positions[p * 3], y = mesh.positions[p * 3 + 1], z = mesh.positions[p * 3 + 2];
      positions.push(
        (t[0] * x + t[4] * y + t[8] * z + t[12]) * ACCESSORY_SCALE,
        (t[1] * x + t[5] * y + t[9] * z + t[13]) * ACCESSORY_SCALE,
        (t[2] * x + t[6] * y + t[10] * z + t[14]) * ACCESSORY_SCALE,
      );
      const src = mesh.normals ?? smooth!;
      const ni = mesh.normals ? n : p;
      const nx = src[ni * 3] ?? 0, ny = src[ni * 3 + 1] ?? 1, nz = src[ni * 3 + 2] ?? 0;
      const wx = t[0] * nx + t[4] * ny + t[8] * nz, wy = t[1] * nx + t[5] * ny + t[9] * nz, wz = t[2] * nx + t[6] * ny + t[10] * nz;
      const len = Math.hypot(wx, wy, wz) || 1;
      normals.push(wx / len, wy / len, wz / len);
      uvs.push(mesh.uvs?.[p * 2] ?? 0, mesh.uvs?.[p * 2 + 1] ?? 0);
      return v;
    };
    mesh.faces.forEach((face, f) => {
      const group = indexGroups[base + Math.min(mesh.faceMaterials[f] ?? 0, meshMaterials.length - 1)];
      const nf = mesh.normalFaces?.[f];
      const corner = (k: number) => vertex(face[k], nf?.[k] ?? 0);
      // Fan-triangulate quads and larger polygons.
      for (let k = 1; k + 1 < face.length; k++) group.push(corner(0), corner(k), corner(k + 1));
    });
  }

  const vertexCount = positions.length / 3;
  if (vertexCount === 0) throw new Error(`${name} 里没有网格`);
  materials.forEach((m, i) => (m.indexCount = indexGroups[i].length));
  const skinWeights = new Float32Array(vertexCount * 4);
  for (let v = 0; v < vertexCount; v++) skinWeights[v * 4] = 1;

  return {
    format: 'X',
    accessory: true,
    header: {
      version: 2, encoding: 'utf-16le', additionalUvCount: 0, vertexIndexSize: vertexCount > 0xffff ? 4 : 2, textureIndexSize: 1,
      materialIndexSize: 1, boneIndexSize: 1, morphIndexSize: 1, rigidBodyIndexSize: 1,
    },
    name: name.replace(/\.x$/i, ''), nameEn: '', comment: '', commentEn: '',
    vertexCount,
    positions: Float32Array.from(positions), normals: Float32Array.from(normals), uvs: Float32Array.from(uvs),
    additionalUvs: [],
    skinTypes: new Uint8Array(vertexCount), skinIndices: new Uint32Array(vertexCount * 4), skinWeights,
    sdef: { vertexIndices: new Uint32Array(0), c: new Float32Array(0), r0: new Float32Array(0), r1: new Float32Array(0) },
    edgeScales: new Float32Array(vertexCount).fill(1),
    indices: Uint32Array.from(indexGroups.flat()),
    textures,
    materials: materials.filter((m) => m.indexCount > 0),
    bones: [{
      name: 'センター', nameEn: 'center', position: [0, 0, 0], parentIndex: -1, layer: 0,
      flags: BoneFlag.Rotatable | BoneFlag.Translatable | BoneFlag.Visible | BoneFlag.Enabled,
      tailBoneIndex: -1, tailOffset: [0, 0, 0], inherit: null, fixedAxis: null, localAxes: null, externalParentKey: -1, ik: null,
    }],
    morphs: [],
    displayFrames: [],
    rigidBodies: [],
    joints: [],
  };
}

/** Area-weighted vertex normals for meshes that come without any. */
function smoothNormals(mesh: XMesh): Float32Array {
  const out = new Float32Array(mesh.positions.length);
  const p = mesh.positions;
  for (const face of mesh.faces) {
    for (let k = 1; k + 1 < face.length; k++) {
      const a = face[0] * 3, b = face[k] * 3, c = face[k + 1] * 3;
      const ux = p[b] - p[a], uy = p[b + 1] - p[a + 1], uz = p[b + 2] - p[a + 2];
      const vx = p[c] - p[a], vy = p[c + 1] - p[a + 1], vz = p[c + 2] - p[a + 2];
      // Clockwise faces front the viewer in a left-handed space.
      const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      for (const i of [a, b, c]) {
        out[i] += nx; out[i + 1] += ny; out[i + 2] += nz;
      }
    }
  }
  return out;
}
