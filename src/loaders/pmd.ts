import { BinaryReader, decodeFixedString, type Vec3Tuple } from './binary-reader';
import {
  BoneFlag, MaterialFlag, MorphKind, SkinType, SphereMode,
  type PmxBone, type PmxMaterial, type PmxModel, type PmxMorph, type PmxRigidBody, type PmxJoint,
} from './pmx';

// PMD (MikuMikuDance 1.x model) parser, converted to the PmxModel structure the runtime
// uses, following MMD's / PmxEditor's conversion rules:
//   * vertices are BDEF2 (weight 0–100 for bone 0), edge flag 1 = no edge;
//   * materials with alpha < 1 are double-sided; alpha 0.98 opts out of self-shadow;
//   * textures are "main.bmp*sphere.sph" (.sph multiply, .spa add);
//   * IK angle limits are stored ÷4; knee ("ひざ") links get MMD's implicit X-only limit;
//   * morphs are offsets relative to the base ("base") morph's vertex list;
//   * rigid body positions are relative to their bone.

const NO_INDEX = 0xffff;
// float32 like every other value read from a model file (so PMD → PMX export is exact).
const KNEE_LIMIT = { min: [Math.fround(-Math.PI), 0, 0] as Vec3Tuple, max: [Math.fround(-0.008726646), 0, 0] as Vec3Tuple };

export function parsePmd(buffer: ArrayBuffer): PmxModel {
  const r = new BinaryReader(buffer);
  const sjis = new TextDecoder('shift-jis');
  const str = (len: number) => decodeFixedString(r.bytes(len), sjis);

  const magic = String.fromCharCode(r.u8(), r.u8(), r.u8());
  if (magic !== 'Pmd') throw new Error(`Not a PMD file (magic "${magic}")`);
  const version = r.f32();
  const name = str(20);
  const comment = str(256);

  // ---- Vertices ----
  const vertexCount = r.u32();
  const positions = new Float32Array(vertexCount * 3);
  const normals = new Float32Array(vertexCount * 3);
  const uvs = new Float32Array(vertexCount * 2);
  const skinTypes = new Uint8Array(vertexCount).fill(SkinType.BDEF2);
  const skinIndices = new Uint32Array(vertexCount * 4);
  const skinWeights = new Float32Array(vertexCount * 4);
  const edgeScales = new Float32Array(vertexCount);
  for (let v = 0; v < vertexCount; v++) {
    positions[v * 3] = r.f32(); positions[v * 3 + 1] = r.f32(); positions[v * 3 + 2] = r.f32();
    normals[v * 3] = r.f32(); normals[v * 3 + 1] = r.f32(); normals[v * 3 + 2] = r.f32();
    uvs[v * 2] = r.f32(); uvs[v * 2 + 1] = r.f32();
    skinIndices[v * 4] = r.u16();
    skinIndices[v * 4 + 1] = r.u16();
    const w = r.u8() / 100;
    skinWeights[v * 4] = w;
    skinWeights[v * 4 + 1] = 1 - w;
    edgeScales[v] = r.u8() === 0 ? 1 : 0;
  }

  // ---- Faces ----
  const indexCount = r.u32();
  const indices = new Uint32Array(indexCount);
  for (let i = 0; i < indexCount; i++) indices[i] = r.u16();

  // ---- Materials ----
  const textures: string[] = [];
  const textureIndex = (path: string): number => {
    let i = textures.indexOf(path);
    if (i < 0) i = textures.push(path) - 1;
    return i;
  };
  const materialCount = r.u32();
  const rawMaterials = [];
  for (let i = 0; i < materialCount; i++) {
    rawMaterials.push({
      diffuse: [r.f32(), r.f32(), r.f32(), r.f32()] as [number, number, number, number],
      specularPower: r.f32(),
      specular: r.vec3(),
      ambient: r.vec3(),
      toon: r.u8(),
      edge: r.u8() === 1,
      indexCount: r.u32(),
      file: str(20),
    });
  }

  // ---- Bones ----
  const boneCount = r.u16();
  const rawBones: { name: string; parent: number; tail: number; type: number; ikParent: number; position: Vec3Tuple }[] = [];
  for (let i = 0; i < boneCount; i++) {
    rawBones.push({ name: str(20), parent: r.u16(), tail: r.u16(), type: r.u8(), ikParent: r.u16(), position: r.vec3() });
  }

  // ---- IK ----
  const ikCount = r.u16();
  const rawIks = [];
  for (let i = 0; i < ikCount; i++) {
    const ikBone = r.u16();
    const target = r.u16();
    const chainLength = r.u8();
    const iterations = r.u16();
    const limit = r.f32();
    const links: number[] = [];
    for (let k = 0; k < chainLength; k++) links.push(r.u16());
    rawIks.push({ ikBone, target, iterations, limit, links });
  }

  // ---- Morphs ("skins") ----
  const morphCount = r.u16();
  const rawMorphs = [];
  for (let i = 0; i < morphCount; i++) {
    const mName = str(20);
    const n = r.u32();
    const type = r.u8();
    const vertexIndices = new Int32Array(n);
    const offsets = new Float32Array(n * 3);
    for (let k = 0; k < n; k++) {
      vertexIndices[k] = r.u32();
      offsets[k * 3] = r.f32(); offsets[k * 3 + 1] = r.f32(); offsets[k * 3 + 2] = r.f32();
    }
    rawMorphs.push({ name: mName, type, vertexIndices, offsets });
  }

  // ---- Display lists (only needed to reach the optional sections after them) ----
  r.skip(r.u8() * 2); // morph display
  const boneFrameNameCount = r.u8();
  const boneFrameNames: string[] = [];
  for (let i = 0; i < boneFrameNameCount; i++) boneFrameNames.push(str(50));
  const boneDisplayCount = r.u32();
  const boneDisplay: { bone: number; frame: number }[] = [];
  for (let i = 0; i < boneDisplayCount; i++) boneDisplay.push({ bone: r.u16(), frame: r.u8() });

  // ---- Optional: English names ----
  let nameEn = '';
  let commentEn = '';
  const boneNamesEn: string[] = [];
  const morphNamesEn: string[] = [];
  if (r.remaining > 0 && r.u8() === 1) {
    nameEn = str(20);
    commentEn = str(256);
    for (let i = 0; i < boneCount; i++) boneNamesEn.push(str(20));
    for (let i = 1; i < morphCount; i++) morphNamesEn.push(str(20));
    r.skip(boneFrameNameCount * 50);
  }

  // ---- Optional: toon texture list (toon01.bmp … by default) ----
  const toonFiles: string[] = [];
  if (r.remaining >= 1000) {
    for (let i = 0; i < 10; i++) toonFiles.push(str(100));
  }

  // ---- Optional: physics ----
  const rigidBodies: PmxRigidBody[] = [];
  if (r.remaining >= 4) {
    const n = r.u32();
    for (let i = 0; i < n; i++) {
      const rbName = str(20);
      const bone = r.u16();
      const group = r.u8();
      const collisionMask = r.u16();
      const shape = r.u8();
      const size = r.vec3();
      const pos = r.vec3();
      const rotation = r.vec3();
      const [mass, linearDamping, angularDamping, restitution, friction] = [r.f32(), r.f32(), r.f32(), r.f32(), r.f32()];
      const mode = r.u8();
      const boneIndex = bone === NO_INDEX || bone >= boneCount ? -1 : bone;
      // Positions are relative to the bone (bone-less bodies: relative to bone 0).
      const base = rawBones[boneIndex >= 0 ? boneIndex : 0]?.position ?? [0, 0, 0];
      rigidBodies.push({
        name: rbName, nameEn: '', boneIndex, group, collisionMask, shape, size,
        position: [pos[0] + base[0], pos[1] + base[1], pos[2] + base[2]],
        rotation, mass, linearDamping, angularDamping, restitution, friction, mode,
      });
    }
  }
  const joints: PmxJoint[] = [];
  if (r.remaining >= 4) {
    const n = r.u32();
    for (let i = 0; i < n; i++) {
      joints.push({
        name: str(20), nameEn: '', type: 0,
        rigidBodyA: r.u32(), rigidBodyB: r.u32(),
        position: r.vec3(), rotation: r.vec3(),
        positionMin: r.vec3(), positionMax: r.vec3(),
        rotationMin: r.vec3(), rotationMax: r.vec3(),
        springPosition: r.vec3(), springRotation: r.vec3(),
      });
    }
  }

  // ---- Convert materials ----
  const materials: PmxMaterial[] = rawMaterials.map((m, i) => {
    let tex = -1;
    let sphere = -1;
    let sphereMode = SphereMode.None;
    for (const part of m.file.split('*').map((p) => p.trim()).filter(Boolean)) {
      const lower = part.toLowerCase();
      if (lower.endsWith('.sph') || lower.endsWith('.spa')) {
        sphere = textureIndex(part);
        sphereMode = lower.endsWith('.spa') ? SphereMode.Add : SphereMode.Multiply;
      } else {
        tex = textureIndex(part);
      }
    }

    // Toon: 0xFF = none; otherwise an entry of the toon list (default toonNN.bmp → shared).
    let sharedToon = false;
    let toonIndex = -1;
    if (m.toon !== 0xff && m.toon < 10) {
      const file = toonFiles[m.toon] || `toon${String(m.toon + 1).padStart(2, '0')}.bmp`;
      const shared = /^toon(0[1-9]|10)\.bmp$/i.exec(file);
      if (shared) {
        sharedToon = true;
        toonIndex = parseInt(shared[1], 10) - 1;
      } else {
        toonIndex = textureIndex(file);
      }
    }

    const alpha = m.diffuse[3];
    let flags = MaterialFlag.GroundShadow;
    if (alpha < 1) flags |= MaterialFlag.DoubleSided;
    if (Math.abs(alpha - 0.98) > 1e-4) flags |= MaterialFlag.CastSelfShadow | MaterialFlag.ReceiveSelfShadow;
    if (m.edge) flags |= MaterialFlag.Edge;

    return {
      name: `材質${i + 1}`, nameEn: '', diffuse: m.diffuse, specular: m.specular, specularPower: m.specularPower,
      ambient: m.ambient, flags, edgeColor: [0, 0, 0, 1], edgeSize: 1,
      textureIndex: tex, sphereTextureIndex: sphere, sphereMode, sharedToon, toonIndex, memo: '',
      indexCount: m.indexCount,
    };
  });

  // ---- Convert bones ----
  const ikByBone = new Map(rawIks.map((ik) => [ik.ikBone, ik]));
  const bones: PmxBone[] = rawBones.map((b, i) => {
    const valid = (x: number) => (x === NO_INDEX || x >= boneCount ? -1 : x);
    let flags: number = BoneFlag.Rotatable | BoneFlag.Enabled;
    if (b.type !== 7) flags |= BoneFlag.Visible; // 7 = invisible
    if (b.type === 1 || b.type === 2) flags |= BoneFlag.Translatable; // 1 = rotate+move, 2 = IK
    const tail = valid(b.tail);
    if (tail > 0) flags |= BoneFlag.TailIsBone;

    // 5 = rotation influenced ("回転影響下"): copies the rotation of the bone in ikParent.
    let inherit: PmxBone['inherit'] = null;
    if (b.type === 5 && valid(b.ikParent) >= 0) {
      flags |= BoneFlag.InheritRotation;
      inherit = { parentIndex: b.ikParent, weight: 1 };
    }

    const ik = ikByBone.get(i);
    if (ik) flags |= BoneFlag.IK;
    return {
      name: b.name, nameEn: boneNamesEn[i] ?? '', position: b.position, parentIndex: valid(b.parent), layer: 0, flags,
      tailBoneIndex: tail > 0 ? tail : -1, tailOffset: [0, 0, 0], inherit, fixedAxis: null, localAxes: null,
      externalParentKey: -1,
      ik: ik
        ? {
            targetIndex: ik.target,
            loopCount: ik.iterations,
            limitAngle: ik.limit * 4,
            links: ik.links.map((l) => ({ boneIndex: l, limits: rawBones[l]?.name.includes('ひざ') ? KNEE_LIMIT : null })),
          }
        : null,
    };
  });

  // ---- Convert morphs: offsets reference the base morph's vertex list ----
  const baseMorph = rawMorphs.find((m) => m.type === 0);
  const morphs: PmxMorph[] = [];
  rawMorphs.forEach((m) => {
    if (m.type === 0 || !baseMorph) return;
    const vertexIndices = new Int32Array(m.vertexIndices.length);
    m.vertexIndices.forEach((k, j) => (vertexIndices[j] = k < baseMorph.vertexIndices.length ? baseMorph.vertexIndices[k] : -1));
    morphs.push({
      name: m.name, nameEn: morphNamesEn[morphs.length] ?? '', panel: m.type,
      kind: MorphKind.Vertex, vertexIndices, offsets: m.offsets,
    });
  });

  return {
    format: 'PMD',
    header: {
      version, encoding: 'utf-8', additionalUvCount: 0, vertexIndexSize: 2, textureIndexSize: 1,
      materialIndexSize: 1, boneIndexSize: 2, morphIndexSize: 2, rigidBodyIndexSize: 4,
    },
    name, nameEn, comment, commentEn,
    vertexCount, positions, normals, uvs, additionalUvs: [], skinTypes, skinIndices, skinWeights,
    sdef: { vertexIndices: new Uint32Array(0), c: new Float32Array(0), r0: new Float32Array(0), r1: new Float32Array(0) },
    edgeScales, indices, textures, materials, bones, morphs,
    displayFrames: boneFrameNames.map((n, fi) => ({
      name: n, nameEn: '', special: false,
      items: boneDisplay.filter((d) => d.frame === fi + 1).map((d) => ({ type: 'bone' as const, index: d.bone })),
    })),
    rigidBodies, joints,
  };
}
