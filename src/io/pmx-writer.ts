import { BoneFlag, MaterialFlag, MorphKind, SkinType, type PmxDisplayFrame, type PmxModel } from '../loaders/pmx';

// PMX serializer (inverse of loaders/pmx.ts), UTF-16LE. Used to export models that didn't
// come from a PMX (PMD → PMX); PMX sources are exported byte-for-byte instead.

class ByteWriter {
  private buf = new Uint8Array(1 << 20);
  private view = new DataView(this.buf.buffer);
  length = 0;

  private ensure(n: number): void {
    if (this.length + n <= this.buf.length) return;
    let cap = this.buf.length * 2;
    while (cap < this.length + n) cap *= 2;
    const next = new Uint8Array(cap);
    next.set(this.buf.subarray(0, this.length));
    this.buf = next;
    this.view = new DataView(next.buffer);
  }
  u8(v: number) { this.ensure(1); this.view.setUint8(this.length, v); this.length += 1; }
  i8(v: number) { this.ensure(1); this.view.setInt8(this.length, v); this.length += 1; }
  u16(v: number) { this.ensure(2); this.view.setUint16(this.length, v, true); this.length += 2; }
  i16(v: number) { this.ensure(2); this.view.setInt16(this.length, v, true); this.length += 2; }
  i32(v: number) { this.ensure(4); this.view.setInt32(this.length, v, true); this.length += 4; }
  f32(...vs: (number | ArrayLike<number>)[]) {
    for (const v of vs) {
      if (typeof v === 'number') { this.ensure(4); this.view.setFloat32(this.length, v, true); this.length += 4; }
      else for (let i = 0; i < v.length; i++) { this.ensure(4); this.view.setFloat32(this.length, v[i], true); this.length += 4; }
    }
  }
  bytes(b: Uint8Array) { this.ensure(b.length); this.buf.set(b, this.length); this.length += b.length; }
  result(): ArrayBuffer { return this.buf.slice(0, this.length).buffer; }
}

const signedSize = (count: number) => (count <= 0x7f ? 1 : count <= 0x7fff ? 2 : 4);
const vertexSize = (count: number) => (count <= 0xff ? 1 : count <= 0xffff ? 2 : 4);

function utf16le(text: string): Uint8Array {
  const out = new Uint8Array(text.length * 2);
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    out[i * 2] = c & 0xff;
    out[i * 2 + 1] = c >> 8;
  }
  return out;
}

/** MMD expects the first two display frames to be the special "Root" and "表情" frames. */
function standardDisplayFrames(model: PmxModel): PmxDisplayFrame[] {
  const frames = model.displayFrames;
  if (frames.length >= 2 && frames[0].special && frames[1].special) return frames;
  const root: PmxDisplayFrame = { name: 'Root', nameEn: 'Root', special: true, items: model.bones.length ? [{ type: 'bone', index: 0 }] : [] };
  const face: PmxDisplayFrame = {
    name: '表情', nameEn: 'Exp', special: true,
    items: model.morphs.map((_, index) => ({ type: 'morph' as const, index })),
  };
  return [root, face, ...frames.filter((f) => !f.special)];
}

export function writePmx(model: PmxModel): ArrayBuffer {
  const w = new ByteWriter();
  const additionalUvCount = model.additionalUvs.length;
  const sizes = {
    vertex: vertexSize(model.vertexCount),
    texture: signedSize(model.textures.length),
    material: signedSize(model.materials.length),
    bone: signedSize(model.bones.length),
    morph: signedSize(model.morphs.length),
    rigid: signedSize(model.rigidBodies.length),
  };
  const needs21 =
    Array.prototype.some.call(model.skinTypes, (t: number) => t === SkinType.QDEF) ||
    model.morphs.some((m) => m.kind === MorphKind.Flip || m.kind === MorphKind.Impulse) ||
    model.materials.some((m) => m.flags & (MaterialFlag.VertexColor | MaterialFlag.PointDraw | MaterialFlag.LineDraw));

  const text = (s: string) => {
    const b = utf16le(s);
    w.i32(b.length);
    w.bytes(b);
  };
  const idx = (size: number, v: number) => (size === 1 ? w.i8(v) : size === 2 ? w.i16(v) : w.i32(v));
  const vIdx = (v: number) => (sizes.vertex === 1 ? w.u8(v) : sizes.vertex === 2 ? w.u16(v) : w.i32(v));
  const bone = (v: number) => idx(sizes.bone, v);

  // ---- Header ----
  w.bytes(new Uint8Array([0x50, 0x4d, 0x58, 0x20])); // "PMX "
  w.f32(needs21 ? 2.1 : 2.0);
  w.u8(8);
  w.bytes(new Uint8Array([0, additionalUvCount, sizes.vertex, sizes.texture, sizes.material, sizes.bone, sizes.morph, sizes.rigid]));
  text(model.name);
  text(model.nameEn);
  text(model.comment);
  text(model.commentEn);

  // ---- Vertices ----
  const sdefOf = new Map<number, number>();
  model.sdef.vertexIndices.forEach((v, i) => sdefOf.set(v, i));
  w.i32(model.vertexCount);
  for (let v = 0; v < model.vertexCount; v++) {
    w.f32(model.positions.subarray(v * 3, v * 3 + 3), model.normals.subarray(v * 3, v * 3 + 3), model.uvs.subarray(v * 2, v * 2 + 2));
    for (const uv of model.additionalUvs) w.f32(uv.subarray(v * 4, v * 4 + 4));
    const type = model.skinTypes[v];
    const bi = model.skinIndices.subarray(v * 4, v * 4 + 4);
    const bw = model.skinWeights.subarray(v * 4, v * 4 + 4);
    const sdefIndex = sdefOf.get(v);
    const effective = type === SkinType.SDEF && sdefIndex === undefined ? SkinType.BDEF2 : type;
    w.u8(effective);
    switch (effective) {
      case SkinType.BDEF1:
        bone(bi[0]);
        break;
      case SkinType.BDEF2:
        bone(bi[0]); bone(bi[1]); w.f32(bw[0]);
        break;
      case SkinType.SDEF: {
        const s = sdefIndex!;
        bone(bi[0]); bone(bi[1]); w.f32(bw[0]);
        w.f32(model.sdef.c.subarray(s * 3, s * 3 + 3), model.sdef.r0.subarray(s * 3, s * 3 + 3), model.sdef.r1.subarray(s * 3, s * 3 + 3));
        break;
      }
      default: // BDEF4, QDEF
        for (let k = 0; k < 4; k++) bone(bi[k]);
        w.f32(bw);
    }
    w.f32(model.edgeScales[v]);
  }

  // ---- Faces ----
  w.i32(model.indices.length);
  for (const i of model.indices) vIdx(i);

  // ---- Textures ----
  w.i32(model.textures.length);
  for (const t of model.textures) text(t);

  // ---- Materials ----
  w.i32(model.materials.length);
  for (const m of model.materials) {
    text(m.name);
    text(m.nameEn);
    w.f32(m.diffuse, m.specular, [m.specularPower], m.ambient);
    w.u8(m.flags);
    w.f32(m.edgeColor, [m.edgeSize]);
    idx(sizes.texture, m.textureIndex);
    idx(sizes.texture, m.sphereTextureIndex);
    w.u8(m.sphereMode);
    w.u8(m.sharedToon ? 1 : 0);
    if (m.sharedToon) w.u8(m.toonIndex);
    else idx(sizes.texture, m.toonIndex);
    text(m.memo);
    w.i32(m.indexCount);
  }

  // ---- Bones ----
  w.i32(model.bones.length);
  for (const b of model.bones) {
    text(b.name);
    text(b.nameEn);
    w.f32(b.position);
    bone(b.parentIndex);
    w.i32(b.layer);
    // Flags must describe exactly the optional fields written below.
    let flags = b.flags & ~(BoneFlag.TailIsBone | BoneFlag.IK | BoneFlag.InheritRotation | BoneFlag.InheritTranslation | BoneFlag.FixedAxis | BoneFlag.LocalAxes | BoneFlag.ExternalParent);
    // Keep the flag even with index -1 (valid PMX: "tail is a bone", but none).
    if (b.tailBoneIndex >= 0 || b.flags & BoneFlag.TailIsBone) flags |= BoneFlag.TailIsBone;
    if (b.inherit) flags |= b.flags & (BoneFlag.InheritRotation | BoneFlag.InheritTranslation);
    if (b.fixedAxis) flags |= BoneFlag.FixedAxis;
    if (b.localAxes) flags |= BoneFlag.LocalAxes;
    if (b.externalParentKey >= 0 && b.flags & BoneFlag.ExternalParent) flags |= BoneFlag.ExternalParent;
    if (b.ik) flags |= BoneFlag.IK;
    w.u16(flags);
    if (flags & BoneFlag.TailIsBone) bone(b.tailBoneIndex);
    else w.f32(b.tailOffset);
    if (flags & (BoneFlag.InheritRotation | BoneFlag.InheritTranslation)) {
      bone(b.inherit!.parentIndex);
      w.f32(b.inherit!.weight);
    }
    if (flags & BoneFlag.FixedAxis) w.f32(b.fixedAxis!);
    if (flags & BoneFlag.LocalAxes) w.f32(b.localAxes!.x, b.localAxes!.z);
    if (flags & BoneFlag.ExternalParent) w.i32(b.externalParentKey);
    if (b.ik) {
      bone(b.ik.targetIndex);
      w.i32(b.ik.loopCount);
      w.f32(b.ik.limitAngle);
      w.i32(b.ik.links.length);
      for (const l of b.ik.links) {
        bone(l.boneIndex);
        w.u8(l.limits ? 1 : 0);
        if (l.limits) w.f32(l.limits.min, l.limits.max);
      }
    }
  }

  // ---- Morphs ----
  w.i32(model.morphs.length);
  for (const m of model.morphs) {
    text(m.name);
    text(m.nameEn);
    w.u8(m.panel);
    w.u8(m.kind);
    switch (m.kind) {
      case MorphKind.Group:
      case MorphKind.Flip:
        w.i32(m.children.length);
        for (const c of m.children) { idx(sizes.morph, c.morphIndex); w.f32(c.weight); }
        break;
      case MorphKind.Vertex:
        w.i32(m.vertexIndices.length);
        m.vertexIndices.forEach((v, k) => { vIdx(v); w.f32(m.offsets.subarray(k * 3, k * 3 + 3)); });
        break;
      case MorphKind.Bone:
        w.i32(m.bones.length);
        for (const b of m.bones) { bone(b.boneIndex); w.f32(b.translation, b.rotation); }
        break;
      case MorphKind.Material:
        w.i32(m.materials.length);
        for (const o of m.materials) {
          idx(sizes.material, o.materialIndex);
          w.u8(o.operation);
          w.f32(o.diffuse, o.specular, [o.specularPower], o.ambient, o.edgeColor, [o.edgeSize], o.textureTint, o.sphereTint, o.toonTint);
        }
        break;
      case MorphKind.Impulse:
        w.i32(m.impulses.length);
        for (const o of m.impulses) { idx(sizes.rigid, o.rigidBodyIndex); w.u8(o.local ? 1 : 0); w.f32(o.velocity, o.torque); }
        break;
      default: // UV and additional UV morphs
        w.i32(m.vertexIndices.length);
        m.vertexIndices.forEach((v, k) => { vIdx(v); w.f32(m.offsets.subarray(k * 4, k * 4 + 4)); });
    }
  }

  // ---- Display frames ----
  const frames = standardDisplayFrames(model);
  w.i32(frames.length);
  for (const f of frames) {
    text(f.name);
    text(f.nameEn);
    w.u8(f.special ? 1 : 0);
    w.i32(f.items.length);
    for (const it of f.items) {
      w.u8(it.type === 'bone' ? 0 : 1);
      idx(it.type === 'bone' ? sizes.bone : sizes.morph, it.index);
    }
  }

  // ---- Rigid bodies ----
  w.i32(model.rigidBodies.length);
  for (const r of model.rigidBodies) {
    text(r.name);
    text(r.nameEn);
    bone(r.boneIndex);
    w.u8(r.group);
    w.u16(r.collisionMask);
    w.u8(r.shape);
    w.f32(r.size, r.position, r.rotation, [r.mass, r.linearDamping, r.angularDamping, r.restitution, r.friction]);
    w.u8(r.mode);
  }

  // ---- Joints ----
  w.i32(model.joints.length);
  for (const j of model.joints) {
    text(j.name);
    text(j.nameEn);
    w.u8(j.type);
    idx(sizes.rigid, j.rigidBodyA);
    idx(sizes.rigid, j.rigidBodyB);
    w.f32(j.position, j.rotation, j.positionMin, j.positionMax, j.rotationMin, j.rotationMax, j.springPosition, j.springRotation);
  }

  if (needs21) w.i32(0); // soft bodies
  return w.result();
}
