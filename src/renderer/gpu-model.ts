import { MaterialFlag, MorphKind, type PmxModel } from '../loaders/pmx';
import { classifyMaterial, MATERIAL_UNIFORM_FLOATS, MATERIAL_UNIFORM_STRIDE, MaterialPart, showsThroughHair } from '../runtime/morph-controller';
import type { MmdModel } from '../runtime/mmd-model';
import { mat4 } from '../math';
import type { ClothCollision } from '../physics/cloth-collision';
import type { PhysicsRig } from '../physics/rig';
import { PREPASS_PART_FORMAT, type MaterialPipelines, type RendererCore } from './renderer';

// GPU resources for one model. Per frame the CPU uploads only bone data, morph
// weights and material uniforms; morphing and skinning happen in one compute pass.

const VERTEX_IN_BYTES = 160; // must match VertexIn in skinning.wgsl
const COLLIDER_FLOATS = 20; // must match Collider in skinning.wgsl
const SKINNED_FLOATS = 10; // position, normal, uv, additional uv 1 (xy)

export interface MaterialTextures {
  texture: GPUTexture | null;
  sphere: GPUTexture | null;
  toon: GPUTexture | null;
}

/** Auxiliary maps a custom material shader can read, per material. */
export interface AuxTextures {
  ramp: GPUTexture | null;
  light: GPUTexture | null;
  face: GPUTexture | null;
}

export class GpuModel {
  readonly skinned: GPUBuffer;
  private readonly buffers: GPUBuffer[] = [];
  private readonly boneBuffer: GPUBuffer;
  private readonly morphWeightBuffer: GPUBuffer;
  private readonly materialBuffer: GPUBuffer;
  private computeBindGroup!: GPUBindGroup;
  private computeBuffers!: GPUBuffer[];
  private clothSlotBuffer: GPUBuffer | null = null;
  private clothDataBuffer: GPUBuffer | null = null;
  private clothActive = false;
  private readonly workgroups: number;
  private readonly paramsBuffer: GPUBuffer;
  /** vertexCount, colliderCount, boneCount, clothActive, then the root matrix. */
  private readonly paramsData = new ArrayBuffer(80);
  private readonly colliderData: Float32Array;
  private shader: MaterialPipelines | null = null;
  private bundle: GPURenderBundle | null = null;
  private bundleEdges: boolean | null = null;
  /** The frame bind group each bundle was recorded with (it changes when the scene is resized). */
  private bundleFrame: GPUBindGroup | null = null;
  private groundFrame: GPUBindGroup | null = null;
  private shadowBundles: GPURenderBundle[] = [];
  private prepassBundle: GPURenderBundle | null = null;
  private groundBundle: GPURenderBundle | null = null;
  private readonly drawRanges: { first: number; count: number }[];
  /** MaterialPart of each material, from its name. */
  private readonly parts: number[];
  /** Materials drawn again through the hair in front of them (eyes, brows). */
  private readonly throughHair: boolean[];
  /** 1 = the user hid this material: it is drawn in no pass. */
  private readonly hidden: Uint8Array;

  private readonly edgeScaleBuffer: GPUBuffer;
  private readonly indexBuffer: GPUBuffer;
  private readonly indexFormat: GPUIndexFormat;
  private materialBindGroups: GPUBindGroup[] = [];

  constructor(
    private readonly core: RendererCore,
    readonly model: MmdModel,
    private readonly textures: MaterialTextures[],
    /** Vertex-level cloth collision setup and the rig its colliders belong to (null = none). */
    private readonly cloth: { data: ClothCollision; rig: PhysicsRig } | null = null,
  ) {
    const pmx = model.pmx;
    const n = pmx.vertexCount;

    // ---- Static vertex input (VertexIn) + morph CSR ----
    const { entries, ranges, uv1Counts } = buildMorphEntries(pmx);
    const uv1 = pmx.additionalUvs[0];
    const vertexIn = new ArrayBuffer(Math.max(n, 1) * VERTEX_IN_BYTES);
    const f32 = new Float32Array(vertexIn);
    const u32 = new Uint32Array(vertexIn);
    const sdefIndexOf = new Int32Array(n).fill(-1);
    pmx.sdef.vertexIndices.forEach((v, i) => (sdefIndexOf[v] = i));
    for (let v = 0; v < n; v++) {
      const o = (v * VERTEX_IN_BYTES) / 4;
      f32[o] = pmx.positions[v * 3]; f32[o + 1] = pmx.positions[v * 3 + 1]; f32[o + 2] = pmx.positions[v * 3 + 2];
      u32[o + 3] = pmx.skinTypes[v];
      f32[o + 4] = pmx.normals[v * 3]; f32[o + 5] = pmx.normals[v * 3 + 1]; f32[o + 6] = pmx.normals[v * 3 + 2];
      u32[o + 7] = Math.max(sdefIndexOf[v], 0);
      f32[o + 8] = pmx.uvs[v * 2]; f32[o + 9] = pmx.uvs[v * 2 + 1];
      f32[o + 10] = pmx.edgeScales[v];
      u32[o + 11] = uv1Counts[v];
      for (let k = 0; k < 4; k++) {
        u32[o + 12 + k] = pmx.skinIndices[v * 4 + k];
        f32[o + 16 + k] = pmx.skinWeights[v * 4 + k];
        u32[o + 20 + k] = ranges[v * 4 + k];
        f32[o + 24 + k] = uv1 ? uv1[v * 4 + k] : 0;
      }
      if (cloth) {
        f32[o + 28] = cloth.data.cloth[v];
        u32[o + 30] = cloth.data.masks[v * 2];
        u32[o + 31] = cloth.data.masks[v * 2 + 1];
        const joint = Math.max(cloth.data.anchorBone[v], 0);
        f32.set(pmx.bones[joint].position, o + 32);
        u32[o + 35] = joint;
        f32.set(cloth.data.clearance.subarray(v * 4, v * 4 + 4), o + 36);
      }
    }
    const vertexInBuffer = this.createBuffer('vertices', f32, GPUBufferUsage.STORAGE);

    // SDEF: precompute (as saba does) so the shader only needs C, CR0, CR1.
    const sdefCount = pmx.sdef.vertexIndices.length;
    const sdef = new Float32Array(Math.max(sdefCount, 1) * 12);
    for (let i = 0; i < sdefCount; i++) {
      const v = pmx.sdef.vertexIndices[i];
      const w0 = pmx.skinWeights[v * 4];
      const w1 = 1 - w0;
      for (let c = 0; c < 3; c++) {
        const C = pmx.sdef.c[i * 3 + c];
        const R0 = pmx.sdef.r0[i * 3 + c];
        const R1 = pmx.sdef.r1[i * 3 + c];
        const rw = R0 * w0 + R1 * w1;
        sdef[i * 12 + c] = C;
        sdef[i * 12 + 4 + c] = (C + (C + R0 - rw)) * 0.5;
        sdef[i * 12 + 8 + c] = (C + (C + R1 - rw)) * 0.5;
      }
    }
    const sdefBuffer = this.createBuffer('sdef', sdef, GPUBufferUsage.STORAGE);
    const morphEntryBuffer = this.createBuffer('morph entries', entries, GPUBufferUsage.STORAGE);

    // ---- Dynamic inputs ----
    // Bones, followed by the cloth colliders (same 20-float layout; see skinning.wgsl).
    const colliderCount = cloth?.data.colliders.length ?? 0;
    this.colliderData = new Float32Array(colliderCount * COLLIDER_FLOATS);
    cloth?.data.colliders.forEach((_, j) => this.colliderData.set(cloth.data.shapes.subarray(j * 4, j * 4 + 4), j * COLLIDER_FLOATS + 16));
    this.boneBuffer = this.createBuffer('bones', model.boneData.byteLength + this.colliderData.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.morphWeightBuffer = this.createBuffer('morph weights', model.morphs.effective.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.materialBuffer = this.createBuffer('materials', model.morphs.materialUniforms.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

    // ---- Output: skinned vertices, consumed as a vertex buffer ----
    this.skinned = this.createBuffer('skinned', Math.max(n, 1) * SKINNED_FLOATS * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX);
    new Uint32Array(this.paramsData, 0, 4).set([n, 0, pmx.bones.length, 0]);
    this.paramsBuffer = this.createBuffer('skin params', this.paramsData.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.computeBuffers = [this.paramsBuffer, vertexInBuffer, sdefBuffer, morphEntryBuffer, this.morphWeightBuffer, this.boneBuffer, this.skinned];
    this.setBakedCloth(null);
    this.workgroups = Math.ceil(n / 64);

    // ---- Render inputs ----
    this.edgeScaleBuffer = this.createBuffer('edge scale', pmx.edgeScales, GPUBufferUsage.VERTEX);
    if (n <= 0xffff) {
      const padded = new Uint16Array(pmx.indices.length + (pmx.indices.length & 1));
      padded.set(pmx.indices);
      this.indexBuffer = this.createBuffer('indices', padded, GPUBufferUsage.INDEX);
      this.indexFormat = 'uint16';
    } else {
      this.indexBuffer = this.createBuffer('indices', pmx.indices, GPUBufferUsage.INDEX);
      this.indexFormat = 'uint32';
    }

    let first = 0;
    this.drawRanges = pmx.materials.map((m) => {
      const r = { first, count: m.indexCount };
      first += m.indexCount;
      return r;
    });
    this.parts = pmx.materials.map((m) => classifyMaterial(m.name, m.nameEn));
    this.throughHair = pmx.materials.map((m) => showsThroughHair(m.name, m.nameEn));
    this.hidden = new Uint8Array(pmx.materials.length);

    this.setAuxTextures(null);
  }

  /**
   * Bind the auxiliary maps custom shaders read (null = none; missing ones read as white).
   * The model does not own them.
   */
  setAuxTextures(aux: AuxTextures[] | null): void {
    const { core, textures } = this;
    const white = core.textures.white;
    this.materialBindGroups = this.model.pmx.materials.map((m, i) =>
      core.device.createBindGroup({
        label: `material ${m.name}`,
        layout: core.materialLayout,
        entries: [
          { binding: 0, resource: { buffer: this.materialBuffer, offset: i * MATERIAL_UNIFORM_STRIDE, size: MATERIAL_UNIFORM_FLOATS * 4 } },
          { binding: 1, resource: (textures[i].texture ?? white).createView() },
          { binding: 2, resource: (textures[i].sphere ?? white).createView() },
          { binding: 3, resource: (textures[i].toon ?? white).createView() },
          { binding: 4, resource: (aux?.[i].ramp ?? white).createView() },
          { binding: 5, resource: (aux?.[i].light ?? white).createView() },
          { binding: 6, resource: (aux?.[i].face ?? white).createView() },
        ],
      }),
    );
    this.model.morphs.setAuxFlags(aux?.map((a) => (a.ramp ? 1 : 0) | (a.light ? 2 : 0) | (a.face ? 4 : 0)) ?? null);
    // Bundles hold the old bind groups.
    this.bundle = null;
    this.shadowBundles = [];
    this.prepassBundle = null;
    this.groundBundle = null;
  }

  /** Create a buffer of `data.byteLength` (or `data` bytes), padded to 4 bytes / at least 16. */
  private createBuffer(label: string, data: ArrayBufferView | number, usage: number): GPUBuffer {
    const byteLength = typeof data === 'number' ? data : data.byteLength;
    const size = Math.max(16, Math.ceil(byteLength / 4) * 4);
    const buffer = this.core.device.createBuffer({ label, size, usage, mappedAtCreation: typeof data !== 'number' });
    if (typeof data !== 'number') {
      new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
      buffer.unmap();
    }
    this.buffers.push(buffer);
    return buffer;
  }

  /**
   * Baked cloth playback: `vertices` lists the model vertices whose position and normal will
   * come from MmdModel.clothFrame (8 floats per listed vertex) instead of skinning. null = none.
   */
  setBakedCloth(vertices: Int32Array | null): void {
    const { device } = this.core;
    this.clothSlotBuffer?.destroy();
    this.clothDataBuffer?.destroy();
    const n = Math.max(this.model.pmx.vertexCount, 1);
    const slots = new Uint32Array(n).fill(0xffffffff);
    vertices?.forEach((v, slot) => (slots[v] = slot));
    this.clothSlotBuffer = device.createBuffer({ label: 'cloth slots', size: slots.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.clothSlotBuffer, 0, slots);
    this.clothDataBuffer = device.createBuffer({ label: 'cloth data', size: Math.max(1, vertices?.length ?? 0) * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.clothActive = !!vertices && vertices.length > 0;
    this.computeBindGroup = device.createBindGroup({
      label: 'skinning',
      layout: this.core.skinningPipeline.getBindGroupLayout(0),
      entries: [
        ...this.computeBuffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
        { binding: 7, resource: { buffer: this.clothSlotBuffer } },
        { binding: 8, resource: { buffer: this.clothDataBuffer } },
      ],
    });
  }

  /** Shade this model with a custom material shader; null returns to MMD's shading. */
  setShader(shader: MaterialPipelines | null): void {
    this.shader = shader;
    this.bundle = null;
  }

  /** Custom shaders may read the scene depth, so the pre-pass is needed. */
  get hasCustomShader(): boolean {
    return this.shader !== null;
  }

  /** Leave these materials out of every pass (parts of the outfit the user hid). */
  setHiddenMaterials(indices: Iterable<number>): void {
    this.hidden.fill(0);
    for (const i of indices) if (i >= 0 && i < this.hidden.length) this.hidden[i] = 1;
    this.bundle = null;
    this.shadowBundles = [];
    this.prepassBundle = null;
    this.groundBundle = null;
  }

  /** Upload this frame's bone / morph / material data, and the colliders if cloth collision is on. */
  upload(clothCollision = true): void {
    const q = this.core.device.queue;
    const cloth = clothCollision ? this.cloth : null;
    const count = cloth?.data.colliders.length ?? 0;
    if (cloth) {
      const bones = this.model.skeleton.bones;
      cloth.data.colliders.forEach((bodyIndex, j) => {
        const body = cloth.rig.bodies[bodyIndex];
        mat4.multiply(tmpMatrix, bones[body.boneIndex].world, body.offset);
        this.colliderData.set(tmpMatrix, j * COLLIDER_FLOATS);
      });
      q.writeBuffer(this.boneBuffer, this.model.boneData.byteLength, this.colliderData as Float32Array<ArrayBuffer>);
    }
    const frame = this.clothActive ? this.model.clothFrame : null;
    if (frame) q.writeBuffer(this.clothDataBuffer!, 0, frame as Float32Array<ArrayBuffer>);
    const header = new Uint32Array(this.paramsData, 0, 4);
    header[1] = count;
    header[3] = frame ? 1 : 0;
    new Float32Array(this.paramsData, 16, 16).set(this.model.skeleton.root);
    q.writeBuffer(this.paramsBuffer, 0, this.paramsData);
    q.writeBuffer(this.boneBuffer, 0, this.model.boneData);
    q.writeBuffer(this.morphWeightBuffer, 0, this.model.morphs.effective);
    q.writeBuffer(this.materialBuffer, 0, this.model.morphs.materialUniforms);
  }

  encodeCompute(pass: GPUComputePassEncoder): void {
    if (this.workgroups === 0) return;
    pass.setPipeline(this.core.skinningPipeline);
    pass.setBindGroup(0, this.computeBindGroup);
    pass.dispatchWorkgroups(this.workgroups);
  }

  /**
   * Draw calls never change between frames (only buffer contents do), so they are
   * recorded once into a render bundle and replayed.
   */
  getBundle(showEdges: boolean): GPURenderBundle {
    const core = this.core;
    if (this.bundle && this.bundleEdges === showEdges && this.bundleFrame === core.frameBindGroup) return this.bundle;
    const enc = core.device.createRenderBundleEncoder({
      label: 'model',
      colorFormats: [core.format, core.auxFormat],
      depthStencilFormat: core.depthFormat,
      sampleCount: core.sampleCount,
    });
    this.bindGeometry(enc);
    enc.setBindGroup(0, core.frameBindGroup);

    const materials = this.model.pmx.materials;
    const ranges = this.drawRanges;
    const set: MaterialPipelines = this.shader ?? core.pipelines;
    const isHair = (i: number) => this.parts[i] === MaterialPart.Hair;

    // MMD draws materials in file order with no transparency sorting.
    let current: GPURenderPipeline | null = null;
    const draw = (pipeline: GPURenderPipeline, i: number) => {
      if (pipeline !== current) enc.setPipeline((current = pipeline));
      enc.setBindGroup(1, this.materialBindGroups[i]);
      enc.drawIndexed(ranges[i].count, 1, ranges[i].first);
    };
    materials.forEach((m, i) => {
      if (this.hidden[i]) return;
      const doubleSided = (m.flags & MaterialFlag.DoubleSided) !== 0;
      let pipeline = doubleSided ? set.mainDoubleSided : set.main;
      if (isHair(i) && set.hair) pipeline = doubleSided ? set.hairDoubleSided! : set.hair;
      draw(pipeline, i);
    });

    if (showEdges) {
      const edge = set.edge ?? core.pipelines.edge;
      materials.forEach((m, i) => {
        if (this.hidden[i] || !(m.flags & MaterialFlag.Edge) || m.edgeSize <= 0) return;
        draw(isHair(i) && set.edgeHair ? set.edgeHair : edge, i);
      });
    }

    // Eyes and brows again, only where hair covers them, at the shader's transparency.
    if (set.through) {
      materials.forEach((_, i) => {
        if (this.throughHair[i] && !this.hidden[i]) draw(set.through!, i);
      });
    }

    this.bundle = enc.finish();
    this.bundleEdges = showEdges;
    this.bundleFrame = core.frameBindGroup;
    return this.bundle;
  }

  /** Every material from the camera: depth and material part (what custom shaders read for screen-space effects). */
  getPrepassBundle(): GPURenderBundle {
    if (this.prepassBundle) return this.prepassBundle;
    const core = this.core;
    const enc = core.device.createRenderBundleEncoder({ label: 'model pre-pass', colorFormats: [PREPASS_PART_FORMAT], depthStencilFormat: core.shadowFormat });
    enc.setPipeline(core.pipelines.prepass);
    this.bindGeometry(enc);
    enc.setBindGroup(0, core.prepassBindGroup);
    this.model.pmx.materials.forEach((_, i) => {
      if (this.hidden[i]) return;
      enc.setBindGroup(1, this.materialBindGroups[i]);
      enc.drawIndexed(this.drawRanges[i].count, 1, this.drawRanges[i].first);
    });
    return (this.prepassBundle = enc.finish());
  }

  /** Depth-only draws of self-shadow casters, from the light, for one cascade. */
  getShadowBundle(cascade: number): GPURenderBundle {
    if (this.shadowBundles[cascade]) return this.shadowBundles[cascade];
    const core = this.core;
    const enc = core.device.createRenderBundleEncoder({
      label: 'model shadow',
      colorFormats: [],
      depthStencilFormat: core.shadowFormat,
    });
    enc.setPipeline(core.pipelines.shadow);
    this.bindGeometry(enc);
    enc.setBindGroup(0, core.cascadeBindGroups[cascade]);
    this.model.pmx.materials.forEach((m, i) => {
      if (this.hidden[i] || !(m.flags & MaterialFlag.CastSelfShadow)) return;
      enc.setBindGroup(1, this.materialBindGroups[i]);
      enc.drawIndexed(this.drawRanges[i].count, 1, this.drawRanges[i].first);
    });
    return (this.shadowBundles[cascade] = enc.finish());
  }

  /** MMD's ground shadow: materials flagged for it, flattened onto the floor. */
  getGroundBundle(): GPURenderBundle {
    const core = this.core;
    if (this.groundBundle && this.groundFrame === core.frameBindGroup) return this.groundBundle;
    const enc = core.device.createRenderBundleEncoder({
      label: 'model ground shadow',
      colorFormats: [core.format, core.auxFormat],
      depthStencilFormat: core.depthFormat,
      sampleCount: core.sampleCount,
    });
    enc.setPipeline(core.pipelines.ground);
    this.bindGeometry(enc);
    enc.setBindGroup(0, core.frameBindGroup);
    this.model.pmx.materials.forEach((m, i) => {
      if (this.hidden[i] || !(m.flags & MaterialFlag.GroundShadow)) return;
      enc.setBindGroup(1, this.materialBindGroups[i]);
      enc.drawIndexed(this.drawRanges[i].count, 1, this.drawRanges[i].first);
    });
    this.groundFrame = core.frameBindGroup;
    return (this.groundBundle = enc.finish());
  }

  private bindGeometry(enc: GPURenderBundleEncoder): void {
    enc.setVertexBuffer(0, this.skinned);
    enc.setVertexBuffer(1, this.edgeScaleBuffer);
    enc.setIndexBuffer(this.indexBuffer, this.indexFormat);
  }

  destroy(): void {
    this.clothSlotBuffer?.destroy();
    this.clothDataBuffer?.destroy();
    for (const b of this.buffers) b.destroy();
  }
}

const tmpMatrix = mat4.create();

/**
 * Build per-vertex CSR ranges for vertex (position), base-UV and additional-UV1 morphs:
 * entries are vec4 (offset.xyz / offset.xy, bitcast morph index in w);
 * ranges are 4 u32 per vertex: [posStart, posCount, uvStart, uvCount]; a vertex's UV1 entries
 * directly follow its UV entries (`uv1Counts` of them).
 */
function buildMorphEntries(pmx: PmxModel): { entries: Float32Array; ranges: Uint32Array; uv1Counts: Uint32Array } {
  const n = pmx.vertexCount;
  const KINDS = [MorphKind.Vertex, MorphKind.UV, MorphKind.AdditionalUV1] as const;
  const counts = KINDS.map(() => new Uint32Array(n));
  const slot = (kind: MorphKind) => KINDS.indexOf(kind as (typeof KINDS)[number]);
  for (const m of pmx.morphs) {
    const k = slot(m.kind);
    if (k < 0 || !('vertexIndices' in m)) continue;
    for (const v of m.vertexIndices) if (v >= 0 && v < n) counts[k][v]++;
  }

  const ranges = new Uint32Array(Math.max(n, 1) * 4);
  const starts = KINDS.map(() => new Uint32Array(n));
  let cursor = 0;
  for (let v = 0; v < n; v++) {
    for (let k = 0; k < KINDS.length; k++) {
      starts[k][v] = cursor;
      cursor += counts[k][v];
    }
    ranges[v * 4] = starts[0][v]; ranges[v * 4 + 1] = counts[0][v];
    ranges[v * 4 + 2] = starts[1][v]; ranges[v * 4 + 3] = counts[1][v];
  }

  const entries = new Float32Array(Math.max(cursor, 1) * 4);
  const entriesU32 = new Uint32Array(entries.buffer);
  const fill = KINDS.map(() => new Uint32Array(n));
  pmx.morphs.forEach((m, mi) => {
    const k = slot(m.kind);
    if (k < 0 || !('vertexIndices' in m)) return;
    const stride = m.kind === MorphKind.Vertex ? 3 : 4;
    m.vertexIndices.forEach((v, j) => {
      if (v < 0 || v >= n) return;
      const e = (starts[k][v] + fill[k][v]++) * 4;
      entries[e] = m.offsets[j * stride]; entries[e + 1] = m.offsets[j * stride + 1];
      if (stride === 3) entries[e + 2] = m.offsets[j * stride + 2];
      entriesU32[e + 3] = mi;
    });
  });
  return { entries, ranges, uv1Counts: counts[2] };
}
