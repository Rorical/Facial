import type { OrbitCamera } from './camera';
import { compileCustomShader, createPipelineChecked, type ShaderError } from './custom-shader';
import type { GpuTiming } from './gpu-timing';
import postWgsl from './shaders/post.wgsl?raw';

// Post-processing after the scene is drawn: optional depth of field, a custom effect shader and
// bloom, then the composite (tone mapping, down-scaling from the render size) onto the canvas.

export type ToneMapping = 'none' | 'aces';

export interface PostSettings {
  bloom: boolean;
  /** 0–2. */
  bloomStrength: number;
  /** Brightness (0–1) above which pixels glow. */
  bloomThreshold: number;
  dof: boolean;
  /** 0–1: how quickly things away from the focus blur. */
  dofStrength: number;
  toneMapping: ToneMapping;
  exposure: number;
}

export const DEFAULT_POST: PostSettings = {
  bloom: false, bloomStrength: 0.6, bloomThreshold: 0.8, dof: false, dofStrength: 0.4, toneMapping: 'none', exposure: 1,
};

/** Declarations a custom effect may use; its `effect` function is called by the epilogue. */
const EFFECT_PRELUDE = `
struct Fx {
  resolution: vec2f,   // scene size in pixels
  time: f32,           // seconds on the timeline
  ortho: f32,
  depthA: f32,
  depthB: f32,
  projScale: f32,      // see pixelsPerUnit()
};

@group(0) @binding(0) var<uniform> fx: Fx;
@group(0) @binding(1) var fxSampler: sampler;
@group(0) @binding(2) var fxColor: texture_2d<f32>;
@group(0) @binding(3) var fxDepth: texture_depth_multisampled_2d;
@group(0) @binding(4) var fxAux: texture_2d<f32>;

struct FxVsOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vsMain(@builtin(vertex_index) i: u32) -> FxVsOut {
  var out: FxVsOut;
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  out.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2f(p.x, 1.0 - p.y);
  return out;
}

/** Scene colour at uv (0–1, origin at the top left). */
fn sceneColor(uv: vec2f) -> vec3f {
  return textureSampleLevel(fxColor, fxSampler, uv, 0.0).rgb;
}

/**
 * Per-pixel data written by the model shading at uv: r = how much the pixel is skin (0–1),
 * g = how thin the surface is (0–1), b = free for custom material shaders. 0 where nothing was drawn.
 */
fn sceneAux(uv: vec2f) -> vec4f {
  return textureSampleLevel(fxAux, fxSampler, uv, 0.0);
}

/** How many pixels one model unit covers at view depth d (for effects sized in the scene, not on screen). */
fn pixelsPerUnit(d: f32) -> f32 {
  if (fx.ortho > 0.5) {
    return fx.projScale;
  }
  return fx.projScale / max(d, 1e-3);
}

/** Distance in front of the camera at uv, in model units (the far plane where nothing was drawn). */
fn sceneDepth(uv: vec2f) -> f32 {
  let size = vec2i(textureDimensions(fxDepth));
  let z = textureLoad(fxDepth, clamp(vec2i(uv * vec2f(size)), vec2i(0), size - vec2i(1)), 0);
  if (fx.ortho > 0.5) {
    return (z - fx.depthB) / fx.depthA;
  }
  return fx.depthB / min(z - fx.depthA, -1e-6);
}
`;

const EFFECT_EPILOGUE = `
@fragment
fn fsCustom(in: FxVsOut) -> @location(0) vec4f {
  return vec4f(effect(in.uv, sceneColor(in.uv)), 1.0);
}
`;

const BLOOM_LEVELS = 5;
const BLOOM_FORMAT: GPUTextureFormat = 'rgba16float';

export class PostProcess {
  settings: PostSettings = { ...DEFAULT_POST };
  private readonly params: GPUBuffer;
  private readonly paramData = new Float32Array(12);
  private readonly sampler: GPUSampler;
  private readonly dofPipeline: GPURenderPipeline;
  private readonly prefilterPipeline: GPURenderPipeline;
  private readonly downPipeline: GPURenderPipeline;
  private readonly upPipeline: GPURenderPipeline;
  private readonly compositePipeline: GPURenderPipeline;
  private readonly black: GPUTexture;

  private textures: GPUTexture[] = [];
  private dofView: GPUTextureView | null = null;
  private dofBind: GPUBindGroup | null = null;
  private bloomViews: GPUTextureView[] = [];
  // Colour sources, in pipeline order: 0 scene, 1 depth of field, 2 custom effect.
  private sources: GPUTextureView[] = [];
  private depthView: GPUTextureView | null = null;
  private auxView: GPUTextureView | null = null;
  private readonly effectLayout: GPUBindGroupLayout;
  private readonly effectParams: GPUBuffer;
  private effectPipeline: GPURenderPipeline | null = null;
  private effectBinds: GPUBindGroup[] = []; // [from scene, from dof]
  private prefilterBinds: GPUBindGroup[] = []; // per source
  private downBinds: GPUBindGroup[] = [];
  private upBinds: GPUBindGroup[] = [];
  private compositeBinds: GPUBindGroup[] = []; // per source, then the same again with bloom
  private width = 1;
  private height = 1;

  constructor(private readonly device: GPUDevice, private readonly sceneFormat: GPUTextureFormat, canvasFormat: GPUTextureFormat) {
    const F = GPUShaderStage.FRAGMENT;
    this.effectLayout = device.createBindGroupLayout({
      label: 'custom effect',
      entries: [
        { binding: 0, visibility: F, buffer: { type: 'uniform' } },
        { binding: 1, visibility: F, sampler: { type: 'filtering' } },
        { binding: 2, visibility: F, texture: { sampleType: 'float' } },
        { binding: 3, visibility: F, texture: { sampleType: 'depth', multisampled: true } },
        { binding: 4, visibility: F, texture: { sampleType: 'float' } },
      ],
    });
    this.effectParams = device.createBuffer({ label: 'custom effect params', size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const module = device.createShaderModule({ label: 'post', code: postWgsl });
    const make = (label: string, entryPoint: string, format: GPUTextureFormat, additive = false) =>
      device.createRenderPipeline({
        label,
        layout: 'auto',
        vertex: { module, entryPoint: 'vsMain' },
        fragment: {
          module,
          entryPoint,
          targets: [{
            format,
            blend: additive ? { color: { srcFactor: 'one', dstFactor: 'one' }, alpha: { srcFactor: 'one', dstFactor: 'one' } } : undefined,
          }],
        },
        primitive: { topology: 'triangle-list' },
      });
    this.dofPipeline = make('dof', 'fsDof', sceneFormat);
    this.prefilterPipeline = make('bloom prefilter', 'fsBloomPrefilter', BLOOM_FORMAT);
    this.downPipeline = make('bloom down', 'fsBloomDown', BLOOM_FORMAT);
    this.upPipeline = make('bloom up', 'fsBloomUp', BLOOM_FORMAT, true);
    this.compositePipeline = make('composite', 'fsComposite', canvasFormat);
    this.params = device.createBuffer({ label: 'post params', size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
    this.black = device.createTexture({ label: 'black', size: [1, 1], format: BLOOM_FORMAT, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT });
  }

  /** (Re)build the intermediate targets for a scene of this size. */
  setTargets(color: GPUTexture, depth: GPUTexture, aux: GPUTexture): void {
    const { device } = this;
    for (const t of this.textures) t.destroy();
    this.textures = [];
    const w = color.width;
    const h = color.height;
    this.width = w;
    this.height = h;
    const target = (label: string, width: number, height: number, format: GPUTextureFormat) => {
      const t = device.createTexture({
        label, format,
        size: [Math.max(1, width), Math.max(1, height)],
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.textures.push(t);
      return t.createView();
    };
    const params = { binding: 0, resource: { buffer: this.params } };
    const sampler = { binding: 1, resource: this.sampler };
    const single = (pipeline: GPURenderPipeline, src: GPUTextureView) =>
      device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [params, sampler, { binding: 2, resource: src }] });

    const colorView = color.createView();
    this.dofView = target('dof', w, h, color.format);
    this.dofBind = device.createBindGroup({
      layout: this.dofPipeline.getBindGroupLayout(0),
      entries: [params, sampler, { binding: 2, resource: colorView }, { binding: 4, resource: depth.createView({ aspect: 'depth-only' }) }],
    });

    this.depthView = depth.createView({ aspect: 'depth-only' });
    this.auxView = aux.createView();
    this.sources = [colorView, this.dofView, target('custom effect', w, h, color.format)];
    this.bindEffect();

    this.bloomViews = Array.from({ length: BLOOM_LEVELS }, (_, i) => target(`bloom ${i}`, w >> (i + 1), h >> (i + 1), BLOOM_FORMAT));
    this.prefilterBinds = this.sources.map((v) => single(this.prefilterPipeline, v));
    this.downBinds = this.bloomViews.slice(0, -1).map((v) => single(this.downPipeline, v));
    this.upBinds = this.bloomViews.slice(1).map((v) => single(this.upPipeline, v));

    const composite = (src: GPUTextureView, bloom: GPUTextureView) =>
      device.createBindGroup({
        layout: this.compositePipeline.getBindGroupLayout(0),
        entries: [params, sampler, { binding: 2, resource: src }, { binding: 3, resource: bloom }],
      });
    const black = this.black.createView();
    this.compositeBinds = [...this.sources.map((v) => composite(v, black)), ...this.sources.map((v) => composite(v, this.bloomViews[0]))];
  }

  private bindEffect(): void {
    if (!this.depthView) return;
    this.effectBinds = this.sources.slice(0, 2).map((src) =>
      this.device.createBindGroup({
        layout: this.effectLayout,
        entries: [
          { binding: 0, resource: { buffer: this.effectParams } },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: src },
          { binding: 3, resource: this.depthView! },
          { binding: 4, resource: this.auxView! },
        ],
      }));
  }

  get hasEffect(): boolean {
    return this.effectPipeline !== null;
  }

  /**
   * Install a custom effect: WGSL defining `fn effect(uv: vec2f, color: vec3f) -> vec3f`, run on
   * the whole picture after depth of field and before bloom / tone mapping. It may call
   * `sceneColor(uv)` and `sceneDepth(uv)` and read `fx.resolution` / `fx.time`.
   * null removes it. On errors the current effect is left as it is.
   */
  async setEffect(shader: { name: string; code: string } | null): Promise<ShaderError[]> {
    if (!shader) {
      this.effectPipeline = null;
      return [];
    }
    const compiled = await compileCustomShader(this.device, `effect ${shader.name}`, EFFECT_PRELUDE, shader.code, EFFECT_EPILOGUE);
    if ('errors' in compiled) return compiled.errors;
    const made = await createPipelineChecked(this.device, () =>
      this.device.createRenderPipelineAsync({
        label: `effect ${shader.name}`,
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.effectLayout] }),
        vertex: { module: compiled.module, entryPoint: 'vsMain' },
        fragment: { module: compiled.module, entryPoint: 'fsCustom', targets: [{ format: this.sceneFormat }] },
        primitive: { topology: 'triangle-list' },
      }));
    if ('errors' in made) return made.errors;
    this.effectPipeline = made.value;
    return [];
  }

  run(encoder: GPUCommandEncoder, canvasView: GPUTextureView, camera: OrbitCamera, timing: GpuTiming | null, time = 0): void {
    const s = this.settings;
    const d = this.paramData;
    d[0] = s.bloomThreshold;
    d[1] = s.bloomStrength / 3; // the blur chain sums several levels
    d[2] = s.exposure;
    d[3] = s.toneMapping === 'aces' ? 1 : 0;
    d[4] = camera.focusDistance;
    d[5] = s.dofStrength * 1.5;
    d[6] = 14 * (this.height / 1080);
    d[7] = camera.orthographic ? 1 : 0;
    d[8] = camera.proj[10];
    d[9] = camera.proj[14];
    d[10] = s.bloom ? 1 : 0;
    this.device.queue.writeBuffer(this.params, 0, d);

    const draw = (label: string, view: GPUTextureView, pipeline: GPURenderPipeline, bind: GPUBindGroup, load = false) => {
      const pass = encoder.beginRenderPass({
        label,
        colorAttachments: [{ view, loadOp: load ? 'load' : 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
        timestampWrites: timing?.pass(label.split(' ')[0]),
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.draw(3);
      pass.end();
    };

    let source = 0;
    if (s.dof) {
      draw('dof', this.sources[1], this.dofPipeline, this.dofBind!);
      source = 1;
    }
    if (this.effectPipeline) {
      const [sw, sh] = [this.width, this.height];
      // Pixels per model unit at depth 1 (perspective) or anywhere (orthographic): half the height × the projection's y scale.
      this.device.queue.writeBuffer(this.effectParams, 0, new Float32Array([sw, sh, time, d[7], d[8], d[9], (sh / 2) * camera.proj[5], 0]));
      draw('effect', this.sources[2], this.effectPipeline, this.effectBinds[source]);
      source = 2;
    }
    if (s.bloom) {
      draw('bloom prefilter', this.bloomViews[0], this.prefilterPipeline, this.prefilterBinds[source]);
      for (let i = 1; i < BLOOM_LEVELS; i++) draw('bloom down', this.bloomViews[i], this.downPipeline, this.downBinds[i - 1]);
      for (let i = BLOOM_LEVELS - 2; i >= 0; i--) draw('bloom up', this.bloomViews[i], this.upPipeline, this.upBinds[i], true);
    }
    draw('composite', canvasView, this.compositePipeline, this.compositeBinds[source + (s.bloom ? this.sources.length : 0)]);
  }
}
