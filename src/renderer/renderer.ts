import { mat4, vec3, type Vec3 } from '../math';
import type { OrbitCamera } from './camera';
import type { GpuModel } from './gpu-model';
import { BackgroundRenderer } from './background';
import { compileCustomShader, createPipelineChecked, detectMaterialHooks, type ShaderError } from './custom-shader';
import { GpuTiming } from './gpu-timing';
import { defaultLighting, EXTRA_LIGHTS, lightDirection, type ExtraLight, type Lighting } from './lighting';
import { PhysicsDebugRenderer } from './physics-debug';
import { PostProcess } from './post';
import { TextureManager } from './textures';
import skinningWgsl from './shaders/skinning.wgsl?raw';
import commonWgsl from './shaders/common.wgsl?raw';
import mmdWgsl from './shaders/mmd.wgsl?raw';
import shadowWgsl from './shaders/shadow.wgsl?raw';

/** Frame uniform size; must match `Frame` in common.wgsl (156 floats). */
const FRAME_UNIFORM_BYTES = 624;
const SHADOW_MAP_SIZE = 2048;
const SHADOW_FORMAT: GPUTextureFormat = 'depth32float';
/** The pre-pass's second output: the material part at each pixel. */
export const PREPASS_PART_FORMAT: GPUTextureFormat = 'r8unorm';
/** Shadow cascades; must match CASCADES in common.wgsl. */
export const CASCADES = 3;
/** The scene is drawn into this format, then post-processed onto the canvas. */
const SCENE_FORMAT: GPUTextureFormat = 'rgba8unorm';

export interface CascadeData {
  /** CASCADES matrices, 16 floats each. */
  matrices: Float32Array;
  splits: Float32Array;
  texels: Float32Array;
  biases: Float32Array;
}

export interface FrameSettings {
  light: LightSettings;
  cascades: CascadeData | null;
  width: number;
  height: number;
  edgeScale: number;
  /** Seconds on the timeline (for custom shaders). */
  time: number;
  /** rgb + opacity; opacity 0 turns the ground shadow off. */
  groundShadow: [number, number, number, number];
  /** Fill / rim lights on top of the main light (EXTRA_LIGHTS of them). */
  extras: ExtraLight[];
}

export interface LightSettings {
  /** Direction the light travels (MMD default: -0.5, -1, 0.5). */
  direction: [number, number, number];
  /** MMD default: 154/255 gray. */
  color: [number, number, number];
}

export interface MaterialPipelines {
  main: GPURenderPipeline;
  mainDoubleSided: GPURenderPipeline;
  /** For hair materials when eyes show through hair: the same shading, marking the hair in the stencil. */
  hair?: GPURenderPipeline;
  hairDoubleSided?: GPURenderPipeline;
  /** Second draw of eyes and brows where hair is in front of them (shader defines `hairTransparency`). */
  through?: GPURenderPipeline;
  /** Outline pass coloured by the shader (defines `outline`); `edgeHair` also marks the hair. */
  edge?: GPURenderPipeline;
  edgeHair?: GPURenderPipeline;
}

/** Stencil bit set by hair, read by the see-through draw of the eyes. Bit 0 belongs to the ground shadow. */
const HAIR_STENCIL = 0x02;
const GROUND_STENCIL = 0x01;
/** Stencil reference for the main pass: bundles cannot set it, so one value serves every draw. */
const STENCIL_REFERENCE = HAIR_STENCIL;

const MATERIAL_EPILOGUE = `
@fragment
fn fsCustom(in: VsOut) -> FragOut {
  let s = surface(in);
  if (s.alpha < 0.001) {
    discard;
  }
  auxOut = defaultAux(s);
  let color = shade(s);
  return FragOut(color, vec4f(auxOut.rgb, color.a));
}
`;

const OUTLINE_EPILOGUE = `
@fragment
fn fsEdgeCustom(in: VsOut) -> @location(0) vec4f {
  return outline(surface(in));
}
`;

const THROUGH_EPILOGUE = `
@fragment
fn fsThrough(in: VsOut) -> FragOut {
  let s = surface(in);
  let through = hairTransparency(s);
  if (s.alpha < 0.001 || through <= 0.0) {
    discard;
  }
  auxOut = defaultAux(s);
  var color = shade(s);
  color.a *= through;
  return FragOut(color, vec4f(auxOut.rgb, color.a));
}
`;

/** How a main-pass pipeline treats depth and stencil. */
type DepthMode = 'normal' | 'markHair' | 'throughHair';

/** Device-level state shared by all models: pipelines, layouts, frame uniforms, shadow map. */
export class RendererCore {
  readonly sampleCount = 4;
  /**
   * Second colour target of the main pass: per-pixel data for post-processing shaders
   * (r = how much the pixel is skin, g = how thin the surface is; see mmd.wgsl).
   */
  readonly auxFormat: GPUTextureFormat = 'rgba8unorm';
  /** With a stencil: the ground shadow uses it to darken each pixel once. */
  readonly depthFormat: GPUTextureFormat = 'depth24plus-stencil8';
  readonly shadowFormat = SHADOW_FORMAT;
  readonly skinningPipeline: GPUComputePipeline;
  readonly materialLayout: GPUBindGroupLayout;
  /** Main / edge passes: frame uniforms, samplers, the shadow map and the pre-pass textures. Replaced by setScenePrepass. */
  frameBindGroup!: GPUBindGroup;
  /** Shadow pass, per cascade: that cascade's matrix (and no shadow map: it's the target). */
  readonly cascadeBindGroups: GPUBindGroup[];
  /** Depth pre-pass: the camera's matrix. */
  readonly prepassBindGroup: GPUBindGroup;
  readonly cascadeViews: GPUTextureView[];
  private readonly cascadeBuffers: GPUBuffer[];
  private readonly prepassBuffer: GPUBuffer;
  readonly pipelines: {
    main: GPURenderPipeline;
    mainDoubleSided: GPURenderPipeline;
    edge: GPURenderPipeline;
    shadow: GPURenderPipeline;
    prepass: GPURenderPipeline;
    ground: GPURenderPipeline;
  };
  /** Builds a main-pass pipeline from a module (used again for custom material shaders). */
  private readonly makeMainPipeline: (label: string, module: GPUShaderModule, fs: string, cullMode: GPUCullMode, mode: DepthMode) => Promise<GPURenderPipeline>;
  private readonly makeEdgePipeline: (label: string, module: GPUShaderModule, markHair: boolean) => Promise<GPURenderPipeline>;
  private readonly frameLayout: GPUBindGroupLayout;
  private readonly frameEntries: GPUBindGroupEntry[];
  private readonly frameBuffer: GPUBuffer;
  private readonly frameData = new Float32Array(FRAME_UNIFORM_BYTES / 4);
  private readonly prepassData = new Float32Array(16);

  private constructor(readonly device: GPUDevice, readonly format: GPUTextureFormat, readonly textures: TextureManager) {
    const skinModule = device.createShaderModule({ label: 'skinning', code: skinningWgsl });
    this.skinningPipeline = device.createComputePipeline({
      label: 'skinning',
      layout: 'auto',
      compute: { module: skinModule, entryPoint: 'main' },
    });

    const VF = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
    const F = GPUShaderStage.FRAGMENT;
    const frameLayout = device.createBindGroupLayout({
      label: 'frame',
      entries: [
        { binding: 0, visibility: VF, buffer: { type: 'uniform' } },
        { binding: 1, visibility: F, sampler: { type: 'filtering' } },
        { binding: 2, visibility: F, sampler: { type: 'filtering' } },
        { binding: 3, visibility: F, texture: { sampleType: 'depth', viewDimension: '2d-array' } },
        { binding: 4, visibility: F, sampler: { type: 'comparison' } },
        { binding: 5, visibility: F, texture: { sampleType: 'depth' } },
        { binding: 6, visibility: F, texture: { sampleType: 'float' } },
      ],
    });
    this.frameLayout = frameLayout;
    const shadowFrameLayout = device.createBindGroupLayout({
      label: 'shadow frame',
      entries: [
        { binding: 0, visibility: VF, buffer: { type: 'uniform' } },
        { binding: 1, visibility: F, sampler: { type: 'filtering' } },
      ],
    });
    this.materialLayout = device.createBindGroupLayout({
      label: 'material',
      entries: [
        { binding: 0, visibility: VF, buffer: { type: 'uniform' } },
        { binding: 1, visibility: F, texture: { sampleType: 'float' } },
        { binding: 2, visibility: F, texture: { sampleType: 'float' } },
        { binding: 3, visibility: F, texture: { sampleType: 'float' } },
        // Auxiliary maps for custom material shaders: colour ramp, light map, face shadow map.
        { binding: 4, visibility: F, texture: { sampleType: 'float' } },
        { binding: 5, visibility: F, texture: { sampleType: 'float' } },
        { binding: 6, visibility: F, texture: { sampleType: 'float' } },
      ],
    });

    this.frameBuffer = device.createBuffer({
      label: 'frame uniforms',
      size: FRAME_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const wrap = device.createSampler({
      addressModeU: 'repeat', addressModeV: 'repeat',
      magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
      maxAnisotropy: 8,
    });
    const clampS = device.createSampler({
      addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
      magFilter: 'linear', minFilter: 'linear',
    });
    const shadowSampler = device.createSampler({
      compare: 'less-equal',
      magFilter: 'linear', minFilter: 'linear',
      addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
    });
    const shadowMap = device.createTexture({
      label: 'shadow map',
      size: { width: SHADOW_MAP_SIZE, height: SHADOW_MAP_SIZE, depthOrArrayLayers: CASCADES },
      format: SHADOW_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.cascadeViews = Array.from({ length: CASCADES }, (_, i) =>
      shadowMap.createView({ dimension: '2d', baseArrayLayer: i, arrayLayerCount: 1 }));

    this.frameEntries = [
      { binding: 0, resource: { buffer: this.frameBuffer } },
      { binding: 1, resource: wrap },
      { binding: 2, resource: clampS },
      { binding: 3, resource: shadowMap.createView({ dimension: '2d-array' }) },
      { binding: 4, resource: shadowSampler },
    ];
    // Until the first frame sizes the real ones: a far-plane scene depth with nothing in it.
    const farDepth = device.createTexture({ label: 'no scene depth', size: [1, 1], format: SHADOW_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    const noPart = device.createTexture({ label: 'no scene part', size: [1, 1], format: PREPASS_PART_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    const clear = device.createCommandEncoder();
    clear.beginRenderPass({
      colorAttachments: [{ view: noPart.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }],
      depthStencilAttachment: { view: farDepth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
    }).end();
    device.queue.submit([clear.finish()]);
    this.setScenePrepass(farDepth.createView(), noPart.createView());

    const depthViewGroup = (label: string, buffer: GPUBuffer) =>
      device.createBindGroup({ label, layout: shadowFrameLayout, entries: [{ binding: 0, resource: { buffer } }, { binding: 1, resource: wrap }] });
    this.cascadeBuffers = Array.from({ length: CASCADES }, (_, i) =>
      device.createBuffer({ label: `cascade ${i}`, size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    this.cascadeBindGroups = this.cascadeBuffers.map((buffer, i) => depthViewGroup(`cascade ${i}`, buffer));
    this.prepassBuffer = device.createBuffer({ label: 'prepass view', size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.prepassBindGroup = depthViewGroup('prepass', this.prepassBuffer);

    const vertexBuffers: GPUVertexBufferLayout[] = [
      {
        arrayStride: 40,
        attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x3' },
          { shaderLocation: 1, offset: 12, format: 'float32x3' },
          { shaderLocation: 2, offset: 24, format: 'float32x2' },
          { shaderLocation: 4, offset: 32, format: 'float32x2' },
        ],
      },
      { arrayStride: 4, attributes: [{ shaderLocation: 3, offset: 0, format: 'float32' }] },
    ];
    const blend: GPUBlendState = {
      color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    };

    const module = device.createShaderModule({ label: 'mmd', code: commonWgsl + mmdWgsl });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [frameLayout, this.materialLayout] });
    const markHair: GPUStencilFaceState = { compare: 'always', passOp: 'replace' };
    const behindHair: GPUStencilFaceState = { compare: 'equal' };
    const depthStencil = (mode: DepthMode): GPUDepthStencilState => {
      const base: GPUDepthStencilState = { format: this.depthFormat, depthWriteEnabled: true, depthCompare: 'less' };
      if (mode === 'markHair') return { ...base, stencilFront: markHair, stencilBack: markHair, stencilWriteMask: HAIR_STENCIL };
      if (mode === 'throughHair') {
        return { ...base, depthWriteEnabled: false, depthCompare: 'greater', stencilFront: behindHair, stencilBack: behindHair, stencilReadMask: HAIR_STENCIL, stencilWriteMask: 0 };
      }
      return base;
    };
    const make = (label: string, vs: string, fs: string, cullMode: GPUCullMode) =>
      device.createRenderPipeline({
        label,
        layout,
        vertex: { module, entryPoint: vs, buffers: vertexBuffers },
        // Only the model shading writes the auxiliary target.
        fragment: { module, entryPoint: fs, targets: [{ format, blend }, fs === 'fsMain' ? { format: this.auxFormat, blend } : { format: this.auxFormat, writeMask: 0 }] },
        // PMX winding is Direct3D's: clockwise triangles face the viewer.
        primitive: { topology: 'triangle-list', frontFace: 'cw', cullMode },
        depthStencil: depthStencil('normal'),
        multisample: { count: this.sampleCount },
      });

    this.makeMainPipeline = (label, customModule, fs, cullMode, mode) =>
      device.createRenderPipelineAsync({
        label,
        layout,
        vertex: { module: customModule, entryPoint: 'vsMain', buffers: vertexBuffers },
        fragment: { module: customModule, entryPoint: fs, targets: [{ format, blend }, { format: this.auxFormat, blend }] },
        primitive: { topology: 'triangle-list', frontFace: 'cw', cullMode },
        depthStencil: depthStencil(mode),
        multisample: { count: this.sampleCount },
      });
    this.makeEdgePipeline = (label, customModule, hair) =>
      device.createRenderPipelineAsync({
        label,
        layout,
        vertex: { module: customModule, entryPoint: 'vsEdge', buffers: vertexBuffers },
        fragment: { module: customModule, entryPoint: 'fsEdgeCustom', targets: [{ format, blend }, { format: this.auxFormat, writeMask: 0 }] },
        primitive: { topology: 'triangle-list', frontFace: 'cw', cullMode: 'front' },
        depthStencil: depthStencil(hair ? 'markHair' : 'normal'),
        multisample: { count: this.sampleCount },
      });

    const shadowModule = device.createShaderModule({ label: 'shadow', code: commonWgsl + shadowWgsl });
    const depthLayout = device.createPipelineLayout({ bindGroupLayouts: [shadowFrameLayout, this.materialLayout] });
    const shadow = device.createRenderPipeline({
      label: 'shadow',
      layout: depthLayout,
      vertex: { module: shadowModule, entryPoint: 'vsShadow', buffers: vertexBuffers },
      fragment: { module: shadowModule, entryPoint: 'fsShadow', targets: [] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: {
        format: SHADOW_FORMAT,
        depthWriteEnabled: true,
        depthCompare: 'less',
        depthBias: 1,
        depthBiasSlopeScale: 1.5,
      },
    });
    const prepass = device.createRenderPipeline({
      label: 'depth pre-pass',
      layout: depthLayout,
      vertex: { module: shadowModule, entryPoint: 'vsPrepass', buffers: vertexBuffers },
      fragment: { module: shadowModule, entryPoint: 'fsPrepass', targets: [{ format: PREPASS_PART_FORMAT }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: SHADOW_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
    });

    // Ground shadow: every covered pixel is darkened exactly once (its own stencil bit), whatever overlaps.
    const once: GPUStencilFaceState = { compare: 'equal', passOp: 'increment-clamp' };
    const ground = device.createRenderPipeline({
      label: 'ground shadow',
      layout,
      vertex: { module, entryPoint: 'vsGround', buffers: vertexBuffers },
      fragment: { module, entryPoint: 'fsGround', targets: [{ format, blend }, { format: this.auxFormat, writeMask: 0 }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: {
        format: this.depthFormat, depthWriteEnabled: false, depthCompare: 'always',
        stencilFront: once, stencilBack: once, stencilReadMask: GROUND_STENCIL, stencilWriteMask: GROUND_STENCIL,
      },
      multisample: { count: this.sampleCount },
    });

    this.pipelines = {
      main: make('mmd', 'vsMain', 'fsMain', 'back'),
      mainDoubleSided: make('mmd double-sided', 'vsMain', 'fsMain', 'none'),
      edge: make('mmd edge', 'vsEdge', 'fsEdge', 'front'),
      shadow,
      prepass,
      ground,
    };
  }

  /** Bind the pre-pass textures custom shaders read. Replaces frameBindGroup: bundles must be rebuilt. */
  setScenePrepass(depth: GPUTextureView, part: GPUTextureView): void {
    this.frameBindGroup = this.device.createBindGroup({
      label: 'frame',
      layout: this.frameLayout,
      entries: [...this.frameEntries, { binding: 5, resource: depth }, { binding: 6, resource: part }],
    });
  }

  static async create(device: GPUDevice): Promise<RendererCore> {
    return new RendererCore(device, SCENE_FORMAT, await TextureManager.create(device));
  }

  /**
   * Compile a custom material shader: WGSL defining `fn shade(s: Surface) -> vec4f`, which
   * replaces MMD's shading for the models it is assigned to (shadows stay built-in).
   * Everything in mmd.wgsl is in scope: the Surface struct, `mmdShade(s)` for the standard
   * result, `sampleToon(lambert)`, `frame` and `material`. Optional hooks: `outline(s)` colours
   * the outline; `hairTransparency(s)` draws eyes and brows through the hair in front of them.
   */
  async createMaterialPipelines(name: string, code: string): Promise<{ pipelines: MaterialPipelines } | { errors: ShaderError[] }> {
    const hooks = detectMaterialHooks(code);
    const epilogue = MATERIAL_EPILOGUE + (hooks.outline ? OUTLINE_EPILOGUE : '') + (hooks.hairTransparency ? THROUGH_EPILOGUE : '');
    const compiled = await compileCustomShader(this.device, `material ${name}`, commonWgsl + mmdWgsl, code, epilogue);
    if ('errors' in compiled) return compiled;
    const { module } = compiled;
    const made = await createPipelineChecked(this.device, async () => {
      const pipelines: MaterialPipelines = {
        main: await this.makeMainPipeline(name, module, 'fsCustom', 'back', 'normal'),
        mainDoubleSided: await this.makeMainPipeline(`${name} double-sided`, module, 'fsCustom', 'none', 'normal'),
      };
      if (hooks.hairTransparency) {
        pipelines.hair = await this.makeMainPipeline(`${name} hair`, module, 'fsCustom', 'back', 'markHair');
        pipelines.hairDoubleSided = await this.makeMainPipeline(`${name} hair double-sided`, module, 'fsCustom', 'none', 'markHair');
        pipelines.through = await this.makeMainPipeline(`${name} through hair`, module, 'fsThrough', 'none', 'throughHair');
      }
      if (hooks.outline) {
        pipelines.edge = await this.makeEdgePipeline(`${name} outline`, module, false);
        if (hooks.hairTransparency) pipelines.edgeHair = await this.makeEdgePipeline(`${name} hair outline`, module, true);
      }
      return pipelines;
    });
    if ('errors' in made) return made;
    return { pipelines: made.value };
  }

  writeFrame(camera: OrbitCamera, f: FrameSettings): void {
    const d = this.frameData;
    d.set(camera.view, 0);
    d.set(camera.proj, 16);
    d.set(camera.viewProj, 32);
    d.set([camera.eye[0], camera.eye[1], camera.eye[2], 1], 48);
    d.set([...f.light.direction, 0], 52);
    d.set([...f.light.color, 1], 56);
    d.set([f.width, f.height, f.edgeScale, f.time], 60);
    d.set([f.cascades ? 1 : 0, 1 / SHADOW_MAP_SIZE, 0, 0], 64);
    if (f.cascades) {
      d.set(f.cascades.splits, 68);
      d.set(f.cascades.texels, 72);
      d.set(f.cascades.biases, 76);
      d.set(f.cascades.matrices, 84);
      for (let i = 0; i < CASCADES; i++) {
        this.device.queue.writeBuffer(this.cascadeBuffers[i], 0, f.cascades.matrices as Float32Array<ArrayBuffer>, i * 16, 16);
      }
    }
    d.set(f.groundShadow, 80);
    for (let i = 0; i < EXTRA_LIGHTS; i++) {
      const e = f.extras[i];
      const o = 132 + i * 8;
      d.set(lightDirection(e.yaw, e.pitch), o);
      d[o + 3] = e.intensity;
      d.set(e.color, o + 4);
      d[o + 7] = e.on ? (e.kind === 'rim' ? 2 : 1) : 0;
    }
    this.device.queue.writeBuffer(this.frameBuffer, 0, d);
    this.prepassData.set(camera.viewProj);
    this.device.queue.writeBuffer(this.prepassBuffer, 0, this.prepassData);
  }
}

export class Renderer {
  readonly context: GPUCanvasContext;
  /** Rigid body / joint wireframes, drawn on top of the models. */
  readonly physicsDebug: PhysicsDebugRenderer;
  /** Panorama behind the scene (none by default). */
  readonly background: BackgroundRenderer;
  readonly post: PostProcess;
  /** GPU time per pass in ms, where the browser allows timestamp queries. */
  readonly timing: GpuTiming | null;
  /** The main light this frame (set per frame from `lighting` or the motion's light keys). */
  light: LightSettings = { direction: [...DEFAULT_LIGHT.direction], color: [...DEFAULT_LIGHT.color] };
  /** The lighting setup: main light and extra lights. Edit in place. */
  readonly lighting: Lighting = defaultLighting();
  clearColor: GPUColor = { r: 1, g: 1, b: 1, a: 1 };
  showEdges = true;
  selfShadow = true;
  /**
   * How far from the camera self-shadows reach (view depth, world units); null fits the range
   * to the models. Set from a VMD's self-shadow keys.
   */
  shadowRange: number | null = null;
  groundShadow = true;
  /** Push cloth / hair vertices out of the body's colliders after skinning (see cloth-collision.ts). */
  clothCollision = true;
  groundShadowColor: [number, number, number, number] = [0, 0, 0, 0.35];
  /** Seconds on the timeline, handed to custom shaders. */
  time = 0;
  /** Scene pixels per canvas pixel (super-sampling). */
  renderScale = 1;
  /** Canvas size in pixels, instead of following the element (for exporting frames). */
  fixedSize: [number, number] | null = null;
  private colorTarget: GPUTexture | null = null;
  private resolveTarget: GPUTexture | null = null;
  private auxTarget: GPUTexture | null = null;
  private auxResolve: GPUTexture | null = null;
  private depthTarget: GPUTexture | null = null;
  /** Camera pre-pass (depth and material part), read by custom material shaders. */
  private sceneDepth: GPUTexture | null = null;
  private scenePart: GPUTexture | null = null;
  private readonly cascades: CascadeData = {
    matrices: new Float32Array(CASCADES * 16),
    splits: new Float32Array(4),
    texels: new Float32Array(4),
    biases: new Float32Array(4),
  };

  static async create(canvas: HTMLCanvasElement): Promise<Renderer> {
    if (!navigator.gpu) throw new Error('此浏览器不支持 WebGPU');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('没有可用的 WebGPU 适配器');
    // BC (DXT) compressed textures let DDS files upload without CPU decoding; timestamp
    // queries give GPU time per pass.
    const requiredFeatures = (['texture-compression-bc', 'timestamp-query'] as GPUFeatureName[]).filter((f) => adapter.features.has(f));
    const device = await adapter.requestDevice({
      requiredFeatures,
      requiredLimits: {
        // Big models (1M+ vertices × 112 B) exceed the default 128 MiB binding limit.
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
    device.lost.then((info) => console.error(`WebGPU device lost: ${info.message}`));
    device.addEventListener('uncapturederror', (e) => console.error('WebGPU error:', (e as GPUUncapturedErrorEvent).error.message));

    const context = canvas.getContext('webgpu');
    if (!context) throw new Error('无法创建 WebGPU canvas context');
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });
    return new Renderer(canvas, device, context, await RendererCore.create(device), format);
  }

  private constructor(
    readonly canvas: HTMLCanvasElement,
    readonly device: GPUDevice,
    context: GPUCanvasContext,
    readonly core: RendererCore,
    canvasFormat: GPUTextureFormat,
  ) {
    this.context = context;
    this.physicsDebug = new PhysicsDebugRenderer(core);
    this.background = new BackgroundRenderer(core);
    this.post = new PostProcess(device, core.format, canvasFormat);
    this.timing = GpuTiming.create(device);
  }

  /** Size of the scene targets in pixels. */
  get sceneSize(): [number, number] {
    return [this.colorTarget?.width ?? 1, this.colorTarget?.height ?? 1];
  }

  /**
   * Match the canvas to its CSS size × devicePixelRatio (or `fixedSize`), and the scene targets
   * to that × `renderScale`.
   */
  resize(): void {
    const dpr = window.devicePixelRatio || 1;
    const max = this.device.limits.maxTextureDimension2D;
    const fit = (v: number) => Math.max(1, Math.min(max, Math.round(v)));
    const [cw, ch] = this.fixedSize ?? [this.canvas.clientWidth * dpr, this.canvas.clientHeight * dpr];
    if (this.canvas.width !== fit(cw)) this.canvas.width = fit(cw);
    if (this.canvas.height !== fit(ch)) this.canvas.height = fit(ch);
    // Keep the 4× MSAA targets within reason on huge canvases.
    const scale = Math.min(this.renderScale, max / Math.max(cw, ch), Math.sqrt(40e6 / Math.max(1, cw * ch)));
    const w = fit(this.canvas.width * scale);
    const h = fit(this.canvas.height * scale);
    if (this.colorTarget?.width === w && this.colorTarget.height === h) return;
    this.colorTarget?.destroy();
    this.resolveTarget?.destroy();
    this.auxTarget?.destroy();
    this.auxResolve?.destroy();
    this.depthTarget?.destroy();
    this.sceneDepth?.destroy();
    this.scenePart?.destroy();
    const size = { width: w, height: h };
    this.colorTarget = this.device.createTexture({
      label: 'msaa color', size, format: this.core.format,
      sampleCount: this.core.sampleCount,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.resolveTarget = this.device.createTexture({
      label: 'scene color', size, format: this.core.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.depthTarget = this.device.createTexture({
      label: 'depth', size, format: this.core.depthFormat,
      sampleCount: this.core.sampleCount,
      // Read back by depth of field.
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.auxTarget = this.device.createTexture({
      label: 'msaa aux', size, format: this.core.auxFormat,
      sampleCount: this.core.sampleCount,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.auxResolve = this.device.createTexture({
      label: 'scene aux', size, format: this.core.auxFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.sceneDepth = this.device.createTexture({
      label: 'scene depth', size, format: this.core.shadowFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.scenePart = this.device.createTexture({
      label: 'scene part', size, format: PREPASS_PART_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.core.setScenePrepass(this.sceneDepth.createView(), this.scenePart.createView());
    this.post.setTargets(this.resolveTarget, this.depthTarget, this.auxResolve);
  }

  /**
   * Cascaded shadow maps: the part of the view that holds the models is cut into CASCADES
   * depth slices, each with its own orthographic light frustum, so shadows stay sharp in a
   * close-up and still cover a whole stage. Returns false if there is nothing to shadow.
   */
  private updateCascades(camera: OrbitCamera, models: GpuModel[]): boolean {
    const inv = mat4.invert(tmpInverse, camera.viewProj);
    if (!inv || models.length === 0) return false;
    const unproject = (out: Vec3, x: number, y: number, z: number): Vec3 => {
      const w = inv[3] * x + inv[7] * y + inv[11] * z + inv[15];
      return vec3.set(out,
        (inv[0] * x + inv[4] * y + inv[8] * z + inv[12]) / w,
        (inv[1] * x + inv[5] * y + inv[9] * z + inv[13]) / w,
        (inv[2] * x + inv[6] * y + inv[10] * z + inv[14]) / w);
    };
    // View depth of a world point (the view matrix's third row).
    const v = camera.view;
    const depthOf = (p: Vec3) => v[2] * p[0] + v[6] * p[1] + v[10] * p[2] + v[14];

    const near = [vec3.create(), vec3.create(), vec3.create(), vec3.create()];
    const far = [vec3.create(), vec3.create(), vec3.create(), vec3.create()];
    [[-1, -1], [1, -1], [-1, 1], [1, 1]].forEach(([x, y], i) => {
      unproject(near[i], x, y, 0);
      unproject(far[i], x, y, 1);
    });
    const viewNear = depthOf(near[0]);
    const viewFar = depthOf(far[0]);

    // Depth range worth shadowing: where the models are, within the shadow range.
    let first = Infinity;
    let last = -Infinity;
    const sceneCenter = vec3.create();
    for (const m of models) {
      const b = m.model.bounds;
      const d = depthOf(b.center);
      first = Math.min(first, d - b.radius);
      last = Math.max(last, d + b.radius);
      vec3.add(sceneCenter, sceneCenter, b.center);
    }
    vec3.scale(sceneCenter, sceneCenter, 1 / models.length);
    let sceneRadius = 0;
    for (const m of models) sceneRadius = Math.max(sceneRadius, vec3.distance(sceneCenter, m.model.bounds.center) + m.model.bounds.radius);

    const range = this.shadowRange ?? Math.max(150, camera.focusDistance * 3);
    const start = Math.min(Math.max(first, viewNear), viewFar);
    const end = Math.min(Math.max(Math.min(last, range), start + 1), viewFar);
    if (end <= start) return false;

    const dir = vec3.normalize(tmpDir, vec3.set(tmpDir, ...this.light.direction));
    const up = Math.abs(dir[1]) > 0.99 ? vec3.set(tmpUp, 0, 0, 1) : vec3.set(tmpUp, 0, 1, 0);
    const lightView = mat4.lookAtLH(tmpLightView, vec3.set(tmpEye, 0, 0, 0), dir, up);
    const toLight = (out: Vec3, p: Vec3): Vec3 =>
      vec3.set(out,
        lightView[0] * p[0] + lightView[4] * p[1] + lightView[8] * p[2],
        lightView[1] * p[0] + lightView[5] * p[1] + lightView[9] * p[2],
        lightView[2] * p[0] + lightView[6] * p[1] + lightView[10] * p[2]);
    const sceneZ = toLight(tmpA, sceneCenter)[2];

    const c = this.cascades;
    const corners = Array.from({ length: 8 }, () => vec3.create());
    let sliceStart = start;
    for (let i = 0; i < CASCADES; i++) {
      // Between logarithmic (even resolution per depth) and uniform splits.
      const t = (i + 1) / CASCADES;
      const log = start > 0 ? start * Math.pow(end / start, t) : start + (end - start) * t;
      const sliceEnd = i === CASCADES - 1 ? end : log * 0.7 + (start + (end - start) * t) * 0.3;

      const center = vec3.set(tmpB, 0, 0, 0);
      for (let k = 0; k < 4; k++) {
        for (const [j, depth] of [[k, sliceStart], [k + 4, sliceEnd]] as const) {
          const s = (depth - viewNear) / (viewFar - viewNear);
          vec3.set(corners[j],
            near[k][0] + (far[k][0] - near[k][0]) * s,
            near[k][1] + (far[k][1] - near[k][1]) * s,
            near[k][2] + (far[k][2] - near[k][2]) * s);
          vec3.add(center, center, corners[j]);
        }
      }
      vec3.scale(center, center, 1 / 8);
      // A sphere around the slice: its size doesn't change as the camera turns, and snapping
      // its centre to whole texels keeps shadow edges from crawling.
      let radius = 0;
      for (const p of corners) radius = Math.max(radius, vec3.distance(center, p));
      radius = Math.ceil(radius * 8) / 8;
      const texel = (radius * 2) / SHADOW_MAP_SIZE;
      const lc = toLight(tmpA, center);
      const cx = Math.round(lc[0] / texel) * texel;
      const cy = Math.round(lc[1] / texel) * texel;
      // Towards the light, reach back to the casters (further ones are flattened onto the near
      // plane by the shadow shader, so they still cast).
      const zNear = Math.max(sceneZ - sceneRadius, lc[2] - radius - 400);
      const zFar = lc[2] + radius;
      mat4.orthoLH(tmpProj, cx - radius, cx + radius, cy - radius, cy + radius, Math.min(zNear, zFar - 1), zFar);
      mat4.multiply(tmpProj, tmpProj, lightView);
      c.matrices.set(tmpProj, i * 16);
      c.splits[i] = sliceEnd;
      c.texels[i] = texel;
      c.biases[i] = (texel * 1.5) / Math.max(zFar - Math.min(zNear, zFar - 1), 1e-3);
      sliceStart = sliceEnd;
    }
    return true;
  }

  render(camera: OrbitCamera, models: GpuModel[]): void {
    this.resize();
    const width = this.colorTarget!.width;
    const height = this.colorTarget!.height;
    camera.update(width / height);
    const shadow = this.selfShadow && this.updateCascades(camera, models);
    // MMD's edge width is in its own screen pixels; scale for denser / larger targets.
    const cssHeight = this.fixedSize ? Math.min(this.fixedSize[1], 1080) : Math.max(1, this.canvas.clientHeight);
    this.core.writeFrame(camera, {
      light: this.light,
      cascades: shadow ? this.cascades : null,
      width, height,
      edgeScale: height / cssHeight,
      time: this.time,
      groundShadow: this.groundShadow ? this.groundShadowColor : [0, 0, 0, 0],
      extras: this.lighting.extras,
    });
    for (const m of models) m.upload(this.clothCollision);

    const timing = this.timing;
    timing?.beginFrame();
    const encoder = this.device.createCommandEncoder({ label: 'frame' });
    if (models.length > 0) {
      const compute = encoder.beginComputePass({ label: 'skinning', timestampWrites: timing?.pass('skinning') });
      for (const m of models) m.encodeCompute(compute);
      compute.end();
    }

    // Custom material shaders may read the scene depth (rim light, hair shadows): draw it first.
    if (models.some((m) => m.hasCustomShader)) {
      const pass = encoder.beginRenderPass({
        label: 'depth pre-pass',
        colorAttachments: [{ view: this.scenePart!.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }],
        depthStencilAttachment: { view: this.sceneDepth!.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
        timestampWrites: timing?.pass('prepass'),
      });
      pass.executeBundles(models.map((m) => m.getPrepassBundle()));
      pass.end();
    }

    if (shadow) {
      for (let i = 0; i < CASCADES; i++) {
        const pass = encoder.beginRenderPass({
          label: 'shadow',
          colorAttachments: [],
          depthStencilAttachment: {
            view: this.core.cascadeViews[i],
            depthClearValue: 1,
            depthLoadOp: 'clear',
            depthStoreOp: 'store',
          },
          timestampWrites: timing?.pass('shadow'),
        });
        pass.executeBundles(models.map((m) => m.getShadowBundle(i)));
        pass.end();
      }
    }

    const pass = encoder.beginRenderPass({
      label: 'main',
      colorAttachments: [{
        view: this.colorTarget!.createView(),
        resolveTarget: this.resolveTarget!.createView(),
        clearValue: this.clearColor,
        loadOp: 'clear',
        storeOp: 'discard',
      }, {
        view: this.auxTarget!.createView(),
        resolveTarget: this.auxResolve!.createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: 'clear',
        storeOp: 'discard',
      }],
      depthStencilAttachment: {
        view: this.depthTarget!.createView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
        stencilClearValue: 0,
        stencilLoadOp: 'clear',
        stencilStoreOp: 'discard',
      },
      timestampWrites: timing?.pass('main'),
    });
    this.background.draw(pass, camera);
    if (models.length > 0) {
      // The reference the stencil tests of every bundle compare against (hair bit set, ground bit clear).
      pass.setStencilReference(STENCIL_REFERENCE);
      // Like MMD: ground shadows first, so every model (and any stage floor) draws over them.
      if (this.groundShadow) pass.executeBundles(models.map((m) => m.getGroundBundle()));
      pass.executeBundles(models.map((m) => m.getBundle(this.showEdges)));
    }
    this.physicsDebug.draw(pass, camera.viewProj);
    pass.end();

    this.post.run(encoder, this.context.getCurrentTexture().createView(), camera, timing, this.time);
    const readTimings = timing?.resolve(encoder);
    this.device.queue.submit([encoder.finish()]);
    readTimings?.();
  }
}

export const DEFAULT_LIGHT: LightSettings = { direction: [-0.5, -1.0, 0.5], color: [154 / 255, 154 / 255, 154 / 255] };

const tmpInverse = mat4.create();
const tmpLightView = mat4.create();
const tmpProj = mat4.create();
const tmpDir = vec3.create();
const tmpUp = vec3.create();
const tmpEye = vec3.create();
const tmpA = vec3.create();
const tmpB = vec3.create();
