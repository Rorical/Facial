import { mat4, type Mat4 } from '../math';
import type { OrbitCamera } from './camera';
import type { RendererCore } from './renderer';
import backgroundWgsl from './shaders/background.wgsl?raw';

/** Panorama (equirectangular image) behind the scene. */
export class BackgroundRenderer {
  private readonly pipeline: GPURenderPipeline;
  private readonly params: GPUBuffer;
  private readonly paramData = new Float32Array(20);
  private readonly sampler: GPUSampler;
  private readonly inverse: Mat4 = mat4.create();
  private texture: GPUTexture | null = null;
  private bindGroup: GPUBindGroup | null = null;

  constructor(private readonly core: RendererCore) {
    const { device } = core;
    const module = device.createShaderModule({ label: 'background', code: backgroundWgsl });
    this.pipeline = device.createRenderPipeline({
      label: 'background',
      layout: 'auto',
      vertex: { module, entryPoint: 'vsMain' },
      fragment: { module, entryPoint: 'fsMain', targets: [{ format: core.format }, { format: core.auxFormat, writeMask: 0 }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: core.depthFormat, depthWriteEnabled: false, depthCompare: 'always' },
      multisample: { count: core.sampleCount },
    });
    this.params = device.createBuffer({ label: 'background params', size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.sampler = device.createSampler({ addressModeU: 'repeat', addressModeV: 'clamp-to-edge', magFilter: 'linear', minFilter: 'linear' });
  }

  get active(): boolean {
    return this.bindGroup !== null;
  }

  /** Takes ownership of `texture`; null removes the background. */
  setTexture(texture: GPUTexture | null): void {
    this.texture?.destroy();
    this.texture = texture;
    this.bindGroup = texture
      ? this.core.device.createBindGroup({
          layout: this.pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.params } },
            { binding: 1, resource: this.sampler },
            { binding: 2, resource: texture.createView() },
          ],
        })
      : null;
  }

  draw(pass: GPURenderPassEncoder, camera: OrbitCamera): void {
    if (!this.bindGroup || !mat4.invert(this.inverse, camera.viewProj)) return;
    this.paramData.set(this.inverse, 0);
    this.paramData.set([camera.eye[0], camera.eye[1], camera.eye[2], 1], 16);
    this.core.device.queue.writeBuffer(this.params, 0, this.paramData);
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3);
  }
}
