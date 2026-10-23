import type { Mat4 } from '../math';
import { buildDebugGeometry, DEBUG_VERTEX_BYTES } from '../physics/debug-geometry';
import type { PhysicsRig } from '../physics/rig';
import type { RendererCore } from './renderer';
import debugWgsl from './shaders/debug-lines.wgsl?raw';

/** Draws rigid body / joint wireframes inside the main pass. */
export class PhysicsDebugRenderer {
  /** Bit N shows wireframe kind N (see debug-geometry.ts). 0 = hidden. */
  mask = 0;
  private readonly pipeline: GPURenderPipeline;
  private readonly params: GPUBuffer;
  private readonly paramData = new Float32Array(20);
  private rig: PhysicsRig | null = null;
  private vertices: GPUBuffer | null = null;
  private bodies: GPUBuffer | null = null;
  private bindGroup: GPUBindGroup | null = null;
  private vertexCount = 0;

  constructor(private readonly core: RendererCore) {
    const { device } = core;
    const module = device.createShaderModule({ label: 'debug lines', code: debugWgsl });
    this.pipeline = device.createRenderPipeline({
      label: 'physics wireframe',
      layout: 'auto',
      vertex: {
        module,
        entryPoint: 'vsMain',
        buffers: [{
          arrayStride: DEBUG_VERTEX_BYTES,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'uint32' },
            { shaderLocation: 2, offset: 16, format: 'uint32' },
          ],
        }],
      },
      fragment: {
        module,
        entryPoint: 'fsMain',
        targets: [{
          format: core.format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }, { format: core.auxFormat, writeMask: 0 }],
      },
      primitive: { topology: 'line-list' },
      depthStencil: { format: core.depthFormat, depthWriteEnabled: false, depthCompare: 'always' },
      multisample: { count: core.sampleCount },
    });
    this.params = device.createBuffer({ label: 'wireframe params', size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  /** (Re)build geometry when the model's rig changes; null clears. */
  setRig(rig: PhysicsRig | null): void {
    if (rig === this.rig) return;
    this.rig = rig;
    this.vertices?.destroy();
    this.bodies?.destroy();
    this.vertices = this.bodies = this.bindGroup = null;
    this.vertexCount = 0;
    if (!rig || rig.bodies.length === 0) return;

    const { device } = this.core;
    const geo = buildDebugGeometry(rig);
    this.vertexCount = geo.vertexCount;
    this.vertices = device.createBuffer({ label: 'wireframe vertices', size: Math.max(geo.data.byteLength, 16), usage: GPUBufferUsage.VERTEX, mappedAtCreation: true });
    new Uint8Array(this.vertices.getMappedRange()).set(new Uint8Array(geo.data));
    this.vertices.unmap();
    this.bodies = device.createBuffer({ label: 'wireframe bodies', size: rig.bodies.length * 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: { buffer: this.bodies } },
      ],
    });
  }

  get visible(): boolean {
    return this.mask !== 0 && this.vertexCount > 0;
  }

  /** Upload this frame's body transforms (16 floats per rig body). */
  updateBodies(transforms: Float32Array): void {
    if (this.bodies) this.core.device.queue.writeBuffer(this.bodies, 0, transforms as Float32Array<ArrayBuffer>);
  }

  draw(pass: GPURenderPassEncoder, viewProj: Mat4): void {
    if (!this.visible || !this.bindGroup || !this.vertices) return;
    this.paramData.set(viewProj, 0);
    new Uint32Array(this.paramData.buffer).set([this.mask, 0, 0, 0], 16);
    this.core.device.queue.writeBuffer(this.params, 0, this.paramData);
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.vertices);
    pass.draw(this.vertexCount);
  }
}
