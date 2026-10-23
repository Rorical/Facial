import { decodeBmp, decodeDdsToRgba, decodeTga, parseDds, type DdsImage, type DecodedImage } from '../loaders/image-decoders';
import { extname, type AssetSource } from '../loaders/asset-source';
import { DEFAULT_SHARED_TOONS } from './default-toons';
import mipmapWgsl from './shaders/mipmap.wgsl?raw';

// Textures stay in rgba8unorm (not -srgb): MMD shades in gamma space, and matching
// its look matters more than physically-correct blending here.

const FORMAT: GPUTextureFormat = 'rgba8unorm';

const BC_FORMATS: Record<Exclude<DdsImage['format'], 'rgba'>, { format: GPUTextureFormat; blockBytes: number }> = {
  bc1: { format: 'bc1-rgba-unorm', blockBytes: 8 },
  bc2: { format: 'bc2-rgba-unorm', blockBytes: 16 },
  bc3: { format: 'bc3-rgba-unorm', blockBytes: 16 },
  bc7: { format: 'bc7-rgba-unorm', blockBytes: 16 },
};

export interface TextureOptions {
  mipmaps: boolean;
}

/** Textures owned by one model: loaded once per path, destroyed together with the model. */
export class TextureScope {
  private readonly cache = new Map<string, Promise<GPUTexture | null>>();

  constructor(private readonly manager: TextureManager) {}

  load(source: AssetSource, baseDir: string, path: string, options: TextureOptions): Promise<GPUTexture | null> {
    const key = `${baseDir}|${path}|${options.mipmaps}`;
    let pending = this.cache.get(key);
    if (!pending) {
      pending = this.manager.decode(source, baseDir, path, options).catch((err) => {
        console.warn(`Texture "${path}" failed to load:`, err);
        return null;
      });
      this.cache.set(key, pending);
    }
    return pending;
  }

  /** Destroy this scope's textures (shared defaults are kept). */
  destroy(): void {
    for (const pending of this.cache.values()) {
      void pending.then((t) => {
        if (t && !this.manager.isShared(t)) t.destroy();
      });
    }
    this.cache.clear();
  }
}

/** Device-wide texture services: decoding, mipmaps, the white fallback and MMD's shared toons. */
export class TextureManager {
  readonly white: GPUTexture;
  /** MMD's shared toon01.bmp … toon10.bmp (index 0–9). Replaceable via setSharedToons. */
  sharedToons: GPUTexture[] = [];
  private readonly mipPipeline: GPURenderPipeline;
  private readonly mipSampler: GPUSampler;
  private readonly bcSupported: boolean;

  private constructor(private readonly device: GPUDevice) {
    this.bcSupported = device.features.has('texture-compression-bc');
    this.white = this.fromPixels({ width: 1, height: 1, data: new Uint8Array([255, 255, 255, 255]) }, false, 'white');

    const module = device.createShaderModule({ label: 'mipmap', code: mipmapWgsl });
    this.mipPipeline = device.createRenderPipeline({
      label: 'mipmap',
      layout: 'auto',
      vertex: { module, entryPoint: 'vsMain' },
      fragment: { module, entryPoint: 'fsMain', targets: [{ format: FORMAT }] },
      primitive: { topology: 'triangle-list' },
    });
    this.mipSampler = device.createSampler({ minFilter: 'linear', magFilter: 'linear' });
  }

  static async create(device: GPUDevice): Promise<TextureManager> {
    const tm = new TextureManager(device);
    tm.sharedToons = await Promise.all(
      DEFAULT_SHARED_TOONS.map(async (url, i) => {
        const blob = await (await fetch(url)).blob();
        return tm.fromBitmap(await decodeWithBrowser(blob), false, `toon${String(i + 1).padStart(2, '0')}`);
      }),
    );
    return tm;
  }

  /** Shared toon index 0–9 for names like "toon03.bmp" that ship with MMD itself. */
  static sharedToonIndex(path: string): number {
    const m = /(?:^|[\\/])toon(0[1-9]|10)\.bmp$/i.exec(path);
    return m ? parseInt(m[1], 10) - 1 : -1;
  }

  createScope(): TextureScope {
    return new TextureScope(this);
  }

  isShared(t: GPUTexture): boolean {
    return t === this.white || this.sharedToons.includes(t);
  }

  /**
   * Replace shared toons with real files (e.g. from MMD's Data folder). Missing indices keep
   * their current texture. Models bound to the old textures must be rebuilt afterwards.
   */
  async setSharedToons(files: Map<number, Blob>): Promise<number> {
    let replaced = 0;
    for (const [i, blob] of files) {
      if (i < 0 || i > 9) continue;
      try {
        const label = `toon${String(i + 1).padStart(2, '0')} (MMD)`;
        const buffer = await blob.arrayBuffer();
        let texture: GPUTexture;
        try {
          texture = this.fromPixels(decodeBmp(buffer), false, label);
        } catch {
          texture = this.fromBitmap(await decodeWithBrowser(new Blob([buffer])), false, label);
        }
        this.sharedToons[i].destroy();
        this.sharedToons[i] = texture;
        replaced++;
      } catch (err) {
        console.warn(`Shared toon ${i + 1} failed to decode:`, err);
      }
    }
    return replaced;
  }

  /** Load and upload one texture (use TextureScope.load for caching / ownership). */
  async decode(source: AssetSource, baseDir: string, path: string, options: TextureOptions): Promise<GPUTexture | null> {
    const blob = await source.resolve(baseDir, path);
    if (!blob) {
      const toon = TextureManager.sharedToonIndex(path);
      if (toon >= 0) return this.sharedToons[toon];
      console.warn(`Texture not found: ${path}`);
      return null;
    }
    const ext = extname(path);
    if (ext === 'tga') return this.fromPixels(decodeTga(await blob.arrayBuffer()), options.mipmaps, path);
    if (ext === 'dds') return this.fromDds(parseDds(await blob.arrayBuffer()), options.mipmaps, path);
    if (ext === 'bmp' || ext === 'sph' || ext === 'spa') {
      const buffer = await blob.arrayBuffer();
      try {
        return this.fromPixels(decodeBmp(buffer), options.mipmaps, path);
      } catch {
        // Some ".bmp"/".sph" files are actually PNG/JPEG: let the browser sniff them.
        return this.fromBitmap(await decodeWithBrowser(new Blob([buffer])), options.mipmaps, path);
      }
    }
    return this.fromBitmap(await decodeWithBrowser(blob), options.mipmaps, path);
  }

  /**
   * DDS with a full mip chain in a BC format → uploaded as-is (no CPU decode, less VRAM).
   * Otherwise decoded on the CPU to RGBA and mipmapped on the GPU.
   */
  /** A plain image file (PNG / JPEG / WebP …) as a texture. The caller owns it. */
  async fromImageBlob(blob: Blob, mipmaps: boolean, label: string): Promise<GPUTexture> {
    return this.fromBitmap(await decodeWithBrowser(blob), mipmaps, label);
  }

  /** Raw RGBA8 pixels as a texture. The caller owns it. */
  fromRgba(width: number, height: number, data: Uint8Array, mipmaps: boolean, label: string): GPUTexture {
    return this.fromPixels({ width, height, data }, mipmaps, label);
  }

  private fromDds(dds: DdsImage, mipmaps: boolean, label: string): GPUTexture {
    const blockAligned = dds.width % 4 === 0 && dds.height % 4 === 0;
    // No CPU decoder for BC7, so upload it even without a mip chain.
    const hasChain = dds.levels.length > 1 || !mipmaps || dds.format === 'bc7';
    if (dds.format !== 'rgba' && this.bcSupported && blockAligned && hasChain) {
      const { format, blockBytes } = BC_FORMATS[dds.format];
      const texture = this.device.createTexture({
        label,
        size: { width: dds.width, height: dds.height },
        format,
        mipLevelCount: mipmaps ? dds.levels.length : 1,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      for (let level = 0; level < texture.mipLevelCount; level++) {
        const { width, height, data } = dds.levels[level];
        const bw = Math.ceil(width / 4);
        const bh = Math.ceil(height / 4);
        this.device.queue.writeTexture(
          { texture, mipLevel: level },
          data as Uint8Array<ArrayBuffer>,
          { bytesPerRow: bw * blockBytes, rowsPerImage: bh },
          { width: bw * 4, height: bh * 4 },
        );
      }
      return texture;
    }
    return this.fromPixels(decodeDdsToRgba(dds), mipmaps, label);
  }

  private createTexture(width: number, height: number, mipmaps: boolean, label: string): GPUTexture {
    const mipLevelCount = mipmaps ? Math.floor(Math.log2(Math.max(width, height))) + 1 : 1;
    return this.device.createTexture({
      label,
      size: { width, height },
      format: FORMAT,
      mipLevelCount,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  private fromPixels(image: DecodedImage, mipmaps: boolean, label: string): GPUTexture {
    const texture = this.createTexture(image.width, image.height, mipmaps, label);
    this.device.queue.writeTexture(
      { texture },
      image.data as Uint8Array<ArrayBuffer>,
      { bytesPerRow: image.width * 4, rowsPerImage: image.height },
      { width: image.width, height: image.height },
    );
    if (mipmaps) this.generateMipmaps(texture);
    return texture;
  }

  private fromBitmap(bitmap: ImageBitmap, mipmaps: boolean, label: string): GPUTexture {
    const texture = this.createTexture(bitmap.width, bitmap.height, mipmaps, label);
    this.device.queue.copyExternalImageToTexture({ source: bitmap }, { texture }, { width: bitmap.width, height: bitmap.height });
    bitmap.close();
    if (mipmaps) this.generateMipmaps(texture);
    return texture;
  }

  private generateMipmaps(texture: GPUTexture): void {
    const encoder = this.device.createCommandEncoder({ label: 'mipmaps' });
    for (let level = 1; level < texture.mipLevelCount; level++) {
      const bindGroup = this.device.createBindGroup({
        layout: this.mipPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: texture.createView({ baseMipLevel: level - 1, mipLevelCount: 1 }) },
          { binding: 1, resource: this.mipSampler },
        ],
      });
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: texture.createView({ baseMipLevel: level, mipLevelCount: 1 }),
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      pass.setPipeline(this.mipPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3);
      pass.end();
    }
    this.device.queue.submit([encoder.finish()]);
  }
}

function decodeWithBrowser(blob: Blob): Promise<ImageBitmap> {
  return createImageBitmap(blob, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
}
