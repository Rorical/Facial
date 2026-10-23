// Decoders for image formats browsers can't (reliably) decode: TGA always, BMP as a
// robust path for MMD's .bmp/.sph/.spa files (32-bit alpha, odd headers).

export interface DecodedImage {
  width: number;
  height: number;
  /** RGBA8, top-down rows. */
  data: Uint8Array;
}

export function decodeTga(buffer: ArrayBuffer): DecodedImage {
  const src = new Uint8Array(buffer);
  const view = new DataView(buffer);
  const idLength = src[0];
  const colorMapType = src[1];
  const imageType = src[2];
  const cmFirst = view.getUint16(3, true);
  const cmLength = view.getUint16(5, true);
  const cmDepth = src[7];
  const width = view.getUint16(12, true);
  const height = view.getUint16(14, true);
  const depth = src[16];
  const descriptor = src[17];

  const rle = imageType >= 9;
  const baseType = rle ? imageType - 8 : imageType; // 1 = color-mapped, 2 = truecolor, 3 = grayscale
  if (baseType < 1 || baseType > 3) throw new Error(`Unsupported TGA image type ${imageType}`);
  if (width === 0 || height === 0) throw new Error('Empty TGA image');

  let p = 18 + idLength;

  let palette: Uint8Array | null = null;
  if (colorMapType === 1) {
    const entryBytes = Math.ceil(cmDepth / 8);
    palette = new Uint8Array((cmFirst + cmLength) * 4);
    for (let i = 0; i < cmLength; i++) {
      readPixel(src, p + i * entryBytes, cmDepth, 2, palette, (cmFirst + i) * 4, null);
    }
    p += cmLength * entryBytes;
  }

  const pixelBytes = Math.ceil(depth / 8);
  const pixelCount = width * height;
  // Decode into file order first, then flip into top-down order.
  const raw = new Uint8Array(pixelCount * 4);
  let i = 0;
  if (!rle) {
    for (; i < pixelCount; i++, p += pixelBytes) readPixel(src, p, depth, baseType, raw, i * 4, palette);
  } else {
    while (i < pixelCount && p < src.length) {
      const packet = src[p++];
      const n = (packet & 0x7f) + 1;
      if (packet & 0x80) {
        readPixel(src, p, depth, baseType, raw, i * 4, palette);
        p += pixelBytes;
        for (let k = 1; k < n && i + k < pixelCount; k++) raw.copyWithin((i + k) * 4, i * 4, i * 4 + 4);
        i += n;
      } else {
        for (let k = 0; k < n && i < pixelCount; k++, i++, p += pixelBytes) {
          readPixel(src, p, depth, baseType, raw, i * 4, palette);
        }
      }
    }
  }

  const bottomUp = (descriptor & 0x20) === 0;
  const rightToLeft = (descriptor & 0x10) !== 0;
  const alphaBits = descriptor & 0x0f;
  const data = new Uint8Array(pixelCount * 4);
  for (let y = 0; y < height; y++) {
    const srcY = bottomUp ? height - 1 - y : y;
    for (let x = 0; x < width; x++) {
      const srcX = rightToLeft ? width - 1 - x : x;
      const s = (srcY * width + srcX) * 4;
      const d = (y * width + x) * 4;
      data[d] = raw[s];
      data[d + 1] = raw[s + 1];
      data[d + 2] = raw[s + 2];
      // 32-bit TGAs that declare no alpha bits are treated as opaque.
      data[d + 3] = depth === 32 && alphaBits === 0 ? 255 : raw[s + 3];
    }
  }
  return { width, height, data };
}

function readPixel(
  src: Uint8Array, p: number, depth: number, baseType: number,
  out: Uint8Array, o: number, palette: Uint8Array | null,
): void {
  if (baseType === 1) {
    const index = depth === 16 ? src[p] | (src[p + 1] << 8) : src[p];
    if (palette && index * 4 + 3 < palette.length) {
      out[o] = palette[index * 4]; out[o + 1] = palette[index * 4 + 1];
      out[o + 2] = palette[index * 4 + 2]; out[o + 3] = palette[index * 4 + 3];
    } else {
      out[o] = out[o + 1] = out[o + 2] = 0; out[o + 3] = 255;
    }
    return;
  }
  if (baseType === 3) {
    out[o] = out[o + 1] = out[o + 2] = src[p];
    out[o + 3] = depth === 16 ? src[p + 1] : 255;
    return;
  }
  switch (depth) {
    case 15:
    case 16: {
      const v = src[p] | (src[p + 1] << 8);
      out[o] = ((v >> 10) & 0x1f) * 255 / 31;
      out[o + 1] = ((v >> 5) & 0x1f) * 255 / 31;
      out[o + 2] = (v & 0x1f) * 255 / 31;
      out[o + 3] = depth === 16 && !(v & 0x8000) ? 0 : 255;
      break;
    }
    case 24:
      out[o] = src[p + 2]; out[o + 1] = src[p + 1]; out[o + 2] = src[p]; out[o + 3] = 255;
      break;
    case 32:
      out[o] = src[p + 2]; out[o + 1] = src[p + 1]; out[o + 2] = src[p]; out[o + 3] = src[p + 3];
      break;
    default:
      throw new Error(`Unsupported TGA depth ${depth}`);
  }
}

export function decodeBmp(buffer: ArrayBuffer): DecodedImage {
  const view = new DataView(buffer);
  const src = new Uint8Array(buffer);
  if (src[0] !== 0x42 || src[1] !== 0x4d) throw new Error('Not a BMP file');
  const dataOffset = view.getUint32(10, true);
  const headerSize = view.getUint32(14, true);
  if (headerSize < 40) throw new Error('OS/2 BMP headers are not supported');
  const width = view.getInt32(18, true);
  const rawHeight = view.getInt32(22, true);
  const bpp = view.getUint16(28, true);
  const compression = view.getUint32(30, true);
  let colorsUsed = view.getUint32(46, true);
  const height = Math.abs(rawHeight);
  const bottomUp = rawHeight > 0;

  // 0 = BI_RGB, 3 = BI_BITFIELDS, 6 = BI_ALPHABITFIELDS
  if (compression !== 0 && compression !== 3 && compression !== 6) {
    throw new Error(`Unsupported BMP compression ${compression}`);
  }

  let masks: [number, number, number, number] | null = null;
  if (bpp === 16 || bpp === 32) {
    if (compression === 3 || compression === 6) {
      const maskBase = 14 + 40; // masks follow a 40-byte header, or are embedded in V4/V5 headers
      masks = [
        view.getUint32(maskBase, true),
        view.getUint32(maskBase + 4, true),
        view.getUint32(maskBase + 8, true),
        compression === 6 || headerSize >= 56 ? view.getUint32(maskBase + 12, true) : 0,
      ];
    } else if (bpp === 16) {
      masks = [0x7c00, 0x03e0, 0x001f, 0];
    }
  }

  let palette: Uint8Array | null = null;
  if (bpp <= 8) {
    if (colorsUsed === 0) colorsUsed = 1 << bpp;
    palette = src.subarray(14 + headerSize, 14 + headerSize + colorsUsed * 4);
  }

  const rowSize = Math.floor((bpp * width + 31) / 32) * 4;
  const data = new Uint8Array(width * height * 4);
  let anyAlpha = false;

  for (let y = 0; y < height; y++) {
    const row = dataOffset + (bottomUp ? height - 1 - y : y) * rowSize;
    for (let x = 0; x < width; x++) {
      const d = (y * width + x) * 4;
      if (bpp === 24) {
        const p = row + x * 3;
        data[d] = src[p + 2]; data[d + 1] = src[p + 1]; data[d + 2] = src[p]; data[d + 3] = 255;
      } else if (bpp === 32 && !masks) {
        const p = row + x * 4;
        data[d] = src[p + 2]; data[d + 1] = src[p + 1]; data[d + 2] = src[p]; data[d + 3] = src[p + 3];
        if (src[p + 3] !== 0) anyAlpha = true;
      } else if (masks) {
        const p = row + x * (bpp / 8);
        const v = bpp === 32 ? view.getUint32(p, true) : view.getUint16(p, true);
        data[d] = applyMask(v, masks[0]);
        data[d + 1] = applyMask(v, masks[1]);
        data[d + 2] = applyMask(v, masks[2]);
        data[d + 3] = masks[3] ? applyMask(v, masks[3]) : 255;
      } else if (palette) {
        let index: number;
        if (bpp === 8) index = src[row + x];
        else if (bpp === 4) index = (src[row + (x >> 1)] >> ((x & 1) ? 0 : 4)) & 0x0f;
        else if (bpp === 1) index = (src[row + (x >> 3)] >> (7 - (x & 7))) & 0x01;
        else throw new Error(`Unsupported BMP depth ${bpp}`);
        data[d] = palette[index * 4 + 2]; data[d + 1] = palette[index * 4 + 1];
        data[d + 2] = palette[index * 4]; data[d + 3] = 255;
      } else {
        throw new Error(`Unsupported BMP depth ${bpp}`);
      }
    }
  }

  // Plain 32-bit BMPs often leave the 4th byte as 0; treat all-zero alpha as opaque.
  if (bpp === 32 && !masks && !anyAlpha) {
    for (let i = 3; i < data.length; i += 4) data[i] = 255;
  }
  return { width, height, data };
}

function applyMask(v: number, mask: number): number {
  if (!mask) return 0;
  let shift = 0;
  while (((mask >>> shift) & 1) === 0) shift++;
  const max = mask >>> shift;
  return Math.round((((v & mask) >>> shift) * 255) / max);
}

// ---------------------------------------------------------------------------------------
// DDS: DXT1/3/5 (BC1/2/3), DX10 BC1/2/3/7, and uncompressed 32/24-bit RGB(A).

export type DdsFormat = 'bc1' | 'bc2' | 'bc3' | 'bc7' | 'rgba';

export interface DdsImage {
  width: number;
  height: number;
  format: DdsFormat;
  /** Mip levels, largest first. Compressed data for bc*, RGBA8 for 'rgba'. */
  levels: { width: number; height: number; data: Uint8Array }[];
}

const BLOCK_BYTES: Record<Exclude<DdsFormat, 'rgba'>, number> = { bc1: 8, bc2: 16, bc3: 16, bc7: 16 };

const fourCC = (s: string) => s.charCodeAt(0) | (s.charCodeAt(1) << 8) | (s.charCodeAt(2) << 16) | (s.charCodeAt(3) << 24);

export function parseDds(buffer: ArrayBuffer): DdsImage {
  const view = new DataView(buffer);
  const src = new Uint8Array(buffer);
  if (view.getUint32(0, true) !== fourCC('DDS ')) throw new Error('Not a DDS file');
  const height = view.getUint32(12, true);
  const width = view.getUint32(16, true);
  const flags = view.getUint32(8, true);
  const mipCount = flags & 0x20000 ? Math.max(1, view.getUint32(28, true)) : 1;
  const pfFlags = view.getUint32(80, true);
  const cc = view.getUint32(84, true);
  let offset = 128;

  let format: DdsFormat;
  let masks: [number, number, number, number] | null = null;
  let bitCount = 0;
  if (pfFlags & 0x4) {
    if (cc === fourCC('DXT1')) format = 'bc1';
    else if (cc === fourCC('DXT2') || cc === fourCC('DXT3')) format = 'bc2';
    else if (cc === fourCC('DXT4') || cc === fourCC('DXT5')) format = 'bc3';
    else if (cc === fourCC('DX10')) {
      const dxgi = view.getUint32(128, true);
      offset += 20;
      if (dxgi === 71 || dxgi === 72) format = 'bc1';
      else if (dxgi === 74 || dxgi === 75) format = 'bc2';
      else if (dxgi === 77 || dxgi === 78) format = 'bc3';
      else if (dxgi === 98 || dxgi === 99) format = 'bc7';
      else if (dxgi === 28 || dxgi === 29) { format = 'rgba'; bitCount = 32; masks = [0xff, 0xff00, 0xff0000, 0xff000000]; }
      else if (dxgi === 87 || dxgi === 91) { format = 'rgba'; bitCount = 32; masks = [0xff0000, 0xff00, 0xff, 0xff000000]; }
      else throw new Error(`Unsupported DDS DXGI format ${dxgi}`);
    } else {
      throw new Error(`Unsupported DDS FourCC ${String.fromCharCode(cc & 255, (cc >> 8) & 255, (cc >> 16) & 255, cc >>> 24)}`);
    }
  } else if (pfFlags & 0x40) {
    format = 'rgba';
    bitCount = view.getUint32(88, true);
    const hasAlpha = (pfFlags & 0x1) !== 0;
    masks = [view.getUint32(92, true), view.getUint32(96, true), view.getUint32(100, true), hasAlpha ? view.getUint32(104, true) : 0];
    if (bitCount !== 32 && bitCount !== 24) throw new Error(`Unsupported DDS bit depth ${bitCount}`);
  } else {
    throw new Error('Unsupported DDS pixel format');
  }

  const levels: DdsImage['levels'] = [];
  let w = width;
  let h = height;
  for (let i = 0; i < mipCount && offset < src.length; i++) {
    if (format === 'rgba') {
      const bpp = bitCount / 8;
      const size = w * h * bpp;
      if (offset + size > src.length) break;
      const data = new Uint8Array(w * h * 4);
      for (let p = 0; p < w * h; p++) {
        const s = offset + p * bpp;
        const v = bpp === 4 ? view.getUint32(s, true) : src[s] | (src[s + 1] << 8) | (src[s + 2] << 16);
        data[p * 4] = applyMask(v, masks![0]);
        data[p * 4 + 1] = applyMask(v, masks![1]);
        data[p * 4 + 2] = applyMask(v, masks![2]);
        data[p * 4 + 3] = masks![3] ? applyMask(v, masks![3]) : 255;
      }
      levels.push({ width: w, height: h, data });
      offset += size;
    } else {
      const size = Math.max(1, Math.ceil(w / 4)) * Math.max(1, Math.ceil(h / 4)) * BLOCK_BYTES[format];
      if (offset + size > src.length) break;
      levels.push({ width: w, height: h, data: src.subarray(offset, offset + size) });
      offset += size;
    }
    w = Math.max(1, w >> 1);
    h = Math.max(1, h >> 1);
  }
  if (levels.length === 0) throw new Error('Truncated DDS file');
  return { width, height, format, levels };
}

/** CPU fallback: decode the top level of a BC1/2/3 (or uncompressed) DDS to RGBA8. */
export function decodeDdsToRgba(dds: DdsImage): DecodedImage {
  const { width, height, format } = dds;
  const level = dds.levels[0];
  if (format === 'rgba') return { width, height, data: level.data };
  if (format === 'bc7') throw new Error('BC7 需要 GPU 支持 texture-compression-bc');
  const out = new Uint8Array(width * height * 4);
  const blockBytes = BLOCK_BYTES[format];
  const bw = Math.ceil(width / 4);
  const bh = Math.ceil(height / 4);
  const colors = new Uint8Array(16);
  const alphas = new Uint8Array(8);
  const src = level.data;
  for (let by = 0; by < bh; by++) {
    for (let bx = 0; bx < bw; bx++) {
      const b = (by * bw + bx) * blockBytes;
      const colorOffset = format === 'bc1' ? b : b + 8;
      decodeColorBlock(src, colorOffset, colors, format === 'bc1');
      if (format === 'bc3') decodeBc3Alpha(src, b, alphas);
      for (let py = 0; py < 4; py++) {
        const y = by * 4 + py;
        if (y >= height) break;
        const bits = src[colorOffset + 4 + py];
        for (let px = 0; px < 4; px++) {
          const x = bx * 4 + px;
          if (x >= width) break;
          const ci = ((bits >> (px * 2)) & 3) * 4;
          const d = (y * width + x) * 4;
          out[d] = colors[ci]; out[d + 1] = colors[ci + 1]; out[d + 2] = colors[ci + 2];
          const i = py * 4 + px;
          if (format === 'bc1') out[d + 3] = colors[ci + 3];
          else if (format === 'bc2') out[d + 3] = ((src[b + (i >> 1)] >> ((i & 1) * 4)) & 0xf) * 17;
          else {
            // 3-bit indices packed little-endian across bytes 2..7
            const bit = i * 3;
            const byte = b + 2 + (bit >> 3);
            const v = (src[byte] | (src[byte + 1] << 8)) >> (bit & 7);
            out[d + 3] = alphas[v & 7];
          }
        }
      }
    }
  }
  return { width, height, data: out };
}

function decodeColorBlock(src: Uint8Array, o: number, out: Uint8Array, bc1: boolean): void {
  const c0 = src[o] | (src[o + 1] << 8);
  const c1 = src[o + 2] | (src[o + 3] << 8);
  const rgb = (c: number, i: number) => {
    out[i] = (((c >> 11) & 31) * 255) / 31;
    out[i + 1] = (((c >> 5) & 63) * 255) / 63;
    out[i + 2] = ((c & 31) * 255) / 31;
    out[i + 3] = 255;
  };
  rgb(c0, 0);
  rgb(c1, 4);
  if (!bc1 || c0 > c1) {
    for (let k = 0; k < 3; k++) {
      out[8 + k] = (2 * out[k] + out[4 + k]) / 3;
      out[12 + k] = (out[k] + 2 * out[4 + k]) / 3;
    }
    out[11] = out[15] = 255;
  } else {
    for (let k = 0; k < 3; k++) out[8 + k] = (out[k] + out[4 + k]) / 2;
    out[11] = 255;
    out[12] = out[13] = out[14] = out[15] = 0; // transparent black
  }
}

function decodeBc3Alpha(src: Uint8Array, o: number, out: Uint8Array): void {
  const a0 = src[o];
  const a1 = src[o + 1];
  out[0] = a0;
  out[1] = a1;
  if (a0 > a1) {
    for (let i = 1; i < 7; i++) out[i + 1] = ((7 - i) * a0 + i * a1) / 7;
  } else {
    for (let i = 1; i < 5; i++) out[i + 1] = ((5 - i) * a0 + i * a1) / 5;
    out[6] = 0;
    out[7] = 255;
  }
}
