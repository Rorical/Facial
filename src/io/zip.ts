// Minimal ZIP reader / writer (store + deflate via the platform's Compression Streams).
// No ZIP64 or encryption. File names: written as UTF-8 (flag bit 11); when reading names
// without that flag we try UTF-8, then Shift-JIS, then GBK — MMD model archives from
// Japanese / Chinese sites are commonly Shift-JIS or GBK.

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

export function crc32(data: Uint8Array, crc = 0): number {
  let c = ~crc >>> 0;
  for (let i = 0; i < data.length; i++) c = CRC_TABLE[(c ^ data[i]) & 0xff] ^ (c >>> 8);
  return ~c >>> 0;
}

async function transform(data: Uint8Array, stream: CompressionStream | DecompressionStream): Promise<Uint8Array> {
  const out = new Blob([data as Uint8Array<ArrayBuffer>]).stream().pipeThrough(stream);
  return new Uint8Array(await new Response(out).arrayBuffer());
}

// ---------------------------------------------------------------------------------------
// Reading

export interface ZipEntry {
  path: string;
  size: number;
  isDirectory: boolean;
  /** Decompress on demand. */
  read(): Promise<Uint8Array>;
}

function decodeName(bytes: Uint8Array, utf8Flag: boolean): string {
  if (utf8Flag) return new TextDecoder('utf-8').decode(bytes);
  for (const enc of ['utf-8', 'shift-jis', 'gbk']) {
    try {
      return new TextDecoder(enc, { fatal: true }).decode(bytes);
    } catch {
      // try the next encoding
    }
  }
  return new TextDecoder('latin1').decode(bytes);
}

export async function readZip(source: Blob | ArrayBuffer): Promise<ZipEntry[]> {
  const buffer = source instanceof ArrayBuffer ? source : await source.arrayBuffer();
  const u8 = new Uint8Array(buffer);
  const dv = new DataView(buffer);

  // End of central directory: scan backwards over a possible comment.
  let eocd = -1;
  for (let i = u8.length - 22; i >= Math.max(0, u8.length - 22 - 0xffff); i--) {
    if (dv.getUint32(i, true) === 0x06054b50) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) throw new Error('不是有效的 zip 文件');
  const count = dv.getUint16(eocd + 10, true);
  let p = dv.getUint32(eocd + 16, true);
  if (p === 0xffffffff || count === 0xffff) throw new Error('暂不支持 ZIP64');

  const entries: ZipEntry[] = [];
  for (let i = 0; i < count; i++) {
    if (dv.getUint32(p, true) !== 0x02014b50) throw new Error('zip 中央目录损坏');
    const flags = dv.getUint16(p + 8, true);
    const method = dv.getUint16(p + 10, true);
    const compressedSize = dv.getUint32(p + 20, true);
    const size = dv.getUint32(p + 24, true);
    const nameLen = dv.getUint16(p + 28, true);
    const extraLen = dv.getUint16(p + 30, true);
    const commentLen = dv.getUint16(p + 32, true);
    const localOffset = dv.getUint32(p + 42, true);
    const path = decodeName(u8.subarray(p + 46, p + 46 + nameLen), (flags & 0x800) !== 0).replace(/\\/g, '/');
    p += 46 + nameLen + extraLen + commentLen;

    if (flags & 0x1) throw new Error(`加密的 zip 不支持：${path}`);
    const localNameLen = dv.getUint16(localOffset + 26, true);
    const localExtraLen = dv.getUint16(localOffset + 28, true);
    const dataStart = localOffset + 30 + localNameLen + localExtraLen;
    const compressed = u8.subarray(dataStart, dataStart + compressedSize);
    const isDirectory = path.endsWith('/');
    entries.push({
      path,
      size,
      isDirectory,
      read: async () => {
        if (method === 0) return compressed.slice();
        if (method === 8) return transform(compressed, new DecompressionStream('deflate-raw'));
        throw new Error(`不支持的 zip 压缩方式 ${method}：${path}`);
      },
    });
  }
  return entries;
}

// ---------------------------------------------------------------------------------------
// Writing

export interface ZipInput {
  path: string;
  data: Uint8Array | Blob | ArrayBuffer;
  /** Deflate this entry (default: yes unless the format is already compressed). */
  compress?: boolean;
}

const STORED_EXTENSIONS = new Set(['png', 'jpg', 'jpeg', 'gif', 'webp', 'mp3', 'ogg', 'm4a', 'aac', 'opus', 'flac', 'zip', 'mp4', 'webm']);

function dosDateTime(d: Date): { time: number; date: number } {
  return {
    time: (d.getHours() << 11) | (d.getMinutes() << 5) | Math.floor(d.getSeconds() / 2),
    date: ((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate(),
  };
}

export async function writeZip(inputs: ZipInput[], onProgress?: (done: number, total: number) => void): Promise<Blob> {
  const parts: BlobPart[] = [];
  const central: Uint8Array[] = [];
  const { time, date } = dosDateTime(new Date());
  const encoder = new TextEncoder();
  let offset = 0;

  for (let i = 0; i < inputs.length; i++) {
    const input = inputs[i];
    const raw = input.data instanceof Blob ? new Uint8Array(await input.data.arrayBuffer())
      : input.data instanceof ArrayBuffer ? new Uint8Array(input.data) : input.data;
    const ext = input.path.slice(input.path.lastIndexOf('.') + 1).toLowerCase();
    const wantDeflate = input.compress ?? !STORED_EXTENSIONS.has(ext);
    let method = 0;
    let body = raw;
    if (wantDeflate && raw.length > 64) {
      const deflated = await transform(raw, new CompressionStream('deflate-raw'));
      if (deflated.length < raw.length) {
        method = 8;
        body = deflated;
      }
    }
    const crc = crc32(raw);
    const name = encoder.encode(input.path);

    const local = new DataView(new ArrayBuffer(30));
    local.setUint32(0, 0x04034b50, true);
    local.setUint16(4, 20, true);
    local.setUint16(6, 0x800, true); // UTF-8 names
    local.setUint16(8, method, true);
    local.setUint16(10, time, true);
    local.setUint16(12, date, true);
    local.setUint32(14, crc, true);
    local.setUint32(18, body.length, true);
    local.setUint32(22, raw.length, true);
    local.setUint16(26, name.length, true);
    local.setUint16(28, 0, true);
    parts.push(local.buffer, name as Uint8Array<ArrayBuffer>, body as Uint8Array<ArrayBuffer>);

    const cd = new DataView(new ArrayBuffer(46 + name.length));
    cd.setUint32(0, 0x02014b50, true);
    cd.setUint16(4, 20, true);
    cd.setUint16(6, 20, true);
    cd.setUint16(8, 0x800, true);
    cd.setUint16(10, method, true);
    cd.setUint16(12, time, true);
    cd.setUint16(14, date, true);
    cd.setUint32(16, crc, true);
    cd.setUint32(20, body.length, true);
    cd.setUint32(24, raw.length, true);
    cd.setUint16(28, name.length, true);
    cd.setUint32(42, offset, true);
    new Uint8Array(cd.buffer).set(name, 46);
    central.push(new Uint8Array(cd.buffer));

    offset += 30 + name.length + body.length;
    if (offset > 0xffffffff) throw new Error('zip 超过 4 GB（不支持 ZIP64）');
    onProgress?.(i + 1, inputs.length);
  }

  const cdSize = central.reduce((s, c) => s + c.length, 0);
  const eocd = new DataView(new ArrayBuffer(22));
  eocd.setUint32(0, 0x06054b50, true);
  eocd.setUint16(8, inputs.length, true);
  eocd.setUint16(10, inputs.length, true);
  eocd.setUint32(12, cdSize, true);
  eocd.setUint32(16, offset, true);
  return new Blob([...parts, ...(central as Uint8Array<ArrayBuffer>[]), eocd.buffer], { type: 'application/zip' });
}
