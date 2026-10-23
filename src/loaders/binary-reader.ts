export type Vec2Tuple = [number, number];
export type Vec3Tuple = [number, number, number];
export type Vec4Tuple = [number, number, number, number];

/** Little-endian sequential reader over an ArrayBuffer. */
export class BinaryReader {
  offset = 0;
  private readonly view: DataView;
  private readonly bytesView: Uint8Array;

  constructor(buffer: ArrayBuffer) {
    this.view = new DataView(buffer);
    this.bytesView = new Uint8Array(buffer);
  }

  get byteLength(): number {
    return this.view.byteLength;
  }

  get remaining(): number {
    return this.view.byteLength - this.offset;
  }

  private ensure(n: number): void {
    if (this.offset + n > this.view.byteLength) {
      throw new RangeError(`Unexpected end of data at byte ${this.offset} (need ${n} more bytes)`);
    }
  }

  u8(): number {
    this.ensure(1);
    return this.view.getUint8(this.offset++);
  }

  i8(): number {
    this.ensure(1);
    return this.view.getInt8(this.offset++);
  }

  u16(): number {
    this.ensure(2);
    const v = this.view.getUint16(this.offset, true);
    this.offset += 2;
    return v;
  }

  i16(): number {
    this.ensure(2);
    const v = this.view.getInt16(this.offset, true);
    this.offset += 2;
    return v;
  }

  u32(): number {
    this.ensure(4);
    const v = this.view.getUint32(this.offset, true);
    this.offset += 4;
    return v;
  }

  i32(): number {
    this.ensure(4);
    const v = this.view.getInt32(this.offset, true);
    this.offset += 4;
    return v;
  }

  f32(): number {
    this.ensure(4);
    const v = this.view.getFloat32(this.offset, true);
    this.offset += 4;
    return v;
  }

  vec2(): Vec2Tuple {
    return [this.f32(), this.f32()];
  }

  vec3(): Vec3Tuple {
    return [this.f32(), this.f32(), this.f32()];
  }

  vec4(): Vec4Tuple {
    return [this.f32(), this.f32(), this.f32(), this.f32()];
  }

  /** Returns a view (not a copy) of the next n bytes. */
  bytes(n: number): Uint8Array {
    this.ensure(n);
    const out = this.bytesView.subarray(this.offset, this.offset + n);
    this.offset += n;
    return out;
  }

  skip(n: number): void {
    this.ensure(n);
    this.offset += n;
  }

  /** PMX-style index: signed for sizes 1/2/4 (with -1 = none). */
  signedIndex(size: number): number {
    switch (size) {
      case 1: return this.i8();
      case 2: return this.i16();
      case 4: return this.i32();
      default: throw new Error(`Invalid index size ${size}`);
    }
  }

  /** PMX vertex index: unsigned for sizes 1/2, signed for 4. */
  vertexIndex(size: number): number {
    switch (size) {
      case 1: return this.u8();
      case 2: return this.u16();
      case 4: return this.i32();
      default: throw new Error(`Invalid vertex index size ${size}`);
    }
  }
}

/**
 * Marks the first half of a double-byte Shift-JIS character that a fixed-width field cut in
 * two: U+F700 + the byte (a private-use code point). MMD cuts names at 15 bytes wherever that
 * falls, so such halves are real data: they are kept so the name can be written back
 * byte-for-byte and matched against a model's equally cut name.
 */
export const CUT_HALF_BASE = 0xf700;

const isSjisLead = (b: number) => (b >= 0x81 && b <= 0x9f) || (b >= 0xe0 && b <= 0xfc);

/**
 * Decode a fixed-width Shift-JIS name. Bytes that don't form a character (typically the first
 * half of one at the very end, see CUT_HALF_BASE) are kept as markers instead of becoming U+FFFD.
 */
export function decodeSjisName(bytes: Uint8Array, decoder: TextDecoder): string {
  let end = bytes.indexOf(0);
  if (end < 0) end = bytes.length;
  const whole = decoder.decode(bytes.subarray(0, end));
  if (!whole.includes('\ufffd')) return whole;
  let out = '';
  for (let i = 0; i < end; ) {
    const pair = isSjisLead(bytes[i]) && i + 1 < end ? decoder.decode(bytes.subarray(i, i + 2)) : '\ufffd';
    if (!pair.includes('\ufffd')) {
      out += pair;
      i += 2;
      continue;
    }
    const single = decoder.decode(bytes.subarray(i, i + 1));
    out += single.includes('\ufffd') ? String.fromCharCode(CUT_HALF_BASE + bytes[i]) : single;
    i++;
  }
  return out;
}

/** Decode a fixed-length, NUL-terminated byte field. */
export function decodeFixedString(bytes: Uint8Array, decoder: TextDecoder): string {
  let end = bytes.indexOf(0);
  if (end < 0) end = bytes.length;
  return decoder.decode(bytes.subarray(0, end));
}
