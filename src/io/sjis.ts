import { CUT_HALF_BASE, decodeSjisName } from '../loaders/binary-reader';
// Shift-JIS encoder for VMD names. Browsers only ship a Shift-JIS *decoder*, so the reverse
// table is built once by decoding every single- and double-byte code.

let table: Map<string, number[]> | null = null;

function buildTable(): Map<string, number[]> {
  const map = new Map<string, number[]>();
  const decoder = new TextDecoder('shift-jis', { fatal: true });
  const tryAdd = (bytes: number[]) => {
    try {
      const ch = decoder.decode(new Uint8Array(bytes));
      // First code wins (some characters have duplicate codes, e.g. NEC/IBM extensions).
      if (ch.length > 0 && !map.has(ch)) map.set(ch, bytes);
    } catch {
      // not a valid code
    }
  };
  for (let b = 0x00; b <= 0x7f; b++) tryAdd([b]);
  for (let b = 0xa1; b <= 0xdf; b++) tryAdd([b]); // half-width katakana
  for (let lead = 0x81; lead <= 0xfc; lead++) {
    if (lead >= 0xa0 && lead <= 0xdf) continue;
    for (let trail = 0x40; trail <= 0xfc; trail++) {
      if (trail === 0x7f) continue;
      tryAdd([lead, trail]);
    }
  }
  return map;
}

/** Look-alike substitutes for common characters Shift-JIS lacks. */
const FALLBACK: Record<string, string> = {
  '\u2014': '\u2015', // — em dash → ― horizontal bar (SJIS 815C)
  '\u301c': '\uff5e', // 〜 wave dash → ～
  '\u2212': '\uff0d', // − minus → －
  '\u00a6': '\uffe4', // ¦ → ￤
};

export interface SjisResult {
  bytes: Uint8Array;
  /** Characters with no Shift-JIS code (written as "?"). */
  unmappable: string[];
  /** True if the text had to be cut to fit. */
  truncated: boolean;
}

/**
 * Encode `text` into at most `maxBytes` bytes without splitting a double-byte character.
 * The result is NUL-padded by the caller (fixed-width VMD fields).
 */
export function encodeSjis(text: string, maxBytes = Infinity): SjisResult {
  table ??= buildTable();
  const out: number[] = [];
  const unmappable: string[] = [];
  let truncated = false;
  for (const ch of text) {
    const cp = ch.codePointAt(0)!;
    // Half of a character that a fixed-width field cut in two: the byte itself.
    const cut = cp > CUT_HALF_BASE && cp <= CUT_HALF_BASE + 0xff ? [cp - CUT_HALF_BASE] : undefined;
    let code = cut ?? table.get(ch) ?? (FALLBACK[ch] ? table.get(FALLBACK[ch]) : undefined);
    if (!code) {
      unmappable.push(ch);
      code = [0x3f];
    }
    if (out.length + code.length > maxBytes) {
      truncated = true;
      break;
    }
    out.push(...code);
  }
  return { bytes: Uint8Array.from(out), unmappable, truncated };
}

let nameDecoder: TextDecoder | undefined;

/**
 * A bone / morph name as a VMD holds it: MMD cuts names at `maxBytes` of Shift-JIS, even
 * through the middle of a character. Motions are matched to models by this form, so longer
 * model names still find their tracks. Names that fit are returned unchanged.
 */
export function vmdName(name: string, maxBytes = 15): string {
  if (name.length * 2 <= maxBytes) return name;
  const r = encodeSjis(name);
  if (r.bytes.length <= maxBytes || r.unmappable.length) return name;
  nameDecoder ??= new TextDecoder('shift-jis');
  return decodeSjisName(r.bytes.subarray(0, maxBytes), nameDecoder);
}
