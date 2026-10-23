// Scene lighting: MMD's one directional light (the main light, which drives the toon shading
// and casts the shadows) plus up to EXTRA_LIGHTS extra directional lights that only add light:
// fill lights that soften the shadow side, and rim / hair lights from behind that catch the
// edges. Angles are from the viewer's point of view: yaw 0 = from the front, +90 = from the
// right, ±180 = from behind; pitch 0 = level, 90 = from straight above.

export const EXTRA_LIGHTS = 3;

export type LightKind = 'fill' | 'rim';

export interface ExtraLight {
  on: boolean;
  kind: LightKind;
  yaw: number;
  pitch: number;
  color: [number, number, number];
  intensity: number;
}

export interface MainLight {
  yaw: number;
  pitch: number;
  /** MMD's light colour: multiplies the diffuse colour (default 154/255 grey). */
  color: [number, number, number];
  /** Scales the colour (also a motion's light keys); 1 = as is. */
  intensity: number;
  /** Take direction and colour from the motion's light keys when it has them. */
  followMotion: boolean;
}

export interface Lighting {
  /** Name of the preset this came from; null once edited. */
  preset: string | null;
  main: MainLight;
  extras: ExtraLight[];
}

const DEG = Math.PI / 180;

/** Direction the light travels (MMD's convention), unit length. */
export function lightDirection(yaw: number, pitch: number): [number, number, number] {
  const y = yaw * DEG, p = pitch * DEG;
  return [-Math.sin(y) * Math.cos(p), -Math.sin(p), Math.cos(y) * Math.cos(p)];
}

/** Yaw / pitch (degrees) of the source of a light travelling along `direction`. */
export function lightAngles(direction: ArrayLike<number>): { yaw: number; pitch: number } {
  const len = Math.hypot(direction[0], direction[1], direction[2]) || 1;
  const sx = -direction[0] / len, sy = -direction[1] / len, sz = -direction[2] / len;
  return { yaw: Math.atan2(sx, -sz) / DEG, pitch: Math.asin(Math.max(-1, Math.min(1, sy))) / DEG };
}

const off = (): ExtraLight => ({ on: false, kind: 'fill', yaw: 90, pitch: 20, color: [1, 1, 1], intensity: 0.4 });
const light = (kind: LightKind, yaw: number, pitch: number, color: [number, number, number], intensity: number): ExtraLight =>
  ({ on: true, kind, yaw, pitch, color, intensity });

export const MMD_LIGHT_COLOR: [number, number, number] = [154 / 255, 154 / 255, 154 / 255];

export const LIGHT_PRESETS: { name: string; main: MainLight; extras: ExtraLight[] }[] = [
  {
    name: 'MMD 默认',
    main: { yaw: 45, pitch: 55, color: MMD_LIGHT_COLOR, intensity: 1, followMotion: true },
    extras: [off(), off(), off()],
  },
  {
    // Key from the front left, a cool fill from the right, a hair light from behind.
    name: '影棚三点光',
    main: { yaw: -35, pitch: 35, color: [0.62, 0.6, 0.58], intensity: 1, followMotion: false },
    extras: [light('fill', 75, 10, [0.55, 0.62, 0.74], 0.35), light('rim', 150, 40, [1, 0.96, 0.9], 0.9), off()],
  },
  {
    name: '侧逆光',
    main: { yaw: 125, pitch: 30, color: [0.64, 0.63, 0.62], intensity: 1, followMotion: false },
    extras: [light('fill', -40, 20, [0.5, 0.56, 0.68], 0.45), off(), off()],
  },
  {
    name: '柔光',
    main: { yaw: -20, pitch: 60, color: [0.6, 0.6, 0.6], intensity: 1, followMotion: false },
    extras: [light('fill', 60, -10, [0.7, 0.66, 0.62], 0.3), light('fill', -90, 10, [0.6, 0.62, 0.66], 0.25), off()],
  },
  {
    // MMD adds the material's ambient (usually 0.5) to the light colour before clamping, so a
    // tinted light has to stay dark to tint anything.
    name: '夕阳',
    main: { yaw: -115, pitch: 12, color: [0.84, 0.48, 0.2], intensity: 0.5, followMotion: false },
    extras: [light('fill', 60, 30, [0.3, 0.4, 0.65], 0.5), light('rim', -150, 25, [1, 0.7, 0.4], 1.0), off()],
  },
];

export function cloneLighting(l: { preset?: string | null; main: MainLight; extras: ExtraLight[] }, preset: string | null = l.preset ?? null): Lighting {
  const extras = Array.from({ length: EXTRA_LIGHTS }, (_, i) => (l.extras[i] ? { ...l.extras[i], color: [...l.extras[i].color] as [number, number, number] } : off()));
  // Projects saved before the main light had an intensity read as 1.
  return { preset, main: { ...l.main, intensity: l.main.intensity ?? 1, color: [...l.main.color] as [number, number, number] }, extras };
}

export function defaultLighting(): Lighting {
  return cloneLighting(LIGHT_PRESETS[0], LIGHT_PRESETS[0].name);
}

/** Copy `from` into `into` (keeping the object other code holds). */
export function assignLighting(into: Lighting, from: Lighting): void {
  const copy = cloneLighting(from);
  into.preset = copy.preset;
  into.main = copy.main;
  into.extras = copy.extras;
}

export function applyPreset(into: Lighting, name: string): boolean {
  const preset = LIGHT_PRESETS.find((p) => p.name === name);
  if (!preset) return false;
  assignLighting(into, cloneLighting(preset, preset.name));
  return true;
}
