import { describe, expect, it } from 'vitest';
import { applyPreset, cloneLighting, defaultLighting, EXTRA_LIGHTS, LIGHT_PRESETS, lightAngles, lightDirection } from '../src/renderer/lighting';
import { DEFAULT_LIGHT } from '../src/renderer/renderer';

describe('lighting', () => {
  it('turns yaw / pitch into a light direction and back', () => {
    for (const [yaw, pitch] of [[0, 0], [45, 55], [-120, 30], [180, -20], [90, 89]]) {
      const d = lightDirection(yaw, pitch);
      expect(Math.hypot(...d)).toBeCloseTo(1, 9);
      const a = lightAngles(d);
      expect(a.pitch).toBeCloseTo(pitch, 6);
      // Yaw is undefined straight overhead; elsewhere it round-trips (mod 360).
      if (pitch < 85) expect(((a.yaw - yaw) % 360 + 540) % 360 - 180).toBeCloseTo(0, 6);
    }
    // From the front (yaw 0): light travels towards +Z, where the model faces away from the viewer.
    expect(lightDirection(0, 0)[2]).toBeCloseTo(1);
    // From the viewer's right (+X): travels towards −X.
    expect(lightDirection(90, 0)[0]).toBeCloseTo(-1);
  });

  it('starts out as MMD’s default light', () => {
    const l = defaultLighting();
    const d = lightDirection(l.main.yaw, l.main.pitch);
    const m = Math.hypot(...DEFAULT_LIGHT.direction);
    DEFAULT_LIGHT.direction.forEach((v, i) => expect(d[i]).toBeCloseTo(v / m, 2));
    expect(l.main.color).toEqual(DEFAULT_LIGHT.color);
    expect(l.main.intensity).toBe(1);
    expect(l.extras).toHaveLength(EXTRA_LIGHTS);
    expect(l.extras.every((e) => !e.on)).toBe(true);
  });

  it('applies presets by name and keeps them independent of the originals', () => {
    const l = defaultLighting();
    expect(applyPreset(l, '影棚三点光')).toBe(true);
    expect(l.preset).toBe('影棚三点光');
    expect(l.extras.filter((e) => e.on).map((e) => e.kind)).toEqual(['fill', 'rim']);
    l.extras[0].color[0] = 0;
    expect(LIGHT_PRESETS[1].extras[0].color[0]).not.toBe(0);
    expect(applyPreset(l, '不存在')).toBe(false);
    const copy = cloneLighting(l, null);
    expect(copy.preset).toBeNull();
    expect(copy.main).toEqual(l.main);
    // Projects saved before the main light had an intensity read as 1.
    const { intensity: _dropped, ...oldMain } = l.main;
    expect(cloneLighting({ main: oldMain as typeof l.main, extras: l.extras }).main.intensity).toBe(1);
  });
});
