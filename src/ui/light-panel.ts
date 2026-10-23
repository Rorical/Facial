import { applyPreset, EXTRA_LIGHTS, LIGHT_PRESETS, type ExtraLight, type Lighting } from '../renderer/lighting';

// Lighting controls in the settings popover: a preset, the main light, and the extra lights.
// Edits go straight into the Lighting object the renderer reads; `sync()` refreshes the
// controls after it was replaced from a project.

const CUSTOM = '自定义';

function el<K extends keyof HTMLElementTagNameMap>(tag: K, props: Partial<HTMLElementTagNameMap[K]> = {}, ...children: (Node | string)[]): HTMLElementTagNameMap[K] {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...children);
  return e;
}

const toHex = (c: [number, number, number]) => '#' + c.map((v) => Math.round(Math.max(0, Math.min(1, v)) * 255).toString(16).padStart(2, '0')).join('');
const fromHex = (h: string): [number, number, number] => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255) as [number, number, number];

export class LightPanel {
  private readonly preset: HTMLSelectElement;
  private readonly syncers: (() => void)[] = [];

  constructor(root: HTMLElement, private readonly lighting: Lighting, private readonly onChange: () => void) {
    this.preset = el('select', { title: '一套布光方案；改动任何一项后变成「自定义」' });
    for (const p of LIGHT_PRESETS) this.preset.append(el('option', { value: p.name, textContent: p.name }));
    this.preset.append(el('option', { value: CUSTOM, textContent: CUSTOM }));
    this.preset.addEventListener('change', () => {
      if (applyPreset(lighting, this.preset.value)) {
        this.sync();
        onChange();
      }
    });
    root.append(el('label', { className: 'setting' }, el('span', {}, '预设'), this.preset));

    // Main light
    const main = () => lighting.main;
    const mainColor = el('input', { type: 'color', title: 'MMD 的灯光颜色：乘在材质的漫反射色上' });
    mainColor.addEventListener('input', () => this.edit(() => (main().color = fromHex(mainColor.value))));
    const follow = el('input', { type: 'checkbox', className: 'switch', title: '动作（镜头 VMD）里带灯光关键帧时按它来' });
    follow.addEventListener('change', () => this.edit(() => (main().followMotion = follow.checked)));
    root.append(el('div', { className: 'setting light-head' }, el('span', {}, '主光'), mainColor));
    root.append(this.slider('方向', -180, 180, 5, '°', () => main().yaw, (v) => (main().yaw = v), '从正面算起：正数从右边来，±180 从背后来'));
    root.append(this.slider('高度', -30, 90, 5, '°', () => main().pitch, (v) => (main().pitch = v), '0 平射，90 从头顶往下'));
    root.append(this.slider('强度', 0, 2, 0.05, '', () => main().intensity, (v) => (main().intensity = v), '乘在灯光颜色上；MMD 会再加上材质的环境色（一般 0.5）并截断，所以高于 1 只在深色材质上看得出'));
    root.append(el('label', { className: 'setting sub' }, el('span', {}, '跟随动作灯光'), follow));
    this.syncers.push(() => {
      mainColor.value = toHex(main().color);
      follow.checked = main().followMotion;
    });

    // Extra lights
    for (let i = 0; i < EXTRA_LIGHTS; i++) {
      const extra = () => lighting.extras[i];
      const on = el('input', { type: 'checkbox', className: 'switch' });
      on.addEventListener('change', () => this.edit(() => (extra().on = on.checked)));
      const kind = el('select', { title: '补光：从那一侧柔和地照亮，减淡暗部；发丝光：从背后打，只亮在边缘和发丝上' });
      kind.append(el('option', { value: 'fill', textContent: '补光' }), el('option', { value: 'rim', textContent: '发丝光 / 轮廓光' }));
      kind.addEventListener('change', () => this.edit(() => (extra().kind = kind.value as ExtraLight['kind'])));
      const color = el('input', { type: 'color' });
      color.addEventListener('input', () => this.edit(() => (extra().color = fromHex(color.value))));
      root.append(el('div', { className: 'setting light-head' }, el('span', {}, `灯 ${i + 1}`), el('span', { className: 'light-controls' }, kind, color, on)));
      root.append(this.slider('方向', -180, 180, 5, '°', () => extra().yaw, (v) => (extra().yaw = v)));
      root.append(this.slider('高度', -30, 90, 5, '°', () => extra().pitch, (v) => (extra().pitch = v)));
      root.append(this.slider('强度', 0, 2, 0.05, '', () => extra().intensity, (v) => (extra().intensity = v)));
      this.syncers.push(() => {
        on.checked = extra().on;
        kind.value = extra().kind;
        color.value = toHex(extra().color);
      });
    }
    this.sync();
  }

  private slider(label: string, min: number, max: number, step: number, unit: string, get: () => number, set: (v: number) => void, title = ''): HTMLElement {
    const input = el('input', { type: 'range', min: String(min), max: String(max), step: String(step), title });
    const value = el('span', { className: 'light-val' });
    const show = () => (value.textContent = `${+get().toFixed(2)}${unit}`);
    input.addEventListener('input', () => this.edit(() => {
      set(Number(input.value));
      show();
    }));
    this.syncers.push(() => {
      input.value = String(get());
      show();
    });
    return el('label', { className: 'setting sub', title }, el('span', {}, label), value, input);
  }

  /** Apply an edit from a control: the setup is now custom. */
  private edit(change: () => void): void {
    change();
    this.lighting.preset = null;
    this.preset.value = CUSTOM;
    this.onChange();
  }

  /** Refresh every control from the lighting state. */
  sync(): void {
    this.preset.value = this.lighting.preset ?? CUSTOM;
    for (const s of this.syncers) s();
  }
}
