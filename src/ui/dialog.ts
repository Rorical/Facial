import { DEFAULT_CLOTH, type ClothSettings } from '../cloth/cloth-sim';
import { BAKE_EXPORT_PRESETS, type BakeExportPreset } from '../physics/bake';

// Small modal dialogs (native <dialog>) and toast notifications.

type Kind = 'info' | 'warn' | 'error';

function toastHost(): HTMLElement {
  let host = document.getElementById('toasts');
  if (!host) {
    host = document.createElement('div');
    host.id = 'toasts';
    document.body.appendChild(host);
  }
  return host;
}

/** Transient message in the corner. Warnings / errors stay longer. */
export function toast(message: string, kind: Kind = 'info', timeout = kind === 'info' ? 3500 : 8000): void {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = message;
  el.addEventListener('click', () => el.remove());
  toastHost().appendChild(el);
  setTimeout(() => el.remove(), timeout);
}

/** A toast that stays until done(): for long operations, optionally cancellable. */
export function progress(message: string, onCancel?: () => void): { update(message: string): void; done(): void } {
  const el = document.createElement('div');
  el.className = 'toast progress';
  const text = document.createElement('span');
  text.textContent = message;
  el.appendChild(text);
  if (onCancel) {
    const cancel = document.createElement('button');
    cancel.className = 'btn small toast-cancel';
    cancel.textContent = '取消';
    cancel.addEventListener('click', (e) => {
      e.stopPropagation();
      onCancel();
    });
    el.appendChild(cancel);
  }
  toastHost().appendChild(el);
  return {
    update: (m) => (text.textContent = m),
    done: () => el.remove(),
  };
}

interface ModalButton<T> {
  label: string;
  value: T;
  primary?: boolean;
}

/** Show a modal; resolves with the clicked button's value, or `cancelValue` on Esc. */
function modal<T>(title: string, body: HTMLElement, buttons: ModalButton<T>[], cancelValue: T): Promise<T> {
  return new Promise((resolve) => {
    const dialog = document.createElement('dialog');
    dialog.className = 'modal';
    const h = document.createElement('h2');
    h.textContent = title;
    const actions = document.createElement('div');
    actions.className = 'actions';
    let result = cancelValue;
    for (const b of buttons) {
      const btn = document.createElement('button');
      btn.className = `btn${b.primary ? ' primary' : ''}`;
      btn.textContent = b.label;
      btn.addEventListener('click', () => {
        result = b.value;
        dialog.close();
      });
      actions.appendChild(btn);
    }
    dialog.append(h, body, actions);
    dialog.addEventListener('close', () => {
      dialog.remove();
      resolve(result);
    });
    document.body.appendChild(dialog);
    dialog.showModal();
    (actions.querySelector('.primary') as HTMLButtonElement | null)?.focus();
  });
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, props: Partial<HTMLElementTagNameMap[K]> = {}, ...children: (Node | string)[]): HTMLElementTagNameMap[K] {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...children);
  return e;
}

export function confirmDialog(title: string, message: string, ok = '确定'): Promise<boolean> {
  return modal(title, el('p', {}, message), [{ label: '取消', value: false }, { label: ok, value: true, primary: true }], false);
}

export interface ExportChoice {
  /** Only the playback range. */
  useRange: boolean;
  /** Shift so the range starts at frame 0. */
  rebase: boolean;
  /** Add the physics bake as bone keys. */
  includeBake?: boolean;
  /** Simplification tolerance for the bake (model units). */
  bakeTolerance?: number;
}

export async function exportMotionDialog(
  title: string,
  summary: string,
  range: [number, number] | null,
  bake?: { from: number; to: number } | null,
): Promise<ExportChoice | null> {
  const all = el('input', { type: 'radio', name: 'range', checked: true });
  const part = el('input', { type: 'radio', name: 'range', disabled: !range });
  const rebase = el('input', { type: 'checkbox', checked: true, disabled: true });
  const sync = () => (rebase.disabled = !part.checked);
  all.addEventListener('change', sync);
  part.addEventListener('change', sync);
  const includeBake = el('input', { type: 'checkbox', checked: !!bake });
  const body = el('div', { className: 'form' },
    el('p', { className: 'muted' }, summary),
    el('label', {}, all, ' 全部关键帧'),
    el('label', {}, part, range ? ` 仅播放区间（第 ${range[0]}–${range[1]} 帧）` : ' 仅播放区间（未设置区间）'),
    el('label', { className: 'indent' }, rebase, ' 区间起点作为第 0 帧'),
  );
  const preset = el('select');
  for (const [key, p] of Object.entries(BAKE_EXPORT_PRESETS)) preset.append(el('option', { value: key, textContent: p.label, selected: key === 'standard' }));
  if (bake) {
    includeBake.addEventListener('change', () => (preset.disabled = !includeBake.checked));
    body.append(
      el('label', { title: '物理骨骼写成关键帧，供 Blender / UE 等其他软件使用；在 MMD 里播放时请关闭物理' },
        includeBake, ` 包含物理烘焙（第 ${bake.from}–${bake.to} 帧）`),
      el('label', { className: 'indent' }, '精简 ', preset),
      el('p', { className: 'muted indent' }, '晃动中的头发 / 布料几乎每帧都在变化，文件会比较大（标准约 0.5 MB / 秒）。'),
    );
  }
  const ok = await modal(title, body, [{ label: '取消', value: false }, { label: '导出', value: true, primary: true }], false);
  return ok
    ? {
        useRange: part.checked,
        rebase: part.checked && rebase.checked,
        includeBake: !!bake && includeBake.checked,
        bakeTolerance: BAKE_EXPORT_PRESETS[preset.value as BakeExportPreset].tolerance,
      }
    : null;
}

export interface BakeChoice {
  from: number;
  to: number;
  warmupSteps: number;
  /** Materials to simulate as cloth, with the solver settings; null = none. */
  cloth: { materials: number[]; settings: ClothSettings } | null;
}

export interface ClothCandidate {
  index: number;
  name: string;
  suggested: boolean;
}

/** Physics bake settings: whole timeline or the playback range, plus settle steps. */
export async function bakeDialog(modelName: string, length: number, range: [number, number] | null, cloth: ClothCandidate[] = []): Promise<BakeChoice | null> {
  const end = Math.max(1, Math.round(length));
  const all = el('input', { type: 'radio', name: 'bake-range', checked: !range });
  const part = el('input', { type: 'radio', name: 'bake-range', checked: !!range, disabled: !range });
  const warmup = el('input', { type: 'number', min: '0', max: '600', step: '10', value: '60' });
  warmup.style.width = '64px';
  const seconds = (frames: number) => (frames / 30).toFixed(1);
  const body = el('div', { className: 'form' },
    el('p', { className: 'muted' },
      `把「${modelName}」的物理按 30 fps 逐帧模拟一次并记录下来。之后播放直接读取结果：可任意拖动、倒放，画面每次都一样，不再实时计算物理。动作改变后需要重新烘焙。`),
    el('label', {}, all, ` 全部：第 0–${end} 帧（${seconds(end)} 秒）`),
    el('label', {}, part, range ? ` 播放区间：第 ${range[0]}–${range[1]} 帧（${seconds(range[1] - range[0])} 秒）` : ' 播放区间（未设置）'),
    el('label', {}, '起始预热 ', warmup, ' 步（1/60 秒一步，让头发 / 裙子先落定）'),
  );

  // Cloth: which materials to simulate vertex by vertex, and how the cloth should feel.
  const boxes = cloth.map((c) => el('input', { type: 'checkbox', checked: c.suggested, value: String(c.index) }));
  const slider = (value: number) => el('input', { type: 'range', min: '0', max: '1', step: '0.05', value: String(value) });
  const stiffness = slider(DEFAULT_CLOTH.stiffness);
  const damping = slider(DEFAULT_CLOTH.damping);
  const shape = slider(DEFAULT_CLOTH.shape);
  const quality = el('select');
  for (const [steps, label] of [[8, '快'], [12, '标准'], [20, '精细（更不易拉伸，较慢）']] as const) {
    quality.append(el('option', { value: String(steps), textContent: label, selected: steps === DEFAULT_CLOTH.substeps }));
  }
  if (cloth.length) {
    body.append(
      el('h3', {}, '布料解算'),
      el('p', { className: 'muted' },
        '勾选的材质按真实布料逐顶点解算（垂坠、褶皱、搭在腿上），代替它原来的骨骼物理。适合裙子、披风、长袖；头发和有固定形状的饰品不要勾。结果只能在这里播放和导出视频，写不进 VMD。'),
      el('div', { className: 'form scroll cloth-list' }, ...cloth.map((c, i) => el('label', {}, boxes[i], ` ${c.name}`))),
      el('label', { className: 'setting' }, el('span', {}, '软 ↔ 硬挺'), stiffness),
      el('label', { className: 'setting', title: '往右：像有衬裙撑着，保持模型上的裙形；往左：完全靠自重垂下，会贴在腿上' }, el('span', {}, '垂顺 ↔ 保持裙形'), shape),
      el('label', { className: 'setting' }, el('span', {}, '飘逸 ↔ 厚重'), damping),
      el('label', { className: 'setting' }, el('span', {}, '精度'), quality),
    );
  }
  const ok = await modal('烘焙物理', body, [{ label: '取消', value: false }, { label: '开始烘焙', value: true, primary: true }], false);
  if (!ok) return null;
  const [from, to] = part.checked && range ? range : [0, end];
  const materials = boxes.filter((b) => b.checked).map((b) => Number(b.value));
  return {
    from, to, warmupSteps: Math.max(0, Math.min(600, Math.round(Number(warmup.value) || 0))),
    cloth: materials.length
      ? { materials, settings: { stiffness: Number(stiffness.value), damping: Number(damping.value), shape: Number(shape.value), substeps: Number(quality.value) } }
      : null,
  };
}

export interface VideoChoice {
  format: 'video' | 'png';
  width: number;
  height: number;
  fps: number;
  useRange: boolean;
  audio: boolean;
}

/** Settings for rendering the timeline to a video file or a PNG sequence. */
export async function exportVideoDialog(info: {
  window: [number, number];
  length: number;
  range: [number, number] | null;
  hasAudio: boolean;
  videoSupported: boolean;
}): Promise<VideoChoice | null> {
  const format = el('select');
  format.append(
    el('option', { value: 'video', textContent: '视频（MP4 / H.264）', disabled: !info.videoSupported, selected: info.videoSupported }),
    el('option', { value: 'png', textContent: 'PNG 序列（zip）', selected: !info.videoSupported }),
  );
  const sizes: [string, number, number][] = [
    [`当前窗口（${info.window[0]}×${info.window[1]}）`, info.window[0], info.window[1]],
    ['1280×720', 1280, 720], ['1920×1080', 1920, 1080], ['2560×1440', 2560, 1440], ['3840×2160', 3840, 2160],
    ['1080×1920（竖屏）', 1080, 1920],
  ];
  const size = el('select');
  sizes.forEach(([label], i) => size.append(el('option', { value: String(i), textContent: label, selected: i === 2 })));
  const fps = el('select');
  fps.append(el('option', { value: '30', textContent: '30 帧 / 秒' }), el('option', { value: '60', textContent: '60 帧 / 秒', selected: true }));
  const all = el('input', { type: 'radio', name: 'video-range', checked: !info.range });
  const part = el('input', { type: 'radio', name: 'video-range', checked: !!info.range, disabled: !info.range });
  const audio = el('input', { type: 'checkbox', checked: info.hasAudio, disabled: !info.hasAudio });
  const note = el('p', { className: 'muted' });
  const seconds = (frames: number) => (frames / 30).toFixed(1);
  const sync = () => {
    const png = format.value === 'png';
    audio.disabled = !info.hasAudio || png;
    const [from, to] = part.checked && info.range ? info.range : [0, info.length];
    const frames = Math.round(((to - from) / 30) * Number(fps.value)) + 1;
    note.textContent = png
      ? `共 ${frames} 张图片，全部保存在内存里再打包，长片段请分段导出。`
      : `共 ${frames} 帧。逐帧渲染，不受实时帧率影响；实时物理会改在主线程逐帧计算，想要每次结果一致请先烘焙物理。`;
  };
  for (const input of [format, fps, all, part]) input.addEventListener('change', sync);
  sync();
  const body = el('div', { className: 'form' },
    el('label', {}, '格式 ', format),
    el('label', {}, '尺寸 ', size),
    el('label', {}, '帧率 ', fps),
    el('label', {}, all, ` 全部：第 0–${info.length} 帧（${seconds(info.length)} 秒）`),
    el('label', {}, part, info.range ? ` 播放区间：第 ${info.range[0]}–${info.range[1]} 帧（${seconds(info.range[1] - info.range[0])} 秒）` : ' 播放区间（未设置）'),
    el('label', {}, audio, info.hasAudio ? ' 包含音频' : ' 包含音频（场景里没有音频）'),
    note,
  );
  const ok = await modal('导出视频', body, [{ label: '取消', value: false }, { label: '开始导出', value: true, primary: true }], false);
  if (!ok) return null;
  const [, width, height] = sizes[Number(size.value)];
  return {
    format: format.value as 'video' | 'png', width, height, fps: Number(fps.value),
    useRange: part.checked && !!info.range, audio: audio.checked && !audio.disabled,
  };
}

export interface AuxRow {
  slot: 'ramp' | 'light' | 'face';
  /** Texture index the map belongs to (-1 for the ramp). */
  texture: number;
  label: string;
  hint: string;
}

/**
 * The auxiliary maps of one model: what each slot holds, with buttons to generate them all,
 * load a file into a slot or clear it. Actions apply at once.
 */
export function auxMapsDialog(
  modelName: string,
  rows: AuxRow[],
  h: {
    status(row: AuxRow): string | null;
    generate(): Promise<void>;
    pick(row: AuxRow): Promise<void>;
    clear(row: AuxRow): void;
  },
): Promise<boolean> {
  const list = el('div', { className: 'form scroll aux-list' });
  const render = () => {
    list.replaceChildren(
      ...rows.map((row) => {
        const status = h.status(row);
        const pick = el('button', { className: 'btn small', textContent: '选择文件…' });
        pick.addEventListener('click', async () => {
          await h.pick(row);
          render();
        });
        const clear = el('button', { className: 'btn small', textContent: '清除', disabled: !status });
        clear.addEventListener('click', () => {
          h.clear(row);
          render();
        });
        return el('div', { className: 'aux-row', title: row.hint },
          el('span', { className: 'grow' }, row.label),
          el('span', { className: status ? '' : 'muted' }, status ?? '无'),
          pick, clear);
      }),
    );
  };
  render();
  const generate = el('button', { className: 'btn', textContent: '全部自动生成' });
  generate.addEventListener('click', async () => {
    generate.disabled = true;
    generate.textContent = '生成中…';
    await h.generate();
    generate.disabled = false;
    generate.textContent = '全部自动生成';
    render();
  });
  const body = el('div', { className: 'form' },
    el('p', { className: 'muted' },
      `材质 shader 可以读取这些贴图。「${modelName}」没有附带的，可以从模型自动生成；有原版或自己画的图片时，用「选择文件」替换对应的一项。`),
    generate,
    list,
  );
  return modal('辅助贴图', body, [{ label: '完成', value: true, primary: true }], true);
}

/** Several models found in one import: let the user pick. */
export async function pickModelsDialog(paths: string[]): Promise<string[] | null> {
  const boxes = paths.map((p, i) => el('input', { type: 'checkbox', checked: i === 0, value: p }));
  const body = el('div', { className: 'form scroll' },
    el('p', { className: 'muted' }, `找到 ${paths.length} 个模型，选择要导入的：`),
    ...paths.map((p, i) => el('label', { title: p }, boxes[i], ` ${p}`)),
  );
  const ok = await modal('导入模型', body, [{ label: '取消', value: false }, { label: '导入所选', value: true, primary: true }], false);
  return ok ? boxes.filter((b) => b.checked).map((b) => b.value) : null;
}

/** List of warnings after an operation. */
export function warningsDialog(title: string, warnings: string[]): Promise<boolean> {
  const list = el('ul', { className: 'warnings' }, ...warnings.slice(0, 50).map((w) => el('li', {}, w)));
  if (warnings.length > 50) list.append(el('li', {}, `…以及另外 ${warnings.length - 50} 条`));
  return modal(title, list, [{ label: '知道了', value: true, primary: true }], true);
}
