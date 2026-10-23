import { MMD_FPS } from '../animation/motion';
import { icon, setIcon, type IconName } from './icons';

// Frame-based timeline: transport controls + a canvas with a ruler, keyframe rows (camera,
// bones, morphs of the selected model) and the audio waveform. View / navigate only.

export interface TimelineTrack {
  label: string;
  color: string;
  /** Sorted keyframe numbers. */
  frames: Uint32Array;
  /** A covered frame range drawn as a bar (e.g. a physics bake). */
  span?: [number, number];
}

export interface TimelineState {
  frame: number;
  length: number;
  playing: boolean;
  range: [number, number] | null;
  loop: boolean;
  tracks: TimelineTrack[];
  /** Peak amplitude (0–1) per frame, or null without audio. */
  waveform: Float32Array | null;
}

export interface TimelineHandlers {
  seek(frame: number): void;
  togglePlay(): void;
  step(delta: number): void;
  jumpKey(dir: 1 | -1): void;
  toStart(): void;
  toEnd(): void;
  setRange(range: [number, number] | null): void;
  setRangeEdge(edge: 'in' | 'out'): void;
  toggleLoop(): void;
}

const LABEL_W = 56;
const RULER_H = 20;
const ROW_H = 18;
const WAVE_H = 34;
const TICK_STEPS = [1, 2, 5, 10, 15, 30, 60, 150, 300, 600, 1800, 3600, 9000, 18000];

const EMPTY_H = 30;

function button(content: IconName | { text: string }, title: string, onClick: () => void): HTMLButtonElement {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'tl-btn';
  if (typeof content === 'string') b.append(icon(content));
  else b.textContent = content.text;
  b.title = title;
  b.setAttribute('aria-label', title);
  b.addEventListener('click', onClick);
  return b;
}

function numberInput(onChange: (v: number) => void, className: string, label: string): HTMLInputElement {
  const i = document.createElement('input');
  i.type = 'number';
  i.min = '0';
  i.step = '1';
  i.className = className;
  i.title = label;
  i.setAttribute('aria-label', label);
  i.addEventListener('change', () => {
    const v = Math.max(0, Math.round(Number(i.value)));
    if (Number.isFinite(v)) onChange(v);
  });
  return i;
}

export function formatTime(frame: number): string {
  const s = frame / MMD_FPS;
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, '0')}:${(s - m * 60).toFixed(2).padStart(5, '0')}`;
}

export class Timeline {
  private state: TimelineState = { frame: 0, length: 0, playing: false, range: null, loop: true, tracks: [], waveform: null };
  private readonly body: HTMLDivElement;
  private readonly canvas: HTMLCanvasElement;
  /** Canvas colours, taken from the stylesheet so they follow the glass tone. */
  private colors = { text: '', faint: '', row: '', beyond: '', accent: '', accentSoft: '', playhead: '' };
  private readonly ctx: CanvasRenderingContext2D;
  private readonly frameInput: HTMLInputElement;
  private readonly lengthLabel: HTMLSpanElement;
  private readonly timeLabel: HTMLSpanElement;
  private readonly playBtn: HTMLButtonElement;
  private readonly loopBtn: HTMLButtonElement;
  private readonly rangeIn: HTMLInputElement;
  private readonly rangeOut: HTMLInputElement;
  /** Frame at the left edge of the track area, and pixels per frame. */
  private viewStart = 0;
  private scale = 2;
  private fitted = false;
  private dirty = true;
  private lastLength = -1;
  private shownPlaying = false;
  private bodyHeight = -1;

  constructor(root: HTMLElement, private readonly h: TimelineHandlers) {
    const bar = document.createElement('div');
    bar.className = 'tl-transport';
    this.playBtn = button('play', '播放 / 暂停（空格）', () => h.togglePlay());
    this.playBtn.classList.add('play');
    this.frameInput = numberInput((v) => h.seek(v), 'tl-frame', '当前帧');
    this.lengthLabel = document.createElement('span');
    this.lengthLabel.className = 'tl-length';
    this.timeLabel = document.createElement('span');
    this.timeLabel.className = 'tl-time';
    this.rangeIn = numberInput((v) => this.commitRange(v, null), 'tl-range', '区间起点');
    this.rangeOut = numberInput((v) => this.commitRange(null, v), 'tl-range', '区间终点');
    this.rangeIn.placeholder = '起点';
    this.rangeOut.placeholder = '终点';
    this.loopBtn = button('loop', '循环播放（L）：有区间时循环区间', () => h.toggleLoop());

    const group = (className: string, ...els: HTMLElement[]) => {
      const g = document.createElement('div');
      g.className = `tl-group ${className}`;
      g.append(...els);
      return g;
    };
    bar.append(
      group('',
        button('toStart', '到开头 / 区间起点（Home）', () => h.toStart()),
        button('prevKey', '上一个关键帧（Alt+←）', () => h.jumpKey(-1)),
        button('stepBack', '上一帧（←）', () => h.step(-1)),
        this.playBtn,
        button('stepForward', '下一帧（→）', () => h.step(1)),
        button('nextKey', '下一个关键帧（Alt+→）', () => h.jumpKey(1)),
        button('toEnd', '到结尾 / 区间终点（End）', () => h.toEnd()),
        this.loopBtn,
      ),
      group('fields', this.frameInput, this.lengthLabel, this.timeLabel),
      group('',
        button('rangeIn', '当前帧设为区间起点（I）', () => h.setRangeEdge('in')),
        this.rangeIn,
        this.rangeOut,
        button('rangeOut', '当前帧设为区间终点（O）', () => h.setRangeEdge('out')),
        button('close', '清除区间', () => h.setRange(null)),
      ),
      group('',
        button('minus', '缩小时间线（⌘/Ctrl + 滚轮）', () => this.zoom(1 / 1.5)),
        button('fit', '显示全部（双击标尺）', () => this.fit()),
        button('plus', '放大时间线（⌘/Ctrl + 滚轮）', () => this.zoom(1.5)),
      ),
    );

    const body = (this.body = document.createElement('div'));
    body.className = 'tl-body';
    this.canvas = document.createElement('canvas');
    body.appendChild(this.canvas);
    root.append(bar, body);
    this.ctx = this.canvas.getContext('2d')!;
    this.readColors();
    window.addEventListener('stagetone', () => this.readColors());
    this.attachPointer();
    new ResizeObserver(() => {
      if (!this.fitted) this.fit();
      this.dirty = true;
    }).observe(body);
  }

  private readColors(): void {
    const css = getComputedStyle(document.documentElement);
    const v = (name: string) => css.getPropertyValue(name).trim();
    this.colors = {
      text: v('--ink-2'), faint: v('--ink-3'), row: v('--tl-row'), beyond: v('--tl-beyond'),
      accent: v('--accent'), accentSoft: v('--accent-soft'), playhead: v('--playhead'),
    };
    this.dirty = true;
  }

  private commitRange(a: number | null, b: number | null): void {
    const cur = this.state.range ?? [0, Math.max(1, Math.round(this.state.length))];
    const from = a ?? cur[0];
    const to = b ?? cur[1];
    this.h.setRange(from < to ? [from, to] : [to, from]);
  }

  private get trackWidth(): number {
    return Math.max(1, this.canvas.clientWidth - LABEL_W);
  }

  private frameToX(f: number): number {
    return LABEL_W + (f - this.viewStart) * this.scale;
  }

  private xToFrame(x: number): number {
    return this.viewStart + (x - LABEL_W) / this.scale;
  }

  fit(): void {
    // Hidden (tracks collapsed): fit once there is a width to fit to.
    if (this.canvas.clientWidth === 0) {
      this.fitted = false;
      return;
    }
    const len = Math.max(this.state.length, 300);
    this.scale = this.trackWidth / (len * 1.02);
    this.viewStart = 0;
    this.fitted = true;
    this.dirty = true;
  }

  private zoom(factor: number, anchorX = LABEL_W + this.trackWidth / 2): void {
    const f = this.xToFrame(anchorX);
    this.scale = Math.min(40, Math.max(0.01, this.scale * factor));
    this.viewStart = Math.max(0, f - (anchorX - LABEL_W) / this.scale);
    this.dirty = true;
  }

  private attachPointer(): void {
    const c = this.canvas;
    let drag: 'scrub' | 'in' | 'out' | null = null;
    const frameAt = (e: PointerEvent | MouseEvent) => {
      const r = c.getBoundingClientRect();
      return Math.max(0, Math.round(this.xToFrame(e.clientX - r.left)));
    };
    c.addEventListener('pointerdown', (e) => {
      const r = c.getBoundingClientRect();
      const x = e.clientX - r.left;
      const y = e.clientY - r.top;
      if (x < LABEL_W) return;
      c.setPointerCapture(e.pointerId);
      const range = this.state.range;
      if (range && y < RULER_H) {
        if (Math.abs(x - this.frameToX(range[0])) < 6) drag = 'in';
        else if (Math.abs(x - this.frameToX(range[1])) < 6) drag = 'out';
      }
      if (!drag) {
        drag = 'scrub';
        this.h.seek(frameAt(e));
      }
    });
    c.addEventListener('pointermove', (e) => {
      if (!drag) {
        const r = c.getBoundingClientRect();
        const range = this.state.range;
        const x = e.clientX - r.left;
        const nearEdge = !!range && e.clientY - r.top < RULER_H && [range[0], range[1]].some((f) => Math.abs(x - this.frameToX(f)) < 6);
        c.style.cursor = nearEdge ? 'ew-resize' : x > LABEL_W ? 'pointer' : 'default';
        return;
      }
      const f = frameAt(e);
      if (drag === 'scrub') this.h.seek(f);
      else if (this.state.range) {
        const [a, b] = this.state.range;
        if (drag === 'in') this.h.setRange([Math.min(f, b - 1), b]);
        else this.h.setRange([a, Math.max(f, a + 1)]);
      }
    });
    const end = (e: PointerEvent) => {
      drag = null;
      if (c.hasPointerCapture(e.pointerId)) c.releasePointerCapture(e.pointerId);
    };
    c.addEventListener('pointerup', end);
    c.addEventListener('pointercancel', end);
    c.addEventListener('dblclick', (e) => {
      if (e.clientY - c.getBoundingClientRect().top < RULER_H) this.fit();
    });
    c.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        if (e.ctrlKey || e.metaKey) {
          this.zoom(Math.exp(-e.deltaY * 0.004), e.clientX - c.getBoundingClientRect().left);
        } else {
          const d = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
          this.viewStart = Math.max(0, this.viewStart + d / this.scale);
          this.dirty = true;
        }
      },
      { passive: false },
    );
  }

  /** Feed new state; cheap to call every frame (redraws only on change). */
  update(state: TimelineState): void {
    const prev = this.state;
    if (
      prev.frame !== state.frame || prev.length !== state.length || prev.range !== state.range ||
      prev.tracks !== state.tracks || prev.waveform !== state.waveform || prev.loop !== state.loop || prev.playing !== state.playing
    ) this.dirty = true;
    this.state = state;

    if (state.length !== this.lastLength) {
      // New content: refit if the old view no longer makes sense.
      if (state.length > this.lastLength * 1.2 || state.length < this.lastLength * 0.5) this.fit();
      this.lastLength = state.length;
    }
    // Follow the playhead while playing.
    if (state.playing) {
      const x = this.frameToX(state.frame);
      if (x > LABEL_W + this.trackWidth - 20 || x < LABEL_W) {
        this.viewStart = Math.max(0, state.frame - 10 / this.scale);
        this.dirty = true;
      }
    }

    const f = Math.round(state.frame);
    if (document.activeElement !== this.frameInput) this.frameInput.value = String(f);
    this.lengthLabel.textContent = `/ ${Math.round(state.length)} 帧`;
    this.timeLabel.textContent = formatTime(state.frame);
    if (state.playing !== this.shownPlaying) setIcon(this.playBtn, (this.shownPlaying = state.playing) ? 'pause' : 'play');
    this.loopBtn.classList.toggle('active', state.loop);
    if (document.activeElement !== this.rangeIn) this.rangeIn.value = state.range ? String(state.range[0]) : '';
    if (document.activeElement !== this.rangeOut) this.rangeOut.value = state.range ? String(state.range[1]) : '';

    if (this.dirty) this.draw();
  }

  private draw(): void {
    this.dirty = false;
    const { canvas, ctx, state } = this;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.font = '11px system-ui, -apple-system, "PingFang SC", sans-serif';
    ctx.textBaseline = 'middle';

    const rows: { label: string; y: number; h: number; draw: (y: number, h: number) => void }[] = [];
    let y = RULER_H;
    for (const t of state.tracks) {
      rows.push({ label: t.label, y, h: ROW_H, draw: (ry, rh) => this.drawKeys(t, ry, rh) });
      y += ROW_H;
    }
    if (state.waveform) {
      const wave = state.waveform;
      rows.push({ label: '音频', y, h: WAVE_H, draw: (ry, rh) => this.drawWave(wave, ry, rh) });
      y += WAVE_H;
    }

    // The dock is only as tall as its rows, so it covers as little of the stage as possible.
    const wanted = (rows.length ? y : RULER_H + EMPTY_H) + 4;
    if (wanted !== this.bodyHeight) this.body.style.height = `${(this.bodyHeight = wanted)}px`;

    const col = this.colors;
    ctx.save();
    ctx.beginPath();
    ctx.rect(LABEL_W, 0, w - LABEL_W, h);
    ctx.clip();

    // Row backgrounds
    ctx.fillStyle = col.row;
    rows.forEach((r, i) => {
      if (i % 2 === 0) ctx.fillRect(LABEL_W, r.y, w - LABEL_W, r.h);
    });
    // Beyond the end
    const endX = this.frameToX(state.length);
    if (endX < w) {
      ctx.fillStyle = col.beyond;
      ctx.fillRect(Math.max(LABEL_W, endX), RULER_H, w - Math.max(LABEL_W, endX), y - RULER_H);
    }
    // Range
    if (state.range) {
      const x0 = Math.max(LABEL_W, this.frameToX(state.range[0]));
      const x1 = Math.min(w, this.frameToX(state.range[1]));
      if (x1 > x0) {
        ctx.fillStyle = col.accentSoft;
        ctx.fillRect(x0, 0, x1 - x0, y);
      }
    }
    for (const r of rows) r.draw(r.y, r.h);

    this.drawRuler(w);

    // Range edges (on top of the ruler)
    if (state.range) {
      ctx.fillStyle = col.accent;
      for (const f of state.range) {
        const x = this.frameToX(f);
        if (x >= LABEL_W && x <= w) ctx.fillRect(x - 1, 0, 2, y);
      }
    }
    if (rows.length === 0) {
      ctx.fillStyle = col.faint;
      ctx.fillText('导入动作、镜头或音频后，这里显示关键帧与波形', LABEL_W + 12, RULER_H + EMPTY_H / 2);
    }

    // Playhead
    const px = this.frameToX(state.frame);
    if (px >= LABEL_W && px <= w) {
      ctx.fillStyle = col.playhead;
      ctx.fillRect(px - 0.75, 0, 1.5, Math.max(y, h));
      ctx.beginPath();
      ctx.moveTo(px - 5, 0);
      ctx.lineTo(px + 5, 0);
      ctx.lineTo(px, 7);
      ctx.fill();
    }
    ctx.restore();

    // Labels
    ctx.fillStyle = col.text;
    for (const r of rows) ctx.fillText(r.label, 10, r.y + r.h / 2);
  }

  private drawRuler(w: number): void {
    const { ctx } = this;
    // Labels need ~60 px each; past the table (very long or far zoomed out) keep doubling.
    let step = TICK_STEPS.find((s) => s * this.scale >= 60) ?? TICK_STEPS[TICK_STEPS.length - 1];
    while (step * this.scale < 60) step *= 2;
    const minor = TICK_STEPS.slice().reverse().find((s) => s < step && step % s === 0 && s * this.scale >= 8) ?? 0;
    const first = Math.floor(this.viewStart / step) * step;
    const last = this.xToFrame(w);
    ctx.fillStyle = this.colors.faint;
    if (minor) {
      for (let f = Math.floor(this.viewStart / minor) * minor; f <= last; f += minor) {
        const x = this.frameToX(f);
        if (x >= LABEL_W) ctx.fillRect(x, RULER_H - 4, 1, 4);
      }
    }
    ctx.fillStyle = this.colors.text;
    for (let f = first; f <= last; f += step) {
      const x = this.frameToX(f);
      if (x < LABEL_W) continue;
      ctx.fillRect(x, RULER_H - 8, 1, 8);
      ctx.fillText(String(f), x + 3, 8);
    }
  }

  private drawKeys(t: TimelineTrack, y: number, h: number): void {
    const { ctx } = this;
    const frames = t.frames;
    const w = this.canvas.clientWidth;
    if (t.span) {
      const x0 = Math.max(LABEL_W, this.frameToX(t.span[0]));
      const x1 = Math.min(w, this.frameToX(t.span[1]));
      if (x1 > x0) {
        ctx.fillStyle = t.color;
        ctx.globalAlpha = 0.55;
        ctx.fillRect(x0, y + h / 2 - 3, x1 - x0, 6);
        ctx.globalAlpha = 1;
      }
    }
    // First key in view (binary search).
    let lo = 0;
    let hi = frames.length;
    const startF = this.xToFrame(LABEL_W - 4);
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (frames[mid] < startF) lo = mid + 1;
      else hi = mid;
    }
    ctx.fillStyle = t.color;
    const cy = y + h / 2;
    const r = Math.min(4, h / 2 - 2);
    let lastX = -Infinity;
    for (let i = lo; i < frames.length; i++) {
      const x = Math.round(this.frameToX(frames[i]));
      if (x > w + 4) break;
      if (x - lastX < 2) continue; // many keys on one pixel: draw once
      lastX = x;
      ctx.beginPath();
      ctx.moveTo(x, cy - r);
      ctx.lineTo(x + r, cy);
      ctx.lineTo(x, cy + r);
      ctx.lineTo(x - r, cy);
      ctx.fill();
    }
  }

  private drawWave(peaks: Float32Array, y: number, h: number): void {
    const { ctx } = this;
    const w = this.canvas.clientWidth;
    ctx.fillStyle = this.colors.accent;
    ctx.globalAlpha = 0.6;
    const mid = y + h / 2;
    for (let x = LABEL_W; x < w; x++) {
      const f0 = Math.max(0, Math.floor(this.xToFrame(x)));
      const f1 = Math.min(peaks.length, Math.max(f0 + 1, Math.floor(this.xToFrame(x + 1))));
      if (f0 >= peaks.length) break;
      let peak = 0;
      for (let f = f0; f < f1; f++) peak = Math.max(peak, peaks[f]);
      const a = Math.max(0.5, peak * (h / 2 - 2));
      ctx.fillRect(x, mid - a, 1, a * 2);
    }
    ctx.globalAlpha = 1;
  }
}

/** Per-frame peak amplitude of an audio file (for the waveform row). */
export async function computeWaveform(blob: Blob): Promise<{ peaks: Float32Array; duration: number }> {
  const ctx = new OfflineAudioContext(1, 1, 44100);
  const audio = await ctx.decodeAudioData(await blob.arrayBuffer());
  const frames = Math.ceil(audio.duration * MMD_FPS);
  const peaks = new Float32Array(frames);
  const perFrame = audio.sampleRate / MMD_FPS;
  for (let ch = 0; ch < audio.numberOfChannels; ch++) {
    const data = audio.getChannelData(ch);
    for (let f = 0; f < frames; f++) {
      const start = Math.floor(f * perFrame);
      const end = Math.min(data.length, Math.floor((f + 1) * perFrame));
      let m = peaks[f];
      for (let i = start; i < end; i++) {
        const v = Math.abs(data[i]);
        if (v > m) m = v;
      }
      peaks[f] = m;
    }
  }
  let max = 0;
  for (const p of peaks) max = Math.max(max, p);
  if (max > 0) for (let f = 0; f < frames; f++) peaks[f] /= max;
  return { peaks, duration: audio.duration };
}
