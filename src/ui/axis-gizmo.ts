import type { Mat4 } from '../math';

// XYZ orientation indicator. World axes are rotated by the view matrix (left-handed view
// space: x right, y up, z into the screen) and drawn back-to-front. Clicking an axis end
// calls `onPick` so the camera can snap to that side.

type Axis = 'x' | 'y' | 'z';

interface Handle {
  axis: Axis;
  sign: 1 | -1;
  x: number;
  y: number;
  depth: number;
}

const COLORS: Record<Axis, string> = { x: '#f0556a', y: '#7ccf3c', z: '#4a8cf5' };

export class AxisGizmo {
  readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly size = 96;
  private handles: Handle[] = [];
  private hover: Handle | null = null;
  private lastKey = '';

  constructor(parent: HTMLElement, private readonly onPick: (axis: Axis, sign: 1 | -1) => void) {
    this.canvas = document.createElement('canvas');
    this.canvas.id = 'axis-gizmo';
    this.canvas.style.width = this.canvas.style.height = `${this.size}px`;
    parent.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d')!;

    this.canvas.addEventListener('pointermove', (e) => {
      const h = this.pick(e);
      if (h !== this.hover) {
        this.hover = h;
        this.lastKey = ''; // force redraw
        this.canvas.style.cursor = h ? 'pointer' : 'default';
      }
    });
    this.canvas.addEventListener('pointerleave', () => {
      this.hover = null;
      this.lastKey = '';
    });
    this.canvas.addEventListener('click', (e) => {
      const h = this.pick(e);
      if (h) this.onPick(h.axis, h.sign);
    });
  }

  private pick(e: PointerEvent | MouseEvent): Handle | null {
    const r = this.canvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    // Front-most handle wins.
    let best: Handle | null = null;
    for (const h of this.handles) {
      if (Math.hypot(h.x - x, h.y - y) <= 10 && (!best || h.depth < best.depth)) best = h;
    }
    return best;
  }

  draw(view: Mat4): void {
    const key = Array.from(view.subarray(0, 11), (v) => v.toFixed(4)).join(',');
    if (key === this.lastKey) return;
    this.lastKey = key;

    const dpr = window.devicePixelRatio || 1;
    const px = Math.round(this.size * dpr);
    if (this.canvas.width !== px) this.canvas.width = this.canvas.height = px;
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, this.size, this.size);

    const c = this.size / 2;
    const len = this.size * 0.34;
    this.handles = [];
    (['x', 'y', 'z'] as Axis[]).forEach((axis, i) => {
      // World axis i in view space = column i of the view rotation.
      const vx = view[i * 4], vy = view[i * 4 + 1], vz = view[i * 4 + 2];
      for (const sign of [1, -1] as const) {
        this.handles.push({ axis, sign, x: c + vx * len * sign, y: c - vy * len * sign, depth: vz * sign });
      }
    });

    // Far to near.
    const sorted = [...this.handles].sort((a, b) => b.depth - a.depth);
    for (const h of sorted) {
      const color = COLORS[h.axis];
      const hovered = this.hover === h;
      if (h.sign > 0) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(c, c);
        ctx.lineTo(h.x, h.y);
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(h.x, h.y, h.sign > 0 ? 8 : 6, 0, Math.PI * 2);
      if (h.sign > 0) {
        ctx.fillStyle = color;
        ctx.fill();
      } else {
        ctx.globalAlpha = 0.3;
        ctx.fillStyle = color;
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
      if (hovered) {
        ctx.strokeStyle = getComputedStyle(this.canvas).color;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      if (h.sign > 0) {
        ctx.fillStyle = '#111';
        ctx.font = 'bold 10px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(h.axis.toUpperCase(), h.x, h.y + 0.5);
      }
    }
  }
}
