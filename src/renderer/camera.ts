import { clamp, mat4, quat, vec3 } from '../math';
import type { CameraPose } from '../animation/motion';

/**
 * Orbit camera in MMD's left-handed space. yaw = 0 looks at the model's front (from -Z).
 * When `pose` is set (camera VMD playback) it drives the view instead of the orbit controls.
 */
export class OrbitCamera {
  readonly target = vec3.create(0, 10, 0);
  distance = 45;
  yaw = 0;
  pitch = 0;
  fovY = (30 * Math.PI) / 180; // MMD default
  near = 0.5;
  /** Far enough for stages and sky domes. */
  far = 10000;

  /**
   * Shifts (NDC units) and scales the picture on screen without changing the perspective, so
   * what would fill the window fills the area the interface leaves free instead. Orbit mode
   * only: a camera VMD is framed for the whole picture.
   */
  readonly lensShift: [number, number] = [0, 0];
  lensScale = 1;

  /** Camera VMD pose; null = user orbit controls. */
  pose: CameraPose | null = null;
  /** Called when the user starts orbiting / zooming (e.g. to leave camera-VMD mode). */
  onInteract: (() => void) | null = null;

  /** True while a camera VMD key with perspective off is in effect. */
  orthographic = false;

  readonly eye = vec3.create();
  readonly view = mat4.create();
  readonly proj = mat4.create();
  readonly viewProj = mat4.create();
  private readonly up = vec3.create(0, 1, 0);
  private readonly tq = quat.create();

  /** Distance from the eye to what the camera looks at (depth of field focuses there). */
  get focusDistance(): number {
    return Math.max(this.pose ? Math.abs(this.pose.distance) : this.distance, this.near);
  }

  update(aspect: number): void {
    this.orthographic = false;
    if (this.pose) {
      this.updateFromPose(this.pose, aspect);
      return;
    }
    vec3.set(this.up, 0, 1, 0);
    const cp = Math.cos(this.pitch);
    vec3.set(
      this.eye,
      this.target[0] + Math.sin(this.yaw) * cp * this.distance,
      this.target[1] + Math.sin(this.pitch) * this.distance,
      this.target[2] - Math.cos(this.yaw) * cp * this.distance,
    );
    mat4.lookAtLH(this.view, this.eye, this.target, this.up);
    mat4.perspectiveLH(this.proj, this.fovY, aspect, this.near, this.far);
    this.proj[0] *= this.lensScale;
    this.proj[5] *= this.lensScale;
    this.proj[8] = this.lensShift[0];
    this.proj[9] = this.lensShift[1];
    mat4.multiply(this.viewProj, this.proj, this.view);
  }

  /**
   * MMD camera (same as babylon-mmd, which is left-handed too):
   *   R = Ry(-ry) · Rx(-rx) · Rz(-rz),  eye = target + R · (0, 0, distance),  up = R · (0, 1, 0)
   */
  private updateFromPose(pose: CameraPose, aspect: number): void {
    const r = quat.fromYawPitchRoll(this.tq, -pose.rotation[0], -pose.rotation[1], -pose.rotation[2]);
    vec3.transformQuat(this.eye, vec3.set(this.eye, 0, 0, pose.distance), r);
    vec3.add(this.eye, this.eye, pose.target);
    vec3.transformQuat(this.up, vec3.set(this.up, 0, 1, 0), r);
    mat4.lookAtLH(this.view, this.eye, pose.target, this.up);
    const fov = (Math.max(pose.fov, 1) * Math.PI) / 180;
    if (pose.perspective === false) {
      // Perspective off: parallel projection showing what the perspective view shows at the target.
      this.orthographic = true;
      const halfH = Math.max(Math.abs(pose.distance), 1e-3) * Math.tan(fov / 2);
      mat4.orthoLH(this.proj, -halfH * aspect, halfH * aspect, -halfH, halfH, -this.far / 2, this.far);
    } else {
      mat4.perspectiveLH(this.proj, fov, aspect, this.near, this.far);
    }
    mat4.multiply(this.viewProj, this.proj, this.view);
  }

  /** Look along a world axis (used by the axis gizmo). `axis` is where the camera sits. */
  snapTo(axis: 'x' | 'y' | 'z', sign: 1 | -1): void {
    if (axis === 'y') {
      this.yaw = 0;
      this.pitch = sign * 1.55;
    } else {
      this.pitch = 0;
      // eye offset = (sin yaw, 0, -cos yaw)
      this.yaw = axis === 'x' ? (sign * Math.PI) / 2 : sign > 0 ? Math.PI : 0;
    }
  }

  /** Frame an axis-aligned bounding box. */
  frame(min: ArrayLike<number>, max: ArrayLike<number>): void {
    vec3.set(this.target, (min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2);
    const height = max[1] - min[1];
    this.distance = Math.max((height * 0.6) / Math.tan(this.fovY / 2), 1);
    this.yaw = 0;
    this.pitch = 0;
  }

  attach(el: HTMLElement): void {
    let mode: 'rotate' | 'pan' | null = null;
    let lastX = 0;
    let lastY = 0;
    el.addEventListener('contextmenu', (e) => e.preventDefault());
    el.addEventListener('pointerdown', (e) => {
      this.onInteract?.();
      mode = e.button === 2 || e.button === 1 || e.shiftKey ? 'pan' : 'rotate';
      lastX = e.clientX;
      lastY = e.clientY;
      el.setPointerCapture(e.pointerId);
    });
    el.addEventListener('pointerup', (e) => {
      mode = null;
      el.releasePointerCapture(e.pointerId);
    });
    el.addEventListener('pointermove', (e) => {
      if (!mode) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      if (mode === 'rotate') {
        this.yaw -= dx * 0.008;
        this.pitch = clamp(this.pitch + dy * 0.008, -1.55, 1.55);
      } else {
        // Pan along the camera's right / up axes (rows 0 and 1 of the view matrix).
        const s = (this.distance * Math.tan(this.fovY / 2) * 2) / el.clientHeight;
        const v = this.view;
        for (let c = 0; c < 3; c++) this.target[c] += (-dx * v[c * 4] + dy * v[c * 4 + 1]) * s;
      }
    });
    el.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        this.onInteract?.();
        this.distance = clamp(this.distance * Math.exp(e.deltaY * 0.001), 0.5, 2000);
      },
      { passive: false },
    );
  }
}
