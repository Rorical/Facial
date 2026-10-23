import { keyframeUnion, MMD_FPS, MmdMotion, splitVmd, type CameraPose } from '../animation/motion';
import { basename, FileSet, UrlAssetSource, type AssetSource } from '../loaders/asset-source';
import { parseVmd } from '../loaders/vmd';
import { vec3 } from '../math';
import { saveBlob } from '../io/download';
import { collectModelFiles, modelArchiveName } from '../io/model-export';
import { buildProject, projectFileName, readProjectFile, type ProjectFile, type ProjectSettings } from '../io/project';
import { deserializeBake } from '../physics/bake';
import { motionToVmd, type ExportRange } from '../io/vmd-export';
import { CAMERA_VMD_MODEL_NAME, writeVmd } from '../io/vmd-writer';
import { writeZip } from '../io/zip';
import { clearSharedToons, saveSharedToons } from '../loaders/toon-store';
import { bodyTransformsFromBones, createPhysicsRunner } from '../physics/runners';
import { BakedPhysics, bakeToVmdBoneFrames } from '../physics/bake';
import { bakeInWorker } from '../physics/bake-client';
import { suggestClothMaterials } from '../cloth/cloth-mesh';
import { OrbitCamera } from '../renderer/camera';
import { detectShaderKind, formatShaderErrors, type CustomShader, type ShaderKind } from '../renderer/custom-shader';
import { DEFAULT_POST } from '../renderer/post';
import type { MaterialPipelines, Renderer } from '../renderer/renderer';
import { assignLighting, defaultLighting, lightDirection } from '../renderer/lighting';
import { TextureManager } from '../renderer/textures';
import { AxisGizmo } from '../ui/axis-gizmo';
import { auxMapsDialog, bakeDialog, confirmDialog, exportMotionDialog, exportVideoDialog, pickModelsDialog, progress, toast, warningsDialog } from '../ui/dialog';
import { pickFiles } from '../ui/menu';
import { ScenePanel } from '../ui/scene-panel';
import { computeWaveform, Timeline, type TimelineTrack } from '../ui/timeline';
import { bakeFaceMaps, bakeVertexOcclusion, generateRamp, imageToPng, occlusionToLightMaps, type RgbaImage } from '../bake/aux-maps';
import { classifyMaterial, MaterialPart } from '../runtime/morph-controller';
import { loadSceneModel, Scene, type AuxSlot, type SceneModel } from './scene';

/** Live physics mode for models without an active bake. */
export type PhysicsMode = 'worker' | 'main' | 'off';

const AUDIO_EXTS = ['mp3', 'wav', 'ogg', 'm4a', 'aac', 'flac', 'opus'];
const MODEL_EXTS = ['pmx', 'pmd', 'x'];
const IMAGE_EXTS = ['jpg', 'jpeg', 'png', 'webp'];

export interface AppDom {
  viewport: HTMLElement;
  canvas: HTMLCanvasElement;
  sceneRoot: HTMLElement;
  timelineRoot: HTMLElement;
  info: HTMLElement;
  audio: HTMLAudioElement;
  controls: {
    edges: HTMLInputElement;
    selfShadow: HTMLInputElement;
    darkBackground: HTMLInputElement;
    physics: HTMLSelectElement;
    wireframe: HTMLInputElement[];
    resetToons: HTMLButtonElement;
  };
  /** Refresh the settings controls after the app changed renderer settings (opening a project). */
  syncControls(): void;
}

export class App {
  readonly scene = new Scene();
  readonly camera = new OrbitCamera();
  private readonly gizmo: AxisGizmo;
  private readonly timeline: Timeline;
  private readonly panel: ScenePanel;

  // Playback
  private playing = false;
  private clockFrame = 0;
  range: [number, number] | null = null;
  loop = true;
  private lastFrame = 0;
  private lastNow = performance.now();
  private fps = 0;
  private cpuMs = 0;

  // Settings
  physicsMode: PhysicsMode = 'worker';
  private useCameraMotion = true;
  /**
   * Fits a camera motion to a model of another height: target height and distance are scaled
   * (so the same part of the body stays framed at the same size), then the target is raised.
   */
  cameraFit = { scale: 1, offsetY: 0 };
  private readonly cameraPose: CameraPose = { target: vec3.create(), rotation: vec3.create(), distance: -45, fov: 30 };
  customToons = 0;

  // Derived data for the timeline
  private tracks: TimelineTrack[] = [];
  private keyFrames: Uint32Array = new Uint32Array(0);
  private waveform: Float32Array | null = null;
  private wireframeScratch = new Float32Array(0);
  private readonly physicsTokens = new Map<number, number>();
  private infoTimer = 0;
  /** While frames are being exported the live loop stands still. */
  private exporting = false;
  private backgroundBlob: Blob | null = null;
  /** Compiled material shaders by source text, shared by models using the same one. */
  private readonly materialShaders = new Map<string, MaterialPipelines>();

  constructor(readonly renderer: Renderer, private readonly dom: AppDom) {
    this.camera.attach(dom.canvas);
    this.camera.onInteract = () => this.setUseCameraMotion(false);
    this.gizmo = new AxisGizmo(dom.viewport, (axis, sign) => {
      this.setUseCameraMotion(false);
      this.camera.snapTo(axis, sign);
    });

    this.panel = new ScenePanel(dom.sceneRoot, {
      select: (id) => this.scene.select(id),
      toggleVisible: (id) => this.toggleVisible(id),
      remove: (id) => this.scene.remove(id),
      clearMotion: (id) => {
        const m = this.scene.models.find((x) => x.id === id);
        if (m) this.motionChanged(m, m.setMotion(null, []));
      },
      clearCamera: () => this.scene.clearCameraMotion(),
      clearAudio: () => this.scene.clearAudio(),
      setMotionScale: (id, scale) => {
        const m = this.scene.models.find((x) => x.id === id);
        if (!m) return;
        const dropped = m.setMotionScale(scale);
        m.model.physics?.requestReset(30);
        if (dropped) this.motionChanged(m, true);
      },
      setCameraFit: (scale, offsetY) => (this.cameraFit = { scale, offsetY }),
      importShader: (kind) => void this.importShaderDialog(kind),
      clearModelShader: (id) => {
        const m = this.scene.models.find((x) => x.id === id);
        if (m) void this.setModelShader(m, null);
      },
      clearPostShader: () => void this.setPostShader(null),
      editAux: (id) => {
        const m = this.scene.models.find((x) => x.id === id);
        if (m) void this.editAuxMaps(m);
      },
      importBackground: () => void this.importBackgroundDialog(),
      clearBackground: () => {
        this.scene.background = null;
        this.scene.changed();
      },
      setTransform: (id, position, rotationY) => {
        const m = this.scene.models.find((x) => x.id === id);
        if (!m) return;
        m.setTransform(position, rotationY);
        m.model.physics?.requestReset(30);
      },
      setMaterialVisible: (id, index, on) => {
        const m = this.scene.models.find((x) => x.id === id);
        if (!m) return;
        m.setMaterialVisible(index, on);
        this.scene.changed();
      },
      showAllMaterials: (id) => {
        const m = this.scene.models.find((x) => x.id === id);
        if (!m) return;
        m.setHiddenMaterials([]);
        this.scene.changed();
      },
      setUseCameraMotion: (on) => this.setUseCameraMotion(on),
      importModel: () => void this.importModelDialog(false),
      importMotion: () => void this.importMotionDialog('model'),
      importCamera: () => void this.importMotionDialog('camera'),
      importAudio: () => void this.importAudioDialog(),
      bake: (id) => void this.bakeModel(id),
      setUseBake: (id, on) => {
        const m = this.scene.models.find((x) => x.id === id);
        if (!m) return;
        m.useBake = on;
        void this.setupPhysics(m);
        this.scene.changed();
      },
      clearBake: (id) => {
        const m = this.scene.models.find((x) => x.id === id);
        if (!m) return;
        m.bake = null;
        void this.setupPhysics(m);
        this.scene.changed();
      },
    });

    this.timeline = new Timeline(dom.timelineRoot, {
      seek: (f) => this.setFrame(f),
      togglePlay: () => this.togglePlay(),
      step: (d) => this.step(d),
      jumpKey: (d) => this.jumpKey(d),
      toStart: () => this.setFrame(this.range?.[0] ?? 0),
      toEnd: () => this.setFrame(this.range?.[1] ?? Math.round(this.length)),
      setRange: (r) => this.setRange(r),
      setRangeEdge: (edge) => this.setRangeEdge(edge),
      toggleLoop: () => (this.loop = !this.loop),
    });

    this.scene.onChange(() => this.onSceneChanged());
    this.onSceneChanged();
    requestAnimationFrame(this.tick);
  }

  // =====================================================================================
  // Scene bookkeeping

  private onSceneChanged(): void {
    const scene = this.scene;
    const sel = scene.selected;
    this.renderer.physicsDebug.setRig(sel?.rig ?? null);
    this.wireframeScratch = new Float32Array((sel?.rig?.bodies.length ?? 0) * 16);

    const tracks: TimelineTrack[] = [];
    const cam = scene.cameraMotion?.cameraTrack;
    if (cam) tracks.push({ label: '镜头', color: '#ff9500', frames: cam.frames });
    const lightKeys = [...(scene.cameraMotion?.lightFrames ?? []), ...(scene.cameraMotion?.selfShadowFrames ?? [])];
    if (lightKeys.length) tracks.push({ label: '灯光', color: '#ffcc00', frames: keyframeUnion([{ frames: Uint32Array.from(lightKeys, (k) => k.frame) }]) });
    if (sel?.motion) {
      tracks.push({ label: '骨骼', color: '#5e5ce6', frames: keyframeUnion(sel.motion.boneTracks.values()) });
      tracks.push({ label: '表情', color: '#ff2d55', frames: keyframeUnion(sel.motion.morphTracks.values()) });
    }
    this.keyFrames = keyframeUnion(tracks);
    if (sel?.bake) tracks.push({ label: '物理', color: sel.useBake ? '#30b0c7' : '#8e8e93', frames: new Uint32Array(0), span: [sel.bake.from, sel.bake.to] });
    this.tracks = tracks;

    // Audio element + waveform follow the scene's audio.
    const audio = this.dom.audio;
    const url = scene.audio?.url ?? '';
    if (audio.src !== url) {
      if (url) {
        audio.src = url;
        audio.currentTime = this.clockFrame / MMD_FPS;
        const blob = scene.audio!.blob;
        this.waveform = null;
        computeWaveform(blob)
          .then(({ peaks }) => {
            if (this.scene.audio?.blob === blob) this.waveform = peaks;
          })
          .catch((err) => console.warn('Waveform failed:', err));
      } else {
        audio.removeAttribute('src');
        audio.load();
        this.waveform = null;
      }
    }

    // The panorama follows the scene's background image.
    const bg = scene.background?.blob ?? null;
    if (bg !== this.backgroundBlob) {
      this.backgroundBlob = bg;
      if (!bg) this.renderer.background.setTexture(null);
      else {
        this.renderer.core.textures.fromImageBlob(bg, false, 'background').then(
          (texture) => {
            if (this.backgroundBlob === bg) this.renderer.background.setTexture(texture);
            else texture.destroy();
          },
          (err) => toast(`背景图片无法读取：${err instanceof Error ? err.message : err}`, 'error'),
        );
      }
    }

    this.panel.useCameraMotion = this.useCameraMotion;
    this.panel.cameraFit = this.cameraFit;
    this.panel.render(scene);
  }

  private toggleVisible(id: number): void {
    const m = this.scene.models.find((x) => x.id === id);
    if (!m) return;
    m.visible = !m.visible;
    if (m.visible) m.model.physics?.requestReset(30);
    this.scene.changed();
  }

  setUseCameraMotion(on: boolean): void {
    if (this.useCameraMotion === on) return;
    this.useCameraMotion = on;
    this.panel.useCameraMotion = on;
    this.panel.render(this.scene);
  }

  // =====================================================================================
  // Playback

  private get hasAudio(): boolean {
    return !!this.scene.audio && !Number.isNaN(this.dom.audio.duration);
  }

  get frame(): number {
    return this.hasAudio ? this.dom.audio.currentTime * MMD_FPS : this.clockFrame;
  }

  get length(): number {
    const audioFrames = this.hasAudio ? this.dom.audio.duration * MMD_FPS : 0;
    return Math.max(this.scene.lastFrame, audioFrames);
  }

  setFrame(f: number): void {
    const frame = Math.max(0, f);
    this.clockFrame = frame;
    if (this.scene.audio) this.dom.audio.currentTime = frame / MMD_FPS;
  }

  togglePlay(): void {
    this.playing = !this.playing;
    // Starting at the end (or outside the range) restarts from the beginning.
    if (this.playing) {
      const [a, b] = this.range ?? [0, this.length];
      if (this.frame >= b - 0.5 || this.frame < a) this.setFrame(a);
    }
    if (this.hasAudio) {
      if (this.playing) void this.dom.audio.play();
      else this.dom.audio.pause();
    }
  }

  private pause(): void {
    if (!this.playing) return;
    this.playing = false;
    this.dom.audio.pause();
  }

  step(delta: number): void {
    this.pause();
    this.setFrame(Math.round(this.frame) + delta);
  }

  jumpKey(dir: 1 | -1): void {
    this.pause();
    const f = Math.round(this.frame);
    const keys = this.keyFrames;
    let target: number | undefined;
    if (dir > 0) target = keys.find((k) => k > f);
    else for (let i = keys.length - 1; i >= 0; i--) if (keys[i] < f) { target = keys[i]; break; }
    if (target !== undefined) this.setFrame(target);
  }

  setRange(range: [number, number] | null): void {
    this.range = range && range[1] > range[0] ? [Math.round(range[0]), Math.round(range[1])] : null;
  }

  setRangeEdge(edge: 'in' | 'out'): void {
    const f = Math.round(this.frame);
    const len = Math.max(Math.round(this.length), f + 1);
    const [a, b] = this.range ?? [0, len];
    if (edge === 'in') this.setRange([f, f < b ? b : len]);
    else this.setRange([f > a ? a : 0, f]);
  }

  private readonly tick = (now: number): void => {
    const dt = Math.min((now - this.lastNow) / 1000, 0.1);
    this.lastNow = now;
    if (this.exporting) {
      requestAnimationFrame(this.tick);
      return;
    }
    this.fps = this.fps * 0.95 + (dt > 0 ? 1 / dt : 0) * 0.05;

    if (this.playing) {
      if (!this.hasAudio) this.clockFrame += dt * MMD_FPS;
      const [a, b] = this.range ?? [0, this.length];
      const ended = this.frame >= b || (this.hasAudio && !this.range && this.dom.audio.ended);
      if (ended && b > 0) {
        if (this.loop) {
          this.setFrame(a);
          if (this.hasAudio && this.dom.audio.paused) void this.dom.audio.play();
        } else {
          this.setFrame(b);
          this.pause();
        }
      }
    }

    try {
      const frame = this.frame;
      // Loops / seeks (not normal playback): snap rigid bodies to the new pose.
      const jumped = frame < this.lastFrame - 0.5 || frame > this.lastFrame + 15;
      this.lastFrame = frame;

      const c0 = performance.now();
      this.renderFrame(frame, dt, jumped);
      this.cpuMs = this.cpuMs * 0.9 + (performance.now() - c0) * 0.1;
      this.gizmo.draw(this.camera.view);
      this.timeline.update({
        frame,
        length: this.length,
        playing: this.playing,
        range: this.range,
        loop: this.loop,
        tracks: this.tracks,
        waveform: this.waveform,
      });
      if (now - this.infoTimer > 250) {
        this.infoTimer = now;
        this.renderInfo();
      }
    } catch (err) {
      console.error(err);
      toast(err instanceof Error ? err.message : String(err), 'error');
      this.pause();
    }
    requestAnimationFrame(this.tick);
  };

  /** Pose every visible model at `frame`, advance physics by `dt` seconds and draw. */
  private renderFrame(frame: number, dt: number, resetPhysics = false): void {
    const visible = this.scene.models.filter((m) => m.visible);
    for (const m of visible) {
      if (resetPhysics) m.model.physics?.requestReset();
      m.model.update(frame, dt);
    }
    const camTrack = this.scene.cameraMotion?.cameraTrack;
    if (camTrack && this.useCameraMotion) {
      camTrack.sample(frame, this.cameraPose);
      this.cameraPose.target[1] = this.cameraPose.target[1] * this.cameraFit.scale + this.cameraFit.offsetY;
      this.cameraPose.distance *= this.cameraFit.scale;
      this.camera.pose = this.cameraPose;
    } else {
      this.camera.pose = null;
    }
    this.applySceneMotion(frame);
    this.renderer.time = frame / MMD_FPS;
    this.updateWireframe();
    this.renderer.render(this.camera, visible.map((m) => m.gpu));
  }

  /** The main light, and the self-shadow keys of the scene motion (MMD saves them with the camera). */
  private applySceneMotion(frame: number): void {
    const r = this.renderer;
    const motion = this.scene.cameraMotion;
    const main = r.lighting.main;
    if (!(main.followMotion && motion?.sampleLight(frame, r.light))) {
      r.light.direction = lightDirection(main.yaw, main.pitch);
      r.light.color = [...main.color];
    }
    for (let c = 0; c < 3; c++) r.light.color[c] *= main.intensity;
    const shadow = motion?.selfShadowAt(frame) ?? null;
    r.selfShadow = this.dom.controls.selfShadow.checked && shadow?.mode !== 0;
    r.shadowRange = shadow ? shadow.range : null;
  }

  private updateWireframe(): void {
    const debug = this.renderer.physicsDebug;
    const sel = this.scene.selected;
    for (const m of this.scene.models) if (m.model.physics) m.model.physics.wantDebug = debug.visible && m === sel;
    if (!debug.visible || !sel?.rig) return;
    const live = sel.model.physics?.enabled ? sel.model.physics.bodyTransforms : null;
    debug.updateBodies(live ?? bodyTransformsFromBones(sel.rig, sel.model.skeleton, this.wireframeScratch));
  }

  private lastInfo = '';
  private renderInfo(): void {
    const lines: string[] = [];
    const sel = this.scene.selected;
    const p = sel?.model.physics;
    if (sel?.rig) {
      lines.push(
        this.physicsMode === 'off' ? `物理: 关闭`
          : p?.kind === 'baked' && sel.bake ? `物理: 烘焙结果（第 ${sel.bake.from}–${sel.bake.to} 帧）`
          : !p ? '物理: 加载中…'
          : `物理: ${p.kind === 'worker' ? 'Worker' : '主线程'} · 步进 ${p.stepMs.toFixed(2)} ms` + (p.kind === 'worker' ? ` · 新结果 ${(p.freshRatio * 100).toFixed(0)}%` : ''),
      );
    }
    lines.push(`共享 toon: ${this.customToons ? `MMD Data ${this.customToons}/10` : '内置'}`);
    lines.push(`${this.fps.toFixed(0)} fps · 主线程 ${this.cpuMs.toFixed(2)} ms`);
    const gpu = this.renderer.timing?.times;
    if (gpu?.size) {
      const names: Record<string, string> = { skinning: '蒙皮', prepass: '深度预绘', shadow: '阴影', main: '绘制', dof: '景深', effect: '自定义后期', bloom: '泛光', composite: '合成' };
      const parts = [...gpu].map(([k, ms]) => `${names[k] ?? k} ${ms.toFixed(2)}`);
      const total = [...gpu.values()].reduce((a, b) => a + b, 0);
      lines.push(`GPU ${total.toFixed(2)} ms（${parts.join(' · ')}）`);
    }
    const [sw, sh] = this.renderer.sceneSize;
    lines.push(`渲染 ${sw}×${sh}`);
    const text = lines.join('\n');
    if (text !== this.lastInfo) this.dom.info.textContent = this.lastInfo = text;
  }

  // =====================================================================================
  // Physics

  async setPhysicsMode(mode: PhysicsMode): Promise<void> {
    this.physicsMode = mode;
    await Promise.all(this.scene.models.map((m) => this.setupPhysics(m)));
  }

  /** (Re)create one model's physics runner for the current mode. */
  private async setupPhysics(m: SceneModel): Promise<void> {
    const token = (this.physicsTokens.get(m.id) ?? 0) + 1;
    this.physicsTokens.set(m.id, token);
    if (!m.rig) return;
    if (this.physicsMode === 'off') {
      m.model.attachPhysics(null);
      return;
    }
    if (m.bake && m.useBake) {
      const current = m.model.physics;
      if (!(current instanceof BakedPhysics && current.bake === m.bake)) {
        const baked = new BakedPhysics(m.bake, m.rig, m.pmx);
        m.model.attachPhysics(baked);
        m.gpu.setBakedCloth(baked.clothMesh?.vertices ?? null);
      }
      return;
    }
    const kind = this.physicsMode;
    if (m.model.physics?.kind === kind) {
      m.model.physics.enabled = true;
      m.model.physics.requestReset(30);
      return;
    }
    m.model.attachPhysics(null);
    try {
      const runner = await createPhysicsRunner(kind, m.pmx, m.rig);
      if (this.physicsTokens.get(m.id) !== token || !this.scene.models.includes(m)) {
        runner.dispose(); // model removed or mode changed while loading
        return;
      }
      runner.requestReset(60); // settle hair / skirts into the current pose first
      m.model.attachPhysics(runner);
    } catch (err) {
      toast(`物理初始化失败：${err instanceof Error ? err.message : err}`, 'error');
    }
  }

  /** After a model's motion changed: a bake made for the old motion is gone. */
  private motionChanged(m: SceneModel, droppedBake: boolean): void {
    if (droppedBake) {
      toast(`「${m.name}」的动作已改变，物理烘焙已删除（需要时请重新烘焙）`, 'warn');
      void this.setupPhysics(m);
    }
    this.scene.changed();
  }

  /** Simulate a model's physics over a frame range in a worker and switch it to the bake. */
  async bakeModel(id: number): Promise<void> {
    const m = this.scene.models.find((x) => x.id === id);
    if (!m?.rig) return;
    const choice = await bakeDialog(m.name, this.length, this.range, suggestClothMaterials(m.pmx, m.rig));
    if (!choice) return;
    const motionAtStart = m.motion;
    const total = choice.to - choice.from + 1;
    const t0 = performance.now();
    let job: ReturnType<typeof bakeInWorker> | null = null;
    const p = progress(`烘焙物理「${m.name}」… 0/${total}`, () => job?.cancel());
    try {
      job = bakeInWorker(m.pmx, m.motion, { ...choice, motionScale: m.motionScale }, (done) => {
        const eta = done > 0 ? ((performance.now() - t0) / done) * (total - done) / 1000 : 0;
        p.update(`烘焙物理「${m.name}」… ${done}/${total} 帧（约剩 ${eta.toFixed(0)} 秒）`);
      });
      const bake = await job.promise;
      if (!this.scene.models.includes(m) || m.motion !== motionAtStart) {
        toast('烘焙期间模型或动作已改变，结果已丢弃', 'warn');
        return;
      }
      m.bake = bake;
      m.useBake = true;
      await this.setupPhysics(m);
      this.scene.changed();
      const clothNote = bake.cloth ? `，布料 ${bake.cloth.particleCount} 个质点` : '';
      toast(`物理烘焙完成：第 ${bake.from}–${bake.to} 帧，${bake.bones.length} 根骨骼${clothNote}，用时 ${((performance.now() - t0) / 1000).toFixed(1)} 秒`);
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') toast('已取消物理烘焙');
      else toast(`物理烘焙失败：${err instanceof Error ? err.message : err}`, 'error');
    } finally {
      p.done();
    }
  }

  resetPhysics(): void {
    for (const m of this.scene.models) m.model.physics?.requestReset(90);
  }

  // =====================================================================================
  // Import

  async importModelDialog(directory: boolean): Promise<void> {
    const files = await pickFiles(directory ? { directory: true } : { accept: '.pmx,.pmd,.x,.zip', multiple: true });
    if (files.length) await this.importFiles(FileSet.fromFileList(files));
  }

  async importMotionDialog(target: 'model' | 'camera'): Promise<void> {
    const files = await pickFiles({ accept: '.vmd', multiple: true });
    for (const f of files) await this.guard(() => this.importVmd(f.name, f, target));
  }

  async importAudioDialog(): Promise<void> {
    const [file] = await pickFiles({ accept: 'audio/*,.mp3,.wav,.ogg,.m4a,.flac' });
    if (file) this.scene.setAudio(file.name, file);
  }

  async importBackgroundDialog(): Promise<void> {
    const [file] = await pickFiles({ accept: 'image/jpeg,image/png,image/webp' });
    if (file) this.setBackground(file.name, file);
  }

  /** A panorama (equirectangular, 2:1) image behind the scene. */
  setBackground(name: string, blob: Blob): void {
    this.scene.background = { name, blob };
    this.scene.changed();
  }

  // =====================================================================================
  // Custom shaders

  async importShaderDialog(kind: ShaderKind): Promise<void> {
    const [file] = await pickFiles({ accept: '.wgsl,.txt' });
    if (file) await this.guard(async () => this.importShader(file.name, await file.text(), kind));
  }

  /** One of the examples shipped in /shaders. */
  async loadExampleShader(file: string): Promise<void> {
    await this.guard(async () => {
      const res = await fetch(`${import.meta.env.BASE_URL}shaders/${file}`);
      if (!res.ok) throw new Error(`无法加载示例 ${file}`);
      await this.importShader(file, await res.text());
    });
  }

  /**
   * Apply a WGSL file: a material shader (defines `shade`) goes to the current model, a
   * post-processing shader (defines `effect`) to the whole picture.
   */
  async importShader(name: string, code: string, expected?: ShaderKind): Promise<void> {
    const kind = detectShaderKind(code);
    if (!kind) {
      toast(`${name}：没有找到 fn shade(…)（材质 shader）或 fn effect(…)（后期 shader）`, 'error');
      return;
    }
    if (expected && kind !== expected) {
      toast(`${name} 是${kind === 'post' ? '后期' : '材质'} shader，已按${kind === 'post' ? '后期' : '材质'} shader 加载`);
    }
    if (kind === 'post') await this.setPostShader({ name, code });
    else if (this.scene.selected) await this.setModelShader(this.scene.selected, { name, code });
    else toast('材质 shader 作用在模型上：先导入并选中一个模型', 'warn');
  }

  /** Shade a model with a custom material shader (null = MMD's shading). Returns false on errors. */
  async setModelShader(m: SceneModel, shader: CustomShader | null): Promise<boolean> {
    let pipelines: MaterialPipelines | null = null;
    if (shader) {
      pipelines = this.materialShaders.get(shader.code) ?? null;
      if (!pipelines) {
        const result = await this.renderer.core.createMaterialPipelines(shader.name, shader.code);
        if ('errors' in result) {
          await warningsDialog(`材质 shader「${shader.name}」有错误，仍使用原来的着色`, formatShaderErrors(result.errors));
          return false;
        }
        pipelines = result.pipelines;
        this.materialShaders.set(shader.code, pipelines);
      }
    }
    if (!this.scene.models.includes(m)) return false;
    m.shader = shader;
    m.gpu.setShader(pipelines);
    this.scene.changed();
    if (shader) toast(`「${m.name}」使用材质 shader：${shader.name}`);
    // A shader that reads auxiliary maps gets generated ones if the model has none yet.
    if (shader && !m.hasAux && /\b(sampleRamp|lightMap|faceMap|thinness)\b/.test(shader.code)) await this.guard(() => this.generateAuxMaps(m));
    return true;
  }

  // =====================================================================================
  // Auxiliary maps for material shaders

  private async installGeneratedAux(m: SceneModel, slot: AuxSlot, textureIndex: number, image: RgbaImage, name: string): Promise<void> {
    const blob = await imageToPng(image);
    if (!this.scene.models.includes(m)) return;
    const texture = this.renderer.core.textures.fromRgba(image.width, image.height, image.data, false, name);
    m.setAux(slot, textureIndex, { name, generated: true, blob, texture });
  }

  /**
   * Generate every auxiliary map from the model itself: colour ramp, ambient occlusion light
   * maps and the face shadow map. Maps loaded from files are kept.
   */
  async generateAuxMaps(m: SceneModel): Promise<void> {
    const p = progress(`生成辅助贴图「${m.name}」…`);
    try {
      if (!m.aux.ramp || m.aux.ramp.generated) await this.installGeneratedAux(m, 'ramp', -1, generateRamp(), '色条（生成）');
      const occlusion = await bakeVertexOcclusion(m.pmx, (msg) => p.update(msg));
      for (const [t, image] of occlusionToLightMaps(m.pmx, occlusion)) {
        if (m.aux.light.get(t)?.generated === false) continue;
        await this.installGeneratedAux(m, 'light', t, image, '环境遮蔽（生成）');
      }
      for (const [t, image] of await bakeFaceMaps(m.pmx, (msg) => p.update(msg))) {
        if (m.aux.face.get(t)?.generated === false) continue;
        await this.installGeneratedAux(m, 'face', t, image, '脸部阴影（生成）');
      }
      this.scene.changed();
    } finally {
      p.done();
    }
  }

  /** The dialog listing a model's auxiliary maps. */
  async editAuxMaps(m: SceneModel): Promise<void> {
    const pmx = m.pmx;
    const textures = [...new Set(pmx.materials.map((x) => x.textureIndex).filter((t) => t >= 0 && t < pmx.textures.length))];
    const faceTextures = new Set(pmx.materials
      .filter((x) => [MaterialPart.Face, MaterialPart.FaceDetail].includes(classifyMaterial(x.name, x.nameEn) as 1 | 2))
      .map((x) => x.textureIndex));
    const rows = [
      { slot: 'ramp' as const, texture: -1, label: '色条', hint: '亮部到暗部换成什么颜色。每行一种材质：布料、皮肤、头发、五官' },
      ...textures.map((t) => ({ slot: 'light' as const, texture: t, label: `光照图 · ${basename(pmx.textures[t])}`, hint: '与这张贴图同样布局的图。自动生成的只有红色通道：环境遮蔽（衣褶、缝隙处偏暗）' })),
      ...textures.filter((t) => faceTextures.has(t)).map((t) => ({ slot: 'face' as const, texture: t, label: `脸部阴影图 · ${basename(pmx.textures[t])}`, hint: '红 / 绿通道：光向右 / 向左转多少之后这个位置变暗' })),
    ];
    await auxMapsDialog(m.name, rows, {
      status: (row) => (row.slot === 'ramp' ? m.aux.ramp : m.aux[row.slot].get(row.texture))?.name ?? null,
      generate: () => this.guard(() => this.generateAuxMaps(m)),
      pick: async (row) => {
        const [file] = await pickFiles({ accept: 'image/png,image/jpeg,image/webp,image/bmp' });
        if (!file) return;
        await this.guard(async () => {
          const texture = await this.renderer.core.textures.fromImageBlob(file, false, file.name);
          m.setAux(row.slot, row.texture, { name: file.name, generated: false, blob: file, texture });
        });
      },
      clear: (row) => m.clearAux(row.slot, row.texture),
    });
    this.scene.changed();
  }

  /** Run a custom shader over the whole picture (null = none). Returns false on errors. */
  async setPostShader(shader: CustomShader | null): Promise<boolean> {
    const errors = await this.renderer.post.setEffect(shader);
    if (errors.length) {
      await warningsDialog(`后期 shader「${shader!.name}」有错误，未启用`, formatShaderErrors(errors));
      return false;
    }
    this.scene.postShader = shader;
    this.scene.changed();
    if (shader) toast(`后期 shader：${shader.name}`);
    return true;
  }

  async openProjectDialog(): Promise<void> {
    const [file] = await pickFiles({ accept: '.zip' });
    if (!file) return;
    await this.guard(async () => {
      const files = await FileSet.fromZip(file);
      const project = await readProjectFile(files);
      if (!project) throw new Error(`${file.name} 不是工程文件`);
      await this.openProject(files, project);
    });
  }

  /** Run an action, reporting failures as a toast. */
  async guard(fn: () => Promise<void>): Promise<void> {
    try {
      await fn();
    } catch (err) {
      console.error(err);
      toast(err instanceof Error ? err.message : String(err), 'error');
    }
  }

  /**
   * Import whatever was dropped / picked: a project archive, model folders / files / zips,
   * VMDs (body → current model, camera → scene), audio, or MMD's shared toon files.
   */
  async importFiles(files: FileSet): Promise<void> {
    await this.guard(async () => {
      // A project archive replaces the scene.
      for (const z of files.filterByExtension('zip')) {
        const inner = await FileSet.fromZip(await z.load());
        const project = await readProjectFile(inner);
        if (project) {
          await this.openProject(inner, project);
          return;
        }
      }
      await files.expandZips();

      let modelPaths = files.filterByExtension(...MODEL_EXTS).map((e) => e.path);
      if (modelPaths.length > 1) modelPaths = (await pickModelsDialog(modelPaths)) ?? [];
      const loaded: SceneModel[] = [];
      for (const path of modelPaths) {
        const m = await this.addModel(files, path, await (await files.get(path)!.load()).arrayBuffer());
        if (m) loaded.push(m);
      }

      // Motions go to the model imported together with them, else to the current model.
      const target = loaded[loaded.length - 1] ?? this.scene.selected;
      for (const e of files.filterByExtension('vmd')) {
        await this.importVmd(basename(e.path), await e.load(), 'auto', target);
      }

      const audio = files.filterByExtension(...AUDIO_EXTS)[0];
      if (audio) this.scene.setAudio(basename(audio.path), await audio.load());

      const shaders = files.filterByExtension('wgsl');
      for (const e of shaders) await this.importShader(basename(e.path), await (await e.load()).text());
      if (shaders.length && !modelPaths.length) return;

      if (!modelPaths.length && !files.filterByExtension('vmd').length && !audio) {
        // An image on its own is a panorama background (next to a model it would be a texture).
        const images = files.filterByExtension(...IMAGE_EXTS);
        if (images.length === 1 && files.entries.length === 1) {
          this.setBackground(basename(images[0].path), await images[0].load());
          return;
        }
        const toons = new Map<number, Blob>();
        for (const e of files.entries) {
          const i = TextureManager.sharedToonIndex(basename(e.path));
          if (i >= 0) toons.set(i, await e.load());
        }
        if (toons.size > 0) {
          const n = await this.setSharedToons(toons);
          await saveSharedToons(toons);
          toast(`已导入 ${n} 张 MMD 共享 toon 贴图`);
        } else if (files.entries.length) {
          toast('没有可导入的文件（支持 PMX / PMD / X、VMD、音频、全景图、zip、工程文件）', 'warn');
        }
      }
    });
  }

  private async addModel(source: AssetSource, path: string, bytes: ArrayBuffer): Promise<SceneModel | null> {
    const p = progress(`加载模型 ${basename(path)}…`);
    try {
      const wasEmpty = this.scene.models.length === 0;
      const m = await loadSceneModel(this.renderer.core, source, path, bytes);
      this.scene.add(m);
      if (wasEmpty) this.frameModel(m);
      void this.setupPhysics(m);
      if (m.missingTextures.length) toast(`${m.name}：缺少 ${m.missingTextures.length} 张贴图`, 'warn');
      return m;
    } finally {
      p.done();
    }
  }

  private frameModel(m: SceneModel): void {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    const pos = m.pmx.positions;
    for (let i = 0; i < pos.length; i++) {
      const c = i % 3;
      if (pos[i] < min[c]) min[c] = pos[i];
      if (pos[i] > max[c]) max[c] = pos[i];
    }
    if (pos.length) this.camera.frame(min, max);
  }

  /**
   * Import a VMD. 'auto' splits it: bones / morphs → `target` (default: current model),
   * camera / light / shadow → the scene. 'model' / 'camera' take only that part.
   */
  async importVmd(name: string, data: Blob, mode: 'auto' | 'model' | 'camera', target = this.scene.selected): Promise<void> {
    const vmd = parseVmd(await data.arrayBuffer());
    const { model, scene } = splitVmd(vmd);
    const hasModelPart = model.boneFrames.length + model.morphFrames.length + model.propertyFrames.length > 0;
    const hasScenePart = scene.cameraFrames.length + scene.lightFrames.length + scene.selfShadowFrames.length > 0;

    if (mode !== 'camera' && hasModelPart) {
      if (target) {
        this.motionChanged(target, target.addMotion(name, model));
        if (target.binding && target.binding.bones === 0 && target.binding.usedBones > 0) {
          toast(`${name}：骨骼名与「${target.name}」都不匹配`, 'warn');
        }
      } else {
        toast(`${name}：场景里没有模型，动作部分未应用`, 'warn');
      }
    }
    if (mode !== 'model' && hasScenePart) {
      this.scene.addCameraMotion(name, scene);
      if (scene.cameraFrames.length) this.setUseCameraMotion(true);
    }
    if (mode === 'camera' && !hasScenePart) toast(`${name} 不含镜头关键帧`, 'warn');
    if (mode === 'model' && !hasModelPart) toast(`${name} 不含骨骼 / 表情关键帧`, 'warn');
  }

  /** ?model=…&motion=…&audio=… (URLs) for hosted assets and quick testing. */
  async importFromQuery(params: URLSearchParams): Promise<void> {
    await this.guard(async () => {
      const modelUrl = params.get('model');
      if (modelUrl) {
        const res = await fetch(modelUrl);
        if (!res.ok) throw new Error(`无法加载 ${modelUrl}: HTTP ${res.status}`);
        // Keep the origin so textures of cross-origin models resolve against the model's host.
        await this.addModel(new UrlAssetSource(), decodeURIComponent(new URL(modelUrl, location.href).href), await res.arrayBuffer());
      }
      for (const url of params.getAll('motion')) {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`无法加载 ${url}: HTTP ${res.status}`);
        await this.importVmd(decodeURIComponent(basename(url)), await res.blob(), 'auto');
      }
      const audioUrl = params.get('audio');
      if (audioUrl) {
        const res = await fetch(audioUrl);
        if (res.ok) this.scene.setAudio(decodeURIComponent(basename(audioUrl)), await res.blob());
      }
    });
  }

  // =====================================================================================
  // Export

  private exportRange(useRange: boolean, rebase: boolean): ExportRange | undefined {
    return useRange && this.range ? { from: this.range[0], to: this.range[1], rebase } : undefined;
  }

  async exportModel(): Promise<void> {
    const m = this.scene.selected;
    if (!m) return;
    await this.guard(async () => {
      const p = progress(`打包 ${m.name}…`);
      try {
        const name = modelArchiveName(m);
        const collected = await collectModelFiles(m, name, true);
        const blob = await writeZip(collected.entries, (d, t) => p.update(`打包 ${m.name}… ${d}/${t}`));
        p.done();
        if (await saveBlob(blob, `${name}.zip`)) {
          toast(`已导出 ${name}.zip（${collected.entries.length} 个文件${m.pmx.format !== 'PMX' ? `，${m.pmx.format} 已转换为 PMX` : ''}）`);
        }
        if (collected.warnings.length) await warningsDialog('导出模型：注意', collected.warnings);
      } finally {
        p.done();
      }
    });
  }

  async exportMotion(): Promise<void> {
    const m = this.scene.selected;
    if (!m?.motion && !m?.bake) return;
    const motion = m.motion ?? new MmdMotion();
    const summary = `「${m.name}」的动作：${motion.boneTracks.size} 个骨骼轨道、${motion.morphTracks.size} 个表情轨道`;
    const choice = await exportMotionDialog('导出动作 VMD', summary, this.range, m.bake);
    if (!choice) return;
    await this.guard(async () => {
      const range = this.exportRange(choice.useRange, choice.rebase);
      const vmd = motionToVmd(motion, m.pmx.name, { model: true, scene: false }, range);
      if (choice.includeBake && m.bake) {
        // Baked physics bones replace any keys those bones had.
        const baked = bakeToVmdBoneFrames(m.bake, m.pmx, { range, tolerance: choice.bakeTolerance });
        const names = new Set(baked.map((k) => k.boneName));
        vmd.boneFrames = [...vmd.boneFrames.filter((k) => !names.has(k.boneName)), ...baked];
      }
      const { buffer, warnings } = writeVmd(vmd);
      const name = `${modelArchiveName(m)}${choice.useRange ? `_${this.range![0]}-${this.range![1]}` : ''}.vmd`;
      if (await saveBlob(new Blob([buffer]), name)) toast(`已导出 ${name}（${vmd.boneFrames.length} 骨骼帧、${vmd.morphFrames.length} 表情帧）`);
      if (warnings.length) await warningsDialog('导出动作：注意', warnings);
    });
  }

  async exportCamera(): Promise<void> {
    const cam = this.scene.cameraMotion;
    if (!cam) return;
    const summary = `${cam.cameraTrack?.frames.length ?? 0} 个镜头关键帧、${cam.lightFrames.length} 个灯光关键帧`;
    const choice = await exportMotionDialog('导出镜头 VMD', summary, this.range);
    if (!choice) return;
    await this.guard(async () => {
      const vmd = motionToVmd(cam, CAMERA_VMD_MODEL_NAME, { model: false, scene: true }, this.exportRange(choice.useRange, choice.rebase));
      const name = `camera${choice.useRange ? `_${this.range![0]}-${this.range![1]}` : ''}.vmd`;
      if (await saveBlob(new Blob([writeVmd(vmd).buffer]), name)) toast(`已导出 ${name}（${vmd.cameraFrames.length} 镜头帧）`);
    });
  }

  /** Render the timeline frame by frame (not in real time) into a video or a PNG sequence. */
  async exportVideo(): Promise<void> {
    if (this.exporting) return;
    const length = Math.round(this.length);
    if (length <= 0) {
      toast('没有可导出的内容：先导入动作、镜头或音频', 'warn');
      return;
    }
    const canvas = this.dom.canvas;
    const choice = await exportVideoDialog({
      window: [canvas.width, canvas.height], length, range: this.range,
      hasAudio: !!this.scene.audio, videoSupported: typeof VideoEncoder !== 'undefined',
    });
    if (!choice) return;

    await this.guard(async () => {
      const [from, to] = choice.useRange && this.range ? this.range : [0, length];
      const r = this.renderer;
      const saved = {
        frame: this.frame, mode: this.physicsMode, fixedSize: r.fixedSize, mask: r.physicsDebug.mask,
        lensScale: this.camera.lensScale, lensShift: [...this.camera.lensShift],
      };
      let cancelled = false;
      const p = progress('准备导出…', () => (cancelled = true));
      this.pause();
      this.exporting = true;
      document.body.classList.add('exporting');
      try {
        const audio = choice.audio && choice.format === 'video' && this.scene.audio
          ? await new OfflineAudioContext(2, 1, 48000).decodeAudioData(await this.scene.audio.blob.arrayBuffer())
          : null;
        // Live physics must step in lockstep with the exported frames: run it on this thread.
        // (Baked physics is unaffected and gives the same result every time.)
        if (saved.mode === 'worker') await this.setPhysicsMode('main');
        r.fixedSize = [choice.width & ~1, choice.height & ~1];
        r.physicsDebug.mask = 0;
        this.camera.lensScale = 1;
        this.camera.lensShift[0] = this.camera.lensShift[1] = 0;
        // Let hair and skirts settle into the first frame's pose.
        for (const m of this.scene.models) m.model.physics?.requestReset(90);
        this.renderFrame(from, 1 / choice.fps);

        // The encoder library is only loaded when something is exported.
        const { exportFrames } = await import('../io/video-export');
        const t0 = performance.now();
        const result = await exportFrames(canvas, (frame, dt) => this.renderFrame(frame, dt), {
          format: choice.format, from, to, fps: choice.fps, audio,
          cancelled: () => cancelled,
          onProgress: (done, total) => {
            const eta = ((performance.now() - t0) / done) * (total - done) / 1000;
            p.update(`导出 ${done}/${total} 帧（约剩 ${eta.toFixed(0)} 秒）`);
          },
        });
        p.done();
        if (!result) {
          toast('已取消导出', 'warn');
          return;
        }
        const name = `${projectFileName(this.scene).replace(/\.mmdproj\.zip$/, '')}_${r.fixedSize[0]}x${r.fixedSize[1]}.${result.extension}`;
        if (await saveBlob(result.blob, name)) toast(`已导出 ${name}（${(result.blob.size / 1048576).toFixed(1)} MB）`);
        if (result.warnings.length) await warningsDialog('导出视频：注意', result.warnings);
      } finally {
        p.done();
        r.fixedSize = saved.fixedSize;
        r.physicsDebug.mask = saved.mask;
        this.camera.lensScale = saved.lensScale;
        this.camera.lensShift[0] = saved.lensShift[0];
        this.camera.lensShift[1] = saved.lensShift[1];
        if (this.physicsMode !== saved.mode) await this.setPhysicsMode(saved.mode);
        this.setFrame(saved.frame);
        document.body.classList.remove('exporting');
        this.exporting = false;
      }
    });
  }

  // =====================================================================================
  // Projects

  projectSettings(): ProjectSettings {
    const c = this.dom.controls;
    return {
      playback: { frame: Math.round(this.frame), range: this.range, loop: this.loop },
      view: {
        target: [this.camera.target[0], this.camera.target[1], this.camera.target[2]],
        distance: this.camera.distance, yaw: this.camera.yaw, pitch: this.camera.pitch,
        useCameraMotion: this.useCameraMotion,
        cameraFit: { ...this.cameraFit },
      },
      render: {
        edges: c.edges.checked, selfShadow: c.selfShadow.checked, darkBackground: c.darkBackground.checked,
        physics: this.physicsMode, wireframe: this.renderer.physicsDebug.mask,
        groundShadow: this.renderer.groundShadow, renderScale: this.renderer.renderScale,
        clothCollision: this.renderer.clothCollision,
        post: { ...this.renderer.post.settings },
      },
      lighting: this.renderer.lighting,
    };
  }

  async saveProject(): Promise<void> {
    if (this.scene.isEmpty) {
      toast('场景是空的', 'warn');
      return;
    }
    await this.guard(async () => {
      const p = progress('保存工程…');
      try {
        const { blob, warnings } = await buildProject(this.scene, this.projectSettings(), (m) => p.update(m));
        p.done();
        const name = projectFileName(this.scene);
        if (await saveBlob(blob, name)) toast(`已保存工程 ${name}（${(blob.size / 1048576).toFixed(1)} MB）`);
        if (warnings.length) await warningsDialog('保存工程：注意', warnings);
      } finally {
        p.done();
      }
    });
  }

  async newProject(): Promise<void> {
    if (!this.scene.isEmpty && !(await confirmDialog('新建工程', '清空当前场景（模型、动作、镜头、音频）？未保存的内容会丢失。', '清空'))) return;
    this.pause();
    this.scene.clear();
    await this.renderer.post.setEffect(null);
    this.range = null;
    this.setFrame(0);
  }

  async openProject(files: FileSet, project: ProjectFile): Promise<void> {
    if (!this.scene.isEmpty && !(await confirmDialog('打开工程', '打开工程会替换当前场景。继续？', '打开'))) return;
    this.pause();
    this.scene.clear();
    const p = progress('打开工程…');
    try {
      let selected: SceneModel | null = null;
      for (const [i, entry] of project.models.entries()) {
        p.update(`加载模型 ${i + 1}/${project.models.length}：${entry.name}`);
        const file = files.get(entry.file);
        if (!file) {
          toast(`工程缺少文件 ${entry.file}`, 'error');
          continue;
        }
        const m = await loadSceneModel(this.renderer.core, files, entry.file, await (await file.load()).arrayBuffer());
        m.visible = entry.visible;
        if (entry.transform) m.setTransform(entry.transform.position, entry.transform.rotationY);
        m.model.motionScale = entry.motionScale ?? 1;
        if (entry.hiddenMaterials) m.setHiddenMaterials(entry.hiddenMaterials);
        if (entry.motion) {
          const vmdFile = files.get(entry.motion);
          if (vmdFile) m.setMotion(MmdMotion.fromVmd(parseVmd(await (await vmdFile.load()).arrayBuffer())), entry.motionNames);
        }
        if (entry.bake) {
          const bakeFile = files.get(entry.bake.file);
          if (bakeFile) {
            const clothFile = entry.bake.clothFile ? files.get(entry.bake.clothFile) : null;
            m.bake = deserializeBake(entry.bake, await (await bakeFile.load()).arrayBuffer(), clothFile ? await (await clothFile.load()).arrayBuffer() : null);
            m.useBake = entry.bake.use;
          }
        }
        this.scene.add(m, false);
        void this.setupPhysics(m);
        for (const a of entry.aux ?? []) {
          const f = files.get(a.file);
          if (!f) continue;
          const blob = await f.load();
          const texture = await this.renderer.core.textures.fromImageBlob(blob, false, a.name);
          m.setAux(a.slot, a.texture, { name: a.name, generated: a.generated, blob, texture });
        }
        const shaderFile = entry.shader ? files.get(entry.shader.file) : null;
        if (shaderFile) await this.setModelShader(m, { name: entry.shader!.name, code: await (await shaderFile.load()).text() });
        if (i === project.selected) selected = m;
      }
      if (selected) this.scene.select(selected.id);

      if (project.camera) {
        const f = files.get(project.camera.file);
        if (f) {
          this.scene.cameraMotion = MmdMotion.fromVmd(parseVmd(await (await f.load()).arrayBuffer()));
          this.scene.cameraNames = project.camera.names;
        }
      }
      if (project.audio) {
        const f = files.get(project.audio.file);
        if (f) this.scene.setAudio(project.audio.name, await f.load());
      }
      const postFile = project.postShader ? files.get(project.postShader.file) : null;
      await this.setPostShader(postFile ? { name: project.postShader!.name, code: await (await postFile.load()).text() } : null);
      if (project.background) {
        const f = files.get(project.background.file);
        if (f) this.scene.background = { name: project.background.name, blob: await f.load() };
      }
      this.applySettings(project);
      this.scene.changed();
      toast(`已打开工程（${project.models.length} 个模型）`);
    } finally {
      p.done();
    }
  }

  private applySettings(s: ProjectSettings): void {
    const c = this.dom.controls;
    c.edges.checked = s.render.edges;
    c.selfShadow.checked = s.render.selfShadow;
    c.darkBackground.checked = s.render.darkBackground;
    for (const input of [c.edges, c.selfShadow, c.darkBackground]) input.dispatchEvent(new Event('change'));
    c.physics.value = s.render.physics;
    void this.setPhysicsMode(s.render.physics);
    c.wireframe.forEach((box, kind) => (box.checked = (s.render.wireframe & (1 << kind)) !== 0));
    this.renderer.physicsDebug.mask = s.render.wireframe;
    this.renderer.groundShadow = s.render.groundShadow ?? true;
    this.renderer.clothCollision = s.render.clothCollision ?? true;
    this.renderer.renderScale = s.render.renderScale ?? 1;
    this.renderer.post.settings = { ...DEFAULT_POST, ...s.render.post };
    assignLighting(this.renderer.lighting, s.lighting ?? defaultLighting());
    this.dom.syncControls();

    this.range = s.playback.range;
    this.loop = s.playback.loop;
    this.setFrame(s.playback.frame);

    vec3.set(this.camera.target, ...s.view.target);
    this.camera.distance = s.view.distance;
    this.camera.yaw = s.view.yaw;
    this.camera.pitch = s.view.pitch;
    this.useCameraMotion = s.view.useCameraMotion;
    this.cameraFit = { scale: 1, offsetY: 0, ...s.view.cameraFit };
    this.panel.cameraFit = this.cameraFit;
    this.panel.useCameraMotion = s.view.useCameraMotion;
  }

  // =====================================================================================
  // Shared toons

  /** Install real MMD shared toons (toon01–10.bmp) and rebuild models to use them. */
  async setSharedToons(files: Map<number, Blob>): Promise<number> {
    const n = await this.renderer.core.textures.setSharedToons(files);
    this.customToons = Math.max(this.customToons, n);
    await this.rebuildModels();
    this.dom.controls.resetToons.disabled = this.customToons === 0;
    return n;
  }

  async resetSharedToons(): Promise<void> {
    await clearSharedToons();
    location.reload();
  }

  /** Reload every model from its source, keeping order, visibility, motion and selection. */
  private async rebuildModels(): Promise<void> {
    const old = [...this.scene.models];
    const selectedIndex = old.findIndex((m) => m.id === this.scene.selectedId);
    for (const [i, m] of old.entries()) {
      const fresh = await loadSceneModel(this.renderer.core, m.source, m.path, m.fileBytes);
      fresh.visible = m.visible;
      fresh.setTransform(m.position, m.rotationY);
      fresh.model.motionScale = m.motionScale;
      fresh.setHiddenMaterials(m.hiddenMaterials);
      m.moveAuxTo(fresh);
      fresh.shader = m.shader;
      fresh.gpu.setShader(m.shader ? (this.materialShaders.get(m.shader.code) ?? null) : null);
      fresh.setMotion(m.motion, m.motionNames);
      this.scene.models[i] = fresh;
      m.dispose();
      void this.setupPhysics(fresh);
    }
    if (selectedIndex >= 0) this.scene.selectedId = this.scene.models[selectedIndex].id;
    this.scene.changed();
  }

  // =====================================================================================
  // Keyboard

  handleKey(e: KeyboardEvent): void {
    // Typing in a field or a dialog shouldn't drive the player. (The target may be window/document.)
    const t = e.target;
    if (t instanceof HTMLInputElement || t instanceof HTMLSelectElement || t instanceof HTMLTextAreaElement) return;
    if (t instanceof Element && t.closest('dialog')) return;
    const mod = e.metaKey || e.ctrlKey;
    if (mod && e.code === 'KeyS') {
      e.preventDefault();
      void this.saveProject();
      return;
    }
    if (mod && e.code === 'KeyO') {
      e.preventDefault();
      void this.openProjectDialog();
      return;
    }
    if (mod) return;
    switch (e.code) {
      case 'Space': this.togglePlay(); break;
      case 'ArrowLeft': if (e.altKey) this.jumpKey(-1); else this.step(e.shiftKey ? -10 : -1); break;
      case 'ArrowRight': if (e.altKey) this.jumpKey(1); else this.step(e.shiftKey ? 10 : 1); break;
      case 'Home': this.setFrame(this.range?.[0] ?? 0); break;
      case 'End': this.setFrame(this.range?.[1] ?? Math.round(this.length)); break;
      case 'KeyI': this.setRangeEdge('in'); break;
      case 'KeyO': this.setRangeEdge('out'); break;
      case 'KeyL': this.loop = !this.loop; break;
      case 'KeyR': this.resetPhysics(); break;
      default: return;
    }
    e.preventDefault();
  }
}
