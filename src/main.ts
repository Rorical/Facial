import { App } from './app/app';
import { FileSet } from './loaders/asset-source';
import { loadSharedToons } from './loaders/toon-store';
import { Renderer } from './renderer/renderer';
import { setStageTone, setupChrome } from './ui/chrome';
import { toast } from './ui/dialog';
import { LightPanel } from './ui/light-panel';
import { MenuBar } from './ui/menu';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

async function main(): Promise<void> {
  const chrome = setupChrome();
  const canvas = $<HTMLCanvasElement>('view');
  let renderer: Renderer;
  try {
    renderer = await Renderer.create(canvas);
  } catch (err) {
    const fatal = $('fatal');
    fatal.textContent = `这个浏览器不支持 WebGPU，无法显示舞台。\n请换用 Chrome / Edge 113+ 或 Safari 26+。\n\n${err instanceof Error ? err.message : err}`;
    fatal.hidden = false;
    return;
  }

  const controls = {
    edges: $<HTMLInputElement>('edges'),
    selfShadow: $<HTMLInputElement>('self-shadow'),
    darkBackground: $<HTMLInputElement>('dark-bg'),
    physics: $<HTMLSelectElement>('physics-mode'),
    wireframe: [0, 1, 2, 3].map((k) => $<HTMLInputElement>(`wf-${k}`)),
    resetToons: $<HTMLButtonElement>('reset-toons'),
  };
  // Display and post-processing settings that live on the renderer.
  const post = renderer.post.settings;
  const groundShadow = $<HTMLInputElement>('ground-shadow');
  const clothCollision = $<HTMLInputElement>('cloth-collision');
  const renderScale = $<HTMLSelectElement>('render-scale');
  const bloom = $<HTMLInputElement>('bloom');
  const bloomStrength = $<HTMLInputElement>('bloom-strength');
  const dof = $<HTMLInputElement>('dof');
  const dofStrength = $<HTMLInputElement>('dof-strength');
  const toneMapping = $<HTMLSelectElement>('tone-mapping');
  const lightPanel = new LightPanel($('lighting'), renderer.lighting, () => {});
  const syncControls = () => {
    lightPanel.sync();
    const s = renderer.post.settings;
    groundShadow.checked = renderer.groundShadow;
    clothCollision.checked = renderer.clothCollision;
    renderScale.value = String(renderer.renderScale);
    bloom.checked = s.bloom;
    bloomStrength.value = String(s.bloomStrength);
    dof.checked = s.dof;
    dofStrength.value = String(s.dofStrength);
    toneMapping.value = s.toneMapping;
    bloomStrength.disabled = !s.bloom;
    dofStrength.disabled = !s.dof;
  };
  const readControls = () => {
    const s = renderer.post.settings;
    renderer.groundShadow = groundShadow.checked;
    renderer.clothCollision = clothCollision.checked;
    renderer.renderScale = Number(renderScale.value) || 1;
    s.bloom = bloom.checked;
    s.bloomStrength = Number(bloomStrength.value);
    s.dof = dof.checked;
    s.dofStrength = Number(dofStrength.value);
    s.toneMapping = toneMapping.value as typeof post.toneMapping;
    syncControls();
  };
  for (const input of [groundShadow, clothCollision, renderScale, bloom, bloomStrength, dof, dofStrength, toneMapping]) input.addEventListener('input', readControls);
  syncControls();

  const app = new App(renderer, {
    viewport: $('viewport'),
    canvas,
    sceneRoot: $('scene'),
    timelineRoot: $('timeline'),
    info: $('info'),
    audio: $<HTMLAudioElement>('audio'),
    controls,
    syncControls,
  });
  // Frame the stage inside the area the floating interface leaves free.
  chrome.onLayout((top, bottom) => {
    const h = Math.max(1, window.innerHeight);
    app.camera.lensScale = Math.max(0.4, (h - top - bottom) / h);
    app.camera.lensShift[1] = (bottom - top) / h;
  });
  if (import.meta.env.DEV) Object.assign(window, { __mmd: { app, renderer } });

  const sel = () => app.scene.selected;
  new MenuBar($('menubar'), [
    {
      label: '文件',
      items: [
        { label: '新建工程', action: () => void app.newProject() },
        { label: '打开工程…', shortcut: '⌘O', action: () => void app.openProjectDialog() },
        { label: '保存工程…', shortcut: '⌘S', action: () => void app.saveProject(), enabled: () => !app.scene.isEmpty },
      ],
    },
    {
      label: '导入',
      items: [
        { label: '模型文件夹…', action: () => void app.importModelDialog(true) },
        { label: '模型 / 配件 / 舞台（PMX、PMD、X、zip）…', action: () => void app.importModelDialog(false) },
        'separator',
        { label: '动作 VMD（当前模型）…', action: () => void app.importMotionDialog('model'), enabled: () => !!sel() },
        { label: '镜头 VMD…', action: () => void app.importMotionDialog('camera') },
        { label: '音频…', action: () => void app.importAudioDialog() },
        { label: '背景全景图…', action: () => void app.importBackgroundDialog() },
      ],
    },
    {
      label: '效果',
      items: [
        { label: '材质 shader（当前模型）…', action: () => void app.importShaderDialog('material'), enabled: () => !!sel() },
        { label: '后期 shader…', action: () => void app.importShaderDialog('post') },
        'separator',
        { label: '示例材质：星铁风格三渲二', action: () => void app.loadExampleShader('hsr-toon.wgsl'), enabled: () => !!sel() },
        { label: '示例材质：边缘光', action: () => void app.loadExampleShader('rim-light.wgsl'), enabled: () => !!sel() },
        { label: '示例材质：两段式卡通', action: () => void app.loadExampleShader('cel-two-tone.wgsl'), enabled: () => !!sel() },
        { label: '示例材质：皮肤通透（卡通）', action: () => void app.loadExampleShader('skin-toon-sss.wgsl'), enabled: () => !!sel() },
        { label: '示例材质：写实皮肤', action: () => void app.loadExampleShader('skin-realistic.wgsl'), enabled: () => !!sel() },
        { label: '示例后期：皮肤次表面散射', action: () => void app.loadExampleShader('skin-sss-blur.wgsl') },
        { label: '示例后期：胶片感', action: () => void app.loadExampleShader('film.wgsl') },
        { label: '示例后期：深度描线', action: () => void app.loadExampleShader('depth-outline.wgsl') },
        'separator',
        { label: '恢复当前模型的 MMD 着色', action: () => sel() && void app.setModelShader(sel()!, null), enabled: () => !!sel()?.shader },
        { label: '移除后期 shader', action: () => void app.setPostShader(null), enabled: () => !!app.scene.postShader },
      ],
    },
    {
      label: '导出',
      items: [
        { label: '当前模型（zip：PMX + 贴图）…', action: () => void app.exportModel(), enabled: () => !!sel() },
        { label: '当前模型的动作 VMD…', action: () => void app.exportMotion(), enabled: () => !!sel()?.motion || !!sel()?.bake },
        { label: '镜头 VMD…', action: () => void app.exportCamera(), enabled: () => !!app.scene.cameraMotion },
        'separator',
        { label: '视频 / 图片序列…', action: () => void app.exportVideo(), enabled: () => app.length > 0 },
      ],
    },
  ]);

  // ---- Settings ----
  controls.edges.addEventListener('change', () => (renderer.showEdges = controls.edges.checked));
  controls.darkBackground.addEventListener('change', () => {
    renderer.clearColor = controls.darkBackground.checked ? { r: 0.1, g: 0.11, b: 0.13, a: 1 } : { r: 1, g: 1, b: 1, a: 1 };
    setStageTone(controls.darkBackground.checked);
  });
  controls.physics.addEventListener('change', () => void app.setPhysicsMode(controls.physics.value as 'worker' | 'main' | 'off'));
  $<HTMLButtonElement>('reset-physics').addEventListener('click', () => app.resetPhysics());
  const updateWireframe = () => (renderer.physicsDebug.mask = controls.wireframe.reduce((m, box, kind) => m | (box.checked ? 1 << kind : 0), 0));
  for (const box of controls.wireframe) box.addEventListener('change', updateWireframe);
  controls.resetToons.addEventListener('click', () => void app.resetSharedToons());

  window.addEventListener('keydown', (e) => app.handleKey(e));

  // ---- Drag & drop anywhere ----
  let dragDepth = 0;
  window.addEventListener('dragenter', (e) => {
    e.preventDefault();
    dragDepth++;
    document.body.classList.add('dragging');
  });
  window.addEventListener('dragleave', () => {
    if (--dragDepth <= 0) document.body.classList.remove('dragging');
  });
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => {
    e.preventDefault();
    dragDepth = 0;
    document.body.classList.remove('dragging');
    if (!e.dataTransfer) return;
    // Entries must be captured synchronously inside the event handler.
    const pending = FileSet.fromDataTransfer(e.dataTransfer);
    void pending.then((files) => app.importFiles(files));
  });

  // ---- Restore imported MMD toons, then URL parameters ----
  try {
    const saved = await loadSharedToons();
    if (saved.size > 0) await app.setSharedToons(saved);
  } catch (err) {
    console.warn('Could not restore saved toon textures:', err);
  }
  const params = new URLSearchParams(location.search);
  if (params.has('model') || params.has('motion') || params.has('audio')) await app.importFromQuery(params);
  else toast('拖入模型文件夹 / PMX / PMD / zip、VMD、音频或工程文件开始');
}

void main();
