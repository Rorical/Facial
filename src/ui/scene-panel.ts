import type { Scene, SceneModel } from '../app/scene';
import { iconButton as makeIconButton, type IconName } from './icons';

// Sidebar list of what's in the scene: models (select / show / remove), their motions,
// the camera motion and the audio track.

export interface ScenePanelHandlers {
  select(id: number): void;
  toggleVisible(id: number): void;
  remove(id: number): void;
  clearMotion(id: number): void;
  clearCamera(): void;
  clearAudio(): void;
  /** Scale of a model's motion translations (steps, hip height). */
  setMotionScale(id: number, scale: number): void;
  /** Camera motion fitted to another height: scale of target height / distance, extra height. */
  setCameraFit(scale: number, offsetY: number): void;
  importShader(kind: 'material' | 'post'): void;
  clearModelShader(id: number): void;
  /** Open the auxiliary maps (ramp, light maps, face map) of a model. */
  editAux(id: number): void;
  clearPostShader(): void;
  importBackground(): void;
  clearBackground(): void;
  /** Offset (model units) and rotation about the vertical axis (degrees). */
  setTransform(id: number, position: [number, number, number], rotationY: number): void;
  /** Show or hide one of a model's materials (a part of the outfit). */
  setMaterialVisible(id: number, index: number, visible: boolean): void;
  showAllMaterials(id: number): void;
  setUseCameraMotion(on: boolean): void;
  importModel(): void;
  importMotion(): void;
  importCamera(): void;
  importAudio(): void;
  bake(id: number): void;
  setUseBake(id: number, on: boolean): void;
  clearBake(id: number): void;
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, props: Partial<HTMLElementTagNameMap[K]> = {}, ...children: (Node | string | null)[]): HTMLElementTagNameMap[K] {
  const e = Object.assign(document.createElement(tag), props);
  for (const c of children) if (c !== null) e.append(c);
  return e;
}

function iconButton(name: IconName, title: string, onClick: (e: MouseEvent) => void): HTMLButtonElement {
  return makeIconButton(name, title, (e) => {
    e.stopPropagation();
    onClick(e);
  });
}

export class ScenePanel {
  useCameraMotion = true;
  cameraFit = { scale: 1, offsetY: 0 };
  /** The selected model's material list is unfolded. */
  private partsOpen = false;
  private scene: Scene | null = null;

  constructor(private readonly root: HTMLElement, private readonly h: ScenePanelHandlers) {}

  render(scene: Scene): void {
    const root = this.root;
    this.scene = scene;
    root.replaceChildren();

    // ---- Models ----
    const header = el('div', { className: 'section-head' }, el('span', {}, '模型', el('span', { className: 'count' }, String(scene.models.length))));
    header.append(iconButton('plus', '导入模型（PMX / PMD / 文件夹 / zip）', () => this.h.importModel()));
    root.append(header);

    if (scene.models.length === 0) {
      root.append(el('div', { className: 'empty' }, '把模型文件夹、PMX / PMD 或 zip 拖到舞台上，或点右上的加号导入。'));
    }
    const list = el('ul', { className: 'model-list' });
    for (const m of scene.models) list.append(this.modelRow(m, m.id === scene.selectedId));
    root.append(list);

    const sel = scene.selected;
    if (sel) root.append(this.selectedDetails(sel));

    // ---- Camera ----
    const cam = scene.cameraMotion?.cameraTrack;
    const camHead = el('div', { className: 'section-head' }, el('span', {}, '镜头'));
    camHead.append(iconButton('plus', '导入镜头 VMD', () => this.h.importCamera()));
    root.append(camHead);
    if (scene.cameraMotion) {
      const use = el('input', { type: 'checkbox', checked: this.useCameraMotion, disabled: !cam });
      use.addEventListener('change', () => this.h.setUseCameraMotion(use.checked));
      root.append(
        el('div', { className: 'item' },
          el('span', { className: 'grow', title: scene.cameraNames.join('\n') }, scene.cameraNames.join(', ') || '（镜头）'),
          iconButton('close', '移除镜头动作', () => this.h.clearCamera()),
        ),
        el('div', { className: 'detail' },
          cam ? `${cam.frames.length} 个镜头关键帧` : '无镜头关键帧（仅灯光 / 阴影）',
          cam ? el('label', { className: 'inline' }, use, ' 使用镜头') : null,
        ),
      );
      if (cam) root.append(this.cameraFitRow(scene.selected));
    } else {
      root.append(el('div', { className: 'empty' }, '还没有镜头动作。带镜头的 VMD 导入后会放在这里。'));
    }

    // ---- Audio ----
    const audioHead = el('div', { className: 'section-head' }, el('span', {}, '音频'));
    audioHead.append(iconButton('plus', '导入音频', () => this.h.importAudio()));
    root.append(audioHead);
    root.append(
      scene.audio
        ? el('div', { className: 'item' }, el('span', { className: 'grow', title: scene.audio.name }, scene.audio.name), iconButton('close', '移除音频', () => this.h.clearAudio()))
        : el('div', { className: 'empty' }, '还没有音频。'),
    );

    // ---- Background ----
    const bgHead = el('div', { className: 'section-head' }, el('span', {}, '背景'));
    bgHead.append(iconButton('plus', '导入全景图（2:1 的等距柱状投影图片）', () => this.h.importBackground()));
    root.append(bgHead);
    root.append(
      scene.background
        ? el('div', { className: 'item' }, el('span', { className: 'grow', title: scene.background.name }, scene.background.name), iconButton('close', '移除背景', () => this.h.clearBackground()))
        : el('div', { className: 'empty' }, '纯色。可以导入一张全景图当天空。'),
    );

    // ---- Post-processing shader ----
    const fxHead = el('div', { className: 'section-head' }, el('span', {}, '后期 shader'));
    fxHead.append(iconButton('plus', '加载后期 shader（WGSL，定义 fn effect）', () => this.h.importShader('post')));
    root.append(fxHead);
    root.append(
      scene.postShader
        ? el('div', { className: 'item' }, el('span', { className: 'grow', title: scene.postShader.name }, scene.postShader.name), iconButton('close', '移除后期 shader', () => this.h.clearPostShader()))
        : el('div', { className: 'empty' }, '无。「效果」菜单里有示例。'),
    );
  }

  private modelRow(m: SceneModel, selected: boolean): HTMLLIElement {
    const li = el('li', { className: `model${selected ? ' selected' : ''}${m.visible ? '' : ' hidden-model'}`, title: m.path });
    li.addEventListener('click', () => this.h.select(m.id));
    const eye = iconButton(m.visible ? 'eye' : 'eyeOff', m.visible ? '隐藏' : '显示', () => this.h.toggleVisible(m.id));
    const name = el('span', { className: 'grow' }, m.name);
    const tag = el('span', { className: 'tag' }, m.pmx.format);
    li.append(eye, name, tag, iconButton('close', '从场景移除', () => this.h.remove(m.id)));
    return li;
  }

  private selectedDetails(m: SceneModel): HTMLElement {
    const pmx = m.pmx;
    const box = el('div', { className: 'details' });
    box.append(
      el('div', { className: 'detail' },
        `${pmx.vertexCount.toLocaleString()} 顶点 · ${pmx.materials.length} 材质 · ${pmx.bones.length} 骨骼 · ${pmx.morphs.length} 表情 · ${pmx.rigidBodies.length} 刚体`),
    );
    if (m.missingTextures.length) {
      box.append(el('div', { className: 'detail warn', title: m.missingTextures.join('\n') }, `⚠ 缺少 ${m.missingTextures.length} 张贴图`));
    }
    const motionRow = el('div', { className: 'item' });
    if (m.motion) {
      motionRow.append(
        el('span', { className: 'grow', title: m.motionNames.join('\n') }, `动作：${m.motionNames.join(', ')}`),
        iconButton('close', '清除此模型的动作', () => this.h.clearMotion(m.id)),
      );
      box.append(motionRow);
      if (m.binding && m.motion) {
        const b = m.binding;
        const none = b.usedBones > 0 && b.bones === 0;
        const partial = b.missingBones.length + b.missingMorphs.length > 0;
        const missing = [
          b.missingBones.length ? `模型没有这些骨骼：${b.missingBones.join('、')}` : '',
          b.missingMorphs.length ? `模型没有这些表情：${b.missingMorphs.join('、')}` : '',
        ].filter(Boolean).join('\n');
        box.append(el('div', { className: `detail${none ? ' warn' : ''}`, title: missing || '动作用到的骨骼和表情这个模型都有' },
          none ? '骨骼名都不匹配，这个动作可能不是给这个模型的'
            : `动作用到的骨骼 ${b.bones}/${b.usedBones}、表情 ${b.morphs}/${b.usedMorphs} 在模型里${partial ? '（悬停看缺少的）' : ''}`));
      }
    } else {
      motionRow.append(el('span', { className: 'grow muted' }, '无动作'));
      motionRow.append(iconButton('plus', '导入动作 VMD（应用到此模型）', () => this.h.importMotion()));
      box.append(motionRow);
    }
    if (m.rig) box.append(this.bakeRow(m));
    box.append(
      m.shader
        ? el('div', { className: 'item' },
            el('span', { className: 'grow', title: m.shader.name }, `着色：${m.shader.name}`),
            iconButton('close', '恢复 MMD 着色', () => this.h.clearModelShader(m.id)))
        : el('div', { className: 'item' },
            el('span', { className: 'grow muted' }, '着色：MMD'),
            iconButton('plus', '加载材质 shader（WGSL，定义 fn shade），应用到此模型', () => this.h.importShader('material'))),
    );
    const auxCount = (m.aux.ramp ? 1 : 0) + m.aux.light.size + m.aux.face.size;
    box.append(el('div', { className: 'item' },
      el('span', { className: `grow${auxCount ? '' : ' muted'}` }, auxCount ? `辅助贴图：${auxCount} 张` : '辅助贴图：无'),
      Object.assign(el('button', { className: 'btn small', textContent: '管理…', title: '色条、光照图、脸部阴影图：供材质 shader 使用，可自动生成或加载图片' }), {
        onclick: () => this.h.editAux(m.id),
      })));
    if (m.motion) box.append(this.motionScaleRow(m));
    box.append(this.transformRow(m));
    box.append(...this.partsRows(m));
    return box;
  }

  /** Show / hide the model's materials: a hat, the shoes, a coat… */
  private partsRows(m: SceneModel): HTMLElement[] {
    const hidden = m.hiddenMaterials.size;
    const toggle = el('button', {
      className: 'btn small', textContent: this.partsOpen ? '收起' : '显示 / 隐藏…',
      title: '按材质显示或隐藏模型的部件（帽子、衣服、鞋子…），随工程保存',
    });
    toggle.addEventListener('click', () => {
      this.partsOpen = !this.partsOpen;
      if (this.scene) this.render(this.scene);
    });
    const row = el('div', { className: 'item' },
      el('span', { className: `grow${hidden ? '' : ' muted'}` }, hidden ? `部件：隐藏了 ${hidden} 个材质` : '部件：全部显示'),
      hidden ? iconButton('eye', '全部显示', () => this.h.showAllMaterials(m.id)) : null,
      toggle,
    );
    if (!this.partsOpen) return [row];
    const chips = el('div', { className: 'chips parts' });
    m.pmx.materials.forEach((mat, i) => {
      const box = el('input', { type: 'checkbox', checked: !m.hiddenMaterials.has(i) });
      box.addEventListener('change', () => this.h.setMaterialVisible(m.id, i, box.checked));
      const name = mat.name || `材质 ${i + 1}`;
      chips.append(el('label', { className: 'chip', title: mat.nameEn && mat.nameEn !== mat.name ? `${name} / ${mat.nameEn}` : name }, box, name));
    });
    return [row, chips];
  }

  private numberField(label: string, value: number, step: string, title: string, onChange: () => void): HTMLLabelElement {
    const input = el('input', { type: 'number', step, value: String(+value.toFixed(3)), title });
    input.setAttribute('aria-label', title);
    input.addEventListener('change', onChange);
    return el('label', { className: 'axis' }, label, input);
  }

  /** Fit the motion's steps and hip height to this model's legs. */
  private motionScaleRow(m: SceneModel): HTMLElement {
    const legs = m.proportions.legs;
    const field = this.numberField('位移 ×', m.motionScale, '0.05', '动作里所有位移（步幅、重心高度）的倍率', () => {
      this.h.setMotionScale(m.id, Math.max(0.01, Number(input.value) || 1));
    });
    const input = field.querySelector('input')!;
    const auto = el('button', {
      className: 'btn small', textContent: legs ? `按腿长 ${legs.toFixed(2)}` : '按腿长',
      disabled: !legs,
      title: legs
        ? '按这个模型的腿长（相对标准 MMD 模型）设置倍率：腿更长的模型步子迈得更大、重心更高，脚不再打滑或悬空'
        : '这个模型没有标准的「左足 / 左足首」骨骼，无法估算',
    });
    auto.addEventListener('click', () => {
      input.value = legs!.toFixed(2);
      this.h.setMotionScale(m.id, +legs!.toFixed(2));
    });
    return el('div', { className: 'item transform', title: '动作是按原模型的体型做的；换了腿长不同的模型，用这个倍率修正' }, field, auto);
  }

  /** Fit the camera motion to the current model's height. */
  private cameraFitRow(sel: SceneModel | null): HTMLElement {
    const commit = () => {
      this.cameraFit = { scale: Math.max(0.01, Number(scale.value) || 1), offsetY: Number(offset.value) || 0 };
      this.h.setCameraFit(this.cameraFit.scale, this.cameraFit.offsetY);
    };
    const scaleField = this.numberField('身高 ×', this.cameraFit.scale, '0.05', '镜头注视高度和距离的倍率', commit);
    const offsetField = this.numberField('抬高', this.cameraFit.offsetY, '0.5', '镜头注视点再抬高多少（模型单位）', commit);
    const scale = scaleField.querySelector('input')!;
    const offset = offsetField.querySelector('input')!;
    const head = sel?.proportions.head ?? null;
    const auto = el('button', {
      className: 'btn small', textContent: head ? `按身高 ${head.toFixed(2)}` : '按身高',
      disabled: !head,
      title: head
        ? '按当前模型头部的高度（相对标准 MMD 模型）设置倍率：镜头对准同一个部位，人物在画面里一样大'
        : '当前模型没有标准的「頭」骨骼，无法估算',
    });
    auto.addEventListener('click', () => {
      scale.value = head!.toFixed(2);
      commit();
    });
    return el('div', { className: 'item transform', title: '镜头是按原模型的身高做的；角色更高或更矮时用这里修正' }, scaleField, offsetField, auto);
  }

  /** Where the model stands: X / Y / Z offset and a turn about the vertical axis. */
  private transformRow(m: SceneModel): HTMLElement {
    const field = (label: string, value: number, step: string, title: string) => {
      const input = el('input', { type: 'number', step, value: String(value), title });
      input.setAttribute('aria-label', title);
      input.addEventListener('change', commit);
      return el('label', { className: 'axis' }, label, input);
    };
    const commit = () => {
      const v = inputs.map((i) => Number(i.value) || 0);
      this.h.setTransform(m.id, [v[0], v[1], v[2]], v[3]);
    };
    const fields = [
      field('X', m.position[0], '1', '左右位置'),
      field('Y', m.position[1], '1', '高度'),
      field('Z', m.position[2], '1', '前后位置'),
      field('↻', m.rotationY, '15', '水平旋转（度）'),
    ];
    const inputs = fields.map((f) => f.querySelector('input')!);
    return el('div', { className: 'item transform', title: '模型在场景里的位置（1 单位约 8 厘米）' }, ...fields);
  }

  private bakeRow(m: SceneModel): HTMLElement {
    const row = el('div', { className: 'item' });
    if (!m.bake) {
      row.append(
        el('span', { className: 'grow muted' }, '物理：实时计算'),
        Object.assign(el('button', { className: 'btn small', textContent: '烘焙物理…', title: '预先逐帧计算物理并记录，播放时直接读取' }), {
          onclick: () => this.h.bake(m.id),
        }),
      );
      return row;
    }
    const use = el('input', { type: 'checkbox', checked: m.useBake });
    use.addEventListener('change', () => this.h.setUseBake(m.id, use.checked));
    row.append(
      el('span', { className: 'grow', title: `${m.bake.bones.length} 根物理骨骼 × ${m.bake.frameCount} 帧` }, `物理烘焙：第 ${m.bake.from}–${m.bake.to} 帧${m.bake.cloth ? ' · 含布料' : ''}`),
      el('label', { className: 'inline', title: '播放时使用烘焙结果（取消则实时计算）' }, use, ' 使用'),
      iconButton('refresh', '重新烘焙', () => this.h.bake(m.id)),
      iconButton('close', '删除烘焙结果', () => this.h.clearBake(m.id)),
    );
    return row;
  }
}
