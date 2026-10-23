import type { Scene } from '../app/scene';
import type { Lighting } from '../renderer/lighting';
import type { PostSettings } from '../renderer/post';
import { basename, extname, type FileSet } from '../loaders/asset-source';
import { serializeBake, type BakeMeta } from '../physics/bake';
import { collectModelFiles } from './model-export';
import { motionToVmd } from './vmd-export';
import { CAMERA_VMD_MODEL_NAME, writeVmd } from './vmd-writer';
import { writeZip, type ZipInput } from './zip';
import { sanitizeFileName } from './download';

// Project file: one zip with everything needed to restore a scene.
//   project.json            settings + references
//   models/<n>/…            each model folder as imported (model file + textures, original bytes)
//   motions/<n>.vmd         each model's motion
//   bakes/<n>.bin           each model's physics bake (raw Float32); <n>-cloth.bin: baked cloth positions
//   camera.vmd              camera / light / self-shadow motion
//   audio/<name>            the audio track
//   background/<name>       the panorama background image
//   shaders/…               custom material / post-processing shaders (WGSL)
//   aux/<n>/…               auxiliary maps for material shaders (ramp, light maps, face maps)

export const PROJECT_FORMAT = 'facial-project';
export const PROJECT_FILE = 'project.json';

export interface ProjectSettings {
  playback: { frame: number; range: [number, number] | null; loop: boolean };
  view: { target: [number, number, number]; distance: number; yaw: number; pitch: number; useCameraMotion: boolean; cameraFit?: { scale: number; offsetY: number } };
  render: {
    edges: boolean; selfShadow: boolean; darkBackground: boolean; physics: 'worker' | 'main' | 'off'; wireframe: number;
    // Added later: absent in older projects.
    groundShadow?: boolean;
    clothCollision?: boolean;
    renderScale?: number;
    post?: PostSettings;
  };
  /** Main light and extra lights (absent in older projects = MMD's default light). */
  lighting?: Lighting;
}

export interface ProjectFile extends ProjectSettings {
  format: typeof PROJECT_FORMAT;
  version: 1;
  models: {
    name: string;
    file: string;
    visible: boolean;
    motion: string | null;
    motionNames: string[];
    /** Physics bake: raw little-endian Float32 data (see physics/bake.ts). */
    bake?: (BakeMeta & { file: string; use: boolean; clothFile?: string | null }) | null;
    /** Placement in the scene (absent = at the origin). */
    transform?: { position: [number, number, number]; rotationY: number };
    /** Auxiliary maps for the material shader (image files in the archive). */
    aux?: { slot: 'ramp' | 'light' | 'face'; texture: number; file: string; name: string; generated: boolean }[];
    /** Custom material shader (WGSL file in the archive). */
    shader?: { file: string; name: string } | null;
    /** Scale of the motion's bone translations (absent = 1). */
    motionScale?: number;
    /** Materials the user hid (absent = none). */
    hiddenMaterials?: number[];
  }[];
  /** Index into models. */
  selected: number;
  camera: { file: string; names: string[] } | null;
  audio: { file: string; name: string } | null;
  /** Panorama background image. */
  background?: { file: string; name: string } | null;
  /** Custom post-processing shader (WGSL file in the archive). */
  postShader?: { file: string; name: string } | null;
}

export async function buildProject(
  scene: Scene,
  settings: ProjectSettings,
  onProgress?: (message: string) => void,
): Promise<{ blob: Blob; warnings: string[] }> {
  const entries: ZipInput[] = [];
  const warnings: string[] = [];
  const project: ProjectFile = {
    format: PROJECT_FORMAT,
    version: 1,
    models: [],
    selected: Math.max(0, scene.models.findIndex((m) => m.id === scene.selectedId)),
    camera: null,
    audio: null,
    ...settings,
  };

  for (const [i, m] of scene.models.entries()) {
    onProgress?.(`收集模型 ${i + 1}/${scene.models.length}：${m.name}`);
    const collected = await collectModelFiles(m, `models/${i}`, false);
    entries.push(...collected.entries);
    warnings.push(...collected.warnings.map((w) => `${m.name}: ${w}`));
    let motion: string | null = null;
    if (m.motion && !m.motion.isEmpty) {
      motion = `motions/${i}.vmd`;
      const written = writeVmd(motionToVmd(m.motion, m.pmx.name, { model: true, scene: false }));
      warnings.push(...written.warnings);
      entries.push({ path: motion, data: written.buffer });
    }
    let bake: ProjectFile['models'][number]['bake'] = null;
    if (m.bake) {
      const { meta, data, cloth } = serializeBake(m.bake);
      bake = { file: `bakes/${i}.bin`, ...meta, use: m.useBake, clothFile: cloth ? `bakes/${i}-cloth.bin` : null };
      entries.push({ path: bake.file, data, compress: true });
      if (cloth) entries.push({ path: bake.clothFile!, data: cloth, compress: true });
    }
    let shader: ProjectFile['models'][number]['shader'] = null;
    if (m.shader) {
      shader = { file: `shaders/model-${i}.wgsl`, name: m.shader.name };
      entries.push({ path: shader.file, data: new TextEncoder().encode(m.shader.code) });
    }
    const aux: NonNullable<ProjectFile['models'][number]['aux']> = [];
    const addAux = (slot: 'ramp' | 'light' | 'face', texture: number, map: { name: string; generated: boolean; blob: Blob }) => {
      const ext = /\.(\w+)$/.exec(map.name)?.[1] ?? 'png';
      const file = `aux/${i}/${slot}${texture >= 0 ? `-${texture}` : ''}.${map.generated ? 'png' : ext}`;
      entries.push({ path: file, data: map.blob });
      aux.push({ slot, texture, file, name: map.name, generated: map.generated });
    };
    if (m.aux.ramp) addAux('ramp', -1, m.aux.ramp);
    for (const [t, map] of m.aux.light) addAux('light', t, map);
    for (const [t, map] of m.aux.face) addAux('face', t, map);
    project.models.push({
      shader,
      aux,
      name: m.name, file: collected.modelPath, visible: m.visible, motion, motionNames: m.motionNames, bake,
      transform: { position: m.position, rotationY: m.rotationY },
      motionScale: m.motionScale,
      hiddenMaterials: [...m.hiddenMaterials],
    });
  }

  if (scene.cameraMotion && !scene.cameraMotion.isEmpty) {
    const written = writeVmd(motionToVmd(scene.cameraMotion, CAMERA_VMD_MODEL_NAME, { model: false, scene: true }));
    warnings.push(...written.warnings);
    entries.push({ path: 'camera.vmd', data: written.buffer });
    project.camera = { file: 'camera.vmd', names: scene.cameraNames };
  }

  if (scene.audio) {
    const file = `audio/${sanitizeFileName(scene.audio.name)}`;
    entries.push({ path: file, data: scene.audio.blob });
    project.audio = { file, name: scene.audio.name };
  }

  if (scene.background) {
    const file = `background/${sanitizeFileName(scene.background.name)}`;
    entries.push({ path: file, data: scene.background.blob });
    project.background = { file, name: scene.background.name };
  }

  if (scene.postShader) {
    const file = 'shaders/post.wgsl';
    entries.push({ path: file, data: new TextEncoder().encode(scene.postShader.code) });
    project.postShader = { file, name: scene.postShader.name };
  }

  entries.unshift({ path: PROJECT_FILE, data: new TextEncoder().encode(JSON.stringify(project, null, 2)) });
  onProgress?.('打包中…');
  const blob = await writeZip(entries, (done, total) => onProgress?.(`打包中… ${done}/${total}`));
  return { blob, warnings: [...new Set(warnings)] };
}

/** The project description if `files` is a project archive, else null. */
export async function readProjectFile(files: FileSet): Promise<ProjectFile | null> {
  const entry = files.get(PROJECT_FILE);
  if (!entry) return null;
  const json = JSON.parse(await (await entry.load()).text()) as Partial<ProjectFile>;
  if (json.format !== PROJECT_FORMAT) return null;
  if (json.version !== 1) throw new Error(`不支持的工程版本 ${json.version}`);
  return json as ProjectFile;
}

export function projectFileName(scene: Scene): string {
  const first = scene.models[0];
  const base = first ? first.name || basename(first.path).replace(/\.\w+$/, '') : 'mmd-project';
  return `${sanitizeFileName(base)}.mmdproj.zip`;
}

/** True for file names this app saves projects as. */
export function looksLikeProject(name: string): boolean {
  return /\.mmdproj\.zip$/i.test(name) || extname(name) === 'mmdproj';
}
