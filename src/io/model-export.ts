import type { SceneModel } from '../app/scene';
import { basename, cleanPath, extname, FileSet, normalizePath } from '../loaders/asset-source';
import { writePmx } from './pmx-writer';
import type { ZipInput } from './zip';
import { sanitizeFileName } from './download';

// Gather a model's files (model + referenced textures) for a zip: model exports and
// project files. PMX models are written byte-for-byte; PMD models are converted to PMX
// on export (or kept as-is inside projects, which are meant to round-trip exactly).

export interface CollectedModel {
  entries: ZipInput[];
  /** Path of the model file inside the archive. */
  modelPath: string;
  warnings: string[];
}

/** Texture paths may climb out of the model folder ("../tex/a.png"); nest the model deep enough. */
function upLevels(paths: string[]): number {
  let max = 0;
  for (const p of paths) {
    let depth = 0;
    let up = 0;
    for (const part of p.replace(/\\/g, '/').split('/')) {
      if (part === '..') {
        depth--;
        up = Math.max(up, -depth);
      } else if (part && part !== '.') depth++;
    }
    max = Math.max(max, up);
  }
  return max;
}

export async function collectModelFiles(m: SceneModel, root: string, convertToPmx: boolean): Promise<CollectedModel> {
  const warnings: string[] = [];
  const pmx = m.pmx;
  const texturePaths = [...new Set(pmx.textures)];
  const modelDir = [root, ...Array(upLevels(texturePaths)).fill('_')].filter(Boolean).join('/');

  // PMD models and .x accessories are written out as PMX.
  const convert = convertToPmx && extname(m.path) !== 'pmx';
  const fileName = convert ? basename(m.path).replace(/\.\w+$/, '.pmx') : basename(m.path);
  const modelBytes = convert ? writePmx(pmx) : m.fileBytes;
  const modelPath = `${modelDir}/${fileName}`;
  const entries: ZipInput[] = [{ path: modelPath, data: new Uint8Array(modelBytes) }];

  const seen = new Set<string>([modelPath.toLowerCase()]);
  for (const tex of texturePaths) {
    const blob = await m.source.resolve(m.baseDir, tex);
    if (!blob) {
      // MMD's own shared toons (toon01.bmp …) are not part of models.
      if (!/(^|[\\/])toon(0[1-9]|10)\.bmp$/i.test(tex)) warnings.push(`缺少贴图：${tex}`);
      continue;
    }
    const path = cleanPath(`${modelDir}/${tex}`);
    if (seen.has(path.toLowerCase())) continue;
    seen.add(path.toLowerCase());
    entries.push({ path, data: blob });
  }
  // Keep the author's readme / terms of use with the model (only possible for local folders / zips).
  if (m.source instanceof FileSet) {
    const dir = normalizePath(m.baseDir);
    for (const e of m.source.entries) {
      const p = normalizePath(e.path);
      const inDir = dir ? p.startsWith(`${dir}/`) : true;
      if (!inDir || !README_EXTENSIONS.has(extname(e.path))) continue;
      const rel = dir ? e.path.replace(/\\/g, '/').slice(m.baseDir.replace(/\\/g, '/').length + 1) : e.path;
      const path = cleanPath(`${modelDir}/${rel}`);
      if (seen.has(path.toLowerCase())) continue;
      seen.add(path.toLowerCase());
      entries.push({ path, data: await e.load() });
    }
  }
  return { entries, modelPath, warnings };
}

const README_EXTENSIONS = new Set(['txt', 'md', 'pdf', 'html', 'htm', 'rtf', 'url']);

export function modelArchiveName(m: SceneModel): string {
  return sanitizeFileName(m.name || basename(m.path).replace(/\.(pmx|pmd|x)$/i, ''));
}
