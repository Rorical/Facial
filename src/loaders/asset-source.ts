import { readZip } from '../io/zip';

// Where model files come from. MMD models are folders: the PMX references textures by
// relative, backslash-separated, case-insensitive paths, so lookups are normalized.

export interface AssetSource {
  /** Resolve `relPath` (as written in a PMX) relative to the directory `baseDir`. */
  resolve(baseDir: string, relPath: string): Promise<Blob | null>;
}

/** NFC (macOS hands out NFD names), forward slashes, collapsed "." / "..", lower case. */
export function normalizePath(path: string): string {
  const parts: string[] = [];
  for (const part of path.normalize('NFC').replace(/\\/g, '/').split('/')) {
    if (part === '' || part === '.') continue;
    if (part === '..') parts.pop();
    else parts.push(part);
  }
  return parts.join('/').toLowerCase();
}

/** Like normalizePath but keeps case: for writing paths into archives. */
export function cleanPath(path: string): string {
  const parts: string[] = [];
  for (const part of path.normalize('NFC').replace(/\\/g, '/').split('/')) {
    if (part === '' || part === '.') continue;
    if (part === '..') parts.pop();
    else parts.push(part);
  }
  return parts.join('/');
}

export function dirname(path: string): string {
  const p = path.replace(/\\/g, '/');
  const i = p.lastIndexOf('/');
  return i >= 0 ? p.slice(0, i) : '';
}

export function basename(path: string): string {
  const p = path.replace(/\\/g, '/');
  return p.slice(p.lastIndexOf('/') + 1);
}

export function extname(path: string): string {
  const name = basename(path);
  const i = name.lastIndexOf('.');
  return i >= 0 ? name.slice(i + 1).toLowerCase() : '';
}

export interface FileEntry {
  path: string;
  size: number;
  load(): Promise<Blob>;
}

/**
 * A virtual folder of files: from drag & drop, <input type="file">, or (lazily extracted)
 * zip archives — which is how most MMD models are distributed.
 */
export class FileSet implements AssetSource {
  readonly entries: FileEntry[] = [];
  private readonly byPath = new Map<string, FileEntry>();
  private readonly byName = new Map<string, FileEntry[]>();

  add(path: string, data: Blob | (() => Promise<Blob>), size?: number): void {
    let cached: Promise<Blob> | null = null;
    const entry: FileEntry = {
      path,
      size: size ?? (data instanceof Blob ? data.size : 0),
      load: () => (cached ??= data instanceof Blob ? Promise.resolve(data) : data()),
    };
    this.entries.push(entry);
    this.byPath.set(normalizePath(path), entry);
    const name = normalizePath(basename(path));
    const list = this.byName.get(name);
    if (list) list.push(entry);
    else this.byName.set(name, [entry]);
  }

  static fromFileList(list: FileList | File[]): FileSet {
    const set = new FileSet();
    for (const file of Array.from(list)) set.add(file.webkitRelativePath || file.name, file);
    return set;
  }

  /** Must be called synchronously inside the drop handler (DataTransfer items expire after). */
  static fromDataTransfer(dt: DataTransfer): Promise<FileSet> {
    const entries: FileSystemEntry[] = [];
    for (const item of Array.from(dt.items)) {
      const entry = item.kind === 'file' ? item.webkitGetAsEntry() : null;
      if (entry) entries.push(entry);
    }
    const plainFiles = Array.from(dt.files);
    return (async () => {
      const set = new FileSet();
      if (entries.length === 0) {
        for (const f of plainFiles) set.add(f.name, f);
        return set;
      }
      await Promise.all(entries.map((e) => walkEntry(e, set)));
      return set;
    })();
  }

  /** Files of a zip archive, extracted on demand, under `prefix/`. */
  static async fromZip(zip: Blob, prefix = ''): Promise<FileSet> {
    const set = new FileSet();
    await set.addZip(zip, prefix);
    return set;
  }

  private async addZip(zip: Blob, prefix: string): Promise<void> {
    for (const e of await readZip(zip)) {
      if (e.isDirectory) continue;
      const path = prefix ? `${prefix}/${e.path}` : e.path;
      this.add(path, async () => new Blob([(await e.read()) as Uint8Array<ArrayBuffer>]), e.size);
    }
  }

  /** Replace every .zip in the set by its contents (in a folder named after the archive). */
  async expandZips(): Promise<void> {
    const zips = this.entries.filter((e) => extname(e.path) === 'zip');
    for (const z of zips) {
      this.remove(z);
      await this.addZip(await z.load(), z.path.replace(/\.zip$/i, ''));
    }
  }

  private remove(entry: FileEntry): void {
    this.entries.splice(this.entries.indexOf(entry), 1);
    this.byPath.delete(normalizePath(entry.path));
    const list = this.byName.get(normalizePath(basename(entry.path)));
    if (list) list.splice(list.indexOf(entry), 1);
  }

  filterByExtension(...exts: string[]): FileEntry[] {
    return this.entries.filter((e) => exts.includes(extname(e.path)));
  }

  get(path: string): FileEntry | undefined {
    return this.byPath.get(normalizePath(path));
  }

  async resolve(baseDir: string, relPath: string): Promise<Blob | null> {
    const direct = this.byPath.get(normalizePath(baseDir ? `${baseDir}/${relPath}` : relPath));
    if (direct) return direct.load();
    // Fallback: flat file uploads lose folder structure; match by file name.
    const candidates = this.byName.get(normalizePath(basename(relPath)));
    return candidates && candidates.length > 0 ? candidates[0].load() : null;
  }
}

async function walkEntry(entry: FileSystemEntry, set: FileSet): Promise<void> {
  const path = entry.fullPath.replace(/^\//, '');
  if (entry.isFile) {
    const file = await new Promise<File>((resolve, reject) => (entry as FileSystemFileEntry).file(resolve, reject));
    set.add(path, file);
    return;
  }
  if (entry.isDirectory) {
    const reader = (entry as FileSystemDirectoryEntry).createReader();
    // readEntries returns results in batches; call until empty.
    for (;;) {
      const batch = await new Promise<FileSystemEntry[]>((resolve, reject) => reader.readEntries(resolve, reject));
      if (batch.length === 0) break;
      await Promise.all(batch.map((e) => walkEntry(e, set)));
    }
  }
}

/** Models hosted over HTTP, e.g. `?model=/models/miku/miku.pmx`. */
export class UrlAssetSource implements AssetSource {
  async resolve(baseDir: string, relPath: string): Promise<Blob | null> {
    const path = relPath.replace(/\\/g, '/');
    const url = new URL(path.split('/').map(encodeURIComponent).join('/'), new URL(`${baseDir}/`, location.href));
    const res = await fetch(url);
    return res.ok ? res.blob() : null;
  }
}
