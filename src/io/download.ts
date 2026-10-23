// Save a Blob to disk: the File System Access "Save as" dialog where available (Chrome,
// Edge), otherwise a regular download.

interface SaveFilePickerOptions {
  suggestedName?: string;
  types?: { description: string; accept: Record<string, string[]> }[];
}
type ShowSaveFilePicker = (options?: SaveFilePickerOptions) => Promise<{
  createWritable(): Promise<{ write(data: Blob): Promise<void>; close(): Promise<void> }>;
}>;

const MIME: Record<string, { description: string; mime: string }> = {
  zip: { description: 'ZIP 压缩包', mime: 'application/zip' },
  vmd: { description: 'VMD 动作', mime: 'application/octet-stream' },
  pmx: { description: 'PMX 模型', mime: 'application/octet-stream' },
};

/** Returns false if the user cancelled the save dialog. */
export async function saveBlob(blob: Blob, suggestedName: string): Promise<boolean> {
  const name = sanitizeFileName(suggestedName);
  const ext = name.slice(name.lastIndexOf('.') + 1).toLowerCase();
  const picker = (window as unknown as { showSaveFilePicker?: ShowSaveFilePicker }).showSaveFilePicker;
  if (picker) {
    try {
      const type = MIME[ext];
      const handle = await picker({
        suggestedName: name,
        types: type ? [{ description: type.description, accept: { [type.mime]: [`.${ext}`] } }] : undefined,
      });
      const writable = await handle.createWritable();
      await writable.write(blob);
      await writable.close();
      return true;
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return false;
      console.warn('Save dialog failed, falling back to download:', err);
    }
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
  return true;
}

export function sanitizeFileName(name: string): string {
  return name.replace(/[\\/:*?"<>|\u0000-\u001f]/g, '_').trim() || 'untitled';
}
