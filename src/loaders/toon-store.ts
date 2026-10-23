// Persists user-provided MMD shared toons (toon01.bmp … toon10.bmp from MMD's Data folder)
// in IndexedDB, so the real textures only need to be dropped in once.

const DB_NAME = 'facial';
const STORE = 'shared-toons';

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function done(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

/** Keyed by shared toon index 0–9. */
export async function saveSharedToons(files: Map<number, Blob>): Promise<void> {
  const db = await open();
  const tx = db.transaction(STORE, 'readwrite');
  for (const [i, blob] of files) tx.objectStore(STORE).put(blob, i);
  await done(tx);
  db.close();
}

export async function loadSharedToons(): Promise<Map<number, Blob>> {
  const db = await open();
  const tx = db.transaction(STORE, 'readonly');
  const store = tx.objectStore(STORE);
  const keys = await new Promise<IDBValidKey[]>((resolve, reject) => {
    const req = store.getAllKeys();
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  const values = await new Promise<Blob[]>((resolve, reject) => {
    const req = store.getAll();
    req.onsuccess = () => resolve(req.result as Blob[]);
    req.onerror = () => reject(req.error);
  });
  db.close();
  return new Map(keys.map((k, i) => [Number(k), values[i]]));
}

export async function clearSharedToons(): Promise<void> {
  const db = await open();
  const tx = db.transaction(STORE, 'readwrite');
  tx.objectStore(STORE).clear();
  await done(tx);
  db.close();
}
