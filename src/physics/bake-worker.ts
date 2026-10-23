// Physics bake worker: runs bakePhysics off the main thread so the UI stays responsive.

import { MmdMotion } from '../animation/motion';
import type { PmxModel } from '../loaders/pmx';
import type { VmdMotion } from '../loaders/vmd';
import type { ClothSettings } from '../cloth/cloth-sim';
import type { ClothBake } from './bake';
import { bakePhysics } from './bake-sim';

export type BakeRequest = {
  pmx: PmxModel; motion: VmdMotion | null; from: number; to: number; warmupSteps: number; motionScale?: number;
  cloth?: { materials: number[]; settings: ClothSettings } | null;
};
export type BakeResponse =
  | { type: 'progress'; done: number; total: number }
  | { type: 'done'; from: number; to: number; bones: Int32Array; data: Float32Array; cloth: ClothBake | null }
  | { type: 'error'; message: string };

const scope = self as unknown as {
  onmessage: ((e: MessageEvent<BakeRequest>) => void) | null;
  postMessage(message: BakeResponse, transfer?: Transferable[]): void;
};

scope.onmessage = (e) => {
  try {
    const { pmx, motion, from, to, warmupSteps, motionScale, cloth } = e.data;
    const bake = bakePhysics(pmx, motion ? MmdMotion.fromVmd(motion) : null, {
      from,
      to,
      warmupSteps,
      motionScale,
      cloth,
      onProgress: (done, total) => scope.postMessage({ type: 'progress', done, total }),
    });
    const transfer: Transferable[] = [bake.bones.buffer, bake.data.buffer];
    if (bake.cloth) transfer.push(bake.cloth.data.buffer);
    scope.postMessage({ type: 'done', from: bake.from, to: bake.to, bones: bake.bones, data: bake.data, cloth: bake.cloth }, transfer);
  } catch (err) {
    scope.postMessage({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};
