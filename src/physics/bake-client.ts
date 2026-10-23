import type { MmdMotion } from '../animation/motion';
import type { PmxModel } from '../loaders/pmx';
import { motionToVmd } from '../io/vmd-export';
import { PhysicsBake } from './bake';
import type { BakeRequest, BakeResponse } from './bake-worker';

export interface BakeJob {
  promise: Promise<PhysicsBake>;
  /** Stop the bake (the promise rejects with an AbortError). */
  cancel(): void;
}

/** Bake in a dedicated worker. The motion travels as VMD keyframes (verbatim, lossless). */
export function bakeInWorker(
  pmx: PmxModel,
  motion: MmdMotion | null,
  range: { from: number; to: number; warmupSteps: number; motionScale?: number; cloth?: BakeRequest['cloth'] },
  onProgress: (done: number, total: number) => void,
): BakeJob {
  const worker = new Worker(new URL('./bake-worker.ts', import.meta.url), { type: 'module', name: 'mmd-physics-bake' });
  let reject!: (err: unknown) => void;
  const promise = new Promise<PhysicsBake>((resolve, rej) => {
    reject = rej;
    worker.onmessage = (e: MessageEvent<BakeResponse>) => {
      const msg = e.data;
      if (msg.type === 'progress') onProgress(msg.done, msg.total);
      else {
        worker.terminate();
        if (msg.type === 'done') resolve(new PhysicsBake(msg.from, msg.to, msg.bones, msg.data, msg.cloth));
        else rej(new Error(msg.message));
      }
    };
    worker.onerror = (e) => {
      worker.terminate();
      rej(new Error(e.message || '物理烘焙 Worker 出错'));
    };
  });
  const request: BakeRequest = {
    pmx,
    motion: motion ? motionToVmd(motion, '', { model: true, scene: false }) : null,
    ...range,
  };
  worker.postMessage(request);
  return {
    promise,
    cancel: () => {
      worker.terminate();
      reject(new DOMException('已取消', 'AbortError'));
    },
  };
}
