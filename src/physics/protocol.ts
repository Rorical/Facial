import type { PhysicsOptions } from './physics-world';
import type { PhysicsModelData } from './rig';

// Messages between the main thread and physics-worker.ts. Per-frame buffers are
// transferred (not copied) and handed back with the result, so they ping-pong forever.

export type WorkerRequest =
  | { type: 'init'; model: PhysicsModelData; options?: Partial<PhysicsOptions> }
  | {
      type: 'step';
      /** Increments on every reset; results from older generations are discarded. */
      generation: number;
      dt: number;
      reset: boolean;
      warmup: number;
      /** Float64Array, 16 per bone. */
      boneWorlds: ArrayBuffer;
      /** Float64Array, 16 per driven bone (filled by the worker). */
      locals: ArrayBuffer;
      /** Float32Array, 16 per body (filled by the worker), or null when not drawing wireframes. */
      debug: ArrayBuffer | null;
    };

export type WorkerResponse =
  | { type: 'ready'; bodyCount: number; jointCount: number }
  | {
      type: 'result';
      generation: number;
      boneWorlds: ArrayBuffer;
      locals: ArrayBuffer;
      debug: ArrayBuffer | null;
      /** Time spent in Bullet inside the worker. */
      stepMs: number;
      /** Warmup steps included in stepMs (such samples are left out of the average). */
      warmup: number;
    }
  | { type: 'error'; message: string };
