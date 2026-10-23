import type { PmxModel } from '../loaders/pmx';
import type { Skeleton } from '../runtime/skeleton';
import { PhysicsWorld } from './physics-world';
import type { WorkerRequest, WorkerResponse } from './protocol';
import { buildPhysicsRig, physicsModelData, type PhysicsRig } from './rig';
import { applyLocals, InlinePhysics, MAX_DT, packBoneWorlds, type PhysicsRunner } from './runner';

// Live physics runners. WorkerPhysics simulates in a Web Worker, pipelined: each frame sends
// the current pose and applies the newest finished result (one frame behind, but
// parent-relative, so physics bones stay attached). The main-thread runner is in runner.ts.

export { bodyTransformsFromBones, InlinePhysics, type PhysicsKind, type PhysicsRunner } from './runner';

export class WorkerPhysics implements PhysicsRunner {
  readonly kind = 'worker';
  enabled = true;
  wantDebug = false;
  stepMs = 0;
  bodyTransforms: Float32Array | null = null;

  freshRatio = 1;
  private fresh = false;
  private latestLocals: Float64Array | null = null;
  private readonly freeWorlds: ArrayBuffer[] = [];
  private readonly freeLocals: ArrayBuffer[] = [];
  private readonly freeDebug: ArrayBuffer[] = [];
  private inFlight = false;
  private pendingDt = 0;
  private resetPending = true;
  private warmupPending = 0;
  private generation = 0;
  private failed = false;

  private constructor(
    private readonly worker: Worker,
    readonly rig: PhysicsRig,
    private readonly boneCount: number,
    readonly bodyCount: number,
    readonly jointCount: number,
  ) {
    worker.onmessage = (e: MessageEvent<WorkerResponse>) => this.onMessage(e.data);
  }

  static create(pmx: PmxModel, rig: PhysicsRig): Promise<WorkerPhysics> {
    const worker = new Worker(new URL('./physics-worker.ts', import.meta.url), { type: 'module', name: 'mmd-physics' });
    return new Promise((resolve, reject) => {
      const fail = (message: string) => {
        worker.terminate();
        reject(new Error(message));
      };
      worker.onerror = (e) => fail(e.message || 'physics worker failed to start');
      worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
        const msg = e.data;
        if (msg.type === 'ready') {
          worker.onerror = (err) => console.error('Physics worker error:', err.message);
          resolve(new WorkerPhysics(worker, rig, pmx.bones.length, msg.bodyCount, msg.jointCount));
        } else if (msg.type === 'error') {
          fail(msg.message);
        }
      };
      const init: WorkerRequest = { type: 'init', model: physicsModelData(pmx) };
      worker.postMessage(init);
    });
  }

  update(skeleton: Skeleton, dt: number): boolean {
    if (this.failed) return false;
    this.pendingDt += dt;
    this.freshRatio = this.freshRatio * 0.98 + (this.fresh ? 0.02 : 0);
    this.fresh = false;
    const applied = this.latestLocals !== null;
    if (applied) applyLocals(skeleton, this.rig, this.latestLocals!);
    if (!this.inFlight) this.send(skeleton);
    return applied;
  }

  private send(skeleton: Skeleton): void {
    const worlds = this.freeWorlds.pop() ?? new ArrayBuffer(this.boneCount * 16 * 8);
    packBoneWorlds(skeleton, new Float64Array(worlds));
    const locals = this.freeLocals.pop() ?? new ArrayBuffer(this.rig.drivenBones.length * 16 * 8);
    const debug = this.wantDebug ? (this.freeDebug.pop() ?? new ArrayBuffer(this.rig.bodies.length * 16 * 4)) : null;
    const msg: WorkerRequest = {
      type: 'step',
      generation: this.generation,
      dt: Math.min(this.pendingDt, MAX_DT),
      reset: this.resetPending,
      warmup: this.warmupPending,
      boneWorlds: worlds,
      locals,
      debug,
    };
    this.worker.postMessage(msg, [worlds, locals, ...(debug ? [debug] : [])]);
    this.pendingDt = 0;
    this.resetPending = false;
    this.warmupPending = 0;
    this.inFlight = true;
  }

  private onMessage(msg: WorkerResponse): void {
    if (msg.type === 'error') {
      console.error('Physics worker:', msg.message);
      this.failed = true;
      return;
    }
    if (msg.type !== 'result') return;
    this.inFlight = false;
    this.freeWorlds.push(msg.boneWorlds);
    if (msg.generation !== this.generation) {
      // Computed before a reset: drop it.
      this.freeLocals.push(msg.locals);
      if (msg.debug) this.freeDebug.push(msg.debug);
      return;
    }
    if (this.latestLocals) this.freeLocals.push(this.latestLocals.buffer as ArrayBuffer);
    this.latestLocals = new Float64Array(msg.locals);
    this.fresh = true;
    if (msg.debug) {
      if (this.bodyTransforms) this.freeDebug.push(this.bodyTransforms.buffer as ArrayBuffer);
      this.bodyTransforms = new Float32Array(msg.debug);
    }
    if (msg.warmup === 0) this.stepMs = this.stepMs * 0.9 + msg.stepMs * 0.1;
  }

  requestReset(warmupSteps = 0): void {
    this.generation++;
    this.resetPending = true;
    this.warmupPending = Math.max(this.warmupPending, warmupSteps);
    // Until the post-reset result arrives, bones follow the animation.
    if (this.latestLocals) this.freeLocals.push(this.latestLocals.buffer as ArrayBuffer);
    this.latestLocals = null;
  }

  dispose(): void {
    this.worker.terminate();
  }
}

/**
 * Create a runner. 'worker' falls back to the main thread if the worker cannot start.
 * `rig` can be passed in when the caller already built one (e.g. for wireframes).
 */
export async function createPhysicsRunner(kind: 'worker' | 'main', pmx: PmxModel, rig = buildPhysicsRig(physicsModelData(pmx))): Promise<PhysicsRunner> {
  if (kind === 'worker' && typeof Worker !== 'undefined') {
    try {
      return await WorkerPhysics.create(pmx, rig);
    } catch (err) {
      console.warn('Physics worker unavailable, falling back to the main thread:', err);
    }
  }
  return new InlinePhysics(new PhysicsWorld(rig), rig, pmx.bones.length);
}
