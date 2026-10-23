// Physics Web Worker: owns one simulation for one model. See runners.ts for the client.

import { PhysicsWorld } from './physics-world';
import type { WorkerRequest, WorkerResponse } from './protocol';
import { buildPhysicsRig } from './rig';

// tsconfig uses the DOM lib; describe just the worker-scope API used here.
const scope = self as unknown as {
  onmessage: ((e: MessageEvent<WorkerRequest>) => void) | null;
  postMessage(message: WorkerResponse, transfer?: Transferable[]): void;
};

let world: PhysicsWorld | null = null;

scope.onmessage = (e) => {
  try {
    handle(e.data);
  } catch (err) {
    scope.postMessage({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};

function handle(msg: WorkerRequest): void {
  if (msg.type === 'init') {
    // Results are shown one frame late, so simulate toward the predicted next pose.
    world = new PhysicsWorld(buildPhysicsRig(msg.model), { predict: true, ...msg.options });
    scope.postMessage({ type: 'ready', bodyCount: world.bodyCount, jointCount: world.jointCount });
    return;
  }

  if (!world) throw new Error('Physics worker received a step before init');
  const t0 = performance.now();
  world.step(
    { boneWorlds: new Float64Array(msg.boneWorlds), dt: msg.dt, reset: msg.reset, warmup: msg.warmup },
    new Float64Array(msg.locals),
  );
  if (msg.debug) world.readBodyTransforms(new Float32Array(msg.debug));
  const transfer = [msg.boneWorlds, msg.locals, ...(msg.debug ? [msg.debug] : [])];
  scope.postMessage(
    {
      type: 'result',
      generation: msg.generation,
      boneWorlds: msg.boneWorlds,
      locals: msg.locals,
      debug: msg.debug,
      stepMs: performance.now() - t0,
      warmup: msg.warmup,
    },
    transfer,
  );
}
