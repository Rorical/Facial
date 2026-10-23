import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, it } from 'vitest';
import { parsePmx } from '../src/loaders/pmx';
import { MmdMotion } from '../src/animation/motion';
import { mat4, quat, vec3 } from '../src/math';
import { PhysicsWorld, type PhysicsOptions } from '../src/physics/physics-world';
import { extrapolatePose } from '../src/physics/pose';
import { buildPhysicsRig, physicsModelData } from '../src/physics/rig';
import { InlinePhysics } from '../src/physics/runners';
import { MmdModel } from '../src/runtime/mmd-model';

function sway(periodFrames: number) {
  const interpolation = new Uint8Array(64);
  for (let c = 0; c < 4; c++) { interpolation[c] = 64; interpolation[4 + c] = 0; interpolation[8 + c] = 64; interpolation[12 + c] = 127; }
  const boneFrames = [];
  for (let i = 0, f = 0; f <= 600; i++, f += periodFrames) {
    const side = i % 2 ? 1 : -1;
    const q = quat.fromAxisAngle(quat.create(), [0, 1, 0], (35 * side * Math.PI) / 180);
    boneFrames.push({ boneName: 'センター', frame: f, position: [3 * side, i % 4 === 1 ? -1.5 : 0, 0] as [number, number, number], rotation: Array.from(q) as [number, number, number, number], interpolation });
  }
  return MmdMotion.fromVmd({ modelName: '', morphFrames: [], cameraFrames: [], lightFrames: [], selfShadowFrames: [], propertyFrames: [], boneFrames });
}

// Opt-in benchmark (slow): PHYSICS_BENCH=1 MMD_SAMPLE_DIR=… pnpm test physics-bench
// Measures visible twitching of physics bones, how far joints come apart, and the worker's
// pose-prediction error.
const pmxPath = (() => {
  const dir = process.env.MMD_SAMPLE_DIR;
  if (!process.env.PHYSICS_BENCH || !dir) return '';
  const name = readdirSync(dir).find((n) => /\.pmx$/i.test(n));
  return name ? join(dir, name) : '';
})();

describe.skipIf(!pmxPath)('physics benchmark', () => it('twitch + prediction', () => {
  const b = readFileSync(pmxPath);
  const pmx = parsePmx(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer);
  const rig = buildPhysicsRig(physicsModelData(pmx));

  /** Joints whose anchors should coincide (no linear play): their frames in each body's space. */
  const attached = rig.joints.filter((j) => j.source.positionMin.every((v, i) => v === 0 && j.source.positionMax[i] === 0));
  /**
   * Visible twitch: how far bone-tip points stray from their own ~66 ms moving average (model
   * units; 1 ≈ 8 cm). Joint gap: how far apart the two anchors of an attached joint are.
   */
  const run = (label: string, opts: Partial<PhysicsOptions>, hz: number, motion: MmdMotion | null) => {
    const model = new MmdModel(pmx, pmx.materials.map(() => ({ hasTexture: false, hasSphere: false, hasToon: false })));
    const physics = new InlinePhysics(new PhysicsWorld(rig, opts), rig, pmx.bones.length);
    physics.wantDebug = true;
    physics.requestReset(60);
    model.attachPhysics(physics);
    model.setMotion(motion);
    const dt = 1 / hz, frames = Math.round(hz * 8), n = rig.drivenBones.length;
    const track = new Float64Array(frames * n * 3);
    const p = vec3.create(), inv = mat4.create();
    const ma = mat4.create(), mb = mat4.create(), pa = vec3.create(), pb = vec3.create();
    let maxGap = 0, sumGap = 0, gaps = 0;
    const t0 = performance.now();
    for (let i = 0; i < frames; i++) {
      model.update(i * dt * 30, dt);
      if (i % 4 === 0 && i >= hz * 2) {
        const bt = physics.bodyTransforms!;
        for (const j of attached) {
          mat4.multiply(ma, bt.subarray(j.bodyA * 16, j.bodyA * 16 + 16) as unknown as Float64Array, j.frameA);
          mat4.multiply(mb, bt.subarray(j.bodyB * 16, j.bodyB * 16 + 16) as unknown as Float64Array, j.frameB);
          const gap = vec3.distance(mat4.getTranslation(pa, ma), mat4.getTranslation(pb, mb));
          maxGap = Math.max(maxGap, gap); sumGap += gap; gaps++;
        }
      }
      rig.drivenBones.forEach((bi, s) => {
        const bone = model.skeleton.bones[bi];
        vec3.transformMat4(p, vec3.set(p, 0, -1, 0), bone.world);
        // physics-only motion: in the space of the nearest non-physics ancestor
        let a = bone.parent;
        while (a && a.physicsActive) a = a.parent;
        if (a) vec3.transformMat4(p, p, mat4.invert(inv, a.world)!);
        track.set(p, (i * n + s) * 3);
      });
    }
    const ms = (performance.now() - t0) / frames;
    const w = Math.max(1, Math.round(hz * 0.033)); // half window
    const dev: number[] = [];
    const perBone = new Float64Array(n);
    for (let s = 0; s < n; s++) {
      for (let i = Math.round(hz * 2) + w; i < frames - w; i++) {
        let sx = 0, sy = 0, sz = 0;
        for (let k = -w; k <= w; k++) { const o = ((i + k) * n + s) * 3; sx += track[o]; sy += track[o + 1]; sz += track[o + 2]; }
        const c = 2 * w + 1, o = (i * n + s) * 3;
        const d = Math.hypot(track[o] - sx / c, track[o + 1] - sy / c, track[o + 2] - sz / c);
        dev.push(d);
        perBone[s] = Math.max(perBone[s], d);
      }
    }
    dev.sort((x, y) => x - y);
    const q = (f: number) => dev[Math.floor(dev.length * f)].toFixed(3);
    const bad = [...perBone].filter((v) => v > 0.1).length;
    console.log(`${label.padEnd(30)} ${String(hz).padStart(3)}Hz  p50 ${q(0.5)}  p99 ${q(0.99)}  p99.9 ${q(0.999)}  max ${dev[dev.length - 1].toFixed(3)}  bones>0.1: ${String(bad).padStart(3)}/${n}  joint gap mean ${(sumGap / gaps).toFixed(4)} max ${maxGap.toFixed(3)}  ${ms.toFixed(2)}ms`);
    model.attachPhysics(null);
  };

  for (const [name, motion] of [['rest', null], ['slow sway (20f)', sway(20)], ['fast sway (8f)', sway(8)]] as const) {
    console.log(`--- ${name}`);
    for (const hz of [60, 120]) {
      run('1/240, 1 pass', { maxIterations: 1 }, hz, motion);
      run('1/240, up to 3 (default)', {}, hz, motion);
      run('1/480, up to 3', { substep: 1 / 480 }, hz, motion);
      run('1/240, always 3', { tolerance: 0 }, hz, motion);
    }
  }

  // Worker pipeline: the worker simulates against pose N, the result is shown at frame N+1.
  // How far is that pose from what's actually shown — without vs. with prediction?
  const model = new MmdModel(pmx, pmx.materials.map(() => ({ hasTexture: false, hasSphere: false, hasToon: false })));
  model.setMotion(sway(8));
  const legs = ['左足', '右足', '左ひざ', '右ひざ', '左足首', '右足首', '下半身'].map((n) => model.skeleton.byName.get(n)!.index);
  const nb = pmx.bones.length;
  const poses: Float64Array[] = [];
  for (let i = 0; i < 240; i++) {
    model.update(i * 0.25, 1 / 120); // 120 Hz
    const w = new Float64Array(nb * 16);
    model.skeleton.bones.forEach((bn, k) => w.set(bn.world, k * 16));
    poses.push(w);
  }
  const pred = new Float64Array(nb * 16);
  let errPlain = 0, errPred = 0, maxPlain = 0, maxPred = 0, count = 0;
  for (let i = 1; i < poses.length - 1; i++) {
    extrapolatePose(poses[i - 1], poses[i], pred);
    for (const k of legs) {
      const o = k * 16 + 12;
      const actual = poses[i + 1];
      const e1 = Math.hypot(actual[o] - poses[i][o], actual[o + 1] - poses[i][o + 1], actual[o + 2] - poses[i][o + 2]);
      const e2 = Math.hypot(actual[o] - pred[o], actual[o + 1] - pred[o + 1], actual[o + 2] - pred[o + 2]);
      errPlain += e1; errPred += e2; maxPlain = Math.max(maxPlain, e1); maxPred = Math.max(maxPred, e2); count++;
    }
  }
  console.log(`--- worker lag, leg bones at 120 Hz (fast sway): pose error without prediction mean ${(errPlain / count).toFixed(4)} max ${maxPlain.toFixed(4)}; with prediction mean ${(errPred / count).toFixed(4)} max ${maxPred.toFixed(4)}`);
}, 900_000));
