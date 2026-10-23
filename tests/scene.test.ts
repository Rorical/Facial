import { describe, expect, it } from 'vitest';
import { MmdMotion, selfShadowRange } from '../src/animation/motion';
import { parseX } from '../src/loaders/x';
import { detectMaterialHooks, detectShaderKind } from '../src/renderer/custom-shader';
import { classifyMaterial, MaterialPart, showsThroughHair } from '../src/runtime/morph-controller';
import { Skeleton } from '../src/runtime/skeleton';
import { mat4, quat, vec3 } from '../src/math';
import type { PmxBone } from '../src/loaders/pmx';

const emptyVmd = { modelName: '', boneFrames: [], morphFrames: [], cameraFrames: [], lightFrames: [], selfShadowFrames: [], propertyFrames: [] };

describe('scene motion', () => {
  it('blends light keys linearly and holds outside them', () => {
    const motion = MmdMotion.fromVmd({
      ...emptyVmd,
      lightFrames: [
        { frame: 10, color: [0.6, 0.6, 0.6], direction: [-0.5, -1, 0.5] },
        { frame: 30, color: [1, 0.2, 0.2], direction: [0.5, -1, 0.5] },
      ],
    });
    const light = { direction: [0, 0, 0], color: [0, 0, 0] };
    expect(motion.sampleLight(20, light)).toBe(true);
    expect(light.direction[0]).toBeCloseTo(0);
    expect(light.color[0]).toBeCloseTo(0.8);
    motion.sampleLight(0, light);
    expect(light.color[1]).toBeCloseTo(0.6);
    motion.sampleLight(99, light);
    expect(light.color[1]).toBeCloseTo(0.2);
    expect(new MmdMotion().sampleLight(0, light)).toBe(false);
  });

  it('adds keys to existing tracks instead of replacing them', () => {
    // A facial motion, then a body motion that only has a zero key per morph at frame 0.
    const motion = MmdMotion.fromVmd({
      ...emptyVmd,
      morphFrames: [{ morphName: 'あ', frame: 0, weight: 0 }, { morphName: 'あ', frame: 10, weight: 1 }],
    });
    motion.merge({ ...emptyVmd, morphFrames: [{ morphName: 'あ', frame: 0, weight: 0 }] });
    expect(motion.morphTracks.get('あ')!.sample(10)).toBe(1);
    // A key on the same frame is replaced by the later file.
    motion.merge({ ...emptyVmd, morphFrames: [{ morphName: 'あ', frame: 10, weight: 0.5 }] });
    expect(motion.morphTracks.get('あ')!.sample(10)).toBe(0.5);
    expect(motion.morphTracks.get('あ')!.frames.length).toBe(2);
  });

  it('steps the self-shadow mode and maps the stored distance to a range', () => {
    const motion = MmdMotion.fromVmd({
      ...emptyVmd,
      selfShadowFrames: [{ frame: 0, mode: 1, distance: 0.01125 }, { frame: 50, mode: 0, distance: 0.01125 }],
    });
    expect(motion.selfShadowAt(49)!.mode).toBe(1);
    expect(motion.selfShadowAt(50)!.mode).toBe(0);
    // MMD's default setting 8875 is stored as 0.1 − 8875 · 10⁻⁵.
    expect(selfShadowRange(0.1 - 8875e-5)).toBeCloseTo(112.5);
    expect(new MmdMotion().selfShadowAt(0)).toBeNull();
  });

  it('carries "perspective off" on camera keys', () => {
    const key = (frame: number, perspective: boolean) => ({
      frame, distance: -45, position: [0, 10, 0] as [number, number, number], rotation: [0, 0, 0] as [number, number, number],
      interpolation: new Uint8Array(24).fill(20), fov: 30, perspective,
    });
    const track = MmdMotion.fromVmd({ ...emptyVmd, cameraFrames: [key(0, true), key(10, false)] }).cameraTrack!;
    const pose = { target: vec3.create(), rotation: vec3.create(), distance: 0, fov: 0, perspective: true };
    track.sample(5, pose);
    expect(pose.perspective).toBe(true);
    track.sample(10, pose);
    expect(pose.perspective).toBe(false);
  });
});

describe('model placement', () => {
  it('moves and turns the whole skeleton through the root transform', () => {
    const bone = (name: string, position: [number, number, number], parentIndex: number): PmxBone => ({
      name, nameEn: '', position, parentIndex, layer: 0, flags: 0, tailBoneIndex: -1, tailOffset: [0, 0, 0],
      inherit: null, fixedAxis: null, localAxes: null, externalParentKey: -1, ik: null,
    });
    const skeleton = new Skeleton([bone('root', [0, 0, 0], -1), bone('child', [1, 2, 0], 0)]);
    // A quarter turn about the vertical axis, then 10 units along X.
    const q = quat.fromAxisAngle(quat.create(), [0, 1, 0], Math.PI / 2);
    mat4.fromRotationTranslation(skeleton.root, q, vec3.create(10, 0, 0));
    skeleton.update();
    const w = skeleton.bones[1].world;
    const turned = vec3.transformQuat(vec3.create(), vec3.create(1, 2, 0), q);
    expect(w[12]).toBeCloseTo(10 + turned[0]);
    expect(w[13]).toBeCloseTo(2);
    expect(w[14]).toBeCloseTo(turned[2]);
  });
});

describe('DirectX .x accessories', () => {
  const X = `xof 0303txt 0032
template Header { <3D82AB43-62DA-11cf-AB39-0020AF71E433> WORD major; WORD minor; DWORD flags; }
Header { 1; 0; 1; }
Material Red { 0.9;0.2;0.2;1.0;; 20.0; 0.5;0.5;0.5;; 0.2;0.05;0.05;; TextureFilename { "tex\\\\a.png*shine.spa"; } }
Frame Root {
  FrameTransformMatrix { 1.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0, 0.0,0.0,1.0,0.0, 0.5,0.0,0.0,1.0;; }
  Mesh Quad {
    4; 0.0;0.0;0.0;, 1.0;0.0;0.0;, 1.0;1.0;0.0;, 0.0;1.0;0.0;;
    2; 4;0,3,2,1;, 3;0,2,1;;
    MeshMaterialList { 2; 2; 0,1;; { Red } Material { 0.2;0.4;0.9;0.5;; 5.0; 0.2;0.2;0.2;; 0.0;0.1;0.2;; } }
    MeshNormals { 1; 0.0;0.0;-1.0;; 2; 4;0,0,0,0;, 3;0,0,0;; }
    MeshTextureCoords { 4; 0.0;1.0;, 1.0;1.0;, 1.0;0.0;, 0.0;0.0;; }
  }
}`;
  const parse = (text: string) => parseX(new TextEncoder().encode(text).buffer as ArrayBuffer, 'prop.x');

  it('reads meshes, materials and frame transforms at MMD scale', () => {
    const m = parse(X);
    expect(m.format).toBe('X');
    expect(m.accessory).toBe(true);
    expect(m.vertexCount).toBe(4);
    // Quad → 2 triangles, plus the triangle.
    expect(Array.from(m.materials, (x) => x.indexCount)).toEqual([6, 3]);
    expect(m.indices.length).toBe(9);
    // (0 + 0.5 frame offset) × 10
    expect(m.positions[0]).toBeCloseTo(5);
    expect(m.positions[6]).toBeCloseTo(15); // third corner used: x = 1
    expect(Array.from(m.normals.subarray(0, 3))).toEqual([0, 0, -1]);
    expect(m.uvs[1]).toBe(1);
    expect(m.materials[0].diffuse).toEqual([0.9, 0.2, 0.2, 1]);
    expect(m.materials[0].ambient).toEqual([0.2, 0.05, 0.05]);
    expect(m.textures).toEqual(['tex\\a.png', 'shine.spa']);
    expect(m.materials[0].sphereMode).toBe(2);
    expect(m.materials[1].diffuse[3]).toBe(0.5);
    expect(m.bones.length).toBe(1);
  });

  it('rejects binary files with a clear message', () => {
    expect(() => parse('xof 0303bin 0032' + '\\0'.repeat(32))).toThrow(/文本格式/);
  });
});

describe('custom shaders', () => {
  it('guesses what part of the character a material is from its name', () => {
    const part = (n: string) => classifyMaterial(n);
    expect([part('顏'), part('顏+'), part('face')]).toEqual([MaterialPart.Face, MaterialPart.Face, MaterialPart.Face]);
    expect([part('眉睫'), part('白目'), part('目光'), part('口'), part('齒')]).toEqual(Array(5).fill(MaterialPart.FaceDetail));
    expect([part('髪'), part('hair_front')]).toEqual([MaterialPart.Hair, MaterialPart.Hair]);
    expect(part('肌')).toBe(MaterialPart.Skin);
    expect([part('袖口'), part('眼鏡'), part('裙'), part('靴')]).toEqual(Array(4).fill(MaterialPart.Other));
  });

  it('draws eyes and brows, but not the mouth or hidden expression sprites, through the hair', () => {
    expect(['眉睫', '白目', '目', '目光', 'eye_L', 'brow'].map((n) => showsThroughHair(n))).toEqual(Array(6).fill(true));
    expect(['口', '齒', '舌', '目影', 'biaoq', '表情', '顏', '髪'].map((n) => showsThroughHair(n))).toEqual(Array(8).fill(false));
  });

  it('finds the optional outline and hair hooks of a material shader', () => {
    expect(detectMaterialHooks('fn shade(s: Surface) -> vec4f { return mmdShade(s); }')).toEqual({ outline: false, hairTransparency: false });
    expect(detectMaterialHooks('fn outline(s: Surface) -> vec4f { return vec4f(0.0); }\nfn hairTransparency(s: Surface) -> f32 { return 0.5; }')).toEqual({ outline: true, hairTransparency: true });
  });

  it('tells material and post-processing shaders apart by the function they define', () => {
    expect(detectShaderKind('fn shade(s: Surface) -> vec4f { return mmdShade(s); }')).toBe('material');
    expect(detectShaderKind('fn helper() {}\nfn effect (uv: vec2f, color: vec3f) -> vec3f { return color; }')).toBe('post');
    expect(detectShaderKind('fn shader() {}')).toBeNull();
  });
});
