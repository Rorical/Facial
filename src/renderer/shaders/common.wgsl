// Shared declarations, prepended to mmd.wgsl and shadow.wgsl. Must match RendererCore.writeFrame
// and MorphController's material block layout.

const CASCADES = 3;

struct Frame {
  view: mat4x4f,
  proj: mat4x4f,
  viewProj: mat4x4f,
  cameraPos: vec4f,
  lightDir: vec4f,       // direction the light travels
  lightColor: vec4f,
  viewport: vec4f,       // width, height, edge scale (pixels per MMD pixel), time in seconds
  shadowParams: vec4f,   // x = self-shadow on, y = 1 / shadow map size
  cascadeSplits: vec4f,  // view depth where each cascade ends
  cascadeTexel: vec4f,   // world size of one shadow texel, per cascade
  cascadeBias: vec4f,    // depth bias in shadow map depth units, per cascade
  groundShadow: vec4f,   // rgb, a = opacity (0 = off)
  lightViewProj: array<mat4x4f, CASCADES>,
  // Extra lights, two vec4 each: (direction the light travels, intensity), (colour, kind: 0 off, 1 fill, 2 rim).
  extraLights: array<vec4f, 6>,
};

struct Material {
  diffuse: vec4f,
  specular: vec4f,       // rgb, power
  ambient: vec4f,
  edgeColor: vec4f,
  edgeParams: vec4f,     // x = edge size
  texMul: vec4f,
  texAdd: vec4f,
  sphereMul: vec4f,
  sphereAdd: vec4f,
  toonMul: vec4f,
  toonAdd: vec4f,
  // x = has texture, y = sphere mode (0 none, 1 mul, 2 add, 3 sub-texture on UV1),
  // z = shading (0 no toon, 1 toon, 2 accessory: plain diffuse lighting), w = receives self-shadow
  modes: vec4u,
  info: vec4f,           // x = material index, y = part (see MaterialPart), z = auxiliary maps present (bits)
  headPos: vec4f,        // the model's head bone in the world; w = 0 if it has none
  headForward: vec4f,
  headUp: vec4f,
};

struct VsIn {
  @location(0) pos: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) edgeScale: f32,
  @location(4) uv1: vec2f,
};
