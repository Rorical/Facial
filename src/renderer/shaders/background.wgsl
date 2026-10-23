// Panorama background: an equirectangular image looked up by view direction, drawn behind
// everything as one full-screen triangle.

struct Sky {
  invViewProj: mat4x4f,
  eye: vec4f,
};

@group(0) @binding(0) var<uniform> sky: Sky;
@group(0) @binding(1) var skySampler: sampler;
@group(0) @binding(2) var skyTex: texture_2d<f32>;

struct VsOut {
  @builtin(position) clip: vec4f,
  @location(0) ndc: vec2f,
};

@vertex
fn vsMain(@builtin(vertex_index) i: u32) -> VsOut {
  var out: VsOut;
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
  out.clip = vec4f(p, 1.0, 1.0);
  out.ndc = p;
  return out;
}

const PI = 3.14159265358979;

@fragment
fn fsMain(in: VsOut) -> @location(0) vec4f {
  let far = sky.invViewProj * vec4f(in.ndc, 1.0, 1.0);
  let near = sky.invViewProj * vec4f(in.ndc, 0.0, 1.0);
  let dir = normalize(far.xyz / far.w - near.xyz / near.w);
  // Left-handed, the image's centre straight ahead (+Z) of the default view.
  let uv = vec2f(atan2(dir.x, dir.z) / (2.0 * PI) + 0.5, acos(clamp(dir.y, -1.0, 1.0)) / PI);
  // Level 0: the wrap-around seam would otherwise pick the smallest mip.
  return vec4f(textureSampleLevel(skyTex, skySampler, uv, 0.0).rgb, 1.0);
}
