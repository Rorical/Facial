// Depth-only passes over the skinned vertex buffer: the self-shadow map from the light (one
// cascade per pass) and the scene depth pre-pass from the camera, which custom material
// shaders read for screen-space effects. Group 0 holds just the pass's matrix.

struct DepthView {
  viewProj: mat4x4f,
};

@group(0) @binding(0) var<uniform> depthView: DepthView;
@group(0) @binding(1) var wrapSampler: sampler;

@group(1) @binding(0) var<uniform> material: Material;
@group(1) @binding(1) var baseTex: texture_2d<f32>;

struct ShadowOut {
  @builtin(position) clip: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vsShadow(in: VsIn) -> ShadowOut {
  var out: ShadowOut;
  out.clip = depthView.viewProj * vec4f(in.pos, 1.0);
  // Casters in front of the near plane still cast: pancake them onto it.
  out.clip.z = max(out.clip.z, 0.0);
  out.uv = in.uv;
  return out;
}

@vertex
fn vsPrepass(in: VsIn) -> ShadowOut {
  var out: ShadowOut;
  out.clip = depthView.viewProj * vec4f(in.pos, 1.0);
  out.uv = in.uv;
  return out;
}

fn cutOut(uv: vec2f) -> bool {
  // Cut-out textures (hair tips, lace) shouldn't cast solid shadows.
  var alpha = material.diffuse.a;
  let tex = textureSample(baseTex, wrapSampler, uv);
  if (material.modes.x != 0u) {
    alpha *= tex.a;
  }
  return alpha < 0.5;
}

@fragment
fn fsShadow(in: ShadowOut) {
  if (cutOut(in.uv)) {
    discard;
  }
}

/** The pre-pass also records what kind of material is in front: its part (see MaterialPart) / 255. */
@fragment
fn fsPrepass(in: ShadowOut) -> @location(0) f32 {
  if (cutOut(in.uv)) {
    discard;
  }
  return material.info.y / 255.0;
}
