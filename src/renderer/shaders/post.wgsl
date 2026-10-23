// Post-processing: depth of field, bloom (dual-filter blur chain) and the final composite /
// tone mapping onto the canvas. Every pass draws one full-screen triangle; each entry point
// only uses the bindings it needs.

struct Post {
  bloomThreshold: f32,
  bloomStrength: f32,
  exposure: f32,
  toneMapping: f32,   // 0 off, 1 ACES
  dofFocus: f32,      // view depth in focus
  dofAperture: f32,
  maxCoc: f32,        // largest blur radius, in scene pixels
  ortho: f32,
  depthA: f32,        // projection terms to turn depth-buffer values into view depth
  depthB: f32,
  bloomOn: f32,
  _pad: f32,
};

@group(0) @binding(0) var<uniform> post: Post;
@group(0) @binding(1) var linearSampler: sampler;
@group(0) @binding(2) var srcTex: texture_2d<f32>;
@group(0) @binding(3) var bloomTex: texture_2d<f32>;
@group(0) @binding(4) var depthTex: texture_depth_multisampled_2d;

struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vsMain(@builtin(vertex_index) i: u32) -> VsOut {
  var out: VsOut;
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  out.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2f(p.x, 1.0 - p.y);
  return out;
}

// ---- Depth of field: one gather pass over a spiral of taps ----

fn viewDepth(coord: vec2i) -> f32 {
  let size = vec2i(textureDimensions(depthTex));
  let z = textureLoad(depthTex, clamp(coord, vec2i(0), size - vec2i(1)), 0);
  if (post.ortho > 0.5) {
    return (z - post.depthB) / post.depthA;
  }
  // z = A + B / depth with A slightly above 1 and B negative: both sides are negative.
  return post.depthB / min(z - post.depthA, -1e-6);
}

/** Blur radius in pixels for something at view depth d. */
fn cocRadius(d: f32) -> f32 {
  return clamp(abs(d - post.dofFocus) / max(d, 1e-3) * post.dofAperture, 0.0, 1.0) * post.maxCoc;
}

const DOF_TAPS = 40;
const GOLDEN_ANGLE = 2.39996323;

@fragment
fn fsDof(in: VsOut) -> @location(0) vec4f {
  let size = vec2f(textureDimensions(srcTex));
  let centerDepth = viewDepth(vec2i(in.pos.xy));
  let centerCoc = cocRadius(centerDepth);
  var sum = textureSampleLevel(srcTex, linearSampler, in.uv, 0.0).rgb;
  var weight = 1.0;
  for (var i = 0; i < DOF_TAPS; i++) {
    let r = sqrt((f32(i) + 0.5) / f32(DOF_TAPS)) * post.maxCoc;
    let a = f32(i) * GOLDEN_ANGLE;
    let offset = vec2f(cos(a), sin(a)) * r;
    let d = viewDepth(vec2i(in.pos.xy + offset));
    // Something in front smears over us by its own blur; something behind only as far as
    // we are blurred ourselves, so a sharp subject keeps a clean edge against the background.
    let sampleCoc = cocRadius(d);
    let reach = select(min(sampleCoc, centerCoc), sampleCoc, d < centerDepth);
    let w = clamp(reach - r + 1.0, 0.0, 1.0);
    sum += textureSampleLevel(srcTex, linearSampler, in.uv + offset / size, 0.0).rgb * w;
    weight += w;
  }
  return vec4f(sum / weight, 1.0);
}

// ---- Bloom ----

fn downsample(uv: vec2f) -> vec3f {
  let h = 0.5 / vec2f(textureDimensions(srcTex));
  var c = textureSampleLevel(srcTex, linearSampler, uv, 0.0).rgb * 4.0;
  c += textureSampleLevel(srcTex, linearSampler, uv + vec2f(-h.x, -h.y) * 2.0, 0.0).rgb;
  c += textureSampleLevel(srcTex, linearSampler, uv + vec2f(h.x, -h.y) * 2.0, 0.0).rgb;
  c += textureSampleLevel(srcTex, linearSampler, uv + vec2f(-h.x, h.y) * 2.0, 0.0).rgb;
  c += textureSampleLevel(srcTex, linearSampler, uv + vec2f(h.x, h.y) * 2.0, 0.0).rgb;
  return c / 8.0;
}

/** Keeps what is brighter than the threshold (soft knee), at half size. */
@fragment
fn fsBloomPrefilter(in: VsOut) -> @location(0) vec4f {
  let c = downsample(in.uv);
  let brightness = max(c.r, max(c.g, c.b));
  let knee = 0.1;
  let soft = clamp(brightness - post.bloomThreshold + knee, 0.0, 2.0 * knee);
  let contribution = max(soft * soft / (4.0 * knee), brightness - post.bloomThreshold) / max(brightness, 1e-4);
  return vec4f(c * contribution, 1.0);
}

@fragment
fn fsBloomDown(in: VsOut) -> @location(0) vec4f {
  return vec4f(downsample(in.uv), post.bloomOn);
}

/** Tent-filtered upsample; blended additively onto the next larger level. */
@fragment
fn fsBloomUp(in: VsOut) -> @location(0) vec4f {
  let h = 0.5 / vec2f(textureDimensions(srcTex));
  var c = textureSampleLevel(srcTex, linearSampler, uv2(in.uv, vec2f(-2.0, 0.0), h), 0.0).rgb;
  c += textureSampleLevel(srcTex, linearSampler, uv2(in.uv, vec2f(2.0, 0.0), h), 0.0).rgb;
  c += textureSampleLevel(srcTex, linearSampler, uv2(in.uv, vec2f(0.0, -2.0), h), 0.0).rgb;
  c += textureSampleLevel(srcTex, linearSampler, uv2(in.uv, vec2f(0.0, 2.0), h), 0.0).rgb;
  c += textureSampleLevel(srcTex, linearSampler, uv2(in.uv, vec2f(-1.0, -1.0), h), 0.0).rgb * 2.0;
  c += textureSampleLevel(srcTex, linearSampler, uv2(in.uv, vec2f(1.0, -1.0), h), 0.0).rgb * 2.0;
  c += textureSampleLevel(srcTex, linearSampler, uv2(in.uv, vec2f(-1.0, 1.0), h), 0.0).rgb * 2.0;
  c += textureSampleLevel(srcTex, linearSampler, uv2(in.uv, vec2f(1.0, 1.0), h), 0.0).rgb * 2.0;
  return vec4f(c / 12.0, post.bloomOn);
}

fn uv2(uv: vec2f, o: vec2f, h: vec2f) -> vec2f {
  return uv + o * h;
}

// ---- Composite ----

fn aces(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

@fragment
fn fsComposite(in: VsOut) -> @location(0) vec4f {
  var c = textureSampleLevel(srcTex, linearSampler, in.uv, 0.0).rgb;
  let bloom = textureSampleLevel(bloomTex, linearSampler, in.uv, 0.0).rgb;
  c += bloom * post.bloomStrength * post.bloomOn;
  if (post.toneMapping > 0.5) {
    // MMD shades in display (gamma) space: tone-map in linear light and come back.
    c = pow(aces(pow(max(c, vec3f(0.0)), vec3f(2.2)) * post.exposure), vec3f(1.0 / 2.2));
  } else {
    c *= post.exposure;
  }
  return vec4f(c, 1.0);
}
