// MMD default shading (MMD / MME full.fx behavior, gamma space like MMD). Shared structs
// live in common.wgsl.
//   no self-shadow:  color = saturate(diffuse·light + ambient) · tex [*/+ sphere] · toon(N·L) + specular
//   self-shadow:     lerp(color · toonBottom, color + specular, min(saturate(3·N·L), shadow))
// Accessories (.x) have no toon: ambient + N·L · diffuse, shadowed down to the ambient part.
// Plus the screen-space inverted-hull edge (outline) pass and MMD's flat ground shadow.

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var wrapSampler: sampler;
@group(0) @binding(2) var clampSampler: sampler;
@group(0) @binding(3) var shadowMap: texture_depth_2d_array;
@group(0) @binding(4) var shadowSampler: sampler_comparison;
// The camera pre-pass (drawn when a custom material shader is in use; far plane / nothing
// otherwise): scene depth and the material part at every pixel, for screen-space effects
// such as rim light and hair shadows.
@group(0) @binding(5) var sceneDepthTex: texture_depth_2d;
@group(0) @binding(6) var scenePartTex: texture_2d<f32>;

@group(1) @binding(0) var<uniform> material: Material;
@group(1) @binding(1) var baseTex: texture_2d<f32>;
@group(1) @binding(2) var sphereTex: texture_2d<f32>;
@group(1) @binding(3) var toonTex: texture_2d<f32>;
// Auxiliary maps for custom material shaders (white when absent).
@group(1) @binding(4) var rampTex: texture_2d<f32>;
@group(1) @binding(5) var lightTex: texture_2d<f32>;
@group(1) @binding(6) var faceTex: texture_2d<f32>;

struct VsOut {
  @builtin(position) clip: vec4f,
  @location(0) worldPos: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) uv1: vec2f,
};

fn surfaceVertex(in: VsIn) -> VsOut {
  var out: VsOut;
  out.clip = frame.viewProj * vec4f(in.pos, 1.0);
  out.worldPos = in.pos;
  out.normal = in.normal;
  out.uv = in.uv;
  out.uv1 = in.uv1;
  return out;
}

@vertex
fn vsMain(in: VsIn) -> VsOut {
  return surfaceVertex(in);
}

fn texMulFactor(c: vec3f, f: vec4f) -> vec3f {
  return mix(vec3f(1.0), c * f.rgb, f.a);
}

fn texAddFactor(c: vec3f, f: vec4f) -> vec3f {
  return clamp(c + (c - vec3f(1.0)) * f.a, vec3f(0.0), vec3f(1.0)) + f.rgb;
}

/**
 * Everything known about the point being shaded. Custom material shaders receive this
 * (see material-shader.ts) and the built-in MMD shading below is written against it too.
 */
struct Surface {
  worldPos: vec3f,
  normal: vec3f,        // unit length
  viewDir: vec3f,       // towards the camera
  lightDir: vec3f,      // towards the light
  lightColor: vec3f,
  uv: vec2f,
  uv1: vec2f,           // additional UV 1
  texture: vec4f,       // base texture with material-morph tints; white if the material has none
  sphere: vec3f,        // sphere map (or sub-texture) colour, tinted
  sphereMode: u32,      // 0 none, 1 multiply, 2 add, 3 sub-texture (multiply)
  toon: vec3f,          // toon ramp at this N·L; white without a toon texture
  toonShadow: vec3f,    // the ramp's darkest colour
  diffuse: vec3f,       // material colours
  ambient: vec3f,
  specular: vec3f,
  shininess: f32,
  alpha: f32,           // material alpha × texture alpha
  lambert: f32,         // N·L, -1 … 1
  shadow: f32,          // self-shadow: 1 = lit … 0 = shadowed (1 when self-shadow is off)
  selfShadow: bool,     // self-shadow is on and this material receives it
  accessory: bool,      // a .x accessory (plain diffuse lighting, no toon)
  materialIndex: u32,
  part: u32,            // PART_… : what the material is, guessed from its name
  hasHead: bool,        // the model has a 頭 bone: the head* fields are valid
  headPos: vec3f,       // head bone position (base of the skull)
  headForward: vec3f,   // where the face looks
  headUp: vec3f,
  time: f32,            // seconds on the timeline
  // Auxiliary maps (see docs/shaders.md): present or not, and their value at this point's UV.
  hasRamp: bool,        // a colour ramp is bound: read it with sampleRamp(light, row)
  hasLightMap: bool,
  lightMap: vec4f,      // generated maps: r = ambient occlusion (1 = open), g = thinness (1 = thin). White if absent.
  hasFaceMap: bool,
  faceMap: vec4f,       // r / g = how far the light may turn to either side before this point goes dark (0–1 = 0–180°)
  screenUv: vec2f,      // where this point is on screen, 0–1 with the origin at the top left
  viewDepth: f32,       // distance in front of the camera, model units
};

/** Distance in front of the camera at a screen position, from the depth pre-pass (huge where nothing was drawn). */
fn sceneDepthAt(uv: vec2f) -> f32 {
  let size = vec2i(textureDimensions(sceneDepthTex));
  let z = textureLoad(sceneDepthTex, clamp(vec2i(uv * vec2f(size)), vec2i(0), size - vec2i(1)), 0);
  let p = frame.proj;
  if (p[3][3] > 0.5) {
    return (z - p[3][2]) / p[2][2];
  }
  // z = A + B / depth with A slightly above 1 and B negative: both sides are negative.
  return p[3][2] / min(z - p[2][2], -1e-6);
}

/** What kind of material (PART_…) is in front at a screen position; PART_OTHER where nothing was drawn. */
fn scenePartAt(uv: vec2f) -> u32 {
  let size = vec2i(textureDimensions(scenePartTex));
  return u32(textureLoad(scenePartTex, clamp(vec2i(uv * vec2f(size)), vec2i(0), size - vec2i(1)), 0).r * 255.0 + 0.5);
}

/** How many screen pixels one model unit covers at view depth d. */
fn pixelsPerUnit(d: f32) -> f32 {
  let scale = frame.viewport.y * 0.5 * frame.proj[1][1];
  if (frame.proj[3][3] > 0.5) {
    return scale;
  }
  return scale / max(d, 1e-3);
}

/**
 * Screen-space rim light: 1 where the surface a little way outwards (along the screen-space
 * normal) lies at least `minGap` model units behind this point — the silhouette against the
 * background or against farther parts — and 0 elsewhere. `width` is the band's thickness in
 * model units (kept between 1 and 8 pixels), so it looks the same on a close-up and a long shot.
 */
fn depthRim(s: Surface, width: f32, minGap: f32) -> f32 {
  let n = (frame.view * vec4f(s.normal, 0.0)).xy;
  let len = length(n);
  if (len < 1e-4) {
    return 0.0;
  }
  let px = clamp(width * pixelsPerUnit(s.viewDepth), 1.0, 8.0);
  // Clip y points up, screen uv y points down.
  let offset = vec2f(n.x, -n.y) / len * px / frame.viewport.xy;
  let behind = sceneDepthAt(s.screenUv + offset) - s.viewDepth;
  return smoothstep(minGap * 0.5, minGap, behind);
}

/**
 * 1 where something lies at least `minGap` model units in front of the surface at `offset`
 * screen pixels from this point (0 = nothing there): a cheap cast shadow of whatever is just
 * in front, such as bangs onto the forehead when `offset` points towards the light.
 */
fn screenOccluded(s: Surface, offset: vec2f, minGap: f32) -> f32 {
  let ahead = s.viewDepth - sceneDepthAt(s.screenUv + offset / frame.viewport.xy);
  return smoothstep(minGap * 0.5, minGap, ahead);
}

/**
 * Shadow of the hair onto what is behind it (the bangs onto the forehead): 1 where hair lies
 * at least `minGap` model units in front of this point, `width` model units away on screen
 * towards the light. The pre-pass tells hair from the rest, so a nose or a hand never counts.
 */
fn hairShadow(s: Surface, width: f32, minGap: f32) -> f32 {
  let dir = screenLightDir(s);
  let px = clamp(width * pixelsPerUnit(s.viewDepth), 2.0, 24.0);
  let uv = s.screenUv + dir * px / frame.viewport.xy;
  if (scenePartAt(uv) != PART_HAIR) {
    return 0.0;
  }
  return smoothstep(minGap * 0.5, minGap, s.viewDepth - sceneDepthAt(uv));
}

/** The light direction on screen (pixels, y down), unit length; (0, 0) when the light points straight at or away from the camera. */
fn screenLightDir(s: Surface) -> vec2f {
  let l = (frame.view * vec4f(s.lightDir, 0.0)).xy;
  let len = length(l);
  if (len < 1e-4) {
    return vec2f(0.0);
  }
  return vec2f(l.x, -l.y) / len;
}

/**
 * Colour ramp: `light` 0 (shadow) … 1 (lit) along the row, `row` picks the material kind
 * (generated ramps: RAMP_CLOTH, RAMP_SKIN, RAMP_HAIR, RAMP_DETAIL).
 */
fn sampleRamp(light: f32, row: f32) -> vec3f {
  let rows = f32(textureDimensions(rampTex).y);
  return textureSampleLevel(rampTex, clampSampler, vec2f(clamp(light, 0.005, 0.995), (row + 0.5) / rows), 0.0).rgb;
}

const RAMP_CLOTH = 0.0;
const RAMP_SKIN = 1.0;
const RAMP_HAIR = 2.0;
const RAMP_DETAIL = 3.0;

const PART_OTHER = 0u;
const PART_FACE = 1u;         // face skin
const PART_FACE_DETAIL = 2u;  // eyes, brows, lashes, mouth, teeth
const PART_HAIR = 3u;
const PART_SKIN = 4u;

/** Toon ramp at any N·L (for shaders that light with their own term). */
fn sampleToon(lambert: f32) -> vec3f {
  let c = textureSampleLevel(toonTex, clampSampler, vec2f(0.0, 0.5 - lambert * 0.5), 0.0).rgb;
  return texAddFactor(texMulFactor(c, material.toonMul), material.toonAdd);
}

/** Gathers the Surface. Samples textures, so it must be called first thing in the fragment stage. */
fn surface(in: VsOut) -> Surface {
  var s: Surface;
  let n = normalize(in.normal);
  s.worldPos = in.worldPos;
  s.normal = n;
  s.viewDir = normalize(frame.cameraPos.xyz - in.worldPos);
  s.lightDir = -normalize(frame.lightDir.xyz);
  s.lightColor = frame.lightColor.rgb;
  s.uv = in.uv;
  s.uv1 = in.uv1;
  s.lambert = dot(n, s.lightDir);

  let texColor = textureSample(baseTex, wrapSampler, in.uv);
  let viewN = (frame.view * vec4f(n, 0.0)).xyz;
  let sphereColor = textureSample(sphereTex, clampSampler, vec2f(viewN.x * 0.5 + 0.5, 0.5 - viewN.y * 0.5)).rgb;
  let subColor = textureSample(sphereTex, wrapSampler, in.uv1).rgb;
  let toonColor = textureSample(toonTex, clampSampler, vec2f(0.0, 0.5 - s.lambert * 0.5)).rgb;
  let toonBottom = textureSample(toonTex, clampSampler, vec2f(0.0, 1.0)).rgb;
  s.lightMap = textureSample(lightTex, wrapSampler, in.uv);
  s.faceMap = textureSample(faceTex, wrapSampler, in.uv);
  let aux = u32(material.info.z);
  s.hasRamp = (aux & 1u) != 0u;
  s.hasLightMap = (aux & 2u) != 0u;
  s.hasFaceMap = (aux & 4u) != 0u;

  s.texture = vec4f(1.0);
  if (material.modes.x != 0u) {
    s.texture = vec4f(texAddFactor(texMulFactor(texColor.rgb, material.texMul), material.texAdd), texColor.a);
  }
  s.sphereMode = material.modes.y;
  s.sphere = texAddFactor(texMulFactor(select(sphereColor, subColor, s.sphereMode == 3u), material.sphereMul), material.sphereAdd);
  let hasToon = material.modes.z == 1u;
  s.toon = select(vec3f(1.0), texAddFactor(texMulFactor(toonColor, material.toonMul), material.toonAdd), hasToon);
  s.toonShadow = select(vec3f(1.0), texAddFactor(texMulFactor(toonBottom, material.toonMul), material.toonAdd), hasToon);

  s.diffuse = material.diffuse.rgb;
  s.ambient = material.ambient.rgb;
  s.specular = material.specular.rgb;
  s.shininess = material.specular.w;
  s.alpha = material.diffuse.a * s.texture.a;
  s.accessory = material.modes.z == 2u;
  s.selfShadow = frame.shadowParams.x > 0.5 && material.modes.w != 0u;
  s.shadow = 1.0;
  if (s.selfShadow) {
    s.shadow = shadowVisibility(in.worldPos, n);
  }
  s.materialIndex = u32(material.info.x);
  s.part = u32(material.info.y);
  s.hasHead = material.headPos.w > 0.5;
  s.headPos = material.headPos.xyz;
  s.headForward = material.headForward.xyz;
  s.headUp = material.headUp.xyz;
  s.time = frame.viewport.w;
  s.screenUv = in.clip.xy / frame.viewport.xy;
  s.viewDepth = (frame.view * vec4f(in.worldPos, 1.0)).z;
  return s;
}

/**
 * Light the extra lights add at this point, as a multiplier of the surface colour. A fill
 * light wraps round the surface (lighting a little past the terminator); a rim light counts
 * where the surface faces it and is seen edge-on, so a light from behind draws a bright
 * edge and catches hair.
 */
fn extraLighting(s: Surface) -> vec3f {
  var sum = vec3f(0.0);
  for (var i = 0; i < 3; i++) {
    let d = frame.extraLights[i * 2];
    let c = frame.extraLights[i * 2 + 1];
    if (c.w < 0.5) {
      continue;
    }
    let l = -normalize(d.xyz);
    let nl = dot(s.normal, l);
    var k: f32;
    if (c.w < 1.5) {
      k = clamp((nl + 0.5) / 1.5, 0.0, 1.0);
    } else {
      let edge = pow(1.0 - max(dot(s.normal, s.viewDir), 0.0), 1.5);
      k = clamp(nl, 0.0, 1.0) * edge * 2.0;
    }
    sum += c.rgb * (k * d.w);
  }
  return sum;
}

/** MMD's own shading (see the top of this file), plus whatever the extra lights add. */
fn mmdShade(s: Surface) -> vec4f {
  var color = clamp(s.diffuse * s.lightColor + s.ambient, vec3f(0.0), vec3f(1.0));
  // What an accessory keeps in shadow; its lit side adds the diffuse term.
  var shade = color;
  if (s.accessory) {
    color += max(s.lambert, 0.0) * s.diffuse * s.lightColor;
  }
  color *= s.texture.rgb;
  shade *= s.texture.rgb;

  if (s.sphereMode == 2u) {
    color += s.sphere;
    shade += s.sphere;
  } else if (s.sphereMode != 0u) {
    color *= s.sphere;
    shade *= s.sphere;
  }

  var specular = vec3f(0.0);
  if (s.shininess > 0.0) {
    let h = normalize(s.viewDir + s.lightDir);
    specular = pow(max(dot(h, s.normal), 0.0), s.shininess) * s.specular * s.lightColor;
  }

  let extra = color * extraLighting(s);
  if (s.accessory) {
    return vec4f(mix(shade, color + specular, s.shadow) + extra, s.alpha);
  }
  // Models always use the toon path; "no toon" means a white toon texture.
  if (s.selfShadow) {
    let comp = min(clamp(s.lambert * 3.0, 0.0, 1.0), s.shadow);
    return vec4f(mix(color * s.toonShadow, color + specular, comp) + extra, s.alpha);
  }
  return vec4f(color * s.toon + specular + extra, s.alpha);
}

/** Colour, plus per-pixel data for post-processing shaders (the auxiliary target). */
struct FragOut {
  @location(0) color: vec4f,
  @location(1) aux: vec4f,
};

/**
 * What goes to the auxiliary target: r = how much this pixel is skin (0–1), g = how thin the
 * surface is there (0–1, from the light map), b = free. A custom material shader may assign it
 * inside shade(); it starts out as defaultAux(s). Post-processing reads it with sceneAux(uv).
 */
var<private> auxOut: vec4f;

/** 0–1: how much this surface is skin — by the material's name, else by how skin-like its colour is. */
fn skinAmount(s: Surface) -> f32 {
  if (s.part == PART_FACE || s.part == PART_SKIN) {
    return 1.0;
  }
  if (s.part == PART_FACE_DETAIL || s.part == PART_HAIR || s.accessory) {
    return 0.0;
  }
  // Bright, warm, not very saturated (bodies often share a texture with the clothes).
  let c = s.texture.rgb;
  let hi = max(c.r, max(c.g, c.b));
  let lo = min(c.r, min(c.g, c.b));
  let warm = smoothstep(0.03, 0.12, c.r - c.b) * smoothstep(0.0, 0.05, c.r - c.g) * smoothstep(-0.04, 0.02, c.g - c.b);
  return warm * smoothstep(0.55, 0.8, hi) * (1.0 - smoothstep(0.32, 0.55, (hi - lo) / max(hi, 1e-3)));
}

/** Thinness of the surface (1 = thin, light shines through) from a generated light map; 0 without one. */
fn thinness(s: Surface) -> f32 {
  return select(0.0, s.lightMap.g, s.hasLightMap);
}

fn defaultAux(s: Surface) -> vec4f {
  return vec4f(skinAmount(s), thinness(s), 0.0, 1.0);
}

@fragment
fn fsMain(in: VsOut) -> FragOut {
  let s = surface(in);
  if (s.alpha < 0.001) {
    discard;
  }
  let color = mmdShade(s);
  return FragOut(color, vec4f(defaultAux(s).rgb, color.a));
}

/**
 * 1 = lit, 0 = in shadow. Picks the cascade covering this view depth, then 3×3 PCF on top of
 * the hardware 2×2 comparison filter.
 */
fn shadowVisibility(worldPos: vec3f, n: vec3f) -> f32 {
  let depth = (frame.view * vec4f(worldPos, 1.0)).z;
  var c = CASCADES;
  for (var i = CASCADES - 1; i >= 0; i--) {
    if (depth < frame.cascadeSplits[i]) {
      c = i;
    }
  }
  if (c == CASCADES) {
    return 1.0;
  }
  // Offset along the normal by about a texel so surfaces don't shadow themselves.
  let p = frame.lightViewProj[c] * vec4f(worldPos + n * frame.cascadeTexel[c] * 1.5, 1.0);
  let uv = vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) || p.z > 1.0) {
    return 1.0;
  }
  let texel = frame.shadowParams.y;
  let depthRef = p.z - frame.cascadeBias[c];
  var sum = 0.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      sum += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2f(f32(x), f32(y)) * texel, c, depthRef);
    }
  }
  return sum / 9.0;
}

// ---- Edge (outline): back faces pushed out along the screen-space normal ----

/** The outline's vertex, with the usual surface data so a custom shader can colour it. */
@vertex
fn vsEdge(in: VsIn) -> VsOut {
  var out = surfaceVertex(in);
  let viewN = (frame.view * vec4f(in.normal, 0.0)).xy;
  let len = length(viewN);
  if (len > 1e-6) {
    // Edge size is in (device-independent) pixels, like MMD.
    let px = material.edgeParams.x * in.edgeScale * frame.viewport.z;
    out.clip = vec4f(out.clip.xy + (viewN / len) * (2.0 / frame.viewport.xy) * px * out.clip.w, out.clip.zw);
  }
  return out;
}

@fragment
fn fsEdge(in: VsOut) -> @location(0) vec4f {
  return material.edgeColor;
}

// ---- Ground shadow: the mesh flattened onto y = 0 along the light, in one flat colour ----

@vertex
fn vsGround(in: VsIn) -> @builtin(position) vec4f {
  let l = frame.lightDir.xyz;
  // Light from below or level casts nothing; a degenerate position drops the triangle.
  if (l.y > -1e-3 || frame.groundShadow.a <= 0.0) {
    return vec4f(0.0, 0.0, 2.0, 1.0);
  }
  let p = in.pos - l * (in.pos.y / l.y);
  return frame.viewProj * vec4f(p.x, 0.0, p.z, 1.0);
}

@fragment
fn fsGround() -> @location(0) vec4f {
  return frame.groundShadow;
}
