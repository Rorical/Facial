// 材质 shader：星铁（崩坏：星穹铁道）风格的三渲二卡通着色
//
// 做了什么：
//   · 明暗只有两档，交界是一条清晰的线，亮部靠近交界处还有一道略暗的半影；暗部不是变灰，
//     而是换一种更饱和的颜色（皮肤偏暖红、布料偏冷紫、头发偏紫红），交界处的皮肤带一点红润
//   · 脸不用凹凸的脸部法线算明暗，而是跟着头的朝向：光从哪一侧来，阴影就整齐地落在另一侧，
//     不会出现鼻子、脸颊上的碎阴影；刘海在额头上投下一条跟着光走的影子
//   · 眼睛、眉毛透过刘海显示（hairTransparency），并且只受很浅的阴影，保持清晰
//   · 轮廓光画在剪影处（按屏幕上的深度差），粗细固定，不会在平缓的面上糊成一片
//   · 描边不是单一的黑，而是用本色压暗、提饱和后的颜色（outline）
//   · 头发有一圈「天使环」高光；金属件用模型自带的高光贴图，再压出一块清晰的亮斑；
//     布料有一点柔和的高光
//   · 光照图里的环境遮蔽让衣褶在亮部也有深浅，在暗部更容易进入阴影
//   · 设置里的补光、发丝光照常起作用（extraLighting）
//
// 辅助贴图（模型的「辅助贴图 → 管理」里生成或加载；第一次加载本 shader 时会自动生成）：
//   · 色条：亮部到暗部换成什么颜色。有色条时下面的 *_SHADOW 颜色不再使用
//   · 光照图：红色通道是环境遮蔽
//   · 脸部阴影图：脸上每个位置在光转到多大角度时变暗，明暗线随光照平滑移动，并带鼻子的遮挡
// 没有对应贴图时退回用计算近似。游戏原版的贴图是手绘的，效果会比自动生成的好；
// 材质属于脸 / 五官 / 头发 / 皮肤，是按材质名字猜的（s.part）。

// ---- 整体亮度 ----
const BRIGHTNESS = 1.0;                            // 整体亮度
const LIT_TINT = vec3f(1.00, 0.995, 0.99);         // 亮部的颜色倍率（白衣服保持白）
const SKIN_LIT_TINT = vec3f(1.00, 0.96, 0.94);     // 皮肤亮部：带一点暖色，不是纸一样的白
const HAIR_LIT_TINT = vec3f(0.96, 0.94, 0.97);     // 头发亮部：略压一点，头顶不至于一片白
const SHADOW_DEPTH = 0.94;                         // 暗部再压暗多少（1 = 不压）

// ---- 明暗 ----
const SHADOW_EDGE = 0.035;                         // 明暗交界的柔和程度，越小越硬
const SHADOW_OFFSET = -0.05;                       // 交界线的位置：越大暗部越多，负数让亮部更多
const HALF_SHADOW_WIDTH = 0.14;                    // 半影的宽度（N·L 的范围），0 = 没有半影；脸上没有半影
const HALF_SHADOW_MIX = 0.3;                       // 半影有多接近暗部的颜色（0 = 和亮部一样）
const CAST_SHADOW = 1.0;                           // 身体上的投影（头发、手臂挡出的影子）保留多少：0 = 不要投影
const SKIN_SHADOW = vec3f(1.00, 0.87, 0.88);       // 皮肤暗部的颜色倍率
const SKIN_BLUSH = vec3f(1.00, 0.62, 0.56);        // 皮肤明暗交界处的红润
const CLOTH_SHADOW = vec3f(0.74, 0.73, 0.86);      // 布料暗部（略偏冷）
const HAIR_SHADOW = vec3f(0.74, 0.60, 0.80);       // 头发暗部
const SHADOW_SATURATION = 1.1;                     // 布料、头发暗部的饱和度（皮肤不加）

const AO_STRENGTH = 0.55;                          // 环境遮蔽把明暗交界往亮部推多少
const AO_IN_LIGHT = 0.2;                           // 亮部里按环境遮蔽压暗多少（衣褶的深浅）
const AO_CHANNEL = 0;                              // 光照图里哪个通道是环境遮蔽：生成的图是 0（红）；游戏原版光照图一般是 1（绿）

// ---- 脸 ----
const FACE_CENTER_UP = 0.9;                        // 头部中心在「頭」骨骼上方多远
const FACE_LIT_BIAS = 0.12;                        // 越大脸越容易受光
const FACE_CAST_SHADOW = 0.0;                      // 头发、手投在脸上的「投影贴图」影子保留多少（0–1）；刘海的影子另外画
const HAIR_SHADOW_WIDTH = 0.35;                    // 刘海影子离头发剪影多远（模型单位，1 ≈ 8 厘米）
const HAIR_SHADOW_GAP = 0.6;                       // 头发要比脸近多少才算挡住（模型单位）
const HAIR_SHADOW_STRENGTH = 1.0;                  // 刘海影子有多深（1 = 和暗部一样）
const EYES_THROUGH_HAIR = 0.55;                    // 眼睛、眉毛透过刘海的程度（0 = 不透）

// ---- 轮廓光、高光、描边 ----
const RIM_WIDTH = 0.1;                             // 轮廓光的宽度（模型单位，换算成 1–8 像素）
const RIM_GAP = 0.8;                               // 剪影外要比这里远多少才算边缘（模型单位）
const RIM_STRENGTH = 0.45;
const RIM_DARK_SIDE = 0.3;                         // 背光一侧的轮廓光保留多少
const RIM_WHITENESS = 0.6;                         // 轮廓光偏白还是偏本色
const HAIR_RING_HEIGHT = 0.60;                     // 天使环的位置
const HAIR_RING_WIDTH = 0.07;
const HAIR_RING_STRENGTH = 0.24;
const SPHERE_GLOSS = 0.6;                          // 高光贴图的强度（MMD 是 1）
const METAL_GLOSS = 0.25;                          // 高光贴图亮过这个值的材质算金属 / 漆面，加一个跟着光走的小亮斑
const METAL_SPOT_STRENGTH = 0.35;
const CLOTH_SPEC_POWER = 32.0;                     // 布料高光的集中程度
const CLOTH_SPEC_STRENGTH = 0.04;                  // 布料高光的强度（很淡；0 = 不要）
const OUTLINE_CLOTH = vec3f(0.42, 0.40, 0.55);     // 描边 = 本色 × 这些倍率（压暗、偏冷）
const OUTLINE_SKIN = vec3f(0.78, 0.45, 0.42);      //   皮肤描边偏红褐
const OUTLINE_HAIR = vec3f(0.55, 0.38, 0.60);
const OUTLINE_MODEL_COLOR = 0.25;                  // 混入多少模型自己设定的描边色

const LUMA = vec3f(0.299, 0.587, 0.114);

fn saturateColor(c: vec3f, amount: f32) -> vec3f {
  return max(mix(vec3f(dot(c, LUMA)), c, amount), vec3f(0.0));
}

/** 这个颜色有多像肤色（0–1）：明亮、偏暖、不太鲜艳。用于没标成皮肤的材质（身体和衣服共用贴图时）。 */
fn skinTone(c: vec3f) -> f32 {
  let hi = max(c.r, max(c.g, c.b));
  let lo = min(c.r, min(c.g, c.b));
  let warm = smoothstep(0.03, 0.12, c.r - c.b) * smoothstep(0.0, 0.05, c.r - c.g) * smoothstep(-0.04, 0.02, c.g - c.b);
  let bright = smoothstep(0.55, 0.8, hi);
  let soft = 1.0 - smoothstep(0.32, 0.55, (hi - lo) / max(hi, 1e-3));
  return warm * bright * soft;
}

/** 材质本色（与 MMD 相同）。 */
fn baseColor(s: Surface) -> vec3f {
  var albedo = clamp(s.diffuse * s.lightColor + s.ambient, vec3f(0.0), vec3f(1.0)) * s.texture.rgb;
  if (s.sphereMode == 1u || s.sphereMode == 3u) {
    albedo *= s.sphere;
  }
  return albedo;
}

/** 这块材质有多像皮肤：名字说了算，否则看颜色。 */
fn skinOf(s: Surface, albedo: vec3f) -> f32 {
  if (s.part == PART_FACE || s.part == PART_SKIN) {
    return 1.0;
  }
  if (s.part == PART_FACE_DETAIL || s.part == PART_HAIR) {
    return 0.0;
  }
  return skinTone(albedo);
}

/** 脸的受光：把脸当成一个绕头部中心的圆柱，只看光在水平方向从哪一侧来。 */
fn faceLight(s: Surface) -> f32 {
  let up = normalize(s.headUp);
  var toLight = s.lightDir - up * dot(s.lightDir, up);
  if (length(toLight) < 1e-3) {
    toLight = s.headForward;
  }
  toLight = normalize(toLight);
  let center = s.headPos + up * FACE_CENTER_UP;
  var around = s.worldPos - center;
  around -= up * dot(around, up);
  // 混进一点真实法线，保留鼻梁、下巴的转折
  var n = s.normal - up * dot(s.normal, up) * 0.8;
  n = normalize(mix(normalize(around + vec3f(1e-5)), normalize(n + vec3f(1e-5)), 0.25));
  return dot(n, toLight) + FACE_LIT_BIAS;
}

/**
 * 用脸部阴影图算脸的受光：图里存的是「光向右 / 向左最多转多少，这个位置还亮着」，
 * 和当前光相对头部正前方转过的角度比较。
 */
fn faceMapLight(s: Surface) -> f32 {
  let up = normalize(s.headUp);
  var toLight = s.lightDir - up * dot(s.lightDir, up);
  if (length(toLight) < 1e-3) {
    return 1.0;
  }
  toLight = normalize(toLight);
  let forward = normalize(s.headForward - up * dot(s.headForward, up));
  let right = cross(up, forward);
  let turned = acos(clamp(dot(forward, toLight), -1.0, 1.0)) / 3.14159265;   // 0–1 = 0–180°
  let limit = select(s.faceMap.g, s.faceMap.r, dot(right, toLight) >= 0.0);
  // 换算成和 faceLight 相近的量纲，两者共用同一个交界宽度
  return (limit - turned) * 2.0 + FACE_LIT_BIAS * 0.5;
}

fn shade(s: Surface) -> vec4f {
  let isFace = (s.part == PART_FACE || s.part == PART_FACE_DETAIL) && s.hasHead;
  let isDetail = s.part == PART_FACE_DETAIL;
  let isHair = s.part == PART_HAIR;
  let albedo = baseColor(s);
  let skin = skinOf(s, albedo);

  // 环境遮蔽：1 = 敞开，0 = 缝隙深处
  var ao = 1.0;
  if (s.hasLightMap) {
    ao = s.lightMap[AO_CHANNEL];
  }
  let crease = (1.0 - ao) * AO_STRENGTH * select(1.0, 0.5, isHair);

  // 受光程度（N·L 的量纲）：正为亮部，负为暗部
  var facing: f32;
  if (isFace) {
    facing = faceLight(s);
    if (s.hasFaceMap) {
      facing = faceMapLight(s);
    }
  } else {
    facing = s.lambert - SHADOW_OFFSET - crease;
  }
  // 受光：0 = 暗部，1 = 亮部；半影：亮部里离交界近的一段
  var lit = smoothstep(-SHADOW_EDGE, SHADOW_EDGE, facing);
  let fullLit = smoothstep(-SHADOW_EDGE, SHADOW_EDGE, facing - HALF_SHADOW_WIDTH);
  // 只有被完全挡住的地方才算投影，半影不算，避免影子边缘外扩
  let occlusion = mix(1.0, smoothstep(0.1, 0.3, s.shadow), CAST_SHADOW);
  if (isFace) {
    lit = min(lit, mix(1.0, occlusion, FACE_CAST_SHADOW));
    // 刘海的影子：往光的方向看，前面是不是头发
    let bangs = hairShadow(s, HAIR_SHADOW_WIDTH, HAIR_SHADOW_GAP) * HAIR_SHADOW_STRENGTH * select(1.0, 0.5, isDetail);
    lit = min(lit, 1.0 - bangs);
  } else {
    lit = min(lit, occlusion);
  }

  // 暗部换色
  var tint = mix(CLOTH_SHADOW, SKIN_SHADOW, skin);
  if (isHair) {
    tint = HAIR_SHADOW;
  }
  if (isDetail) {
    tint = mix(tint, vec3f(1.0), 0.6);   // 五官只要很浅的阴影
  }
  var color: vec3f;
  if (s.hasRamp) {
    // 色条：每行一种材质，横向从暗到亮
    var ramp = mix(sampleRamp(lit, RAMP_CLOTH), sampleRamp(lit, RAMP_SKIN), skin);
    if (isHair) {
      ramp = sampleRamp(lit, RAMP_HAIR);
    }
    if (isDetail) {
      ramp = sampleRamp(lit, RAMP_DETAIL);
    }
    color = mix(saturateColor(albedo * ramp, mix(SHADOW_SATURATION, 1.0, skin)), albedo * ramp, lit);
  } else {
    color = mix(saturateColor(albedo * tint, mix(SHADOW_SATURATION, 1.0, skin)), albedo, lit);
    // 皮肤的明暗交界带一点红
    let edge = lit * (1.0 - lit) * 4.0;
    color = mix(color, color * SKIN_BLUSH, edge * skin * 0.4);
  }
  // 半影：亮部靠近交界的一段略暗，带一点暗部的颜色（脸要干净的一条线，不加）
  let halfTint = mix(vec3f(1.0), tint, HALF_SHADOW_MIX * select(1.0, 0.6, isHair));
  color = mix(color, color * halfTint, (1.0 - fullLit) * lit * select(1.0, 0.0, isFace));

  // 亮部、暗部各自的整体倍率；亮部里按环境遮蔽压出衣褶
  var litTint = mix(LIT_TINT, SKIN_LIT_TINT, skin);
  if (isHair) {
    litTint = HAIR_LIT_TINT;
  }
  color *= mix(vec3f(SHADOW_DEPTH), litTint, lit) * BRIGHTNESS;
  color *= mix(1.0, ao, AO_IN_LIGHT * lit * select(1.0, 0.4, skin > 0.5 || isHair));

  let h = normalize(s.viewDir + s.lightDir);
  let spot = pow(max(dot(h, s.normal), 0.0), max(s.shininess, 1.0));
  // 加算的高光贴图：受光面完整，暗部减弱。贴图亮的材质（金属、漆面）再加一个跟着光走的小亮斑，
  // 而不是把贴图亮的地方整片提亮
  if (s.sphereMode == 2u) {
    let gloss = dot(s.sphere, LUMA);
    color += s.sphere * mix(0.3, 1.0, lit) * SPHERE_GLOSS;
    let shiny = smoothstep(METAL_GLOSS, METAL_GLOSS + 0.15, gloss);
    color += vec3f(1.0) * smoothstep(0.6, 0.75, pow(max(dot(h, s.normal), 0.0), 48.0)) * shiny * METAL_SPOT_STRENGTH * lit;
  }

  // 材质自带的高光：一块边缘清晰的亮斑
  if (s.shininess > 0.0 && dot(s.specular, LUMA) > 0.01) {
    color += s.specular * smoothstep(0.45, 0.5, spot) * lit;
  }
  // 布料：一点很淡的高光，让受光面有起伏
  if (!isHair && !isFace && skin < 0.5 && CLOTH_SPEC_STRENGTH > 0.0) {
    let soft = pow(max(dot(h, s.normal), 0.0), CLOTH_SPEC_POWER);
    color += (albedo * 0.5 + vec3f(0.5)) * smoothstep(0.55, 0.7, soft) * CLOTH_SPEC_STRENGTH * lit;
  }

  let toCamera = max(dot(s.normal, s.viewDir), 0.0);

  // 头发的天使环：绕着头顶的一圈高光，随视角移动
  if (isHair && s.hasHead) {
    let up = normalize(s.headUp);
    let height = dot(s.normal, normalize(up + s.viewDir * 0.45));
    let ring = smoothstep(HAIR_RING_HEIGHT - HAIR_RING_WIDTH, HAIR_RING_HEIGHT, height)
      * (1.0 - smoothstep(HAIR_RING_HEIGHT, HAIR_RING_HEIGHT + HAIR_RING_WIDTH, height));
    color += (albedo * 0.5 + vec3f(0.5)) * ring * HAIR_RING_STRENGTH * smoothstep(0.25, 0.6, toCamera) * mix(0.35, 1.0, lit);
  }

  // 轮廓光：剪影处一条固定粗细的亮边，受光一侧亮，背光一侧只留一点
  if (!isDetail) {
    let rim = depthRim(s, RIM_WIDTH, RIM_GAP);
    let side = mix(RIM_DARK_SIDE, 1.0, smoothstep(-0.2, 0.3, s.lambert));
    color += mix(albedo, vec3f(1.0), RIM_WHITENESS) * rim * RIM_STRENGTH * side * select(1.0, 0.5, isFace) * mix(0.5, 1.0, occlusion);
  }

  // 设置里的补光 / 发丝光
  color += albedo * extraLighting(s) * select(1.0, 0.6, isDetail);

  return vec4f(color, s.alpha);
}

/** 描边的颜色：本色压暗、提饱和，皮肤偏红褐、头发偏紫，再混一点模型自己设定的描边色。 */
fn outline(s: Surface) -> vec4f {
  let albedo = baseColor(s);
  let skin = skinOf(s, albedo);
  var tint = mix(OUTLINE_CLOTH, OUTLINE_SKIN, skin);
  if (s.part == PART_HAIR) {
    tint = OUTLINE_HAIR;
  }
  let line = saturateColor(albedo * tint, 1.3);
  return vec4f(mix(line, material.edgeColor.rgb, OUTLINE_MODEL_COLOR), material.edgeColor.a);
}

/** 眼睛、眉毛透过刘海显示的程度：只在镜头大致正对脸时。 */
fn hairTransparency(s: Surface) -> f32 {
  if (s.part != PART_FACE_DETAIL || !s.hasHead) {
    return 0.0;
  }
  let facing = dot(normalize(s.headForward), s.viewDir);
  return EYES_THROUGH_HAIR * smoothstep(0.2, 0.5, facing);
}
