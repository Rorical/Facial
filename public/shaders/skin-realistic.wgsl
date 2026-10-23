// 材质 shader：写实皮肤
//
// 不用卡通的明暗分界，整个模型改成柔和的渐变光照；皮肤另外加上：
//   · 红光比绿光、蓝光绕得更远：明暗过渡处自然偏红（预积分皮肤的近似）
//   · 两层高光：一层窄而亮（表面油脂），一层宽而弱（皮肤本身），掠射角更强
//   · 薄处逆光透红（需要辅助贴图里的厚度，自动生成）
// 建议同时加载后期 shader「皮肤次表面散射」，把皮肤上的光再柔化一次。
// 适合写实或半写实的模型；贴图是平涂卡通风格的模型用它会显得像蜡像。

const SHADOW_COLOR = vec3f(0.64, 0.64, 0.69);       // 背光处的颜色倍率（衣服、头发等）
const SKIN_SHADOW_COLOR = vec3f(0.88, 0.68, 0.62);  // 皮肤背光处：光在皮肤里散射后偏暖红，而不是发灰
const OCCLUSION = 0.18;                            // 环境遮蔽最多压暗多少
const SKIN_WRAP = vec3f(0.55, 0.28, 0.16);         // 红 / 绿 / 蓝各绕过明暗线多远
const SPECULAR_STRENGTH = 0.55;
const SHEEN = 0.28;                                // 宽的那层高光相对窄的那层的比例
const GLOW_COLOR = vec3f(1.00, 0.30, 0.18);
const GLOW_STRENGTH = 0.8;

fn shade(s: Surface) -> vec4f {
  let skin = skinAmount(s);
  // 投影的边缘放软一些
  let shadow = smoothstep(0.0, 1.0, s.shadow);

  // 漫反射：每个颜色通道各自的半兰伯特
  let wrap = mix(vec3f(0.25), SKIN_WRAP, skin);
  let light = clamp((vec3f(s.lambert) + wrap) / (vec3f(1.0) + wrap), vec3f(0.0), vec3f(1.0)) * shadow;
  var albedo = clamp(s.diffuse * s.lightColor + s.ambient, vec3f(0.0), vec3f(1.0)) * s.texture.rgb;
  if (s.sphereMode == 1u || s.sphereMode == 3u) {
    albedo *= s.sphere;
  }
  // 非皮肤的暗部沿用模型 toon 贴图的暗色（头发、衣服的色调是靠它定的）
  var color = albedo * mix(mix(SHADOW_COLOR * mix(vec3f(1.0), s.toonShadow, 0.85), SKIN_SHADOW_COLOR, skin), vec3f(1.0), light);
  // 环境遮蔽（有光照图时）
  if (s.hasLightMap) {
    color *= mix(1.0 - OCCLUSION, 1.0, s.lightMap.r);
  }
  if (s.sphereMode == 2u) {
    color += s.sphere * mix(0.35, 1.0, light.g);
  }

  let h = normalize(s.viewDir + s.lightDir);
  let nh = max(dot(s.normal, h), 0.0);
  let nv = max(dot(s.normal, s.viewDir), 0.0);
  let lit = clamp(s.lambert, 0.0, 1.0) * shadow;
  if (skin > 0.0) {
    // 菲涅耳：正面几乎不反光，掠射角明显
    let fresnel = 0.028 + 0.972 * pow(1.0 - max(dot(s.viewDir, h), 0.0), 5.0);
    let lobes = pow(nh, 64.0) * 1.0 + pow(nh, 10.0) * SHEEN;
    color += s.lightColor * lobes * (0.35 + fresnel * 6.0) * SPECULAR_STRENGTH * lit * skin * (0.4 + 0.6 * nv);

    let through = pow(clamp(dot(s.viewDir, -normalize(s.lightDir + s.normal * 0.35)), 0.0, 1.0), 3.0);
    color += GLOW_COLOR * s.lightColor * through * thinness(s) * GLOW_STRENGTH * skin;
  } else if (s.shininess > 0.0) {
    color += pow(nh, s.shininess) * s.specular * s.lightColor * lit;
  }
  color += albedo * extraLighting(s);
  return vec4f(color, s.alpha);
}
