// 材质 shader：卡通皮肤通透感（次表面散射的卡通做法）
//
// 在 MMD 原有着色上，只对皮肤加三样东西，其余材质不变：
//   · 明暗交界处一条偏红的过渡带：光在皮肤里散开后从暗部边缘透出来的颜色。只有紧贴交界线
//     的一小段，不会把整个背光面染红
//   · 逆光时薄的地方（耳朵、手指、鼻翼）透出暖光
//   · 薄的地方在暗部也略带血色
// 哪里是皮肤：按材质名和颜色判断（skinAmount）。哪里薄：来自辅助贴图里光照图的绿色通道，
// 第一次加载本 shader 时会自动生成；没有的话只有第一样。

const BAND_COLOR = vec3f(1.08, 0.70, 0.62);        // 交界带的颜色倍率（略亮、偏红）
const BAND_STRENGTH = 0.6;
const BAND_INTO_SHADOW = 0.2;                      // 交界带往暗部延伸多远（N·L 的范围）
const BAND_INTO_LIGHT = 0.1;                       // 往亮部延伸多远
const GLOW_COLOR = vec3f(1.00, 0.36, 0.24);        // 透光的颜色
const GLOW_STRENGTH = 0.9;
const GLOW_FOCUS = 3.0;                            // 越大越只在正逆光时出现
const FLUSH_STRENGTH = 0.12;                       // 薄处在暗部的血色

fn shade(s: Surface) -> vec4f {
  let base = mmdShade(s);
  let skin = skinAmount(s);
  if (skin <= 0.0) {
    return base;
  }
  var color = base.rgb;

  // 交界带：以明暗交界线（N·L ≈ 0）为中心，往暗部多延伸一些，往亮部少一些
  let turn = smoothstep(-BAND_INTO_SHADOW, -0.02, s.lambert) * (1.0 - smoothstep(-0.02, BAND_INTO_LIGHT, s.lambert));
  // 投影的边缘也带一点；边缘很窄，全强度会变成一条红线，所以减半
  let edge = s.shadow * (1.0 - s.shadow) * 4.0;
  var band = clamp(max(turn * smoothstep(0.0, 0.4, s.shadow), edge * 0.45), 0.0, 1.0);
  // 只有暗部真的变暗的地方才有这条带：脸的 toon 贴图常常接近全白，那里本来就看不到明暗线，
  // 硬画一条红带会像鼻梁上的一道疤
  band *= smoothstep(0.03, 0.25, 1.0 - dot(s.toonShadow, vec3f(0.299, 0.587, 0.114)));
  color = mix(color, color * BAND_COLOR, band * BAND_STRENGTH * skin);

  // 逆光透光：光从表面背后照来，视线正对着光
  let thin = smoothstep(0.3, 0.9, thinness(s));
  let through = pow(clamp(dot(s.viewDir, -normalize(s.lightDir + s.normal * 0.35)), 0.0, 1.0), GLOW_FOCUS);
  color += GLOW_COLOR * s.lightColor * through * thin * GLOW_STRENGTH * skin;

  // 薄处的暗部带一点血色
  let dark = 1.0 - smoothstep(-0.1, 0.3, s.lambert) * s.shadow;
  color = mix(color, color * vec3f(1.10, 0.88, 0.84), dark * thin * FLUSH_STRENGTH * skin);

  return vec4f(color, base.a);
}
