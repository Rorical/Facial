// 后期 shader：皮肤次表面散射（屏幕空间）
//
// 光进入皮肤后会在里面散开一小段距离再出来，红光走得最远。这里在画面上对皮肤像素做一次
// 按颜色分开的模糊：红色模糊得最宽，蓝色最窄，于是明暗交界变软并带一圈红晕，毛孔级的细节被抹平。
// 模糊半径按场景里的实际尺寸算（离镜头越远，屏幕上的半径越小），不会跨过深度差很大的边缘，
// 也不会把衣服、头发的颜色混进皮肤。
// 哪些像素是皮肤来自 sceneAux(uv).r：由材质着色写入（按材质名和颜色判断）。
// 和材质 shader「写实皮肤」一起用效果最好；单独配卡通着色也可以，只是更淡。

const RADIUS = 0.16;                               // 红光的散射半径，模型单位（1 单位约 8 厘米）
const CHANNEL_RADIUS = vec3f(1.0, 0.42, 0.24);     // 红 / 绿 / 蓝的相对半径
const STRENGTH = 0.85;
const TAPS = 28;
const GOLDEN_ANGLE = 2.39996323;

fn effect(uv: vec2f, color: vec3f) -> vec3f {
  let skin = sceneAux(uv).r;
  if (skin < 0.02) {
    return color;
  }
  let depth = sceneDepth(uv);
  let radius = RADIUS * pixelsPerUnit(depth);      // 屏幕上的像素半径
  if (radius < 0.75) {
    return color;
  }
  var sum = color;
  var weight = vec3f(1.0);
  for (var i = 0; i < TAPS; i++) {
    let r = sqrt((f32(i) + 0.5) / f32(TAPS));      // 0–1，在圆盘上均匀分布
    let a = f32(i) * GOLDEN_ANGLE;
    let at = uv + vec2f(cos(a), sin(a)) * r * radius / fx.resolution;
    // 每个通道按自己的半径衰减（高斯）
    let falloff = exp(-(r * r) / (2.0 * CHANNEL_RADIUS * CHANNEL_RADIUS * 0.2));
    // 只收集同一块皮肤上的光：别的材质、前后隔得远的表面不算
    let same = sceneAux(at).r * (1.0 - smoothstep(RADIUS * 1.5, RADIUS * 4.0, abs(sceneDepth(at) - depth)));
    let w = falloff * same;
    sum += sceneColor(at) * w;
    weight += w;
  }
  return mix(color, sum / weight, skin * STRENGTH);
}
