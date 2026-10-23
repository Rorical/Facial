// 材质 shader 示例：边缘光
// 在 MMD 原有着色的基础上，给剪影加一圈暖色的光：按屏幕上的深度差找到轮廓，
// 所以光圈粗细固定，不会在平缓的面上糊成一片。受光的一侧更亮。
//
// 材质 shader 要定义 fn shade(s: Surface) -> vec4f，返回这个像素的颜色和不透明度。
// mmdShade(s) 是 MMD 原本的结果，可以直接在它上面改。

const RIM_COLOR = vec3f(1.0, 0.86, 0.72);
const RIM_STRENGTH = 0.7;
const RIM_WIDTH = 0.12;   // 模型单位（1 ≈ 8 厘米），换算成 1–8 像素
const RIM_GAP = 0.8;      // 轮廓外要比这里远多少才算边缘（模型单位）

fn shade(s: Surface) -> vec4f {
  let base = mmdShade(s);
  let rim = depthRim(s, RIM_WIDTH, RIM_GAP);
  // 背光、处在阴影里的一侧减弱
  let lit = smoothstep(-0.3, 0.5, s.lambert) * (0.4 + 0.6 * s.shadow);
  return vec4f(base.rgb + RIM_COLOR * rim * RIM_STRENGTH * lit, base.a);
}
