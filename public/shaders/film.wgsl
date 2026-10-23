// 后期 shader 示例：胶片感（暗角 + 颗粒 + 轻微调色）
//
// 后期 shader 要定义 fn effect(uv: vec2f, color: vec3f) -> vec3f。
// uv 是画面坐标（0–1，左上角为原点），color 是这个位置原本的颜色。
// 可用：sceneColor(uv) 取画面任意位置的颜色，sceneDepth(uv) 取到镜头的距离，
//       fx.resolution 画面像素尺寸，fx.time 时间线上的秒数。

const VIGNETTE = 0.45;
const GRAIN = 0.035;

fn hash(p: vec2f) -> f32 {
  return fract(sin(dot(p, vec2f(12.9898, 78.233))) * 43758.5453);
}

fn effect(uv: vec2f, color: vec3f) -> vec3f {
  var c = color;
  // 暗部偏青、亮部偏暖
  let luma = dot(c, vec3f(0.299, 0.587, 0.114));
  c = mix(c * vec3f(0.94, 1.0, 1.05), c * vec3f(1.05, 1.0, 0.94), smoothstep(0.2, 0.8, luma));
  // 暗角
  let d = length((uv - 0.5) * vec2f(fx.resolution.x / fx.resolution.y, 1.0));
  c *= 1.0 - VIGNETTE * smoothstep(0.35, 0.95, d);
  // 每帧变化的颗粒
  c += (hash(uv * fx.resolution + fx.time * 60.0) - 0.5) * GRAIN;
  return c;
}
