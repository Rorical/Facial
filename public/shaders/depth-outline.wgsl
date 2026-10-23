// 后期 shader 示例：深度描线
// 比较相邻像素到镜头的距离，距离突变的地方（物体轮廓、前后重叠处）画上线。

const LINE_COLOR = vec3f(0.12, 0.09, 0.14);
const LINE_WIDTH = 1.5;      // 像素
const SENSITIVITY = 0.02;    // 越小线越多

fn effect(uv: vec2f, color: vec3f) -> vec3f {
  let px = LINE_WIDTH / fx.resolution;
  let d = sceneDepth(uv);
  let dx = abs(sceneDepth(uv + vec2f(px.x, 0.0)) - sceneDepth(uv - vec2f(px.x, 0.0)));
  let dy = abs(sceneDepth(uv + vec2f(0.0, px.y)) - sceneDepth(uv - vec2f(0.0, px.y)));
  // 按距离归一化：远处的物体不会因为深度数值大而全是线
  let edge = smoothstep(SENSITIVITY, SENSITIVITY * 3.0, (dx + dy) / max(d, 1e-3));
  return mix(color, LINE_COLOR, edge);
}
