// 材质 shader 示例：两段式卡通着色
// 不用模型自带的 toon 贴图，明暗只有两档，交界是一条清晰的线，暗部偏冷色。

const SHADOW_TINT = vec3f(0.70, 0.68, 0.84);   // 暗部颜色倍率
const EDGE_SOFTNESS = 0.03;                    // 明暗交界的柔和程度

fn shade(s: Surface) -> vec4f {
  // 材质本色：贴图 × (漫反射 × 灯光 + 环境色)，与 MMD 相同
  var albedo = clamp(s.diffuse * s.lightColor + s.ambient, vec3f(0.0), vec3f(1.0)) * s.texture.rgb;
  if (s.sphereMode == 2u) {
    albedo += s.sphere;
  } else if (s.sphereMode != 0u) {
    albedo *= s.sphere;
  }

  // 受光：法线朝向灯光，并且没有被别的东西挡住
  let facing = smoothstep(0.0, EDGE_SOFTNESS, s.lambert);
  let light = min(facing, smoothstep(0.4, 0.6, s.shadow));
  var color = albedo * mix(SHADOW_TINT, vec3f(1.0), light);

  // 一块边缘清晰的高光
  if (s.shininess > 0.0) {
    let h = normalize(s.viewDir + s.lightDir);
    let spot = smoothstep(0.0, 0.02, pow(max(dot(h, s.normal), 0.0), s.shininess) - 0.5);
    color += s.specular * s.lightColor * spot * light;
  }
  // 设置里的补光 / 发丝光
  color += albedo * extraLighting(s);
  return vec4f(color, s.alpha);
}
