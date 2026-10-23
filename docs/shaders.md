# 自定义 shader

用 WGSL 写，从「效果」菜单加载，或把 `.wgsl` 文件拖进窗口（按里面定义的函数自动判断是材质 shader 还是后期 shader）。写错时会显示出错的行号，并继续使用原来的着色。shader 随工程保存。

`public/shaders/` 里的示例也在「效果」菜单里，每个文件开头都有说明和可调参数：

| 文件 | 类型 | 内容 |
|---|---|---|
| `hsr-toon.wgsl` | 材质 | 星铁风格三渲二：两档明暗加半影、脸部阴影图、刘海投影、眉眼透发、深度轮廓光、本色描边、天使环和金属亮斑 |
| `cel-two-tone.wgsl` | 材质 | 两段式卡通着色，不用 toon 贴图 |
| `rim-light.wgsl` | 材质 | 在 MMD 着色上加固定粗细的边缘光 |
| `skin-toon-sss.wgsl` | 材质 | 卡通皮肤通透感：交界处的红色过渡带、薄处逆光透红 |
| `skin-realistic.wgsl` | 材质 | 写实皮肤：预积分散射的近似、双层高光、薄处透光 |
| `skin-sss-blur.wgsl` | 后期 | 屏幕空间皮肤次表面散射模糊 |
| `depth-outline.wgsl` | 后期 | 按深度差画线 |
| `film.wgsl` | 后期 | 暗角 + 颗粒 + 调色 |

## 材质 shader

替换一个模型的着色（阴影贴图、地面阴影仍是内置的）。定义

```wgsl
fn shade(s: Surface) -> vec4f   // 返回颜色和不透明度
```

`Surface` 里有：

| 字段 | 含义 |
|---|---|
| `worldPos`、`normal`、`viewDir`、`lightDir` | 世界坐标、法线、指向镜头和指向灯光的方向 |
| `lightColor` | 主光颜色（含强度） |
| `uv`、`uv1` | 贴图坐标、附加 UV1 |
| `texture` | 贴图颜色，含材质 morph 的调色 |
| `sphere`、`sphereMode` | sphere 贴图颜色和模式（0 无、1 乘算、2 加算、3 子贴图） |
| `toon`、`toonShadow` | 当前明暗处的 toon 颜色、toon 最暗的颜色 |
| `diffuse`、`ambient`、`specular`、`shininess`、`alpha` | 材质参数 |
| `lambert` | N·L |
| `shadow`、`selfShadow` | 自阴影（1 受光到 0 被遮挡）、该材质是否接收自阴影 |
| `accessory`、`materialIndex` | 是否为 `.x` 配件、材质序号 |
| `part` | 材质是脸 / 五官 / 头发 / 皮肤（按名字猜的） |
| `hasHead`、`headPos`、`headForward`、`headUp` | 模型有没有头骨骼，以及头的位置和朝向 |
| `screenUv`、`viewDepth` | 在屏幕上的位置、离镜头多远 |
| `hasRamp`、`hasLightMap`、`lightMap`、`hasFaceMap`、`faceMap` | 辅助贴图是否存在及其在这一点的值（见下） |
| `time` | 时间线秒数 |

可调用的函数：

- `mmdShade(s)`：MMD 原本的结果，可以在它上面改。
- `sampleToon(lambert)`：按任意明暗值取 toon 颜色。
- `skinAmount(s)`：这一点有多像皮肤（0–1，按材质名和颜色判断）。
- `thinness(s)`：这里有多薄（0–1，来自生成的光照图）。
- `extraLighting(s)`：补光 / 发丝光在这一点叠加的光，乘在本色上即可。MMD 着色和自带的示例都已经加上了。
- 屏幕空间：用了材质 shader 的场景会先多画一遍深度（含每个像素是什么材质）。`sceneDepthAt(uv)` 取任意屏幕位置到镜头的距离，`scenePartAt(uv)` 取那里是什么材质，`pixelsPerUnit(depth)` 取该距离上一个模型单位占多少像素。现成的组合：`depthRim(s, 宽度, 深度差)`（剪影处固定粗细的轮廓光，0–1）、`hairShadow(s, 距离, 深度差)`（朝光的方向看前面是不是头发：刘海投在额头上的影子）、`screenOccluded(s, 像素偏移, 深度差)`（前面有没有东西）、`screenLightDir(s)`（灯光在屏幕上的方向）。

材质着色还会往一张附加画面里写每个像素的数据，供后期 shader 用 `sceneAux(uv)` 读取：红 = 皮肤程度，绿 = 薄的程度，蓝空闲。要自己写的话在 `shade` 里给 `auxOut` 赋值。

两个可选函数：

```wgsl
fn outline(s: Surface) -> vec4f        // 描边的颜色和不透明度（不定义就用模型自己的描边色）
fn hairTransparency(s: Surface) -> f32 // 眼睛、眉毛透过前面的头发显示的程度（0 = 不透）
```

定义了 `hairTransparency` 时，眼睛、眉毛、睫毛（按材质名）会在头发之后再画一遍，只画被头发挡住的部分，按返回值的不透明度叠在头发上；头发在模板缓冲里做标记，所以挡住眼睛的手不会透。

## 辅助贴图

材质 shader 用。选中模型后点「辅助贴图 → 管理」。可以「全部自动生成」，也可以给每一项单独选图片文件（比如游戏原版或自己画的）；加载一个用到辅助贴图的 shader 而模型还没有时，会自动生成。随工程保存。

| 贴图 | shader 里怎么读 | 自动生成的内容 |
|---|---|---|
| 色条（整个模型一张） | `sampleRamp(light, row)`，`light` 0 暗到 1 亮；行号 `RAMP_CLOTH` / `RAMP_SKIN` / `RAMP_HAIR` / `RAMP_DETAIL` | 每行从暗部颜色倍率过渡到白，皮肤行在交界处带一点红 |
| 光照图（每张颜色贴图一张，布局相同） | `s.lightMap`，`s.hasLightMap` | 红 = 环境遮蔽（1 敞开、0 缝隙深处），绿 = 薄的程度（1 很薄，如耳朵、手指；0 背后有约 12 厘米以上的厚度）。按静止姿势对每个顶点做射线检测后画到 UV 上 |
| 脸部阴影图（脸用到的贴图各一张） | `s.faceMap`，`s.hasFaceMap` | 红 / 绿通道 = 光相对头部正前方向右 / 向左最多转多少（0–1 对应 0–180°）这个位置还亮着，含鼻子、脸颊的遮挡；蓝 = 脸的范围 |

自动生成的图是算出来的近似，不如手绘的原版。脸部阴影图里按眼睛骨骼和鼻尖的位置画了两样细节：侧光时背光一侧脸颊上的三角形亮斑，以及鼻梁背光侧的阴影线和鼻尖的亮点；形状是固定公式，不会像原版那样贴合每个角色。光照图只有环境遮蔽，没有高光和材质分区。

## 后期 shader

处理整幅画面，在景深之后、泛光和色调映射之前。定义

```wgsl
fn effect(uv: vec2f, color: vec3f) -> vec3f   // uv 为 0–1，左上角是原点
```

可调用 `sceneColor(uv)` 取任意位置的颜色、`sceneDepth(uv)` 取到镜头的距离（模型单位）、`sceneAux(uv)` 取材质着色写的附加数据、`pixelsPerUnit(depth)` 取该距离上一个模型单位占多少像素，读取 `fx.resolution`（像素尺寸）和 `fx.time`。
