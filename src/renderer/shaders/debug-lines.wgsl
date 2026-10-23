// Rigid body / joint wireframes. Vertices live in their body's local space and are moved by
// that body's world transform; `mask` bit N shows kind N (0 follow bone, 1 physics,
// 2 physics + bone, 3 joint). Drawn on top of the model (no depth test), like PMX Editor.

struct Params {
  viewProj: mat4x4f,
  mask: vec4u,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> bodies: array<mat4x4f>;

struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec4f,
};

@vertex
fn vsMain(@location(0) p: vec3f, @location(1) body: u32, @location(2) kind: u32) -> VsOut {
  var out: VsOut;
  if ((params.mask.x & (1u << kind)) == 0u) {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0); // outside the depth range: clipped
    out.color = vec4f(0.0);
    return out;
  }
  // Function-scope var: runtime indexing of a module const array isn't portable.
  var colors = array<vec4f, 4>(
    vec4f(0.30, 0.62, 1.00, 0.85), // follow bone: blue
    vec4f(1.00, 0.36, 0.26, 0.85), // physics: red-orange
    vec4f(0.45, 0.92, 0.35, 0.85), // physics + bone: green
    vec4f(1.00, 0.84, 0.20, 0.95), // joint: yellow
  );
  out.pos = params.viewProj * (bodies[body] * vec4f(p, 1.0));
  out.color = colors[min(kind, 3u)];
  return out;
}

@fragment
fn fsMain(in: VsOut) -> @location(0) vec4f {
  return in.color;
}
