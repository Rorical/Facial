// Morph + skinning compute pass. Runs once per frame; the output vertex buffer is
// shared by the main, edge, shadow and ground shadow passes, so skinning is never repeated.

struct VertexIn {
  pos: vec3f,
  skinType: u32,       // 0 BDEF1, 1 BDEF2, 2 BDEF4, 3 SDEF, 4 QDEF (dual quaternion)
  normal: vec3f,
  sdefIndex: u32,
  uv: vec2f,
  edgeScale: f32,
  uv1MorphCount: u32,  // additional-UV1 morphs follow the uv morphs: [z+w, z+w+count)
  bones: vec4u,
  weights: vec4f,
  morphRange: vec4u,   // position morphs [x, x+y), uv morphs [z, z+w) in morphEntries
  uv1: vec4f,          // additional UV 1
  cloth: f32,          // 0 = not cloth … 1 = pushed fully out of colliders (see cloth-collision.ts)
  _pad1: f32,
  colliderMask: vec2u, // which colliders apply
  anchor: vec3f,       // rest position of the joint of the physics bone this vertex follows
  anchorBone: u32,
  // Distance to keep from colliders at each point tested along the joint-to-vertex line
  // (w = the vertex itself); negative: may stay that deep inside, as in the rest pose.
  clearance: vec4f,
};


struct Sdef {
  c: vec4f,
  r0: vec4f,           // precomputed (C + R0') / 2, see gpu-model.ts
  r1: vec4f,
};

// Also holds the cloth colliders, after the bones (from index params.boneCount): the model's
// "follow bone" rigid bodies, with `skin` = the body's world transform (rotation + translation)
// and `rot` = (shape: 0 sphere, 1 box, 2 capsule; size xyz). One buffer, because a shader stage
// may only bind 8 storage buffers.
struct Bone {
  skin: mat4x4f,
  rot: vec4f,          // world rotation quaternion (xyzw)
};

struct Params {
  vertexCount: u32,
  colliderCount: u32,  // 0 = cloth collision off
  boneCount: u32,
  clothActive: u32,    // 1 = baked cloth positions replace skinning for vertices that have them
  root: mat4x4f,       // the model's placement in the scene (baked cloth is stored without it)
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> vertices: array<VertexIn>;
@group(0) @binding(2) var<storage, read> sdefs: array<Sdef>;
// xyz (or xy for UV) = offset, w = bitcast<f32>(morph index)
@group(0) @binding(3) var<storage, read> morphEntries: array<vec4f>;
@group(0) @binding(4) var<storage, read> morphWeights: array<f32>;
@group(0) @binding(5) var<storage, read> bones: array<Bone>;
// 10 floats per vertex: position.xyz, normal.xyz, uv.xy, uv1.xy
@group(0) @binding(6) var<storage, read_write> outVertices: array<f32>;
// Baked cloth: per vertex its slot in clothData (0xffffffff = none); per slot two vec4f:
// position and normal in model space.
@group(0) @binding(7) var<storage, read> clothSlots: array<u32>;
@group(0) @binding(8) var<storage, read> clothData: array<vec4f>;

/**
 * Furthest a cloth vertex is moved by collision, in model units (about 6 cm). This corrects what
 * physics leaves over; cloth that is far inside the body (physics off, a leg right through a
 * rigid skirt) is left alone rather than blown outwards.
 */
const MAX_PUSH = 0.8;

/**
 * Moves p out of collider c if it is closer than `clearance`. Returns the corrected point.
 * Shapes as in Bullet / MMD: sphere (radius x), box (half extents), capsule along Y (radius x, height y).
 */
fn pushOut(p: vec3f, c: Bone, clearance: f32) -> vec3f {
  let rot = mat3x3f(c.skin[0].xyz, c.skin[1].xyz, c.skin[2].xyz);
  let l = (p - c.skin[3].xyz) * rot; // into the collider's space (rotation inverse = transpose)
  let size = c.rot.yzw;
  var dist: f32;
  var normal: vec3f;
  if (c.rot.x > 0.5 && c.rot.x < 1.5) {
    let q = abs(l) - size;
    let outside = max(q, vec3f(0.0));
    dist = length(outside) + min(max(q.x, max(q.y, q.z)), 0.0);
    if (dist > 0.0) {
      normal = normalize(outside) * sign(l);
    } else if (q.x > q.y && q.x > q.z) {
      normal = vec3f(sign(l.x), 0.0, 0.0);
    } else if (q.y > q.z) {
      normal = vec3f(0.0, sign(l.y), 0.0);
    } else {
      normal = vec3f(0.0, 0.0, sign(l.z));
    }
  } else {
    // Sphere, or a capsule: a sphere around the nearest point of its axis.
    var q = l;
    if (c.rot.x > 1.5) {
      q.y -= clamp(l.y, -size.y * 0.5, size.y * 0.5);
    }
    let len = length(q);
    dist = len - size.x;
    normal = select(vec3f(1.0, 0.0, 0.0), q / len, len > 1e-6);
  }
  if (dist >= clearance) {
    return p;
  }
  return p + rot * normal * (clearance - dist);
}

fn quatRotate(q: vec4f, v: vec3f) -> vec3f {
  let t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}

fn slerp(a: vec4f, bIn: vec4f, t: f32) -> vec4f {
  var b = bIn;
  var d = dot(a, b);
  if (d < 0.0) {
    b = -b;
    d = -d;
  }
  if (d > 0.9995) {
    return normalize(mix(a, b, t));
  }
  let theta = acos(d);
  return (a * sin((1.0 - t) * theta) + b * sin(t * theta)) / sin(theta);
}

fn quatMul(a: vec4f, b: vec4f) -> vec4f {
  return vec4f(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

/** Dual part of the dual quaternion for rotation q followed by translation t. */
fn dualPart(q: vec4f, t: vec3f) -> vec4f {
  return 0.5 * quatMul(vec4f(t, 0.0), q);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.vertexCount) {
    return;
  }
  let v = vertices[i];

  // Morphs: CSR ranges per vertex, so no atomics are needed.
  var pos = v.pos;
  var uv = v.uv;
  let r = v.morphRange;
  for (var k = r.x; k < r.x + r.y; k++) {
    let e = morphEntries[k];
    pos += e.xyz * morphWeights[bitcast<u32>(e.w)];
  }
  for (var k = r.z; k < r.z + r.w; k++) {
    let e = morphEntries[k];
    uv += e.xy * morphWeights[bitcast<u32>(e.w)];
  }
  var uv1 = v.uv1.xy;
  for (var k = r.z + r.w; k < r.z + r.w + v.uv1MorphCount; k++) {
    let e = morphEntries[k];
    uv1 += e.xy * morphWeights[bitcast<u32>(e.w)];
  }

  var outPos: vec3f;
  var outNrm: vec3f;
  if (v.skinType == 3u) {
    // SDEF: rotate around the precomputed center with the slerped bone rotation.
    let s = sdefs[v.sdefIndex];
    let b0 = bones[v.bones.x];
    let b1 = bones[v.bones.y];
    let w0 = v.weights.x;
    let w1 = v.weights.y;
    let q = slerp(b0.rot, b1.rot, w1);
    outPos = quatRotate(q, pos - s.c.xyz)
      + (b0.skin * vec4f(s.r0.xyz, 1.0)).xyz * w0
      + (b1.skin * vec4f(s.r1.xyz, 1.0)).xyz * w1;
    outNrm = quatRotate(q, v.normal);
  } else if (v.skinType == 4u) {
    // QDEF: blend the bones' rigid transforms as dual quaternions (no volume loss on twists).
    // A bone's skin transform is its world rotation plus the skin matrix's translation.
    var real = vec4f(0.0);
    var dual = vec4f(0.0);
    let pivot = bones[v.bones.x].rot;
    for (var k = 0; k < 4; k++) {
      let w = v.weights[k];
      if (w == 0.0) {
        continue;
      }
      let b = bones[v.bones[k]];
      let s = select(w, -w, dot(b.rot, pivot) < 0.0);  // keep all on one hemisphere
      real += b.rot * s;
      dual += dualPart(b.rot, b.skin[3].xyz) * s;
    }
    let norm = length(real);
    if (norm > 1e-8) {
      real /= norm;
      dual /= norm;
    } else {
      real = vec4f(0.0, 0.0, 0.0, 1.0);
      dual = vec4f(0.0);
    }
    // translation = 2 · dual · conj(real)
    let t = 2.0 * quatMul(dual, vec4f(-real.xyz, real.w)).xyz;
    outPos = quatRotate(real, pos) + t;
    outNrm = quatRotate(real, v.normal);
  } else {
    let m = bones[v.bones.x].skin * v.weights.x
      + bones[v.bones.y].skin * v.weights.y
      + bones[v.bones.z].skin * v.weights.z
      + bones[v.bones.w].skin * v.weights.w;
    outPos = (m * vec4f(pos, 1.0)).xyz;
    outNrm = (m * vec4f(v.normal, 0.0)).xyz;
  }
  // Cloth collision: keep cloth vertices out of the body's colliders. Two rounds, so a vertex
  // pushed out of one collider into another still ends up outside both.
  let clothSlot = clothSlots[i];
  if (params.clothActive != 0u && clothSlot != 0xffffffffu) {
    // Baked cloth: this vertex's position and normal were simulated ahead of time.
    outPos = (params.root * vec4f(clothData[clothSlot * 2u].xyz, 1.0)).xyz;
    outNrm = (params.root * vec4f(clothData[clothSlot * 2u + 1u].xyz, 0.0)).xyz;
  } else if (params.colliderCount > 0u && v.cloth > 0.0) {
    var p = outPos;
    // The line from the joint the vertex hangs from to the vertex stands in for the cloth in
    // between: a few points along it are tested too, and the vertex is swung out (around the
    // joint) far enough to clear them. Otherwise a leg passes through the middle of long skirt
    // triangles whose corners are all outside it.
    let joint = (bones[v.anchorBone].skin * vec4f(v.anchor, 1.0)).xyz;
    for (var iter = 0; iter < 2; iter++) {
      for (var j = 0u; j < params.colliderCount; j++) {
        if ((v.colliderMask[j >> 5u] & (1u << (j & 31u))) == 0u) {
          continue;
        }
        // Of the points along the line, act on the one that needs the vertex moved furthest
        // (adding them all up would overshoot).
        var swing = vec3f(0.0);
        for (var k = 1; k <= 4; k++) {
          let t = f32(k) * 0.25;
          let at = mix(joint, p, t);
          // Moving the vertex by d moves this point by d·t; keep the leverage near the joint small.
          let d = (pushOut(at, bones[params.boneCount + j], v.clearance[k - 1]) - at) / max(t, 0.75);
          if (dot(d, d) > dot(swing, swing)) {
            swing = d;
          }
        }
        p += swing;
      }
    }
    // A vertex this far inside something can't be put right by pushing; don't fling it away.
    let shift = p - outPos;
    let moved = length(shift);
    if (moved > MAX_PUSH) {
      p = outPos + shift * (MAX_PUSH / moved);
    }
    outPos = mix(outPos, p, v.cloth);
  }

  let len = length(outNrm);
  if (len > 1e-8) {
    outNrm = outNrm / len;
  }

  let o = i * 10u;
  outVertices[o] = outPos.x;
  outVertices[o + 1u] = outPos.y;
  outVertices[o + 2u] = outPos.z;
  outVertices[o + 3u] = outNrm.x;
  outVertices[o + 4u] = outNrm.y;
  outVertices[o + 5u] = outNrm.z;
  outVertices[o + 6u] = uv.x;
  outVertices[o + 7u] = uv.y;
  outVertices[o + 8u] = uv1.x;
  outVertices[o + 9u] = uv1.y;
}
