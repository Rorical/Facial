import { describe, expect, it } from 'vitest';
import { quat } from '../src/math';
import { BOX, CAPSULE, collide, shapeAabb, SPHERE, type Contact } from '../src/physics/solver/collision';
import { mat3FromQuat } from '../src/physics/solver/rotation';

// Narrowphase checks: separation distances, and that every reported contact is consistent
// (the points lie `depth` apart along the normal, on either surface).

const out: Contact = { p1x: 0, p1y: 0, p1z: 0, p2x: 0, p2y: 0, p2z: 0, nx: 0, ny: 0, nz: 0, depth: 0 };

interface Shape { shape: number; size: [number, number, number]; at: [number, number, number]; axisAngle?: [[number, number, number], number] }

function pose(s: Shape): { x: Float64Array; R: Float64Array } {
  const q = s.axisAngle ? quat.fromAxisAngle(quat.create(), s.axisAngle[0], s.axisAngle[1]) : quat.create();
  return { x: Float64Array.from(s.at), R: mat3FromQuat(q, 0, new Float64Array(9)) };
}

function test(a: Shape, b: Shape): number {
  const pa = pose(a), pb = pose(b);
  return collide(a.shape, a.size, pa.x, 0, pa.R, 0, b.shape, b.size, pb.x, 0, pb.R, 0, out);
}

/** The contact's own bookkeeping: p1 − p2 = depth · n, with n a unit vector. */
function expectConsistent(): void {
  expect(Math.hypot(out.nx, out.ny, out.nz)).toBeCloseTo(1, 6);
  expect(out.depth).toBeGreaterThan(0);
  expect(out.p1x - out.p2x).toBeCloseTo(out.depth * out.nx, 6);
  expect(out.p1y - out.p2y).toBeCloseTo(out.depth * out.ny, 6);
  expect(out.p1z - out.p2z).toBeCloseTo(out.depth * out.nz, 6);
}

const sphere = (r: number, at: [number, number, number]): Shape => ({ shape: SPHERE, size: [r, 0, 0], at });
const capsule = (r: number, h: number, at: [number, number, number], axisAngle?: Shape['axisAngle']): Shape => ({ shape: CAPSULE, size: [r, h, 0], at, axisAngle });
const box = (e: [number, number, number], at: [number, number, number], axisAngle?: Shape['axisAngle']): Shape => ({ shape: BOX, size: e, at, axisAngle });

describe('rigid body narrowphase', () => {
  it('measures sphere and capsule separations exactly', () => {
    expect(test(sphere(1, [0, 0, 0]), sphere(0.5, [3, 0, 0]))).toBeCloseTo(1.5);
    // Capsule along Y (half height 2): the sphere beside its middle, then above its cap.
    expect(test(sphere(0.5, [2, 1, 0]), capsule(0.5, 4, [0, 0, 0]))).toBeCloseTo(1);
    expect(test(sphere(0.5, [0, 4, 0]), capsule(0.5, 4, [0, 0, 0]))).toBeCloseTo(1);
    // Two crossed capsules (one along Y, one along Z) whose axes pass 1 apart.
    expect(test(capsule(0.4, 4, [0, 0, 0]), capsule(0.4, 4, [1, 0, 0], [[1, 0, 0], Math.PI / 2]))).toBeCloseTo(0.2);
  });

  it('reports a sphere pushing into a capsule with the right normal', () => {
    expect(test(sphere(1, [1.5, 0.5, 0]), capsule(1, 4, [0, 0, 0]))).toBeCloseTo(-0.5);
    expectConsistent();
    expect(out.nx).toBeCloseTo(-1); // from the sphere toward the capsule axis
    expect(out.depth).toBeCloseTo(0.5);
  });

  it('handles a sphere against a box face, edge and corner', () => {
    expect(test(sphere(1, [0, 3, 0]), box([2, 1, 2], [0, 0, 0]))).toBeCloseTo(1);
    expect(test(sphere(1, [0, 1.5, 0]), box([2, 1, 2], [0, 0, 0]))).toBeCloseTo(-0.5);
    expectConsistent();
    expect(out.ny).toBeCloseTo(-1);
    // Corner: the nearest point is the corner itself.
    expect(test(sphere(1, [3, 2, 3]), box([2, 1, 2], [0, 0, 0]))).toBeCloseTo(Math.sqrt(3) - 1);
    // Center inside the box: pushed out through the nearest face (+Y, 0.8 away).
    expect(test(sphere(0.5, [0.2, 0.2, 0.3]), box([2, 1, 2], [0, 0, 0]))).toBeLessThan(0);
    expectConsistent();
    expect(out.ny).toBeCloseTo(-1);
    expect(out.depth).toBeCloseTo(0.8 + 0.5);
  });

  it('finds the nearest point of a tilted capsule to a box', () => {
    // Capsule tilted 45° in the XY plane, its lower end 0.3 above the box top.
    const c = capsule(0.25, 4, [0, 2 + 0.25 + 0.3 + Math.SQRT2, 0], [[0, 0, 1], Math.PI / 4]);
    expect(test(c, box([3, 2, 3], [0, 0, 0]))).toBeCloseTo(0.3, 4);
    // Lowered by 0.5: penetrating by 0.2.
    c.at[1] -= 0.5;
    expect(test(c, box([3, 2, 3], [0, 0, 0]))).toBeCloseTo(-0.2, 4);
    expectConsistent();
    expect(out.ny).toBeCloseTo(-1);
  });

  it('pushes a capsule whose axis runs through a box out the shortest way', () => {
    // Horizontal capsule through a tall thin box: out along X (the box is 0.5 wide, 3 tall).
    expect(test(capsule(0.3, 4, [0, 0, 0], [[0, 0, 1], Math.PI / 2]), box([0.5, 3, 3], [0, 0.2, 0]))).toBeLessThan(0);
    expectConsistent();
    expect(Math.abs(out.nx)).toBeCloseTo(1);
    expect(out.depth).toBeCloseTo(0.5 + 2 + 0.3); // half width + reach of the segment + radius
  });

  it('separates and contacts boxes by faces and edges', () => {
    expect(test(box([1, 1, 1], [0, 0, 0]), box([1, 1, 1], [2.5, 0, 0]))).toBeCloseTo(0.5);
    expect(test(box([1, 1, 1], [0, 0, 0]), box([1, 1, 1], [1.8, 0.3, 0]))).toBeCloseTo(-0.2);
    expectConsistent();
    expect(out.nx).toBeCloseTo(1);
    expect(out.depth).toBeCloseTo(0.2);
    // Box 2 rotated 45° about Z sits on box 1 with an edge: it rests at height 1 + √2.
    expect(test(box([1, 1, 1], [0, 0, 0]), box([1, 1, 1], [0, 1 + Math.SQRT2 + 0.1, 0], [[0, 0, 1], Math.PI / 4]))).toBeCloseTo(0.1, 4);
    expect(test(box([1, 1, 1], [0, 0, 0]), box([1, 1, 1], [0, 1 + Math.SQRT2 - 0.1, 0], [[0, 0, 1], Math.PI / 4]))).toBeCloseTo(-0.1, 4);
    expectConsistent();
    expect(out.ny).toBeCloseTo(1);
    // Crossed edges: two boxes turned 45° about different axes, touching edge to edge.
    const a = box([1, 1, 1], [0, 0, 0], [[0, 0, 1], Math.PI / 4]);
    const b = box([1, 1, 1], [0, 2 * Math.SQRT2 - 0.1, 0], [[1, 0, 0], Math.PI / 4]);
    expect(test(a, b)).toBeCloseTo(-0.1, 4);
    expectConsistent();
    expect(out.ny).toBeCloseTo(1);
  });

  it('is symmetric in its arguments', () => {
    const a = capsule(0.4, 2, [0.5, 1.2, 0.3], [[1, 0, 1], 0.7]);
    const b = box([1, 0.5, 0.8], [0, 0, 0], [[0, 1, 0], 0.3]);
    const ab = test(a, b);
    const p1 = [out.p1x, out.p1y, out.p1z], n = [out.nx, out.ny, out.nz];
    const ba = test(b, a);
    expect(ba).toBeCloseTo(ab, 9);
    expect([out.p2x, out.p2y, out.p2z]).toEqual(p1);
    expect([out.nx, out.ny, out.nz]).toEqual(n.map((v) => -v));
  });

  it('bounds every shape conservatively', () => {
    const aabb = new Float64Array(6);
    const c = pose(capsule(0.5, 3, [1, 2, 3], [[0, 0, 1], Math.PI / 2]));
    shapeAabb(CAPSULE, [0.5, 3, 0], c.x, 0, c.R, 0, aabb, 0);
    expect(Array.from(aabb)).toEqual([1 - 2, 2 - 0.5, 3 - 0.5, 1 + 2, 2 + 0.5, 3 + 0.5].map((v) => expect.closeTo(v, 9)));
    const b = pose(box([1, 1, 1], [0, 0, 0], [[0, 0, 1], Math.PI / 4]));
    shapeAabb(BOX, [1, 1, 1], b.x, 0, b.R, 0, aabb, 0);
    expect(aabb[3]).toBeCloseTo(Math.SQRT2);
    expect(aabb[4]).toBeCloseTo(Math.SQRT2);
    expect(aabb[5]).toBeCloseTo(1);
  });
});
