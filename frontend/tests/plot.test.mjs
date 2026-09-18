import assert from "node:assert/strict";
import { test } from "node:test";
import { plotValueAt } from "../src/workspace/plotModel.ts";

test("a step target remains active until the next timestamp", () => {
  const points = [{ x: 0, y: 100 }, { x: 0.25, y: 200 }, { x: 0.5, y: 0 }];
  assert.equal(plotValueAt(points, -0.01, true), null);
  for (const x of [0, 0.1, 0.24]) assert.equal(plotValueAt(points, x, true), points[0]);
  for (const x of [0.25, 0.251, 0.49]) assert.equal(plotValueAt(points, x, true), points[1]);
  for (const x of [0.5, 0.6]) assert.equal(plotValueAt(points, x, true), points[2]);
});

test("duplicate interval endpoints display the new target at the boundary", () => {
  const points = [
    { x: 0, y: 100 },
    { x: 0.25, y: 100 },
    { x: 0.25, y: 200 },
    { x: 0.5, y: 200 },
    { x: 0.5, y: 0 },
    { x: 0.75, y: 0 },
  ];
  for (const x of [0.25, 0.251, 0.375, 0.499]) {
    assert.equal(plotValueAt(points, x, true), points[2]);
  }
  for (const x of [0.5, 0.501, 0.625]) {
    assert.equal(plotValueAt(points, x, true), points[4]);
  }
});

test("ordinary series preserve nearest-sample selection and first-point ties", () => {
  const points = [{ x: 1, y: 10 }, { x: 2, y: 20 }, { x: 2, y: 30 }];
  assert.equal(plotValueAt(points, 0), points[0]);
  assert.equal(plotValueAt(points, 1.5), points[0]);
  for (const x of [1.9, 2, 3]) assert.equal(plotValueAt(points, x), points[1]);
});

test("empty series and invalid inspection times have no displayed value", () => {
  for (const step of [false, true]) {
    assert.equal(plotValueAt([], 1, step), null);
    assert.equal(plotValueAt([{ x: 1, y: 10 }], Number.NaN, step), null);
  }
});
