import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import { test } from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";

// Render the actual shared component without adding a browser or test dependency.
const require = createRequire(import.meta.url);
const source = readFileSync(new URL("../src/workspace/Plot.tsx", import.meta.url), "utf8");
let compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2023, jsx: ts.JsxEmit.ReactJSX },
}).outputText;
for (const specifier of ["react", "react/jsx-runtime"]) {
  compiled = compiled.replaceAll(`from "${specifier}"`, `from "${pathToFileURL(require.resolve(specifier))}"`);
}
for (const module of ["format", "plotModel"]) {
  compiled = compiled.replaceAll(`from "./${module}"`, `from "${new URL(`../src/workspace/${module}.ts`, import.meta.url)}"`);
}
const { Plot } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`);
const render = (props = {}) => renderToStaticMarkup(createElement(Plot, {
  title: "Diagnostic", unit: "ratio", series: [{ label: "Recorded", color: "red", points: [{ x: 8192, y: 0.002 }, { x: 16384, y: 0.006 }] }], ...props,
}));
const ticks = (markup) => [...markup.matchAll(/<text[^>]*>([^<]*)<\/text>/g)].slice(0, 3).map((match) => match[1]);
const circles = (markup) => [...markup.matchAll(/<circle\b/g)].length;

test("default plot preserves its existing y range, integer labels and marker behavior", () => {
  const markup = render();
  assert.deepEqual(ticks(markup), ["0", "1", "1"]);
  assert.equal(circles(markup), 0);
  const one = render({ series: [{ label: "One point", color: "red", points: [{ x: 0, y: 1 }] }] });
  assert.equal(circles(one), 1);
  assert.match(one, /r="4"/);
});

test("opt-in fractional range separates KL ticks and shows only measured points", () => {
  const markup = render({ yDomain: [0, 0.006], valueDigits: 4, showPoints: true });
  assert.deepEqual(ticks(markup), ["0", "0.003", "0.006"]);
  assert.equal(circles(markup), 2);
  assert.match(markup, /cy="25"/);
  assert.doesNotMatch(markup, /NaN|Infinity/);
});

test("small and negative diagnostics preserve their nonzero sign", () => {
  assert.deepEqual(ticks(render({ yDomain: [0, 4e-9], valueDigits: 6 })), ["0", "2.0e-9", "4.0e-9"]);
  assert.deepEqual(ticks(render({ yDomain: [-0.3, 0], valueDigits: 3 })), ["-0.3", "-0.15", "0"]);
});

test("invalid or zero-width domains fall back to the safe existing scale", () => {
  for (const yDomain of [[0, 0], [1, 0], [NaN, 1], [0, Infinity]]) {
    const markup = render({ yDomain });
    assert.deepEqual(ticks(markup), ["0", "1", "1"]);
    assert.doesNotMatch(markup, /NaN|Infinity/);
  }
});

test("actual KL diagnostics keep all eight recorded values visible", () => {
  const evidence = JSON.parse(readFileSync(new URL("../../backend/src/plant_backend/data/adaptive_learning.json", import.meta.url), "utf8"));
  const policy = evidence.policies.find((row) => row.id === "A-mpc-residual-scalar");
  const points = policy.rollouts.map((row) => ({ x: row.transitions, y: row.approx_kl }));
  const high = Math.max(...points.map((point) => point.y));
  const markup = render({ series: [{ label: "Approximate KL", color: "red", points }], yDomain: [0, high], valueDigits: 6, showPoints: true });
  assert.equal(circles(markup), 8);
  assert.equal(new Set(ticks(markup)).size, 3);
  assert.ok(high < 0.1);
  assert.match(ticks(markup)[2], /^0\.0/);
  assert.match(markup, /cy="25"/);
});
