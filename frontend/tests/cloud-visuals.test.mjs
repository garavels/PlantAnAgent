import assert from "node:assert/strict";
import { test } from "node:test";
import { deriveCloudVisuals } from "../src/components/plant-3d/cloudVisuals.ts";

const site = (cover, extra = {}) => ({
  reference_kind: "forecast_pv",
  site_sample: { cloud_cover_pct: cover },
  status: "observed",
  attenuation: null,
  ...extra,
});

test("forecast clear sky suppresses clouds even with the illustrative fallback enabled", () => {
  assert.deepEqual(deriveCloudVisuals(site(0), true), { density: 0, source: "forecast", castShadow: false });
  assert.deepEqual(deriveCloudVisuals(site(100)), { density: 1, source: "forecast", castShadow: false });
});

test("missing or invalid cover stays unknown and needs explicit fallback opt-in", () => {
  for (const cover of [undefined, null, NaN, Infinity, -Infinity, -1, 100.1, "50"]) {
    assert.deepEqual(deriveCloudVisuals(site(cover)), { density: 0, source: "none", castShadow: false });
    assert.deepEqual(deriveCloudVisuals(site(cover), true), { density: 0.55, source: "illustrative", castShadow: false });
  }
  assert.equal(deriveCloudVisuals(site(undefined, { site_sample: undefined })).source, "none");
});

test("forecast density is monotonic and independent of stress, irradiance and night", () => {
  for (const cover of [0, 10, 25, 50, 75, 90, 100]) {
    for (const status of ["observed", "night", "unattributed"]) {
      for (const stress of [0.1, 0.7, 1]) {
        for (const solar of [0, 500, 5000]) {
          const result = deriveCloudVisuals(site(cover, { status, stress_factor: stress, clear_sky_kw: solar, available_kw: solar * stress, attenuation: 0.99 }));
          assert.equal(result.density, cover / 100);
          assert.equal(result.castShadow, false);
        }
      }
    }
  }
});

test("overlapping cloud layers are not summed or substituted for missing total cover", () => {
  const weather = site(undefined, { site_sample: { cloud_cover_low_pct: 90, cloud_cover_mid_pct: 80, cloud_cover_high_pct: 100 } });
  assert.equal(deriveCloudVisuals(weather).source, "none");
  assert.equal(deriveCloudVisuals({ ...weather, site_sample: { ...weather.site_sample, cloud_cover_pct: 95 } }).density, 0.95);
});

test("synthetic weather retains observed attenuation and shadow behavior", () => {
  const observed = { status: "observed", attenuation: 0.4 };
  assert.deepEqual(deriveCloudVisuals(observed), { density: 0.4, source: "synthetic", castShadow: true });
  assert.equal(deriveCloudVisuals({ ...observed, attenuation: null }).density, 0);
  for (const status of ["night", "unattributed"]) {
    assert.deepEqual(deriveCloudVisuals({ ...observed, status }, true), { density: 0, source: "none", castShadow: false });
  }
});
