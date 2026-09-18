import { Component, lazy, Suspense } from "react";
import type { ReactNode } from "react";
import { PlantFlow } from "../components/overview/PlantFlow";
import type { V2DemoFrame, V2DemoSnapshot } from "../types/v2";
import { useWorkspace } from "./context";
import "../pages/plant-3d.css";
const PlantScene = lazy(() =>
  import("../components/plant-3d/PlantScene").then((m) => ({
    default: m.PlantScene,
  })),
);
class SceneBoundary extends Component<
  { children: ReactNode; fallback: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? (
      <>
        <p className="w-note">
          3D is unavailable. The process diagram shows the same state.
        </p>
        {this.props.fallback}
      </>
    ) : (
      this.props.children
    );
  }
}
export function ProcessDiagram({
  frame,
  snapshot,
}: {
  frame: V2DemoFrame;
  snapshot: V2DemoSnapshot;
}) {
  const min = Number(
    frame.commitment_profile?.min_stable_load_frac ??
      snapshot.config.electrolyser.min_load_frac,
  );
  return (
    <div className="w-diagram">
      <PlantFlow
        state={frame.state}
        commitmentMode={frame.commitment?.mode}
        config={{
          ...snapshot.config,
          electrolyser_min_load_kw: min * snapshot.config.electrolyser.rated_kw,
          electrolyser: { ...snapshot.config.electrolyser, min_load_frac: min },
        }}
      />
    </div>
  );
}
export function Scene({
  overhead = false,
  compact = false,
  illustrativeClouds = false,
}: {
  overhead?: boolean;
  compact?: boolean;
  illustrativeClouds?: boolean;
}) {
  const w = useWorkspace();
  if (!w.frame || !w.snapshot) return null;
  return (
    <div className={`w-scene${compact ? " w-scene--compact" : ""}`}>
      <SceneBoundary
        fallback={<ProcessDiagram frame={w.frame} snapshot={w.snapshot} />}
      >
        <Suspense
          fallback={
            <div className="w-scene-loading">Building the plant view…</div>
          }
        >
          <PlantScene
            key={overhead ? "overhead" : "plant"}
            state={w.frame.state}
            config={w.snapshot.config}
            weather={w.frame.weather}
            site={w.snapshot.site_forecast?.site}
            commitment={w.frame.commitment}
            minLoadFrac={Number(
              w.frame.commitment_profile?.min_stable_load_frac ?? 0.4,
            )}
            playing={w.running || w.playing}
            overhead={overhead}
            illustrativeClouds={illustrativeClouds}
            hideLabels={compact}
            selectedUnit={compact ? null : w.selectedUnit}
            onSelectUnit={compact ? undefined : w.setSelectedUnit}
            agentAction={w.agent.visible && !overhead ? (w.agent.action ?? { unit: "hub", label: "Awaiting commands" }) : null}
          />
        </Suspense>
      </SceneBoundary>
    </div>
  );
}
