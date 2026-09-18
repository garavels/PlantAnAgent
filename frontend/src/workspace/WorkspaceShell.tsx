import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { useWorkspace } from "./context";
import { duration } from "./format";
import { missionDescription } from "./recording";
const PANELS = [
  ["/", "Overview", "◈"],
  ["/plant", "Plant", "⬡"],
  ["/weather", "Weather", "☁"],
  ["/site", "Location study", "◎"],
  ["/scenarios", "Scenarios", "◇"],
  ["/planning", "Planning", "≋"],
  ["/deliveries", "Deliveries & resources", "◫"],
  ["/economics", "Economics", "€"],
  ["/events", "Events", "≡"],
  ["/models", "Models", "∿"],
] as const;
export function WorkspaceShell({ children }: { children: ReactNode }) {
  const w = useWorkspace(),
    location = useLocation();
  const [setup, setSetup] = useState(false),
    [controller, setController] = useState<"H" | "MPC">("MPC");
  const [menu, setMenu] = useState(false);
  const siteStudy = location.pathname === "/site";
  const scenarios = location.pathname === "/scenarios";
  const models = ["/models", "/learning", "/evidence"].includes(location.pathname);
  const configuration = siteStudy || models || scenarios;
  const { setPlaying, setRunning } = w;
  useEffect(() => {
    if (models || scenarios) {
      setPlaying(false);
      setRunning(false);
    }
  }, [models, scenarios, setPlaying, setRunning]);
  useEffect(() => {
    document.querySelector(".workspace-main")?.scrollTo(0, 0);
    setMenu(false);
  }, [location.pathname]);
  const modal = useRef<HTMLElement>(null);
  useEffect(() => {
    if (!setup) return;
    const previous = document.activeElement as HTMLElement | null;
    modal.current?.querySelector<HTMLButtonElement>("button")?.focus();
    return () => previous?.focus();
  }, [setup]);
  const frame = w.frame,
    snapshot = w.snapshot;
  const historical = w.index < w.latestIndex;
  const activeSite = snapshot?.site_forecast?.site;
  return (
    <div className="workspace">
      <aside
        className={`workspace-sidebar${menu ? " workspace-sidebar--open" : ""}`}
      >
        <NavLink to="/" className="w-brand">
          <span className="w-brand-mark" aria-hidden="true">
            <svg viewBox="0 0 28 28">
              <rect
                x="0.75"
                y="0.75"
                width="26.5"
                height="26.5"
                rx="9"
                fill="none"
                stroke="var(--accent)"
              />
              <path
                d="M9.5 5.5h4.5a8.5 8.5 0 0 1 0 17H9.5Z"
                fill="var(--accent)"
              />
            </svg>
          </span>
          <span>
            Plant An Agent<small>Plant operations</small>
          </span>
        </NavLink>
        <nav aria-label="Workspace panels">
          {PANELS.map(([path, label, glyph]) => (
            <NavLink key={path} to={path} end={path === "/"}>
              <span aria-hidden>{glyph}</span>
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="w-sidebar-foot">
          <span className="w-status-dot" />
          Synthetic methane plant<small>Renewable power to fuel</small>
          <button className="w-button" onClick={() => setSetup(true)}>
            Mission
          </button>
        </div>
      </aside>
      <div className="workspace-body">
        <header className="workspace-header">
          <button
            className="w-menu"
            aria-label="Toggle panel navigation"
            onClick={() => setMenu(!menu)}
          >
            ☰
          </button>
          <div>
            <strong>{scenarios ? "Scenario studio" : models ? "Plant replay" : activeSite ? `${activeSite.name} · renewable methane` : "Renewable methane"}</strong>
            {!models && <span>{scenarios ? "Describe conditions. Review the plan. Run the plant." : siteStudy && !activeSite ? "10-day location study" : missionDescription(snapshot, w.recording)}</span>}
          </div>
          {models || scenarios ? null : siteStudy && !activeSite ? <div className="w-header-status"><span className="w-mode">Weather-driven plant simulation</span></div> : <div className="w-header-status">
            <span className="w-mode">
              {activeSite ? "Site forecast simulation" : w.recording ? "Recorded simulation" : "Interactive simulation"}
            </span>
            <strong>
              {frame ? w.time(frame.clock.elapsed_hours) : "Connecting…"}
            </strong>
            <span>
              {snapshot?.controller.label ?? "Connecting…"}
            </span>
          </div>}
        </header>
        {!configuration && (w.notice || (!w.recording && (w.live.error || snapshot?.error))) && (
          <div className="w-banner" role="alert">
            <span>{w.notice ?? w.live.error ?? snapshot?.error}</span>
            <button
              className="w-button"
              disabled={w.busy}
              onClick={() => void w.live.refresh()}
            >
              Refresh simulation
            </button>
          </div>
        )}
        {!configuration && w.recording?.snapshot.error && (
          <div className="w-banner" role="status">
            <span>
              {snapshot?.controller.id} stopped at{" "}
              {w.time(w.recording.snapshot.frame.clock.elapsed_hours)} ·{" "}
              {w.recording.snapshot.error.match(/\(([^)]+)\)/)?.[1] ??
                "Controller error"}{" "}
              · no fallback
            </span>
            <NavLink to="/events">View events</NavLink>
          </div>
        )}
        <main className="workspace-main" aria-busy={!configuration && w.reading}>
          {!configuration && (!frame || !snapshot) ? (
            <div className="w-loading">
              <h1>Connecting</h1>
              <p>{w.live.error ?? "Loading state…"}</p>
              <button
                className="w-button"
                onClick={() => void w.live.refresh()}
              >
                Reconnect
              </button>
            </div>
          ) : (
            children
          )}
        </main>
        {!configuration && <footer className="workspace-timeline">
          <div className="w-transport">
            <span className="w-transport-state">
              {w.busy
                ? "Reading…"
                : w.running
                  ? "Simulating"
                  : w.playing
                    ? "Playing recording"
                    : historical
                      ? "Inspecting history"
                      : "Paused"}
              <small>{duration(w.index / 4)} elapsed</small>
            </span>
            {w.recording ? (
              <>
              <button
                className="w-button w-button--primary"
                disabled={w.busy || w.latestIndex === 0}
                onClick={() => {
                  if (!w.playing && w.index >= w.latestIndex) void w.selectFrame(0);
                  w.setPlaying(!w.playing);
                }}
              >
                {w.playing ? "Pause" : "Play recording"}
              </button>
              <button className="w-button" disabled={w.busy || w.index >= w.latestIndex}
                onClick={() => { w.setPlaying(false); void w.selectFrame(w.index + 1); }}>
                +15 min
              </button>
              <button className="w-button w-hour" disabled={w.busy || w.index >= w.latestIndex}
                onClick={() => { w.setPlaying(false); void w.selectFrame(Math.min(w.latestIndex, w.index + 4)); }}>
                +1 hour
              </button>
              </>
            ) : (
              <>
                <button
                  className="w-button w-button--primary"
                  disabled={
                    !w.running && (w.busy || w.live.stale || snapshot?.status !== "ready" || historical)
                  }
                  onClick={() => w.setRunning(!w.running)}
                >
                  {w.running ? "Pause" : "Run simulation"}
                </button>
                <button
                  className="w-button"
                  disabled={
                    w.busy ||
                    w.running ||
                    historical ||
                    w.live.stale ||
                    snapshot?.status !== "ready"
                  }
                  onClick={() => void w.advance(1)}
                >
                  +15 min
                </button>
                <button
                  className="w-button w-hour"
                  disabled={
                    w.busy ||
                    w.running ||
                    historical ||
                    w.live.stale ||
                    snapshot?.status !== "ready" ||
                    w.latestIndex > (snapshot?.environment.horizon_hours ?? 168) * 4 - 4
                  }
                  onClick={() => void w.advance(4)}
                >
                  +1 hour
                </button>
              </>
            )}
            <div className="w-scrubber">
              <label htmlFor="mission-time">
                {w.recording ? "Recording timeline" : "Completed history"}
                <span>
                  {w.time(0)} — {w.time(w.latestIndex / 4)}
                </span>
              </label>
              <input
                id="mission-time"
                aria-label="Inspect completed decision"
                type="range"
                min="0"
                max={w.latestIndex}
                step="1"
                value={w.index}
                disabled={!w.latestIndex || w.running}
                onChange={(e) => {
                  w.setPlaying(false);
                  void w.selectFrame(Number(e.target.value));
                }}
              />
            </div>
            <button
              className="w-button"
              disabled={w.reading}
              onClick={() => {
                w.setPlaying(false);
                void w.selectFrame(w.latestIndex);
              }}
            >
              {w.recording ? "Final result" : "Latest"}
            </button>
            <button className="w-button" onClick={() => setSetup(true)}>
              Mission
            </button>
          </div>
        </footer>}
      </div>
      {setup && (
        <div
          className="w-modal-backdrop"
          onClick={(e) => {
            if (e.target === e.currentTarget) setSetup(false);
          }}
        >
          <section
            ref={modal}
            className="w-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="mission-title"
            onKeyDown={(e) => {
              if (e.key === "Escape") setSetup(false);
              if (e.key === "Tab") {
                const controls = Array.from(
                  e.currentTarget.querySelectorAll<HTMLElement>(
                    "button:not(:disabled), select:not(:disabled), a[href]",
                  ),
                );
                const first = controls[0],
                  last = controls.at(-1);
                if (e.shiftKey && document.activeElement === first) {
                  e.preventDefault();
                  last?.focus();
                }
                if (!e.shiftKey && document.activeElement === last) {
                  e.preventDefault();
                  first?.focus();
                }
              }
            }}
          >
            <div className="w-panel-heading">
              <h2 id="mission-title">Mission</h2>
              <button className="w-button" onClick={() => setSetup(false)}>
                Close
              </button>
            </div>
            <p>{activeSite ? `${activeSite.name} · ` : ""}{missionDescription(snapshot, w.recording)} · {snapshot?.frame.operating_policy ? "daily inventory review" : "daily inventory returns"}</p>
            {w.recording?.schema === "plant-research-playback-v1" && <p className="w-note">Recorded playback is read-only. Open or reset a separate interactive simulation below.</p>}
            <div className="w-mission-actions">
              <button
                className="w-button"
                onClick={() => {
                  w.useSimulation();
                  setSetup(false);
                }}
              >
                Open current simulation
              </button>
              <button
                className="w-button"
                onClick={() => void w.exportRecording()}
              >
                Download recording
              </button>
            </div>
            <h3>{activeSite ? "Restart site simulation" : "New simulation"}</h3>
            <p className="w-note">
              {activeSite ? `Restart ${activeSite.name} with the same downloaded forecast and plant capacities. Select the controller below.` : "Interactive mission: 240 kg every 6 hours. Previous 3 runs retained until server restart."}
            </p>
            <div className="w-mission-actions">
              <label>
                Controller
                <select
                  value={controller}
                  onChange={(e) =>
                    setController(e.target.value as "H" | "MPC")
                  }
                >
                  {w.live.snapshot?.available_controllers.map((c) => (
                    <option key={c.id} value={c.id} disabled={!c.available}>
                      {c.label}
                    </option>
                  ))}
                </select>
              </label>
              <button
                className="w-button w-button--primary"
                disabled={
                  w.busy ||
                  w.live.stale ||
                  !w.live.snapshot?.available_controllers.find(
                    (c) => c.id === controller,
                  )?.available
                }
                onClick={() => {
                  void w.reset(controller);
                  setSetup(false);
                }}
              >
                {activeSite ? "Restart site simulation" : "Reset simulation"}
              </button>
            </div>
            <NavLink className="w-link" to="/models" onClick={() => setSetup(false)}>
              Choose a model replay
            </NavLink>
          </section>
        </div>
      )}
    </div>
  );
}
