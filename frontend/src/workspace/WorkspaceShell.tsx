import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { useWorkspace } from "./context";
import { duration, time } from "./format";
const PANELS = [
  ["/", "Overview", "◈"],
  ["/plant", "Plant", "⬡"],
  ["/weather", "Weather", "☁"],
  ["/planning", "Planning", "≋"],
  ["/deliveries", "Deliveries & resources", "◫"],
  ["/economics", "Economics", "€"],
  ["/events", "Events", "≡"],
  ["/evidence", "Evidence", "◇"],
] as const;
export function WorkspaceShell({ children }: { children: ReactNode }) {
  const w = useWorkspace(),
    location = useLocation();
  const [setup, setSetup] = useState(false),
    [controller, setController] = useState<"H" | "MPC">("MPC");
  const [menu, setMenu] = useState(false);
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
            Mission & recordings
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
            <strong>Renewable methane</strong>
            <span>240 kg every 6 hours · 7-day mission</span>
          </div>
          <div className="w-header-status">
            <span className="w-mode">
              {w.recording ? "Recorded simulation" : "Interactive simulation"}
            </span>
            <strong>
              {frame ? time(frame.clock.elapsed_hours) : "Connecting…"}
            </strong>
            <span>
              {snapshot?.controller.id === "H" ? "Heuristic" : "Economic MPC"}
            </span>
          </div>
        </header>
        {(w.notice || (!w.recording && (w.live.error || snapshot?.error))) && (
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
        {w.recording?.snapshot.error && (
          <div className="w-banner" role="status">
            <span>
              {snapshot?.controller.id} stopped at{" "}
              {time(w.recording.snapshot.frame.clock.elapsed_hours)} ·{" "}
              {w.recording.snapshot.error.match(/\(([^)]+)\)/)?.[1] ??
                "Controller error"}{" "}
              · no fallback
            </span>
            <NavLink to="/evidence">Failure details</NavLink>
          </div>
        )}
        <main className="workspace-main" aria-busy={w.reading}>
          {!frame || !snapshot ? (
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
        <footer className="workspace-timeline">
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
              <button
                className="w-button w-button--primary"
                disabled={w.busy || w.latestIndex === 0}
                onClick={() => {
                  if (w.index >= w.latestIndex) void w.selectFrame(0);
                  w.setPlaying(!w.playing);
                }}
              >
                {w.playing ? "Pause" : "Play recording"}
              </button>
            ) : (
              <>
                <button
                  className="w-button w-button--primary"
                  disabled={
                    (w.busy && !w.running) ||
                    w.live.stale ||
                    snapshot?.status !== "ready" ||
                    historical
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
                    w.latestIndex > 668
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
                  {time(0)} — {time(w.latestIndex / 4)}
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
              Latest
            </button>
            <button className="w-button" onClick={() => setSetup(true)}>
              Mission
            </button>
          </div>
        </footer>
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
              <h2 id="mission-title">Mission & recordings</h2>
              <button className="w-button" onClick={() => setSetup(false)}>
                Close
              </button>
            </div>
            <p>168 hours · 240 kg CH₄ every 6 hours · inventory recovery</p>
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
                Download public recording
              </button>
            </div>
            <h3>New simulation</h3>
            <p className="w-note">
              Previous 3 runs retained until server restart.
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
                Reset simulation
              </button>
            </div>
            <div className="w-panel-heading">
              <h3>Available recordings</h3>
              <button
                className="w-button"
                onClick={() => void w.reloadCatalog()}
              >
                Refresh list
              </button>
            </div>
            <div className="w-recordings">
              {w.catalog.length ? (
                w.catalog.map((c) => (
                  <button
                    key={c.id}
                    disabled={!c.available || w.reading}
                    onClick={() => {
                      void w.loadRecording(c.id);
                      setSetup(false);
                    }}
                  >
                    <strong>
                      {c.controller ?? "Unreadable recording"}
                      <span>
                        {c.hours ?? 0} h · {c.status}
                      </span>
                    </strong>
                    <small>
                      {c.source === "saved"
                        ? "Saved recording"
                        : "Previous interactive run"}{" "}
                      · {c.id.slice(0, 8)}
                    </small>
                  </button>
                ))
              ) : (
                <p className="w-note">No saved recordings.</p>
              )}
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
