import { useWorkspace } from "./context";
import "./agent-guide.css";

const power = (value: number) => value.toLocaleString("en-GB", value > 0 && value < 1
  ? { maximumSignificantDigits: 3 } : { maximumFractionDigits: 1 });

export function AgentViewToggle() {
  const { agent } = useWorkspace();
  return (
    <button className="w-agent-toggle" aria-pressed={agent.visible} onClick={agent.toggle}
      title="Follow recorded commands around the plant. Playback pauses at each component.">
      <span className="w-agent-dot" aria-hidden="true" />
      Agent view
      <span className="w-agent-switch" aria-hidden="true"><i /></span>
    </button>
  );
}

export function AgentGuide() {
  const w = useWorkspace();
  const { agent, frame } = w;
  if (!agent.visible || !frame) return null;
  const decision = frame.decision;
  const action = agent.action;
  const choose = (index: number) => {
    w.setPlaying(false);
    w.setRunning(false);
    agent.select(index);
  };
  return (
    <section className="w-agent-guide" aria-label="Agent actions">
      <div className="w-agent-heading">
        <span><span className="w-agent-dot" aria-hidden="true" />{decision ? `${decision.controller_id} dispatch` : "Agent ready"}</span>
        <small>{decision ? `${w.time(decision.at_hours)} → ${w.time(frame.clock.elapsed_hours)}` : "Before the first command"}</small>
      </div>
      {action ? <>
        <div className="w-agent-stops" aria-label="Inspect a recorded command">
          {agent.actions.map((stop, index) => <button key={stop.unit} aria-pressed={index === agent.index}
            onClick={() => choose(index)}>
            <span className="w-agent-stop-dot" aria-hidden="true" />{stop.label}
          </button>)}
        </div>
        <div className="w-agent-explanation" aria-live={w.playing || w.running ? "off" : "polite"} aria-atomic="true">
          <div>
            <h3>{action.title}</h3>
            <p>{action.explanation}</p>
          </div>
          <dl>
            <div><dt>{action.requestLabel}</dt><dd>{power(action.requestedKw)} <small>kW</small></dd></div>
            <div><dt>{action.appliedLabel}</dt><dd>{power(action.appliedKw)} <small>kW</small></dd></div>
          </dl>
        </div>
        {action.detail && <p className="w-agent-detail">{action.detail}</p>}
        <div className="w-agent-footer">
          <p>One completed interval. Commands were issued together; the orb visits each to explain them.</p>
          <button className="w-agent-inspect" onClick={() => {
            w.setPlaying(false);
            w.setRunning(false);
            w.setSelectedUnit(action.unit);
          }}>Inspect {action.label.toLowerCase()}</button>
        </div>
      </> : <p className="w-agent-empty">{decision
        ? "No equipment targets were retained for this interval. Advance to another decision to follow its commands."
        : "Play or step forward to follow the agent’s first recorded commands. The orb will visit each machine and show what was requested and what happened."}</p>}
    </section>
  );
}
