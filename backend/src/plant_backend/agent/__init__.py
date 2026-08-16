"""V6: the supervisory agent.

    frontend
        |
    backend ---- SupervisorAgent  (slow loop: "what is happening, and why?")
        |           |  tools: observe, inspect, retune the envelope
        |           v
        +------- controller       (fast loop: "what should the plant do now?")
        |           v
        +------- safety.validate  (authoritative, always)
        v
    simulator

The agent sits *beside* the controller, not in front of it. It never produces an
`Action` and never touches the simulator: it changes the envelope the controller
plans inside, and every action still goes through the same validation guard as
before. Turn the agent off and V5 behaviour is exactly what remains.

Modules:
    observations  what the supervisor is allowed to see (telemetry, not truth)
    tools         the only surface it can act through (`AgentToolbox`)
    triggers      when it is worth waking up
    risk          V10: HEPA's forecast as levels, trends and one more trigger
    engines       how it decides: `LLMEngine` (Agents SDK) or `RuleEngine`
    prompts       the supervisory brief
    sdk           the Agents SDK adapter: toolbox methods as function tools
    trace         what it did, as operational entries and incidents
    supervisor    the slow loop that ties those together
"""

from __future__ import annotations

from .engines import Diagnosis, LLMEngine, RuleEngine
from .risk import EventRisk, RiskMonitor, RiskPolicy, Thresholds
from .supervisor import SupervisorAgent, default_engine
from .tools import AgentToolbox
from .trace import AgentTrace, Incident, TraceEntry
from .triggers import Trigger, TriggerWatch

__all__ = [
    "AgentToolbox",
    "AgentTrace",
    "Diagnosis",
    "EventRisk",
    "Incident",
    "LLMEngine",
    "RiskMonitor",
    "RiskPolicy",
    "RuleEngine",
    "SupervisorAgent",
    "Thresholds",
    "TraceEntry",
    "Trigger",
    "TriggerWatch",
    "default_engine",
]
