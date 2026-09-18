"""Hard mission MPC with explicit causal coverage recovery after typed rejection.

Accepted MPC plans retain every original hard mission row. A recovery action is
labelled as coverage control and never described as a feasible MPC plan.
"""

from types import SimpleNamespace

from docs.research.tools.offgrid_coverage_v1 import OffgridCoverageController
from plant_backend.mission_mpc import MissionMPCControllerV1
from plant_backend.rl.v2.economic_mpc_schema import (
    EconomicMPCInputError,
    EconomicMPCPlanError,
    EconomicMPCSolveError,
)
from plant_backend.rl.v2.mpc_execution import public_observation_sha256
from plant_backend.rl.v2.mpc_schema import ServiceMPCInputError, domain_sha256
from plant_backend.rl.v2.setpoint_command import issue_process_setpoint_command


SCHEMA = "mission-mpc-coverage-recovery-decision-v1"


class MissionMPCRecoveryControllerV1:
    id = "mission-mpc-coverage-recovery-v1"

    def __init__(self, **mpc_configuration):
        self.mpc = MissionMPCControllerV1(**mpc_configuration)
        self.coverage = OffgridCoverageController()
        self.reset()

    def reset(self):
        self.mpc.reset()
        self.coverage.reset()
        self._events = []

    def to_dict(self):
        return {
            "id": self.id,
            "mpc": self.mpc.to_dict(),
            "coverage": self.coverage.to_dict(),
            "recovery_trigger": "typed model input, solver or selected-plan rejection",
            "recovery_information": "same current public observation and actuator authority",
            "retry": "retry the unchanged hard MPC at the next decision boundary",
            "hard_constraints_relaxed": False,
            "recovery_is_a_feasible_mpc_plan": False,
        }

    @property
    def decision_events(self):
        return tuple(self._events)

    def compute_command(self, observation, *, constraints, config, commitment_profile=None):
        arguments = {
            "constraints": constraints,
            "config": config,
            "commitment_profile": commitment_profile,
        }
        # Clock/reference integrity errors are not a reason to mask bad input
        # with a fallback. Validate those invariants outside the rejection catch.
        reference = self.mpc._validate(observation, config)
        self.mpc._cycle_boundary(observation)
        index = self.mpc._next_decision
        rejection = None
        try:
            selected = self.mpc.compute_command(observation, **arguments)
            mode = "hard_mpc_plan"
            selected_evidence = selected.evidence.to_dict()
        except (
            EconomicMPCInputError,
            EconomicMPCSolveError,
            EconomicMPCPlanError,
            ServiceMPCInputError,
        ) as exc:
            if self.mpc._next_decision != index:
                raise RuntimeError("rejected MPC unexpectedly advanced its clock") from exc
            rejection = {
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "stages": [stage.to_dict() for stage in getattr(exc, "stages", ())],
            }
            self.coverage.decisions = index
            selected = self.coverage.compute_command(observation, **arguments)
            mode = "coverage_recovery"
            selected_evidence = selected.evidence
            self.mpc._reference = reference
            self.mpc._next_decision = index + 1
        evidence = {
            "schema": SCHEMA,
            "decision": index,
            "controller": self.to_dict(),
            "observation_sha256": observation.sha256,
            "decision_mode": mode,
            "audited_hard_return_plan": mode == "hard_mpc_plan",
            "realized_mission_passed": None,
            "rejection": rejection,
            "selected_controller_evidence": selected_evidence,
            "targets": selected.command.targets.to_dict(),
        }
        command = issue_process_setpoint_command(
            selected.command.targets,
            decision_index=index,
            observation_sha256=public_observation_sha256(observation.plant),
            hold_duration_s=self.mpc.settings.interval_s,
            issuer_id=self.id,
            source_schema=SCHEMA,
            source_sha256=domain_sha256(SCHEMA, evidence),
            constraints=constraints,
            config=config,
        )
        self._events.append(evidence)
        return SimpleNamespace(command=command, evidence=evidence, invalid=selected.invalid)
