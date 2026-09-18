"""Independent audit accepts real methanator faults and rejects forged health."""
from copy import deepcopy
import numpy as np
from docs.research.tools.tail_stress_v1.environment import TailStressEnv, SEEDS
from docs.research.tools.tail_stress_v1.audit import MinuteAudit

def test_target_aware_audit_accepts_fault_and_detects_false_component_health():
    env=TailStressEnv('meth24',record_minutes=True)
    try:
        env.reset(seed=SEEDS[0]); audit=MinuteAudit(env); audit.add(env.simulator.state)
        corruptions=[]
        for _ in range(345):
            env.step(-np.ones(7))
            for state in env.last_minute_truth_states[1:]:
                if state.step in (1,3721,5161):
                    bad=MinuteAudit(env); bad.count=state.step; bad.previous=deepcopy(audit.previous)
                    payload=state.to_dict()
                    payload['health']['methanator']['offline']=not payload['health']['methanator']['offline']
                    bad.add(payload)
                    corruptions.append(any(f['rule']=='declared_health_schedule' for f in bad.failures))
                audit.add(state)
        assert corruptions==[True,True,True]
        assert audit.result()['all_physical_invariants_passed'],audit.result()
        assert audit.count==5176
    finally:
        env.close()
