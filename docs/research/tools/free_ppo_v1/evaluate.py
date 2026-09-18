"""One-attempt direct-control evaluation with source gates and minute audits.

Every minute is checked and hashed. Full raw inputs/actions/minutes are retained
on the prospectively selected first scenario, while every case retains compact
accounts, final stocks, descriptive daily deficits and execution failures.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from docs.research.tools.free_ppo_v1.analyze import SCHEMA, BASELINE, UNTRAINED, RESOURCES, canonical_sha, file_pin, publish


def environment():
    from docs.research.tools.free_ppo_v1 import environment as module
    return module


def training():
    environment()  # Bind the frozen physics modules before importing SB3/trainer.
    from docs.research.tools.free_ppo_v1 import train
    return train


def sources():
    env = environment()
    paths = {Path(row["path"]) for row in training().source_records()}
    paths.update(Path(__file__).parent / name for name in ("evaluate.py", "analyze.py"))
    paths.update(env.SOURCE_ROOT / "docs/research/tools" / name for name in ("offgrid_coverage_v1.py", "feasible_mission_controller.py"))
    protocol = ROOT / "docs/research/FREE_PPO_V1_PROTOCOL.md"
    if protocol.exists():
        paths.add(protocol)
    return [file_pin(path) for path in sorted(paths)]


def label_for(kind, optimizer_seed=None, checkpoint_steps=None):
    if kind == "coverage":
        if optimizer_seed is not None or checkpoint_steps is not None:
            raise ValueError("Coverage has no optimizer checkpoint")
        return BASELINE
    if kind == "untrained":
        if optimizer_seed not in (None, 101) or checkpoint_steps not in (None, 0):
            raise ValueError("Prospectively fixed untrained comparator is seed101")
        return UNTRAINED
    if kind != "ppo" or optimizer_seed is None:
        raise ValueError("PPO requires an optimizer seed")
    result = f"free-ppo-v1-{optimizer_seed}"
    return result if checkpoint_steps is None else f"{result}-step-{checkpoint_steps}"


def selection(plan):
    trained = [label_for("ppo", seed) for seed in plan["optimizer_seeds"]]
    checkpoints = plan.get("evaluation_checkpoint_timesteps", [0,65536,196608,plan["total_timesteps"]])
    return {"primary_controllers": [BASELINE, UNTRAINED, *trained], "trained_controllers":trained,
            "optimizer_seeds":plan["optimizer_seeds"], "final_timesteps":plan["total_timesteps"],
            "development_checkpoint_timesteps":checkpoints,
            "development_seeds":list(environment().DEVELOPMENT_SEEDS), "held_out_seeds":list(environment().HELD_OUT_SEEDS),
            "untrained": "Initial seed101 with its own frozen initial observation-normalization statistics",
            "baseline": "Coverage targets translated once at each public decision boundary to solar shares and battery support; same direct15-minute runtime. No minute target dispatcher during execution.",
            "outcomes": "Full-week completion, all28deliveries, exact-minute physical invariants; no daily-return or forced-stop success gate",
            "evidence": "Every minute audited and streaming-hashed. Full minute/input/action traces on first scenario per panel; compact accounts and decision traces for every case.",
            "attempts_per_case":1, "bootstrap":{"seed":20260917,"replicates":10000,"unit":"paired environmental scenario; average optimizer deltas within scenario"}}


def freeze_evaluation(training_root, output):
    plan = training().require_plan(Path(training_root))
    payload = {"schema":SCHEMA+"-freeze", "created_at":datetime.now(timezone.utc).isoformat(),
               "training_root":str(Path(training_root).resolve()), "training_plan":file_pin(Path(training_root)/"plan.json"),
               "environment_identity_sha256":environment().environment_identity_sha256(),
               "source_identity_sha256":environment().source_identity_sha256(), "sources":sources(), "selection":selection(plan)}
    payload["sha256"] = canonical_sha(payload)
    publish(output,payload)
    return payload


def require_freeze(path):
    payload=json.loads(Path(path).read_text())
    if payload.get("schema") != SCHEMA+"-freeze" or payload.get("sha256") != canonical_sha({k:v for k,v in payload.items() if k != "sha256"}):
        raise ValueError("Evaluation freeze identity/checksum differs")
    plan=training().require_plan(Path(payload["training_root"]))
    if payload["training_plan"] != file_pin(Path(payload["training_root"])/"plan.json") or payload["selection"] != selection(plan):
        raise ValueError("Plan or prospective comparison changed")
    if payload["sources"] != sources() or payload["environment_identity_sha256"] != environment().environment_identity_sha256() or payload["source_identity_sha256"] != environment().source_identity_sha256():
        raise ValueError("Executed source changed after comparison freeze")
    return payload


def checkpoint_path(root, kind, optimizer_seed=None, checkpoint_steps=None):
    root=Path(root)
    if kind=="coverage":return None
    if kind=="untrained":return root/"seed-101/initial-policy.zip"
    directory=root/f"seed-{optimizer_seed}"
    if checkpoint_steps == 0:return directory/"initial-policy.zip"
    return directory/"final-policy.zip" if checkpoint_steps is None else directory/"checkpoints"/f"step-{checkpoint_steps:06d}.zip"


def checkpoint_record(path):
    return {"checkpoint":file_pin(path), "metadata":file_pin(Path(path).with_suffix(".json")),
            "normalizer":file_pin(Path(path).with_suffix(".vecnormalize.pkl"))}


def freeze_endpoints(freeze_path, output):
    freeze=require_freeze(freeze_path);root=Path(freeze["training_root"]);records={}
    trainer=training();trainer.configure_torch()
    closure=json.loads((root/"campaign-closure.json").read_text())
    expected_seeds=freeze["selection"]["optimizer_seeds"]
    runs=closure.get("runs",[])
    if (closure.get("schema")!="free-direct-scalar-ppo-v1" or closure.get("all_fixed_endpoints_completed") is not True
        or closure.get("plan_sha256")!=trainer.sha(trainer.require_plan(root))
        or sorted(r["optimizer_seed"] for r in runs)!=sorted(expected_seeds)
        or any(r.get("exit_code")!=0 or r.get("status")!="completed" or r.get("num_timesteps")!=freeze["selection"]["final_timesteps"] or r.get("result")!=file_pin(r["result"]["path"]) for r in runs)):
        raise ValueError("All selected training processes must close successfully before endpoint freeze")
    for seed in freeze["selection"]["optimizer_seeds"]:
        result=json.loads((root/f"seed-{seed}/result.json").read_text())
        if result["status"]!="completed" or result["num_timesteps"]!=freeze["selection"]["final_timesteps"] or not result["actor_parameters_changed"]:
            raise ValueError(f"Endpoint seed{seed} did not close at the fixed budget")
        path=checkpoint_path(root,"ppo",seed);model=trainer.load_checkpoint(path,root/"plan.json")
        try:
            if model.seed!=seed or model.num_timesteps!=freeze["selection"]["final_timesteps"]:
                raise ValueError("Loaded final checkpoint identity differs")
        finally:model.close()
        records[label_for("ppo",seed)]={**checkpoint_record(path),"result":file_pin(root/f"seed-{seed}/result.json")}
    path=checkpoint_path(root,"untrained");model=trainer.load_checkpoint(path,root/"plan.json")
    try:
        if model.seed!=101 or model.num_timesteps!=0:raise ValueError("Untrained comparator is not initial seed101")
    finally:model.close()
    records[UNTRAINED]=checkpoint_record(path)
    payload={"schema":SCHEMA+"-endpoints", "evaluation_freeze_sha256":freeze["sha256"], "created_at":datetime.now(timezone.utc).isoformat(),"endpoints":records,"campaign_closure":file_pin(root/"campaign-closure.json")}
    payload["sha256"]=canonical_sha(payload);publish(output,payload);return payload


def require_endpoints(path, freeze):
    payload=json.loads(Path(path).read_text())
    if payload.get("schema")!=SCHEMA+"-endpoints" or payload.get("evaluation_freeze_sha256")!=freeze["sha256"] or payload.get("sha256")!=canonical_sha({k:v for k,v in payload.items() if k!="sha256"}):
        raise ValueError("Endpoint freeze differs")
    expected={UNTRAINED,*freeze["selection"]["trained_controllers"]}
    if set(payload["endpoints"])!=expected:raise ValueError("Endpoint set changed")
    if payload["campaign_closure"]!=file_pin(payload["campaign_closure"]["path"]):raise ValueError("Training closure record changed")
    for item in payload["endpoints"].values():
        for pin in item.values():
            if file_pin(pin["path"])!=pin:raise ValueError("Frozen endpoint bytes changed")
    return payload


def admit(freeze, *, split, kind, optimizer_seed, checkpoint_steps, gate, endpoints):
    if split not in ("train","development","held_out"):raise ValueError("Unknown split")
    label=label_for(kind,optimizer_seed,checkpoint_steps)
    if kind=="ppo" and optimizer_seed not in freeze["selection"]["optimizer_seeds"]:raise ValueError("Unselected optimizer")
    if checkpoint_steps is not None and kind=="ppo" and (split!="development" or checkpoint_steps not in freeze["selection"]["development_checkpoint_timesteps"] or checkpoint_steps == freeze["selection"]["final_timesteps"]):
        raise ValueError("Intermediate checkpoint admitted only on prescribed development panel")
    endpoint_record=None
    if split=="held_out":
        if not endpoints:raise ValueError("Held-out access requires all endpoints frozen")
        endpoint_record=require_endpoints(endpoints,freeze)
    if split!="train":
        environment()._admit(split,Path(gate) if gate else None)
    return label,endpoint_record


class MinuteAudit:
    """Independent bounds/energy/forcing checks; no discretionary return targets."""
    def __init__(self, env):
        self.env=env;self.count=0;self.failures=[];self.failure_count=0
        self.sha=hashlib.sha256();self.minimum_soc=float("inf");self.max_power_balance_residual=0.0
        self.max_battery_balance_residual=0.0;self.previous=None
        self.max_gas_balance_residual={r:0.0 for r in ("h2","co2","ch4")}
        self.delivery_cache_decision=None;self.deliveries_by_step={};self.delivered_checked_kg=0.0

    def fail(self, rule, value, step):
        self.failure_count+=1
        if len(self.failures)<25:self.failures.append({"rule":rule,"value":value,"step":step})

    def add(self,state):
        env=self.env;s=state.to_dict() if hasattr(state,"to_dict") else state;step=self.count
        raw=json.dumps(s,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
        self.sha.update(len(raw).to_bytes(8,"big"));self.sha.update(raw)
        if s["step"]!=step or abs(s["time_s"]-(env.episode_config.start_hour*3600+step*env.config.dt_s))>1e-7:self.fail("clock",s["time_s"],step)
        for name,level,capacity in (("battery","battery_energy_kwh","battery_capacity_kwh"),("h2","h2_level_kg","h2_capacity_kg"),("co2","co2_level_kg","co2_capacity_kg"),("ch4","ch4_level_kg","ch4_capacity_kg")):
            if not math.isfinite(s[level]) or not -1e-8<=s[level]<=s[capacity]+1e-7:self.fail(name+"_bounds",s[level],step)
        self.minimum_soc=min(self.minimum_soc,s["battery_soc"])
        if s["battery_soc"]<env.config.battery.min_soc-1e-9:self.fail("physical_battery_floor",s["battery_soc"],step)
        limits={"electrolyser_power_kw":env.config.electrolyser.rated_kw,"dac_power_kw":env.config.dac.rated_kw,"methanator_power_kw":env.config.methanator.rated_kw,"solar_to_battery_kw":env.config.battery.max_charge_kw,"battery_to_electrolyser_kw":env.config.battery.max_discharge_kw}
        for field,limit in limits.items():
            if not math.isfinite(s[field]) or not -1e-8<=s[field]<=limit+1e-6:self.fail(field+"_limit",s[field],step)
        if step:
            if self.delivery_cache_decision!=env.decisions_taken:
                self.deliveries_by_step={}
                for settlement in env.simulator.delivery_audit()["settlements"]:
                    due=settlement["due_step"]
                    self.deliveries_by_step[due]=self.deliveries_by_step.get(due,0.0)+settlement["delivered_kg"]
                    if (settlement["delivered_kg"]<0 or settlement["unmet_kg"]<0
                        or abs(settlement["requested_kg"]-settlement["delivered_kg"]-settlement["unmet_kg"])>1e-7):
                        self.fail("delivery_mass_accounting",due,step)
                self.delivery_cache_decision=env.decisions_taken
            methane_produced=s["ch4_produced_total_kg"]-self.previous["ch4_produced_total_kg"]
            h2_produced=s["h2_produced_total_kg"]-self.previous["h2_produced_total_kg"]
            co2_captured=s["co2_captured_total_kg"]-self.previous["co2_captured_total_kg"]
            delivered=self.deliveries_by_step.get(step,0.0);self.delivered_checked_kg+=delivered
            expected_changes={"h2":h2_produced-methane_produced*env.config.methanator.h2_kg_per_kg_ch4,
                              "co2":co2_captured-methane_produced*env.config.methanator.co2_kg_per_kg_ch4,
                              "ch4":methane_produced-delivered}
            for resource,change in expected_changes.items():
                residual=abs(s[resource+"_level_kg"]-self.previous[resource+"_level_kg"]-change)
                self.max_gas_balance_residual[resource]=max(self.max_gas_balance_residual[resource],residual)
                if residual>1e-6:self.fail(resource+"_mass_balance",residual,step)
            for resource,produced in (("h2",h2_produced),("co2",co2_captured),("ch4",methane_produced)):
                if produced < -1e-8:self.fail(resource+"_cumulative_production_decreased",produced,step)
            expected=env.frozen_source.expected_solar_kw[step-1]
            if abs(s["solar_available_kw"]-expected)>1e-7:self.fail("solar_forcing",s["solar_available_kw"]-expected,step)
            solar_total=sum(s[k] for k in ("solar_to_battery_kw","solar_to_electrolyser_kw","solar_to_dac_kw","solar_to_methanator_kw","solar_curtailed_kw"))
            residual=abs(solar_total-s["solar_available_kw"]);self.max_power_balance_residual=max(self.max_power_balance_residual,residual)
            if residual>1e-6:self.fail("solar_power_balance",residual,step)
            # No refused process solar may be silently redirected into charging.
            explicit_charge=s["action"]["battery_share"]*s["solar_available_kw"]
            if s["solar_to_battery_kw"]>explicit_charge+1e-6:self.fail("unrequested_battery_charge",s["solar_to_battery_kw"]-explicit_charge,step)
            if abs(s["electrolyser_power_kw"]-s["solar_to_electrolyser_kw"]-s["battery_to_electrolyser_kw"])>1e-6:self.fail("electrolyser_power_balance",s["electrolyser_power_kw"],step)
            delta=(s["solar_to_battery_kw"]*env.config.battery.charge_efficiency-s["battery_to_electrolyser_kw"]/env.config.battery.discharge_efficiency)*env.config.dt_h
            residual=abs(s["battery_energy_kwh"]-self.previous["battery_energy_kwh"]-delta);self.max_battery_balance_residual=max(self.max_battery_balance_residual,residual)
            if residual>1e-6:self.fail("battery_energy_balance",residual,step)
            schedule=env.frozen_source.original_manifest["disturbances"]["schedule"]
            outage=any(f["start_s"]<=(step-1)*env.config.dt_s<f["start_s"]+f["duration_s"] for f in schedule)
            for unit,health in s["health"].items():
                if health["offline"]!=(outage if unit=="dac" else False) or health["degradation"]!=0:self.fail("declared_health_schedule",unit,step)
            for unit,field in (("electrolyser","electrolyser_power_kw"),("dac","dac_power_kw"),("methanator","methanator_power_kw")):
                if s["health"][unit]["offline"] and abs(s[field])>1e-7:self.fail("offline_power",unit,step)
        self.previous=s;self.count+=1

    def result(self):
        return {"all_physical_invariants_passed":self.failure_count==0,"states_checked":self.count,"failure_count":self.failure_count,"first_failures":self.failures,
                "minimum_battery_soc":self.minimum_soc if self.count else None,"physical_battery_floor":self.env.config.battery.min_soc,
                "maximum_solar_power_balance_residual_kw":self.max_power_balance_residual,"maximum_battery_energy_balance_residual_kwh":self.max_battery_balance_residual,
                "maximum_gas_mass_balance_residual_kg":self.max_gas_balance_residual,"settled_delivery_mass_checked_kg":self.delivered_checked_kg,
                "stream_sha256":self.sha.hexdigest(),"stream_encoding":"SHA256 over each canonical JSON state prefixed by unsigned8-byte big-endian byte length",
                "daily_returns_gate":False,"forced_stops_gate":False,"critical_status_gate":False}


def economic_audit(env, initial, final, reward_sum):
    e=env.economic_metrics.to_dict();c=e["components_eur"];a=env.economic_assumptions
    def equivalent(state):return state["ch4_level_kg"]+min(state["h2_level_kg"]/env.config.methanator.h2_kg_per_kg_ch4,state["co2_level_kg"]/env.config.methanator.co2_kg_per_kg_ch4)
    methane=max(0,equivalent(initial)-equivalent(final));battery=max(0,initial["battery_energy_kwh"]-final["battery_energy_kwh"])
    residuals={"component_sum_eur":e["net_operating_value_eur"]-sum(c.values()),"interval_reward_sum_eur":reward_sum-e["net_operating_value_eur"],
               "methane_terminal_liability_eur":c["methane_endowment_liability_change"]+methane*a.methane_endowment_value_eur_per_kg_ch4_equivalent,
               "battery_terminal_liability_eur":c["battery_endowment_liability_change"]+battery*a.battery_endowment_value_eur_per_kwh,
               "delivered_service_value_eur":c["delivered_service_value"]-e["delivered_kg"]*a.delivered_service_value_eur_per_kg,
               "unmet_delivery_penalty_eur":c["unmet_delivery_penalty"]+e["unmet_kg"]*a.unmet_delivery_penalty_eur_per_kg,
               "solar_opportunity_cost_eur":c["solar_opportunity_cost"]}
    return {"passed":all(math.isfinite(x) and abs(x)<=1e-6 for x in residuals.values()),"residuals":residuals,
            "initial_methane_equivalent_kg":equivalent(initial),"final_methane_equivalent_kg":equivalent(final),"terminal_methane_equivalent_deficit_kg":methane,
            "terminal_battery_deficit_kwh":battery,"interpretation":"Liability changes telescope to ending deficits. Positive excess terminal stocks earn no extra salvage credit. Daily individual deficits are not scalar penalties."}


class CoverageTranslation:
    def __init__(self):
        from docs.research.tools.offgrid_coverage_v1 import OffgridCoverageController
        self.prior=OffgridCoverageController();self.prior.reset()

    def action(self,env):
        from plant_backend.rl.v2.setpoint_command import dispatch_process_setpoint_command
        decision=self.prior.compute_command(env.cyclic_market_controller_observation,constraints=env.constraints,config=env.config,commitment_profile=env.episode_config.electrolyser_commitment)
        public=env.telemetry
        dispatch=dispatch_process_setpoint_command(decision.command,public,solar_available_kw=public.solar_available_kw,constraints=env.constraints,config=env.config)
        vector=environment().action_to_vector(dispatch.allocation.action,max_discharge_kw=env.config.battery.max_discharge_kw)
        return vector,{"baseline_targets":decision.command.targets.to_dict(),"translation_boundary_step":public.step,"translation_solar_kw":public.solar_available_kw,"translated_action":dispatch.allocation.action.to_dict()}


def evaluate_case(*,freeze_path,gate_path,endpoints_path,kind,split,seed,optimizer_seed=None,checkpoint_steps=None,output,retain_minutes=False,max_decisions=None):
    import numpy as np
    freeze=require_freeze(freeze_path);label,endpoint=admit(freeze,split=split,kind=kind,optimizer_seed=optimizer_seed,checkpoint_steps=checkpoint_steps,gate=gate_path,endpoints=endpoints_path)
    if split!="train" and seed not in freeze["selection"][split+"_seeds"]:raise ValueError("Scenario is outside selected split")
    if max_decisions is not None and (split!="train" or not 1<=max_decisions<=672):raise ValueError("Engineering fixture length must be1..672 on training seeds")
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    started=time.monotonic();env=model=audit=None;failure=None;commands=[];states=[];latencies=[];reward_sum=0.0;terminated=truncated=False
    input_digest=hashlib.sha256();clipped_means=0;checkpoint=None;checkpoint_steps_actual=0
    try:
        if kind!="coverage":
            trainer=training();trainer.configure_torch();path=checkpoint_path(freeze["training_root"],kind,optimizer_seed,checkpoint_steps)
            model=trainer.load_checkpoint(path,Path(freeze["training_root"])/"plan.json");checkpoint=checkpoint_record(path);checkpoint_steps_actual=model.num_timesteps
            expected_steps=0 if kind=="untrained" else checkpoint_steps if checkpoint_steps is not None else freeze["selection"]["final_timesteps"]
            if model.num_timesteps!=expected_steps or model.research_identity!=environment().environment_identity():raise ValueError("Wrong checkpoint budget or runtime identity")
            if endpoint and checkpoint!={k:v for k,v in endpoint["endpoints"][label].items() if k!="result"}:raise ValueError("Held-out model not identical to frozen endpoint")
        env=environment().DirectPlantEnv(split=split,gate_path=gate_path,record_minutes=True)
        obs,_=env.reset(seed=seed);initial=env.simulator.state.to_dict();audit=MinuteAudit(env);audit.add(initial)
        if retain_minutes:states.append(initial)
        baseline=CoverageTranslation() if kind=="coverage" else None
        for index in range(min(max_decisions or 672,672)):
            if not np.array_equal(obs,env.current_policy_observation):raise ValueError("Evaluation input differs from actual current observation")
            t=time.perf_counter()
            if baseline:action,evidence=baseline.action(env)
            else:
                details=model.deterministic_details(obs.copy());action,_=model.predict(obs.copy(),deterministic=True)
                if not np.array_equal(action,details["clipped_action"]):raise ValueError("Recorded inference differs from predicted action")
                mean=np.asarray(details["unclipped_mean"]);clipped_means+=int(np.any(mean!=action))
                evidence={"observation_sha256":canonical_sha(obs.tolist()),"normalized_observation_sha256":canonical_sha(np.asarray(details["normalized_input"]).tolist()),"raw_mean":mean.tolist(),"action":np.asarray(action).tolist()}
                if retain_minutes:evidence.update(observation=obs.tolist(),normalized_observation=np.asarray(details["normalized_input"]).tolist())
            elapsed=time.perf_counter()-t;latencies.append(elapsed)
            input_digest.update(json.dumps(evidence,sort_keys=True,separators=(",",":"),allow_nan=False).encode())
            obs,reward,terminated,truncated,info=env.step(action);reward_sum+=reward
            trace=env.last_minute_truth_states
            if trace is None or len(trace)!=16:raise ValueError("Missing15-minute exact physical trace")
            for state in trace[1:]:
                audit.add(state)
                if retain_minutes:states.append(state.to_dict())
            commands.append({"decision_index":index,"action":np.asarray(action).tolist(),"evidence":evidence,"reward_eur":reward,
                             "direct_control":info["direct_control"],"state":{k:getattr(env.simulator.state,k) for k in ("battery_energy_kwh","battery_soc","h2_level_kg","co2_level_kg","ch4_level_kg","solar_available_kw","electrolyser_power_kw","dac_power_kw","methanator_power_kw")}})
            if terminated or truncated:break
    except BaseException as exc:
        failure={"type":type(exc).__name__,"message":str(exc)}
    try:
        if env is None or audit is None:return failed_case(seed,label,split,failure or {"type":"SetupFailure"},freeze,output,time.monotonic()-started)
        physical=env.metrics.to_dict();economic=env.economic_metrics.to_dict();delivery=env.simulator.delivery_audit();cycles=[s.to_dict() for s in env.cycle_settlements]
        complete=failure is None and terminated and not truncated and env.decisions_taken==672 and env.physics_steps_taken==10080 and audit.count==10081
        audit_result=audit.result();econcheck=economic_audit(env,initial,env.simulator.state.to_dict(),reward_sum)
        delivery_pass=complete and len(delivery["settlements"])==28 and all(r["unmet_kg"]<=1e-9 for r in delivery["settlements"])
        physical_pass=complete and audit_result["all_physical_invariants_passed"]
        manifest=env.audit_manifest()
        row={"schema":SCHEMA+"-case","seed":seed,"controller":label,"split":split,"optimizer_seed":optimizer_seed if kind=="ppo" else 101 if kind=="untrained" else None,
             "checkpoint_timesteps":checkpoint_steps_actual,"checkpoint":checkpoint,"evaluation_freeze_sha256":freeze["sha256"],"endpoint_freeze_sha256":endpoint["sha256"] if endpoint else None,
             "execution_complete":bool(complete),"every_delivery_passed":bool(delivery_pass),"physical_invariants_passed":bool(physical_pass),"delivery_and_physics_passed":bool(delivery_pass and physical_pass),
             "failure":failure,"decisions":env.decisions_taken,"physical_steps":env.physics_steps_taken,"wall_time_s":time.monotonic()-started,
             "delivery_settlements":len(delivery["settlements"]),"maximum_delivery_unmet_kg":max((r["unmet_kg"] for r in delivery["settlements"]),default=0),"daily_returns_met":sum(c["feasible"] for c in cycles),
             "stock_deficits":{r:{"observed_daily_deficits":[c["deficits"][r] for c in cycles],"sum":sum(c["deficits"][r] for c in cycles),"max":max((c["deficits"][r] for c in cycles),default=0)} for r in RESOURCES},
             "physical_metrics":physical,"economic_metrics":economic,"minute_audit":audit_result,"economic_audit":econcheck,
             "initial_state":initial,"final_state":env.simulator.state.to_dict(),"exogenous_episode_sha256":manifest["beta_exogenous_episode_sha256"],"manifest_sha256":manifest["manifest_sha256"],
             "controller_latency_s":{"mean":statistics.mean(latencies) if latencies else None,"max":max(latencies,default=0),"calls":len(latencies)},
             "inference":{"verified_decisions":len(commands),"actual_observation_inputs":True,"raw_mean_clipped_decisions":clipped_means,"witness_sha256":input_digest.hexdigest()},
             "retained_minute_truth_sha256":canonical_sha(states) if states else None}
        record={"schema":SCHEMA+"-recording","summary":row,"decision_trace":commands,"cycle_settlements":cycles,"delivery":delivery,"manifest":manifest,
                "minute_truth_states":states if states else None,"trace_selection":"Firstprospectivelyselectedscenario" if retain_minutes else "Everyminuteauditedandstreaminghashed;rawminutesnotretained"}
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open("xb") as stream:stream.write(gzip.compress(json.dumps(record,separators=(",",":"),allow_nan=False).encode(),mtime=0))
        return {**row,"recording":file_pin(output)}
    finally:
        if env is not None:env.close()
        if model is not None:model.close()


def failed_case(seed,label,split,failure,freeze,output,elapsed=0):
    return {"schema":SCHEMA+"-case","seed":seed,"controller":label,"split":split,"optimizer_seed":None,"checkpoint_timesteps":None,"evaluation_freeze_sha256":freeze["sha256"],
            "execution_complete":False,"every_delivery_passed":False,"physical_invariants_passed":False,"delivery_and_physics_passed":False,
            "failure":failure,"decisions":0,"physical_steps":0,"wall_time_s":elapsed,"delivery_settlements":0,"maximum_delivery_unmet_kg":None,"daily_returns_met":0,
            "stock_deficits":{r:{"observed_daily_deficits":[],"sum":0,"max":0} for r in RESOURCES},"physical_metrics":{},"economic_metrics":{},
            "minute_audit":{"all_physical_invariants_passed":False,"states_checked":0},"economic_audit":{"passed":False},"recording":None}


def run_panel(args):
    freeze=require_freeze(args.freeze);label,_=admit(freeze,split=args.split,kind=args.controller,optimizer_seed=args.optimizer_seed,checkpoint_steps=args.checkpoint_steps,gate=args.gate,endpoints=args.endpoints)
    seeds=list(args.seeds) if args.seeds is not None else freeze["selection"][args.split+"_seeds"] if args.split!="train" else []
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError("Empty or duplicate panel; training fixtures need explicit seeds")
    if args.split!="train" and seeds!=freeze["selection"][args.split+"_seeds"]:raise ValueError("Evaluation must retain whole prescribed panel")
    if not 1<=args.workers<=3 or args.case_timeout<=0:raise ValueError("Invalid bounded concurrency or timeout")
    args.output.mkdir(parents=True,exist_ok=False);free=shutil.disk_usage(args.output).free
    if free<args.min_free_gib*1024**3:raise OSError("Insufficient disk reserve")
    publish(args.output/"attempt.json",{"controller":label,"split":args.split,"requested_seeds":seeds,"evaluation_freeze_sha256":freeze["sha256"],"pid":os.getpid(),"case_timeout_s":args.case_timeout,"workers":args.workers,"free_disk_bytes":free})
    active={};lock=threading.Lock();cancelled=threading.Event()
    def stop_active(signum=None,frame=None):
        cancelled.set()
        with lock:
            for process in active.values():
                if process.poll() is None:
                    try:os.killpg(process.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
    previous_handler=signal.getsignal(signal.SIGTERM)
    if threading.current_thread() is threading.main_thread():signal.signal(signal.SIGTERM,stop_active)
    def one(seed):
        out=args.output/f"case-{seed}.json.gz";resultpath=args.output/f"case-{seed}.result.json";log=args.output/f"case-{seed}.log"
        command=[sys.executable,str(Path(__file__).resolve()),"case","--freeze",str(args.freeze),"--controller",args.controller,"--split",args.split,"--seed",str(seed),"--output",str(out),"--result",str(resultpath)]
        for key in ("gate","endpoints","optimizer_seed","checkpoint_steps","max_decisions"):
            value=getattr(args,key,None)
            if value is not None:command.extend(["--"+key.replace("_","-"),str(value)])
        if seed==seeds[0]:command.append("--retain-minutes")
        start=time.monotonic()
        if cancelled.is_set() or shutil.disk_usage(args.output).free<args.min_free_gib*1024**3:
            row=failed_case(seed,label,args.split,{"type":"PanelCancelledOrDiskReserve","message":"No attempt started"},freeze,out)
            publish(resultpath,row);return row
        with log.open("x") as stream:
            envvars={**os.environ,"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
            with lock:
                process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=envvars,start_new_session=True)
                active[seed]=process
                if cancelled.is_set():os.killpg(process.pid,signal.SIGTERM)
            try:
                stop_at=start+args.case_timeout;terminated_at=None
                while process.poll() is None:
                    if cancelled.is_set() or time.monotonic()>=stop_at:
                        if terminated_at is None:
                            terminated_at=time.monotonic()
                            try:os.killpg(process.pid,signal.SIGTERM)
                            except ProcessLookupError:pass
                        elif time.monotonic()-terminated_at>15:
                            try:os.killpg(process.pid,signal.SIGKILL)
                            except ProcessLookupError:pass
                    try:process.wait(timeout=.25)
                    except subprocess.TimeoutExpired:pass
            finally:
                with lock:active.pop(seed,None)
        if resultpath.exists():
            row=json.loads(resultpath.read_text())
        else:
            row=failed_case(seed,label,args.split,{"type":"WorkerFailed","returncode":process.returncode,"log":file_pin(log)},freeze,out,time.monotonic()-start)
            publish(resultpath,row)
        print(json.dumps({"controller":label,"seed":seed,"complete":row["execution_complete"],"deliveries":row["every_delivery_passed"],"seconds":round(time.monotonic()-start,2)}),flush=True)
        return row
    rows=[]
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(one,seed) for seed in seeds]
            try:
                for future in as_completed(futures):rows.append(future.result())
            except BaseException:
                stop_active()
                raise
    finally:
        stop_active()
        if threading.current_thread() is threading.main_thread():signal.signal(signal.SIGTERM,previous_handler)
    unchanged=freeze["sources"]==sources()
    summary={"schema":SCHEMA+"-panel","controller":label,"split":args.split,"requested_seeds":seeds,"runs":sorted(rows,key=lambda r:r["seed"]),
             "evaluation_freeze_sha256":freeze["sha256"],"source_files_unchanged_during_panel":unchanged,"completed_at":datetime.now(timezone.utc).isoformat()}
    publish(args.output/"summary.json",summary)
    if not unchanged:raise ValueError("Source changed during panel")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("freeze");p.add_argument("--training-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p=sub.add_parser("freeze-endpoints");p.add_argument("--freeze",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    for name in ("panel","case"):
        p=sub.add_parser(name);p.add_argument("--freeze",type=Path,required=True);p.add_argument("--gate",type=Path);p.add_argument("--endpoints",type=Path)
        p.add_argument("--controller",choices=("coverage","untrained","ppo"),required=True);p.add_argument("--split",choices=("train","development","held_out"),required=True)
        p.add_argument("--optimizer-seed",type=int);p.add_argument("--checkpoint-steps",type=int);p.add_argument("--max-decisions",type=int);p.add_argument("--output",type=Path,required=True)
        if name=="case":p.add_argument("--seed",type=int,required=True);p.add_argument("--result",type=Path,required=True);p.add_argument("--retain-minutes",action="store_true")
        else:p.add_argument("--seeds",type=int,nargs="+");p.add_argument("--workers",type=int,default=3);p.add_argument("--case-timeout",type=float,default=300);p.add_argument("--min-free-gib",type=float,default=2)
    args=parser.parse_args()
    if args.command=="freeze":result=freeze_evaluation(args.training_root,args.output)
    elif args.command=="freeze-endpoints":result=freeze_endpoints(args.freeze,args.output)
    elif args.command=="panel":result=run_panel(args)
    else:
        def interrupted(signum,frame):raise TimeoutError(f"Parent watchdog signal{signum}; partial execution retained")
        signal.signal(signal.SIGTERM,interrupted)
        result=evaluate_case(freeze_path=args.freeze,gate_path=args.gate,endpoints_path=args.endpoints,kind=args.controller,split=args.split,seed=args.seed,optimizer_seed=args.optimizer_seed,checkpoint_steps=args.checkpoint_steps,output=args.output,retain_minutes=args.retain_minutes,max_decisions=args.max_decisions)
        publish(args.result,result)
    print(json.dumps({"schema":result.get("schema"),"output":str(args.output),"sha256":result.get("sha256")}),flush=True)


if __name__=="__main__":main()
