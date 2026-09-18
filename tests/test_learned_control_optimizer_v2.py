"""Independent optimizer groups, KL stopping, transfer and serialization."""
from datetime import datetime, timedelta, timezone
import copy
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from docs.research.tools.learned_control_optimizer_v2 import training as t
from docs.research.tools.learned_control_optimizer_v2.algorithm import GroupedPPO
from docs.research.tools.learned_control_optimizer_v2.policy import GroupedHybridPolicy, parameter_partition
from docs.research.tools.learned_control_hybrid_v1.policy import HybridActorCriticPolicy
from docs.research.tools.learned_control_asymmetric_v1.policy import encode_conditions


class ToyEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-10,10,shape=(149,),dtype=np.float32)
        self.action_space = gym.spaces.Box(-1,1,shape=(7,),dtype=np.float32)
        self.training_scenario = None
    def reset(self, *, seed=None, options=None):
        self.steps = 0
        self.training_scenario = {"condition":"ordinary", "seed":50_300_000}
        return np.zeros(149,dtype=np.float32), {"training_scenario":dict(self.training_scenario)}
    def step(self, action):
        self.steps += 1
        obs=np.zeros(149,dtype=np.float32);obs[0]=self.steps/8
        return obs, float(1-.1*np.square(action).sum()), self.steps==8, False, {"training_scenario":dict(self.training_scenario)}


def toy(grouped=True, **kwargs):
    t.configure_torch()
    algorithm,policy=(GroupedPPO,GroupedHybridPolicy) if grouped else (PPO,HybridActorCriticPolicy)
    options=dict(n_steps=8,batch_size=4,n_epochs=1,seed=172,learning_rate=3e-7,
        target_kl=None,policy_kwargs={"net_arch":{"pi":[16,16],"vf":[16,16]},
            "min_on_std":.0001,"max_on_std":.1,"initial_on_std":.01})
    options.update(kwargs)
    model=algorithm(policy,DummyVecEnv([ToyEnv]),**options)
    model.set_logger(configure(folder=None,format_strings=[]))
    return model


def test_partition_is_exhaustive_disjoint_and_two_groups_reopen(tmp_path):
    model=toy()
    try:
        partition=parameter_partition(model.policy)
        ids=[id(p) for values in partition.values() for _,p in values]
        assert len(ids)==len(set(ids))==len(list(model.policy.parameters()))
        assert [group['name'] for group in model.policy.optimizer.param_groups]==['actor','critic']
        model.learn(8)
        path=tmp_path/'grouped.zip';model.save(path)
        loaded=PPO.load(path,device='cpu')
        assert t.common.policy_parameter_sha256(loaded)==t.common.policy_parameter_sha256(model)
        assert t.common.optimizer_sha256(loaded)==t.common.optimizer_sha256(model)
        assert len(loaded.policy.optimizer.param_groups)==2
        obs=np.zeros((3,149),dtype=np.float32)
        np.testing.assert_array_equal(model.predict(obs,deterministic=True)[0],loaded.predict(obs,deterministic=True)[0])
    finally:
        model.env.close()


def test_actor_kl_stop_preserves_nonempty_adam_state_while_critic_finishes():
    model=toy()
    try:
        model.learn(8)
        before=t.common.actor_parameter_sha256(model)
        actor_names=[name for name,_ in parameter_partition(model.policy)['actor']]
        state=t.optimizer_named_state(model.policy,actor_names)
        counts=model.optimizer_counters()
        # Force a clear stop at the first check, with an already initialized Adam.
        data=model.rollout_buffer
        with torch.no_grad():
            data.log_probs=model.policy.get_distribution(torch.as_tensor(data.observations)).log_prob(torch.as_tensor(data.actions)).numpy()-20
        model.target_kl=.02;model.n_epochs=3
        model.train()
        assert t.common.actor_parameter_sha256(model)==before
        assert t.optimizer_named_state(model.policy,actor_names)==state
        assert model.actor_minibatches_this_run==counts['actor_minibatches_this_run']
        assert model.critic_minibatches_this_run==counts['critic_minibatches_this_run']+6
        assert model.optimizer_counters()['critic_adam_step_min']==counts['critic_adam_step_min']+6
        assert model.logger.name_to_value['train/actor_kl_stopped']==1
        assert model.logger.name_to_value['train/critic_accepted_minibatches']==6
    finally:
        model.env.close()


def fill_identical_buffer(model, value_target):
    rng=np.random.default_rng(491)
    buffer=model.rollout_buffer
    for index in range(8):
        obs=rng.normal(size=(1,149)).astype(np.float32)
        context=encode_conditions(['ordinary'])
        with torch.no_grad():
            actions,values,logp=model.policy.forward_with_context(torch.from_numpy(obs),torch.from_numpy(context))
        buffer.add(obs,actions.numpy(),np.array([1.]),np.array([index==0]),values,logp,critic_context=context)
    buffer.returns[:]=value_target
    buffer.advantages[:,0]=np.linspace(-1,1,8)


def test_large_critic_gradients_do_not_rescale_actor_update():
    one,two=toy(batch_size=8),toy(batch_size=8)
    try:
        torch.manual_seed(22);fill_identical_buffer(one,1.)
        torch.manual_seed(22);fill_identical_buffer(two,10000.)
        np.random.seed(5);one.train()
        np.random.seed(5);two.train()
        assert t.common.actor_parameter_sha256(one)==t.common.actor_parameter_sha256(two)
        assert two.logger.name_to_value['train/critic_preclip_gradient_norm']>1000*one.logger.name_to_value['train/critic_preclip_gradient_norm']
        assert one.logger.name_to_value['train/actor_learning_rate']==3e-7
        assert one.logger.name_to_value['train/critic_learning_rate']==1e-4
    finally:
        one.env.close();two.env.close()


def test_named_single_to_two_group_transfer_preserves_existing_moments_and_actor():
    source,target=toy(grouped=False),toy()
    try:
        source.learn(8)
        original=t.common.actor_parameter_sha256(source)
        transfer=t.transfer_hybrid(source,target)
        assert transfer['actor_tensors_exact'] and transfer['actor_logprob_probe_exact']
        assert t.common.actor_parameter_sha256(target)==original
        assert transfer['existing_optimizer_named_states_exact']
        assert len(target.policy.optimizer.param_groups)==2
        assert target.optimizer_counters()['actor_adam_step_min']==2
    finally:
        source.env.close();target.env.close()


def test_actor_actions_and_logprob_do_not_depend_on_private_context():
    model=toy()
    try:
        obs=torch.zeros(7,149)
        contexts=torch.eye(7)
        with torch.no_grad():
            model.policy.private_value_head[-1].weight.fill_(.1)
            torch.manual_seed(111);a,v,logp=model.policy.forward_with_context(obs,contexts)
            torch.manual_seed(111);b,w,other=model.policy.forward_with_context(obs,torch.flip(contexts,[0]))
        assert torch.equal(a,b) and torch.equal(logp,other)
    finally:
        model.env.close()


def test_real_transfer_tiny_training_and_existing_evaluator_reopening(tmp_path):
    parent=t.ROOT/'docs/research/results/learned_control_20260918/calibration/hybrid-relative03/training/calibrated-policy.zip'
    if not parent.exists():
        pytest.skip('The campaign calibrated parent is local-only')
    r=t.default_recipe(initialization='continue',initial_checkpoint=str(parent),optimizer_seed=903,
        scenario_sampler_seed=9100,total_timesteps=32,checkpoint_steps=32,n_steps=8,environments=4,
        deadline_utc=(datetime.now(timezone.utc)+timedelta(minutes=5)).isoformat(),freeze_observation_statistics=True)
    r['ppo']['batch_size']=8;r['ppo']['n_epochs']=2
    r['ppo']['policy_kwargs'].update(net_arch={'pi':[256,256],'vf':[128,128]},min_on_std=.0001,max_on_std=.1,initial_on_std=.001)
    run=tmp_path/'run'
    plan=t.freeze_plan(run,r)
    result=t.train(run)
    assert result['status']=='completed'
    assert result['optimizer_counters']['critic_minibatches_this_run']==8
    assert result['optimizer_counters']['actor_minibatches_this_run']>0
    assert result['initial_checkpoint']['initialization_transfer']['actor_logprob_probe_exact']
    assert result['endpoint']['optimizer_learning_rates']==pytest.approx({'actor':1e-7,'critic':3e-5})
    loaded=t.load_checkpoint(run/'training/final-policy.zip',run/'plan.json')
    try:
        assert loaded.model.policy.observation_space.shape==(149,)
        assert loaded.model.policy.action_space.shape==(7,)
        assert loaded.model.critic_minibatches_this_run==8
    finally:
        loaded.close()
    records=[json.loads(line) for line in (run/'training/optimization.jsonl').read_text().splitlines()]
    assert len(records)==1 and records[0]['additional_timesteps']==32
    assert records[0]['optimizer_counters']['critic_minibatches_this_run']==8
    assert t.require_plan(run)==plan


def test_optimization_deadline_does_not_advance_parameters_or_adam():
    model=toy()
    try:
        model.learn(8)
        parameters=t.common.policy_parameter_sha256(model)
        states=t.optimizer_named_state(model.policy)
        counts=model.optimizer_counters()
        model.research_training_deadline_utc='2020-01-01T00:00:00Z'
        model.train()
        assert model.optimizer_deadline_reached
        assert t.common.policy_parameter_sha256(model)==parameters
        assert t.optimizer_named_state(model.policy)==states
        assert model.optimizer_counters()==counts
        assert model.collect_rollouts(None,None,None,None) is False
    finally:
        model.env.close()
