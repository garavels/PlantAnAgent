from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from docs.research.tools.learned_control_transfer_v1 import transfer as t
from docs.research.tools.learned_control_release_v1.portable import checkpoint_plan


def test_real_algebraic_merge_preserves_selected_functions_and_reopens(tmp_path):
    root=t.refit.ROOT/'docs/research/results/learned_control_20260918'
    h=root/'runs/hybrid-dagger-fidelity/seed-903/training/initial-policy.zip'
    g=root/'runs/gaussian-dagger-cohort/seed-903/training/initial-policy.zip'
    dataset=root/'demonstrations/original-plus-dagger/demonstrations.npz'
    if not h.exists() or not g.exists():pytest.skip('Local campaign parents required')
    run=tmp_path/'merge'
    plan=t.freeze_plan(run,h,g,dataset,deadline_utc=(datetime.now(timezone.utc)+timedelta(minutes=5)).isoformat())
    result=t.train(run)
    assert result['status']=='completed'
    witness=result['witness']
    assert witness['exact_observation_witnesses']==80640
    assert witness['gaussian_mean_outputs_bitwise_equal']==[1,2]
    assert witness['preserved_action_coordinates_bitwise_equal']==[0,3,4,5,6]
    assert witness['all_parameters_trainable']
    loaded=t.hybrid.load_checkpoint(run/'training/initial-policy.zip')
    try:
        model=loaded.model
        assert model.policy_kwargs['net_arch']['pi']==[512,512]
        weight=model.policy.mlp_extractor.policy_net[2].weight
        assert not torch.count_nonzero(weight[:256,256:])
        assert not torch.count_nonzero(weight[256:,:256])
        assert model.policy.action_net.weight.shape==(21,512)
        assert model.num_timesteps==0 and not model.policy.optimizer.state
    finally:loaded.close()
    metadata=run/'training/initial-policy.json'
    assert checkpoint_plan(metadata,json.loads(metadata.read_text()),tmp_path)==run/'plan.json'
    assert t.require_plan(run)==plan
