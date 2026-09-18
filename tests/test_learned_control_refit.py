from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import numpy as np
import pytest
from docs.research.tools.learned_control_refit_v1 import refit as r
from docs.research.tools.learned_control_release_v1.portable import checkpoint_plan


def test_ridge_correction_uses_only_selected_rows_and_centers_on_parent():
    rng=np.random.default_rng(17)
    x=rng.normal(size=(60,4));weight=np.array([.1,-.3,.7,.2]);bias=.3
    target=x@weight+bias
    selected=np.arange(60)%5!=0
    unchanged=r.ridge_correction(x,target,weight,bias,selected,.001)
    np.testing.assert_array_equal(unchanged[0],weight.astype(np.float32))
    assert unchanged[1]==np.float32(bias)
    desired=target+.2*x[:,0]
    one=r.ridge_correction(x,desired,weight,bias,selected,.001)
    desired[~selected]=10000
    two=r.ridge_correction(x,desired,weight,bias,selected,.001)
    np.testing.assert_array_equal(one[0],two[0]);assert one[1]==two[1]
    assert one[2]['fit_rows']==48


def test_real_refit_preserves_other_coordinates_and_portable_plan_closure(tmp_path):
    parent=r.ROOT/'docs/research/results/learned_control_20260918/runs/hybrid-dagger-fidelity/seed-903/training/initial-policy.zip'
    dataset=r.ROOT/'docs/research/results/learned_control_20260918/demonstrations/original-plus-dagger/demonstrations.npz'
    if not parent.exists():pytest.skip('Local campaign parent required')
    root=tmp_path/'refit'
    plan=r.freeze_plan(root,parent,dataset,deadline_utc=(datetime.now(timezone.utc)+timedelta(minutes=5)).isoformat())
    result=r.train(root)
    assert result['status']=='completed'
    assert result['witness']['other_action_coordinates_bitwise_unchanged']==[0,2,3,4,5,6]
    assert result['witness']['ppo_optimizer_exactly_preserved']
    assert not(set(result['witness']['fit_episode_indices'])&set(result['witness']['repair_holdout_episode_indices']))
    metadata_path=root/'training/initial-policy.json'
    metadata=json.loads(metadata_path.read_text())
    assert checkpoint_plan(metadata_path,metadata,tmp_path)==root/'plan.json'
    assert metadata['num_timesteps']==metadata['ppo_updates_this_run']==0
    assert metadata['normalizer']['sha256']==r.common.file_record(parent.with_suffix('.vecnormalize.pkl'))['sha256']
    assert r.require_plan(root)==plan
