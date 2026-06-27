"""eval_max_chunks should match env TimeLimit / action_chunk_horizon."""

from utils.trl_critic_config import eval_max_chunks_for_env


def test_humanoidmaze_eval_max_chunks():
    assert eval_max_chunks_for_env('hmm', action_chunk_horizon=5) == 400
    assert eval_max_chunks_for_env('hml', action_chunk_horizon=5) == 400


def test_antmaze_eval_max_chunks():
    assert eval_max_chunks_for_env('amm', action_chunk_horizon=5) == 200
    assert eval_max_chunks_for_env('aml', action_chunk_horizon=5) == 200


def test_manip_eval_max_chunks():
    assert eval_max_chunks_for_env('p3', action_chunk_horizon=5) == 100
