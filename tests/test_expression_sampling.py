"""Unit tests for the train-time sequence sampling in genphoto/data/expression_dataset.py.

Run with ``python -m pytest tests/`` or ``python tests/test_expression_sampling.py``.
"""

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genphoto.data.expression_dataset import (  # noqa: E402
    _build_bridge_pairs,
    _build_constant_pool,
    _normalize_mode_probs,
    _select_bridge_clips,
    _select_bridge_progression,
    _select_constant_frames,
    _select_intensity_progression,
    _select_permuted_progression,
)

N = 5
LEVELS = [i / 10 for i in range(11)]


def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


def test_normalize_mode_probs():
    assert _normalize_mode_probs({"ramp": 5, "constant": 3, "permuted": 2}) == {
        "ramp": 0.5, "constant": 0.3, "permuted": 0.2,
    }
    assert _normalize_mode_probs({"ramp": 1, "permuted": 0}) == {"ramp": 1.0}
    assert _normalize_mode_probs({"ramp": 1, "bridge": 1}) == {"ramp": 0.5, "bridge": 0.5}
    assert _raises(lambda: _normalize_mode_probs({"ramp": 1, "reverse": 1}))
    assert _raises(lambda: _normalize_mode_probs({"ramp": -1, "constant": 2}))
    assert _raises(lambda: _normalize_mode_probs({"ramp": 0}))


def test_constant_pool_drops_unsupported_levels_and_respects_tolerance():
    clips = [
        [0.02, 0.05, 0.08, 0.04, 0.06, 0.9, 0.95],   # 5 frames near 0.0, only 2 near 0.9/1.0
        [0.5] * 3 + [0.95] * 6,                      # only 3 near 0.5 -> cannot fill 0.5
    ]
    pool = dict(_build_constant_pool(clips, N, LEVELS, tolerance=0.1))
    assert set(pool) == {0.0, 0.1, 0.9, 1.0}
    assert [c for c, _ in pool[0.0]] == [0]
    assert pool[1.0] == [(1, [3, 4, 5, 6, 7, 8])]
    for level, candidates in pool.items():
        for clip_idx, near in candidates:
            assert len(near) >= N
            assert all(abs(clips[clip_idx][i] - level) <= 0.1 for i in near)


def test_constant_selection_rebalances_levels_uniformly():
    # One rare low-intensity clip vs many high-intensity clips: level choice must stay uniform.
    clips = [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05]] + [[0.95] * 8 for _ in range(50)]
    pool = _build_constant_pool(clips, N, LEVELS, tolerance=0.1)
    rng = random.Random(0)
    draws = [_select_constant_frames(pool, N, rng) for _ in range(6000)]
    picked_low = sum(1 for clip_idx, _ in draws if clip_idx == 0)
    # levels {0.0, 0.1} come only from clip 0; {0.9, 1.0} only from the rest -> ~50/50.
    assert 0.45 < picked_low / len(draws) < 0.55
    for clip_idx, frames in draws[:200]:
        assert len(frames) == N == len(set(frames))
        assert frames == sorted(frames)


def test_permuted_uses_ramp_frames_in_non_identity_order():
    rng = random.Random(1)
    intensities = [0.9, 0.1, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4]
    ramp = _select_intensity_progression(intensities, N)
    orders = Counter()
    for _ in range(2000):
        perm = _select_permuted_progression(intensities, N, rng)
        assert sorted(perm) == sorted(ramp)
        assert perm != ramp
        orders[tuple(perm)] += 1
    assert len(orders) > 100  # most of the 119 non-identity orders appear


def test_permuted_single_frame_is_ramp():
    assert _select_permuted_progression([0.3, 0.6], 1, random.Random(0)) == _select_intensity_progression([0.3, 0.6], 1)


def test_bridge_pairs_need_both_emotions_per_actor():
    actors = ["A", "A", "B", "A", "C", "C", "B"]
    emotions = ["neutral", "happy", "happy", "Happy", "neutral", "sad", "happy"]
    assert _build_bridge_pairs(actors, emotions, "neutral", "happy") == [([0], [1, 3])]
    assert _build_bridge_pairs(actors, emotions, "neutral", "angry") == []


def test_bridge_clip_selection_is_uniform_over_actors_and_same_actor():
    actors = ["A"] * 40 + ["B"] * 4
    emotions = (["neutral"] * 10 + ["happy"] * 30) + ["neutral", "happy", "happy", "happy"]
    pairs = _build_bridge_pairs(actors, emotions, "neutral", "happy")
    rng = random.Random(0)
    draws = [_select_bridge_clips(pairs, rng) for _ in range(4000)]
    for low, high in draws:
        assert actors[low] == actors[high]
        assert (emotions[low], emotions[high]) == ("neutral", "happy")
    assert 0.45 < sum(actors[low] == "B" for low, _ in draws) / len(draws) < 0.55


def test_bridge_progression_ramps_from_low_clip_to_high_clip():
    low = [0.05, 0.0, 0.3, 0.01, 0.02, 0.1]
    high = [0.4, 0.9, 0.6, 0.95, 0.8, 0.5]
    picks = _select_bridge_progression(low, high, N)
    values = [(low, high)[side][i] for side, i in picks]
    assert values == sorted(values)
    assert picks[0] == (0, 1) and picks[-1] == (1, 3)  # global min (low clip) .. global max (high clip)
    assert len(set(picks)) == N
    assert _select_bridge_progression([0.1, 0.2, 0.3], [0.7, 0.8, 0.9], 6) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
