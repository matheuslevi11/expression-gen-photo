"""Unit tests for scripts/normalize_labels_per_actor.py.

Run with ``python -m pytest tests/`` or ``python tests/test_label_normalization.py``.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from normalize_labels_per_actor import actor_baselines, normalize  # noqa: E402


def test_normalize_maps_zero_baseline_and_one():
    for b in (0.05, 0.27, 0.73):
        assert normalize(0.0, b, 0.02) == 0.0
        assert abs(normalize(b, b, 0.02) - 0.02) < 1e-12
        assert abs(normalize(1.0, b, 0.02) - 1.0) < 1e-12
    assert abs(normalize(0.5, 0.25, 0.0) - 1 / 3) < 1e-12  # floor 0 = the plain clip((x - b) / (1 - b))


def test_normalize_is_continuous_and_strictly_monotone():
    for b in (0.05, 0.27, 0.73):
        x = np.linspace(0, 1, 2001)
        y = np.array([normalize(v, b, 0.02) for v in x])
        assert np.all(np.diff(y) > 0)
        eps = 1e-9
        assert abs(normalize(b - eps, b, 0.02) - normalize(b + eps, b, 0.02)) < 1e-6


def test_actor_baselines_use_only_neutral_frames():
    recs = [
        {"actor": "A", "emotion": "neutral", "frames": [{"au12": v} for v in (0.1, 0.2, 0.9)]},
        {"actor": "A", "emotion": "happy", "frames": [{"au12": 0.0}]},
        {"actor": "B", "emotion": "neutral", "frames": [{"au12": 0.4}]},
        {"actor": "C", "emotion": "happy", "frames": [{"au12": 0.5}]},
    ]
    assert actor_baselines(recs) == {"A": (0.2, 3), "B": (0.4, 1)}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
