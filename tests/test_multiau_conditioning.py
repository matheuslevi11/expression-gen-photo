"""Unit tests for multi-AU conditioning in genphoto/data/expression_dataset.py (E1, 2026-11-19).

CLIP is stubbed (no weights load). Run with ``python -m pytest tests/`` or
``python tests/test_multiau_conditioning.py``.
"""

import json
import random
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_num_threads(1)

import genphoto.data.expression_dataset as ed  # noqa: E402

H, W = 256, 384
TEMPLATES = ["<smile intensity: {value:.3f}>", "<brow raise intensity: {value:.3f}>",
             "<brow lower intensity: {value:.3f}>"]
KEYS = ["au12", "brow_raise", "brow_lower"]
RAMP_KEYS = {"happy": "au12", "surprised": "brow_raise", "angry": "brow_lower"}


class FakeTok:
    model_max_length = 77

    def __call__(self, prompts, **kw):
        ids = torch.tensor([[sum(map(ord, p)) * (i + 1) % 997 for i in range(77)] for p in prompts])
        return types.SimpleNamespace(input_ids=ids)


class FakeEnc:
    _proj = torch.randn(997, 768, generator=torch.Generator().manual_seed(0))

    def __call__(self, input_ids):
        return types.SimpleNamespace(last_hidden_state=self._proj[input_ids])

    def eval(self):
        return self

    def parameters(self):
        return []


ed.CLIPTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeTok())
ed.CLIPTextModel = types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeEnc())
ed.cv2 = types.SimpleNamespace(imread=lambda p: np.zeros((8, 8, 3), np.uint8), cvtColor=lambda im, c: im,
                               COLOR_BGR2RGB=None)


def _dataset(label_keys, templates, **kw):
    rng = np.random.default_rng(0)
    clips = []
    for a in ("A0", "A1", "A2"):
        for emo in ("neutral", "happy", "surprised", "angry"):
            for c in range(2):
                frames = [{"path": f"{a}/{emo}/{c}/{k}.jpg", "au12": float(rng.uniform()),
                           "brow_raise": float(rng.uniform()), "brow_lower": float(rng.uniform())}
                          for k in range(24)]
                clips.append({"clip_id": f"{a}_{emo}_{c}", "actor": a, "emotion": emo, "frames": frames})
    tmp = Path(tempfile.mkdtemp())
    (tmp / "ann.json").write_text(json.dumps(clips))
    args = dict(root_path=str(tmp), annotation_json="ann.json", clip_model_path="x", sample_size=(H, W),
                emotion_filter="all", label_keys=label_keys, prompt_templates=templates, ramp_keys=RAMP_KEYS,
                ramp_emotions=["happy", "surprised", "angry"],
                bridge_emotions=["neutral", "happy", "surprised", "angry"],
                sequence_mode_probs={"ramp": 0.25, "bridge": 0.25, "constant": 0.3, "permuted": 0.2})
    args.update(kw)
    return ed.ExpressionMEAD(**args)


def test_physical_channels_carry_one_key_each():
    v = torch.tensor([[0.1, 0.5, 0.9]] * 5)
    emb = ed.create_intensity_embedding(v, 4, 6)
    assert emb.shape == (5, 3, 4, 6)
    for k in range(3):
        assert torch.all(emb[:, k] == v[0, k])
    single = ed.create_intensity_embedding(torch.tensor([[0.3]] * 5), 4, 6)
    assert torch.all(single == 0.3)


def test_three_templates_give_one_ccl_map_per_key():
    v = torch.rand(5, 3)
    emb = ed.build_conditioning(v, FakeTok(), FakeEnc(), H, W, TEMPLATES)
    assert emb.shape == (5, 6, H, W)
    for k in range(3):
        ref = ed.build_ccl_embedding(v[:, k:k + 1], FakeTok(), FakeEnc(), H, W, prompt_template=TEMPLATES[k])
        assert torch.equal(emb[:, 3 + k], ref[:, 0])
        assert torch.all(emb[:, k] == v[:, k].view(5, 1, 1))
    assert not torch.equal(emb[:, 3], emb[:, 4])


def test_one_template_is_the_single_axis_embedding():
    v = torch.rand(5, 1)
    emb = ed.build_conditioning(v, FakeTok(), FakeEnc(), H, W, TEMPLATES[:1])
    ref = torch.cat((ed.create_intensity_embedding(v, H, W),
                     ed.build_ccl_embedding(v, FakeTok(), FakeEnc(), H, W, prompt_template=TEMPLATES[0])), dim=1)
    assert torch.equal(emb, ref)


def test_ramps_are_ordered_by_the_emotions_key():
    ds = _dataset(KEYS, TEMPLATES)
    for idx, clip in enumerate(ds.dataset):
        if clip["emotion"] == "neutral":
            continue
        key = RAMP_KEYS[clip["emotion"]]
        picks = ds._train_frames(idx, "ramp")
        vals = [clip["frames"][i][key] for _, i in picks]
        assert all(c == idx for c, _ in picks) and vals == sorted(vals), (clip["emotion"], vals)


def test_bridges_cover_every_high_emotion_within_one_actor():
    ds = _dataset(KEYS, TEMPLATES)
    random.seed(0)
    seen = set()
    for _ in range(300):
        picks = ds._train_frames(0, "bridge")
        clips = [ds.dataset[c] for c, _ in picks]
        assert len({c["actor"] for c in clips}) == 1
        highs = {c["emotion"] for c in clips} - {"neutral"}
        assert len(highs) <= 1
        if highs:
            high = highs.pop()
            seen.add(high)
            vals = [ds.dataset[c]["frames"][i][RAMP_KEYS[high]] for c, i in picks]
            assert vals == sorted(vals)
    assert seen == {"happy", "surprised", "angry"}


def test_constant_sequences_draw_every_key_within_tolerance():
    ds = _dataset(KEYS, TEMPLATES, constant_level_tolerance=0.1)
    assert sorted(ds.constant_pools) == ["au12", "brow_lower", "brow_raise"]  # neutral orders by au12
    random.seed(1)
    for _ in range(200):
        picks = ds._train_frames(0, "constant")
        frames = [ds.dataset[c]["frames"][i] for c, i in picks]
        assert any(max(f[k] for f in frames) - min(f[k] for f in frames) <= 0.2 + 1e-9 for k in KEYS)


def test_arms_differing_only_in_label_keys_see_the_same_frames():
    t = _dataset(KEYS, TEMPLATES)
    c = _dataset(["au12"], TEMPLATES[:1])
    for seed in range(50):
        random.seed(seed)
        bt = t[seed % len(t)]
        random.seed(seed)
        bc = c[seed % len(c)]
        assert bt["clip_name"] == bc["clip_name"]
        assert torch.equal(bt["intensity_values"][:, :1], bc["intensity_values"])
        assert bt["camera_embedding"].shape == bc["camera_embedding"].shape == (5, 6, H, W)


def test_validation_targets_drive_the_ordering_channel():
    ds = _dataset(KEYS, TEMPLATES, is_Train=False, emotion_filter="all", sequence_mode_probs=None)
    targets = [0.0, 0.25, 0.5, 0.75, 1.0]
    for clip in ds.dataset:
        clip["intensity_list"] = targets
    for idx, clip in enumerate(ds.dataset):
        _, _, values, _ = ds._load_clip(idx)
        assert values.shape == (5, 3)
        key = RAMP_KEYS.get(clip["emotion"], "au12")
        col = KEYS.index(key)
        assert values[:, col].tolist() == targets
        order = ed._select_intensity_progression([f[key] for f in clip["frames"]], 5)
        for j in range(3):
            if j != col:
                own = [clip["frames"][i][KEYS[j]] for i in order]
                assert torch.allclose(values[:, j], torch.tensor(own))


def test_validation_targets_drive_the_clips_own_channel():
    ds = _dataset(KEYS, TEMPLATES, is_Train=False, sequence_mode_probs=None, emotion_filter="happy")
    targets = [0.0, 0.25, 0.5, 0.75, 1.0]
    for c in ds.dataset:
        c["intensity_list"] = targets
    b = ds[0]
    v = b["intensity_values"]
    assert v.shape == (5, 3) and v[:, 0].tolist() == targets  # happy -> smile channel
    shown = ds._clip_intensities(0)
    picks = ed._select_intensity_progression(shown, 5)
    assert v[:, 1].tolist() == [torch.tensor(ds.dataset[0]["frames"][i]["brow_raise"]).float().item() for i in picks]


def test_mismatched_keys_and_templates_are_rejected():
    for keys, temps in ((KEYS, TEMPLATES[:1]), (KEYS[:2], TEMPLATES[:2])):
        try:
            _dataset(keys, temps)
        except ValueError:
            continue
        raise AssertionError(f"accepted {keys} / {temps}")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
