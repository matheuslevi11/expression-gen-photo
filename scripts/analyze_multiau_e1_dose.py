"""Multi-AU E1 dose-response: does T's smile-amplitude cost against C shrink with training?

Pre-registered in docs/experiments/2026-11-21-multi-au-e1-dose-response-design.md. Reads the
`calibration_jitter` protocol of T and C at several EMA checkpoints (same units, same noise).

    python scripts/analyze_multiau_e1_dose.py \
        --t 10000=inference_output/batch_eval/e1/multiau_t_ema10000 ... 25000=.../multiau_t_ema25000 \
        --c 10000=inference_output/batch_eval/e1/multiau_c_ema10000 ... 25000=.../multiau_c_ema25000 \
        --out report.json

Per unit (prompt, seed: 11 levels x 5 frames) the response slope is the OLS slope of the detected
smile on the commanded level. gap(s) = mean over units of slope_T(s) - slope_C(s).
Primary: d_gap = gap(25k) - gap(10k) on MediaPipe, 2,000-draw bootstrap over units, all models
and steps resampled together (paired). Category: closing (CI > 0), widening (CI < 0), constant
(CI includes 0). The same rule on pooled CLS is the check: if the two disagree, inconclusive.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_multiau_e1 as A  # noqa: E402

JUDGES = {"mediapipe": "smile_mp", "pyfeat": "smile_pf", "libreface": "smile_lf"}
PRIMARY = "mediapipe"
FIRST, LAST, PREV = 10000, 25000, 20000
N_BOOT, SEED = 2000, 0
N_UNITS, N_LEVELS = 90, 11


def load(model_dir: Path) -> dict:
    units = A.load_experiment(model_dir, "calibration_jitter")
    groups = {}
    for u in units.values():
        groups.setdefault((u["prompt_idx"], u["seed"]), []).append(u)
    keys = sorted(groups)
    rows = [sorted(groups[k], key=lambda u: u["level"]) for k in keys]
    if len(keys) != N_UNITS or any(len(r) != N_LEVELS for r in rows):
        raise SystemExit(f"{model_dir}: {len(keys)} units, level counts {sorted({len(r) for r in rows})}; "
                         f"expected {N_UNITS} x {N_LEVELS}")
    out = {"keys": keys, "x": np.array([[[u["level"]] * 5 for u in r] for r in rows], float)}  # [n, 11, 5]
    for j, m in JUDGES.items():
        out[j] = np.array([[u[m] for u in r] for r in rows], float)
    return out


def slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-unit OLS slope of y on x over finite frames; x, y are [n, 11, 5]."""
    m = np.isfinite(y)
    cnt = m.sum(axis=(1, 2))
    xm = np.where(m, x, 0).sum(axis=(1, 2)) / cnt
    ym = np.where(m, y, 0).sum(axis=(1, 2)) / cnt
    dx = np.where(m, x - xm[:, None, None], 0)
    dy = np.where(m, y - ym[:, None, None], 0)
    return (dx * dy).sum(axis=(1, 2)) / (dx * dx).sum(axis=(1, 2))


def cls(x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> float:
    xs, ys = x[idx].ravel(), y[idx].ravel()
    m = np.isfinite(ys)
    return float(np.corrcoef(xs[m], ys[m])[0, 1])


def ci(point: float, draws: np.ndarray) -> dict:
    draws = draws[np.isfinite(draws)]
    return dict(value=float(point), lo=float(np.percentile(draws, 2.5)), hi=float(np.percentile(draws, 97.5)))


def category(d: dict) -> str:
    return "closing" if d["lo"] > 0 else "widening" if d["hi"] < 0 else "constant"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t", nargs="+", required=True, help="step=dir, one per checkpoint")
    ap.add_argument("--c", nargs="+", required=True, help="step=dir, the same steps as --t")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    dirs = {name: {int(s.split("=", 1)[0]): Path(s.split("=", 1)[1]) for s in specs}
            for name, specs in (("T", args.t), ("C", args.c))}
    steps = sorted(dirs["T"])
    if sorted(dirs["C"]) != steps or not {FIRST, PREV, LAST} <= set(steps):
        raise SystemExit(f"T and C need the same steps, including {FIRST}, {PREV}, {LAST}")

    data = {(n, s): load(dirs[n][s]) for n in dirs for s in steps}
    keys = data[("T", steps[0])]["keys"]
    if any(d["keys"] != keys for d in data.values()):
        raise SystemExit("units differ between directories")
    for d in data.values():
        if not np.array_equal(d["x"], data[("T", steps[0])]["x"]):
            raise SystemExit("commanded levels differ between directories")
    x = data[("T", steps[0])]["x"]

    rng = np.random.default_rng(SEED)
    boot = rng.integers(0, N_UNITS, size=(N_BOOT, N_UNITS))
    rep = {"steps": steps, "units": N_UNITS, "judges": {}}

    print(f"== {N_UNITS} units x {N_LEVELS} levels, steps {steps}")
    for j in JUDGES:
        sl = {k: slopes(x, d[j]) for k, d in data.items()}
        # per-unit slope means and the T - C gap, with paired unit resampling
        rj = {"slope": {}, "gap": {}, "cls": {}, "cls_gap": {}, "curve": {}}
        for k, v in sl.items():
            rj["slope"][f"{k[0]}@{k[1]}"] = ci(np.nanmean(v), np.nanmean(v[boot], axis=1))
        gap = {s: sl[("T", s)] - sl[("C", s)] for s in steps}
        for s in steps:
            rj["gap"][s] = ci(np.nanmean(gap[s]), np.nanmean(gap[s][boot], axis=1))
        dg = gap[LAST] - gap[FIRST]
        rj["d_gap"] = ci(np.nanmean(dg), np.nanmean(dg[boot], axis=1))
        for n in ("T", "C"):
            dd = sl[(n, LAST)] - sl[(n, PREV)]
            rj[f"{n}_last_minus_prev"] = ci(np.nanmean(dd), np.nanmean(dd[boot], axis=1))

        # pooled CLS, same resampling
        allidx = np.arange(N_UNITS)
        cpt = {k: cls(x, d[j], allidx) for k, d in data.items()}
        cb = {k: np.array([cls(x, d[j], b) for b in boot]) for k, d in data.items()}
        for k in data:
            rj["cls"][f"{k[0]}@{k[1]}"] = ci(cpt[k], cb[k])
        for s in steps:
            rj["cls_gap"][s] = ci(cpt[("T", s)] - cpt[("C", s)], cb[("T", s)] - cb[("C", s)])
        rj["cls_d_gap"] = ci((cpt[("T", LAST)] - cpt[("C", LAST)]) - (cpt[("T", FIRST)] - cpt[("C", FIRST)]),
                             (cb[("T", LAST)] - cb[("C", LAST)]) - (cb[("T", FIRST)] - cb[("C", FIRST)]))
        for k, d in data.items():
            rj["curve"][f"{k[0]}@{k[1]}"] = [float(v) for v in np.nanmean(d[j], axis=(0, 2))]
        rep["judges"][j] = rj

        f = lambda d: f"{d['value']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"
        print(f"\n-- {j}")
        print("   step     slope T                      slope C                      gap T - C")
        for s in steps:
            print(f"   {s:6d}   {f(rj['slope'][f'T@{s}'])}  {f(rj['slope'][f'C@{s}'])}  {f(rj['gap'][s])}")
        print(f"   d_gap {LAST} - {FIRST}: {f(rj['d_gap'])} -> {category(rj['d_gap'])}")
        print(f"   T {LAST} - {PREV}: {f(rj['T_last_minus_prev'])}   C {LAST} - {PREV}: {f(rj['C_last_minus_prev'])}")
        print("   step     CLS T                        CLS C                        CLS gap")
        for s in steps:
            print(f"   {s:6d}   {f(rj['cls'][f'T@{s}'])}  {f(rj['cls'][f'C@{s}'])}  {f(rj['cls_gap'][s])}")
        print(f"   CLS d_gap {LAST} - {FIRST}: {f(rj['cls_d_gap'])} -> {category(rj['cls_d_gap'])}")
        for k in sorted(rj["curve"]):
            print(f"   curve {k:8s} " + " ".join(f"{v:.3f}" for v in rj["curve"][k]))

    p = rep["judges"][PRIMARY]
    a, b = category(p["d_gap"]), category(p["cls_d_gap"])
    rep["outcome"] = a if a == b else "inconclusive"
    rep["outcome_detail"] = dict(slope=a, cls=b)
    print(f"\n== OUTCOME ({PRIMARY}): slope {a}, CLS {b} -> {rep['outcome'].upper()}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=1, default=lambda v: None if isinstance(v, float) and math.isnan(v) else v))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
