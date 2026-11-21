"""Multi-AU E1 analysis: the pre-registered readings of
docs/experiments/2026-11-19-multi-au-e1-design.md.

Each model directory holds one sub-directory per experiment with
    manifest.jsonl            commands (``intensities`` = smile, ``brow_raise``, ``brow_lower``)
    results.jsonl             batch_eval score: py-feat AU12, MediaPipe smile, identity
    results_allau.jsonl       rescore_allau.py: 20 py-feat AUs + FaceMesh geometry
    results_libreface.jsonl   rescore_libreface.py: 12 LibreFace AU intensities

    python scripts/analyze_multiau_e1.py --t <T dir> --c <C dir> --m <main-model dir> --out e1_report.json

Δ = T − C paired over units (prompt, seed), 2,000-draw bootstrap, 95% percentile CI. Per-unit
metrics are averaged over units valid in both models; pooled CLS is recomputed on each
resample of units.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

N_BOOT, SEED = 2000, 0
RAMP_MIN_R, SMILE_TOL, ID_TOL = 0.5, -0.05, -0.03
ETHNIC_PROMPTS = range(14, 24)
CMD = {"smile": "intensities", "raise": "brow_raise", "lower": "brow_lower"}


# ----------------------------------------------------------------------------- loading

def _jsonl(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["id"]] = d
    return out


def _nanlist(v, n=5):
    if v is None:
        return [math.nan] * n
    return [math.nan if x is None else float(x) for x in v]


def load_experiment(model_dir: Path, exp: str) -> dict:
    """id -> unit record with commands and per-frame measures (lists of 5, NaN when missing)."""
    d = model_dir / exp
    man, res, allau, lf = (_jsonl(d / f) for f in
                           ("manifest.jsonl", "results.jsonl", "results_allau.jsonl", "results_libreface.jsonl"))
    units = {}
    for uid, m in man.items():
        r, a, l = res.get(uid, {}), allau.get(uid, {}), lf.get(uid, {})
        pa, la, geo = a.get("aus") or {}, l.get("aus") or {}, a.get("facemesh_geo") or {}
        mean2 = lambda x, y: list(np.nanmean([_nanlist(x), _nanlist(y)], axis=0)) if x is not None and y is not None else _nanlist(None)
        gap = geo.get("AU4_inner_brow_gap")
        CLUSTER[uid] = (m["prompt_idx"], m["seed"])
        units[uid] = dict(
            prompt_idx=m["prompt_idx"], seed=m["seed"], level=m.get("level"),
            cmd={k: m.get(v) for k, v in CMD.items()},
            smile_mp=_nanlist(r.get("mp_smile")), smile_pf=_nanlist(r.get("au12")), smile_lf=_nanlist(la.get("AU12")),
            raise_lf=mean2(la.get("AU01"), la.get("AU02")), raise_pf=mean2(pa.get("AU01"), pa.get("AU02")),
            raise_geo=_nanlist(geo.get("AU1_2_brow_raise")),
            lower_lf=_nanlist(la.get("AU04")), lower_pf=_nanlist(pa.get("AU04")),
            lower_geo=[-x for x in _nanlist(gap)],
            au25_lf=_nanlist(la.get("AU25")), au26_lf=_nanlist(la.get("AU26")),
            id_min=r.get("id_min_adjacent", math.nan), faces=r.get("id_faces_found", math.nan),
            scored=dict(results=uid in res, allau=uid in allau, libreface=uid in lf),
        )
    return units


# ----------------------------------------------------------------------------- statistics

def corr(a, b, spearman=False) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() < 1e-12 or b[m].std() < 1e-12:
        return math.nan
    return float((stats.spearmanr if spearman else stats.pearsonr)(a[m], b[m])[0])


CLUSTER = {}  # id -> (prompt_idx, seed), filled by load_experiment


def per_unit(units: dict, f) -> dict:
    return {uid: f(u) for uid, u in units.items()}


def _cluster_means(vals: dict, ids) -> np.ndarray:
    groups = {}
    for i in ids:
        groups.setdefault(CLUSTER[i], []).append(vals[i])
    return np.array([np.mean(v) for v in groups.values()])


def boot_mean_delta(t: dict, c: dict, rng) -> dict:
    """Paired mean of t − c over ids finite in both. Ids of one (prompt, seed) unit (e.g. the 3
    permuted lists) are averaged first; the CI resamples units."""
    ids = [i for i in t if i in c and np.isfinite(t[i]) and np.isfinite(c[i])]
    tv, cv = _cluster_means(t, ids), _cluster_means(c, ids)
    if len(tv) < 2:
        return dict(n=len(tv), t=math.nan, c=math.nan, delta=math.nan, lo=math.nan, hi=math.nan)
    d = tv - cv
    bs = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
    return dict(n=len(d), t=float(tv.mean()), c=float(cv.mean()), delta=float(d.mean()),
                lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))


def boot_mean(v: dict, rng) -> dict:
    x = _cluster_means(v, [i for i, y in v.items() if np.isfinite(y)])
    if len(x) < 2:
        return dict(n=len(x), mean=math.nan, lo=math.nan, hi=math.nan)
    bs = x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(axis=1)
    return dict(n=len(x), mean=float(x.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))


def pooled_cls(units: dict, measure: str, ids) -> float:
    lv, dv = [], []
    for i in ids:
        u = units[i]
        lv += [u["level"]] * 5
        dv += u[measure]
    return corr(lv, dv)


def boot_cls_delta(t: dict, c: dict, measure: str, rng) -> dict:
    """Pooled CLS difference; resampling (prompt, seed) clusters with all their levels."""
    ids = [i for i in t if i in c]
    clusters = {}
    for i in ids:
        clusters.setdefault((t[i]["prompt_idx"], t[i]["seed"]), []).append(i)
    keys = list(clusters)
    if len(keys) < 2:
        return dict(n=len(keys), t=math.nan, c=math.nan, delta=math.nan, lo=math.nan, hi=math.nan)
    ct, cc = pooled_cls(t, measure, ids), pooled_cls(c, measure, ids)
    bs = []
    for draw in rng.integers(0, len(keys), size=(N_BOOT, len(keys))):
        sel = [i for k in draw for i in clusters[keys[k]]]
        bs.append(pooled_cls(t, measure, sel) - pooled_cls(c, measure, sel))
    bs = np.array(bs)
    bs = bs[np.isfinite(bs)]
    return dict(n=len(keys), t=ct, c=cc, delta=ct - cc,
                lo=float(np.percentile(bs, 2.5)) if len(bs) else math.nan,
                hi=float(np.percentile(bs, 97.5)) if len(bs) else math.nan)


def ramp_r(axis: str, measure: str, spearman=False):
    return lambda u: corr(u["cmd"][axis], u[measure], spearman) if u["cmd"][axis] is not None else math.nan


def fmt(d: dict) -> str:
    if "delta" in d:
        return f"T {d['t']:+.3f}  C {d['c']:+.3f}  Δ {d['delta']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]  (n {d['n']})"
    return f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]  (n {d['n']})"


# ----------------------------------------------------------------------------- report

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t", type=Path, required=True)
    ap.add_argument("--c", type=Path, required=True)
    ap.add_argument("--m", type=Path, default=None, help="main model (smile protocols only)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    exps = ["scaled", "calibration_jitter", "permuted", "raise_scaled", "raise_calibration_jitter",
            "lower_scaled", "lower_calibration_jitter", "raise_lower_scaled"]
    T = {e: load_experiment(args.t, e) for e in exps}
    C = {e: load_experiment(args.c, e) for e in exps}
    M = {e: load_experiment(args.m, e) for e in ("scaled", "calibration_jitter", "permuted")} if args.m else {}
    rep = {"coverage": {}, "readings": {}, "guardrails": {}, "secondary": {}}

    print("== coverage (units: manifest / results / allau / libreface)")
    for name, model in (("T", T), ("C", C), ("M", M)):
        for e, units in model.items():
            cov = [len(units)] + [sum(u["scored"][k] for u in units.values()) for k in ("results", "allau", "libreface")]
            rep["coverage"][f"{name}/{e}"] = cov
            print(f"   {name} {e:26s} {cov}")

    def paired_ramp(exp, axis, measure, spearman=False):
        f = ramp_r(axis, measure, spearman)
        return boot_mean_delta(per_unit(T[exp], f), per_unit(C[exp], f), rng)

    R = rep["readings"]
    print("\n== readings")
    R["1_raise_control"] = r1 = paired_ramp("raise_scaled", "raise", "raise_lf")
    r1["pass"] = bool(r1["lo"] > 0 and r1["t"] >= RAMP_MIN_R)
    print(f"1 brow raise ramp r (LibreFace)   {fmt(r1)} -> {'pass' if r1['pass'] else 'FAIL'}")
    R["2_lower_control"] = r2 = paired_ramp("lower_scaled", "lower", "lower_lf")
    r2["pass"] = bool(r2["lo"] > 0 and r2["t"] >= RAMP_MIN_R)
    print(f"2 brow lower ramp r (LibreFace)   {fmt(r2)} -> {'pass' if r2['pass'] else 'FAIL'}")
    for chn, exp, meas in (("raise", "raise_calibration_jitter", "raise_lf"), ("lower", "lower_calibration_jitter", "lower_lf")):
        r3 = boot_cls_delta(T[exp], C[exp], meas, rng)
        r3["pass"] = bool(r3["lo"] > 0)
        R[f"3_{chn}_calibration"] = r3
        print(f"3 {chn} CLS (LibreFace)            {fmt(r3)} -> {'pass' if r3['pass'] else 'FAIL'}")
    r4 = {"cls_mp": boot_cls_delta(T["calibration_jitter"], C["calibration_jitter"], "smile_mp", rng),
          "ramp_mp": paired_ramp("scaled", "smile", "smile_mp"),
          "permuted_mp": paired_ramp("permuted", "smile", "smile_mp")}
    for k, v in r4.items():
        v["pass"] = bool(v["lo"] > SMILE_TOL)
        print(f"4 smile {k:12s}              {fmt(v)} -> {'pass' if v['pass'] else 'FAIL'}")
    r4["pass"] = all(v["pass"] for v in r4.values())
    R["4_smile_kept"] = r4
    leak = ramp_r("smile", "raise_lf", spearman=True)
    lt, lc = per_unit(T["scaled"], leak), per_unit(C["scaled"], leak)
    r5 = boot_mean_delta(lt, lc, rng)
    c_leak = boot_mean(lc, rng)
    r5["c_leak"] = c_leak
    r5["informative"] = bool(c_leak["lo"] > 0 or c_leak["hi"] < 0)
    r5["pass"] = bool(r5["informative"] and r5["hi"] < 0)
    R["5_localization"] = r5
    print(f"5 smile→brow-raise leakage (LF ρ) {fmt(r5)}; C alone {fmt(c_leak)} -> "
          f"{'pass' if r5['pass'] else ('uninformative' if not r5['informative'] else 'FAIL')}")

    if r1["pass"] and r2["pass"] and r4["pass"] and r5["pass"]:
        decision = "Full success"
    elif r1["pass"] and r2["pass"] and r4["pass"]:
        decision = "Brow control without localization"
    elif (r1["pass"] or r2["pass"]) and r4["pass"]:
        decision = f"Partial ({'brow raise' if r1['pass'] else 'brow lower'} kept)"
    else:
        decision = "Fail"
    rep["decision"] = decision
    print(f"\n== DECISION: {decision}")

    # ------------------------------------------------------------------ guardrails
    G = rep["guardrails"]
    print("\n== guardrails")
    for exp in ("scaled", "raise_scaled", "lower_scaled"):
        g = boot_mean_delta(per_unit(T[exp], lambda u: u["id_min"]), per_unit(C[exp], lambda u: u["id_min"]), rng)
        g["flag"] = bool(g["delta"] < ID_TOL and g["hi"] < 0)
        G[f"identity_{exp}"] = g
        print(f"   identity min-adjacent {exp:14s} {fmt(g)}{'  TRADE-OFF' if g['flag'] else ''}")
    for exp, axis, meas in (("scaled", "smile", "smile_mp"), ("raise_scaled", "raise", "raise_lf"),
                            ("lower_scaled", "lower", "lower_lf")):
        for name, model in (("T", T), ("C", C)):
            rs = list(per_unit(model[exp], ramp_r(axis, meas)).values())
            fail = float(np.mean([not (np.isfinite(r) and r >= RAMP_MIN_R) for r in rs])) if rs else math.nan
            undefined = float(np.mean([not np.isfinite(r) for r in rs])) if rs else math.nan
            G[f"fail_{name}_{exp}"] = dict(fail_rate=fail, undefined_rate=undefined, n=len(rs))
            print(f"   fail rate {name} {exp:14s} {fail:.3f} (undefined r {undefined:.3f}, n {len(rs)})")
    for name, model in (("T", T), ("C", C), ("M", M)):
        for exp, units in model.items():
            fr = lambda key: float(np.mean([not np.isfinite(x) for u in units.values() for x in u[key]])) if units else math.nan
            G[f"noface_{name}_{exp}"] = nf = dict(mediapipe=fr("smile_mp"), libreface=fr("smile_lf"),
                                                  facenet_lt5=float(np.mean([u["faces"] != 5 for u in units.values()])) if units else math.nan)
            print(f"   no-face {name} {exp:26s} MediaPipe {nf['mediapipe']:.3f}  LibreFace {nf['libreface']:.3f}  "
                  f"facenet<5 {nf['facenet_lt5']:.3f}")

    # ------------------------------------------------------------------ secondary
    S = rep["secondary"]
    print("\n== secondary")
    if M:
        for k, (exp, f) in {"ramp_mp": ("scaled", ramp_r("smile", "smile_mp")),
                            "permuted_mp": ("permuted", ramp_r("smile", "smile_mp")),
                            "leak_raise_lf": ("scaled", ramp_r("smile", "raise_lf", True)),
                            "leak_lower_lf": ("scaled", ramp_r("smile", "lower_lf", True))}.items():
            d = boot_mean_delta(per_unit(C[exp], f), per_unit(M[exp], f), rng)
            S[f"C_minus_M_{k}"] = d
            print(f"   C − M {k:14s} {fmt(d).replace('T ', 'C ').replace('  C ', '  M ', 1)}")
        d = boot_cls_delta(C["calibration_jitter"], M["calibration_jitter"], "smile_mp", rng)
        S["C_minus_M_cls_mp"] = d
        print(f"   C − M cls_mp         {fmt(d).replace('T ', 'C ').replace('  C ', '  M ', 1)}")
    for exp, axis in (("raise_scaled", "raise"), ("lower_scaled", "lower")):
        for meas in ("smile_mp", "smile_lf", "au25_lf", "au26_lf"):
            d = paired_ramp(exp, axis, meas, spearman=True)
            S[f"drag_{exp}_{meas}"] = d
            print(f"   {axis} command drags {meas:9s} (ρ)  {fmt(d)}")
    d = paired_ramp("scaled", "smile", "lower_lf", spearman=True)
    S["leak_lower_lf"] = d
    print(f"   smile→brow-lower leakage (LF ρ)  {fmt(d)}")
    for axis, meas in (("raise", "raise_lf"), ("lower", "lower_lf")):
        d = paired_ramp("raise_lower_scaled", axis, meas)
        S[f"combined_{axis}"] = d
        print(f"   combined ramp: {axis} r (LF)        {fmt(d)}")
    for exp, axis, meas in (("raise_scaled", "raise", "raise_pf"), ("lower_scaled", "lower", "lower_pf"),
                            ("lower_scaled", "lower", "lower_geo")):
        d = paired_ramp(exp, axis, meas)
        S[f"other_judge_{exp}_{meas}"] = d
        print(f"   {exp} ramp r ({meas})  {fmt(d)}")
    for chn, exp, meas in (("raise", "raise_calibration_jitter", "raise_pf"), ("lower", "lower_calibration_jitter", "lower_pf")):
        d = boot_cls_delta(T[exp], C[exp], meas, rng)
        S[f"labeler_cls_{chn}"] = d
        print(f"   {chn} CLS (py-feat)              {fmt(d)}")
    for exp, axis, meas in (("raise_scaled", "raise", "raise_lf"), ("raise_scaled", "raise", "raise_geo"),
                            ("lower_scaled", "lower", "lower_lf"), ("lower_scaled", "lower", "lower_geo")):
        for gname, keep in (("ethnicity_named", True), ("other", False)):
            sub = lambda model: {i: u for i, u in model[exp].items() if (u["prompt_idx"] in ETHNIC_PROMPTS) == keep}
            f = ramp_r(axis, meas)
            d = boot_mean_delta(per_unit(sub(T), f), per_unit(sub(C), f), rng)
            S[f"group_{exp}_{meas}_{gname}"] = d
            print(f"   {exp} {meas:9s} {gname:15s} {fmt(d)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=1, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
