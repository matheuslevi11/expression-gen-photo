"""Multi-AU E0 analysis: the pre-registered criteria of
docs/experiments/2026-11-15-multi-au-e0-design.md, computed from

  * py-feat all-AU labels   <processed>/<emotion>/shard_*/annotations/train.json (preprocess --all-aus)
  * geometry scores         geo.jsonl        (score_mead_frames.py --scorer geo)
  * LibreFace scores        libreface.jsonl  (score_mead_frames.py --scorer libreface)

Frame keys are the records' relative paths (frames/<actor>/<emotion>/level_<n>/<clip>/frame_k.jpg).

    python scripts/analyze_multiau_e0.py --processed /petrobr/crono/.../MEAD_multiau_processed \
        --geo .../geo.jsonl --libreface .../libreface.jsonl \
        --existing /petrobr/crono/.../MEAD_processed --out .../e0_report.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DETECTORS = ("pyfeat", "libreface", "geo")
CHANNELS = ("smile", "raise", "lower")
TARGET = {"raise": "surprised", "lower": "angry"}
# V passes with >= 90% of actors: 9/10 on the primary panel, 34/37 on the replication (amendment of 2026-11-18)
V_MIN_FRAC, A_MIN_R, PYFEAT_TIE, MERGE_MIN_R, S_MAX_R = 0.9, 0.4, 0.05, 0.5, 0.7


# ----------------------------------------------------------------------------- loading

def load_frames(processed: Path) -> pd.DataFrame:
    rows = []
    for ann in sorted(processed.glob("*/shard_*/annotations/train.json")) + \
               sorted(processed.glob("*/shard_*/annotations/validation.json")):
        for rec in json.loads(ann.read_text()):
            for k, f in enumerate(rec["frames"]):
                aus = f.get("aus") or {}
                row = {"path": f["path"], "actor": rec["actor"], "emotion": rec["emotion"],
                       "level": 0 if rec["emotion"] == "neutral" else int(rec["intensity_level"]),
                       "clip_key": f'{rec["actor"]}/{rec["emotion"]}/{rec["intensity_level"]}/{rec["clip_id"]}',
                       "frame": k, "au12_field": f["au12"], "pf_face": bool(f.get("aus")),
                       "abs_path": str(ann.parent.parent / f["path"])}
                row.update({f"pf_{a}": (np.nan if v is None else v) for a, v in aus.items()})
                rows.append(row)
    df = pd.DataFrame(rows)
    assert not df.empty, f"no annotations under {processed}"
    assert df["path"].is_unique
    return df


def load_scores(path: Path, key: str, prefix: str) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            rows.append({"path": d["path"], **{f"{prefix}{k}": (np.nan if v is None else v)
                                               for k, v in d[key].items()}})
    return pd.DataFrame(rows).drop_duplicates("path", keep="last")


def channel_columns(df: pd.DataFrame) -> pd.DataFrame:
    c = pd.DataFrame(index=df.index)
    c["smile_pyfeat"], c["smile_libreface"], c["smile_geo"] = df["pf_AU12"], df["lf_AU12"], df["geo_AU12_mouth_corner"]
    c["raise_pyfeat"] = df[["pf_AU01", "pf_AU02"]].mean(axis=1, skipna=False)
    c["raise_libreface"] = df[["lf_AU01", "lf_AU02"]].mean(axis=1, skipna=False)
    c["raise_geo"] = df["geo_AU1_2_brow_raise"]
    c["lower_pyfeat"], c["lower_libreface"], c["lower_geo"] = df["pf_AU04"], df["lf_AU04"], -df["geo_AU4_inner_brow_gap"]
    # single-AU brow-raise variants for the AU01/AU02 merge rule
    for d, p in (("pyfeat", "pf_"), ("libreface", "lf_")):
        c[f"au01_{d}"], c[f"au02_{d}"] = df[f"{p}AU01"], df[f"{p}AU02"]
    return c


# ----------------------------------------------------------------------------- statistics

def centered(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] - df.groupby("actor")[col].transform("mean")


def pooled_r(df: pd.DataFrame, a: str, b: str, mask=None) -> tuple[float, int]:
    d = df if mask is None else df[mask]
    x, y = centered(d, a), centered(d, b)
    m = x.notna() & y.notna()
    if m.sum() < 3:
        return float("nan"), int(m.sum())
    return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())


def clip_values(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.groupby(["actor", "emotion", "level", "clip_key"], as_index=False)[col].mean()


def validity(df, col, target):
    cv = clip_values(df, col)
    per = {}
    for a, g in cv.groupby("actor"):
        t = g[(g.emotion == target) & (g.level == 3)][col].dropna()
        n = g[g.emotion == "neutral"][col].dropna()
        per[a] = float(t.median() - n.median()) if len(t) and len(n) else float("nan")
    n_pos = sum(1 for v in per.values() if v > 0)
    need = math.ceil(V_MIN_FRAC * len(per) - 1e-9)
    return {"per_actor_diff": per, "n_positive": n_pos, "n_actors": len(per), "n_required": need,
            "sign_test_p_one_sided": float(stats.binomtest(n_pos, len(per), 0.5, alternative="greater").pvalue)
            if per else float("nan"), "pass": bool(per) and n_pos >= need}


def dose(df, col, target):
    cv = clip_values(df, col)
    rs = []
    for _, g in cv.groupby("actor"):
        g = g[(g.emotion == "neutral") | (g.emotion == target)].dropna(subset=[col])
        if g.level.nunique() > 1:
            rs.append(stats.spearmanr(g.level, g[col]).correlation)
    return {"median_spearman": float(np.median(rs)) if rs else float("nan"), "n_actors": len(rs)}


def actor_scaled(df, col):
    neu = df[df.emotion == "neutral"].groupby("actor")[col].median()
    p99 = df.groupby("actor")[col].quantile(0.99)
    den = (p99 - neu).where(lambda s: s.abs() > 1e-9)
    return (df[col] - df["actor"].map(neu)) / df["actor"].map(den)


# ----------------------------------------------------------------------------- report

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--geo", type=Path, required=True)
    ap.add_argument("--libreface", type=Path, required=True)
    ap.add_argument("--existing", type=Path, default=None, help="MEAD_processed root for label continuity")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--luma-frames", type=int, default=40)
    args = ap.parse_args()

    df = load_frames(args.processed)
    df = df.merge(load_scores(args.geo, "geo", "geo_"), on="path", how="left")
    df = df.merge(load_scores(args.libreface, "libreface", "lf_"), on="path", how="left")
    ch = channel_columns(df)
    df = pd.concat([df, ch], axis=1)
    rep = {"n_frames": len(df), "n_clips": int(df["clip_key"].nunique()), "actors": sorted(df.actor.unique()),
           "clips_per_emotion": df.groupby("emotion")["clip_key"].nunique().to_dict()}
    print(f"{len(df)} frames, {rep['n_clips']} clips, actors {rep['actors']}")

    # 4. detector failures
    fail = {}
    for e, g in df.groupby("emotion"):
        fail[e] = {"pyfeat_noface": float((~g.pf_face).mean()),
                   "libreface_nan": float(g["lf_AU12"].isna().mean()),
                   "geo_nan": float(g["geo_AU1_2_brow_raise"].isna().mean())}
    rep["failures"] = fail
    print("\n== failure rates (pyfeat no-face / libreface NaN / geo NaN)")
    for e, v in fail.items():
        print(f"   {e:10s} {v['pyfeat_noface']:.3f} {v['libreface_nan']:.3f} {v['geo_nan']:.3f}")

    # A yardstick: smile
    rep["A_smile_yardstick"] = {f"{a}~{b}": pooled_r(df, f"smile_{a}", f"smile_{b}")[0]
                                for a, b in (("pyfeat", "geo"), ("pyfeat", "libreface"), ("libreface", "geo"))}
    print("\n== A yardstick (smile, actor-centered frame r):", {k: round(v, 3) for k, v in rep["A_smile_yardstick"].items()})

    # V, A, D per brow channel x labeler
    crit = {}
    for chn in ("raise", "lower"):
        crit[chn] = {}
        print(f"\n== {chn} (target {TARGET[chn]} level 3 vs neutral)")
        for d in DETECTORS:
            col = f"{chn}_{d}"
            v = validity(df, col, TARGET[chn])
            others = {o: pooled_r(df, col, f"{chn}_{o}")[0] for o in DETECTORS if o != d}
            a_pass = any(r >= A_MIN_R for r in others.values())
            crit[chn][d] = {"V": v, "A": {"r": others, "mean": float(np.nanmean(list(others.values()))), "pass": a_pass},
                            "D": dose(df, col, TARGET[chn]), "pass": v["pass"] and a_pass}
            print(f"   {d:9s} V {v['n_positive']}/{v['n_actors']} (p {v['sign_test_p_one_sided']:.3f}) "
                  f"A {', '.join(f'{o} {r:+.3f}' for o, r in others.items())} -> {'pass' if crit[chn][d]['pass'] else 'FAIL'}"
                  f"   D median rho {crit[chn][d]['D']['median_spearman']:+.2f}")
    rep["criteria"] = crit

    # labeler choice
    choice = {}
    for chn in ("raise", "lower"):
        ok = {d: c["A"]["mean"] for d, c in crit[chn].items() if c["pass"]}
        if not ok:
            choice[chn] = None
            continue
        best = max(ok, key=ok.get)
        if "pyfeat" in ok and ok[best] - ok["pyfeat"] <= PYFEAT_TIE:
            best = "pyfeat"
        choice[chn] = best
    rep["labeler"] = choice
    print("\n== labeler choice:", choice)

    # AU01/AU02 merge rule (on the chosen brow-raise labeler; geometry has a single brow-raise measure)
    merge = None
    if choice["raise"] in ("pyfeat", "libreface"):
        d = choice["raise"]
        r12 = pooled_r(df, f"au01_{d}", f"au02_{d}")[0]
        if r12 >= MERGE_MIN_R:
            merge = {"r_AU01_AU02": r12, "decision": "mean(AU01, AU02)"}
        else:
            a_single = {}
            for au in ("au01", "au02"):
                rs = [pooled_r(df, f"{au}_{d}", f"raise_{o}")[0] for o in DETECTORS if o != d]
                a_single[au] = max(rs)
            pick = "AU02" if a_single["au02"] >= a_single["au01"] else "AU01"
            merge = {"r_AU01_AU02": r12, "A_single": a_single, "decision": pick}
        print("== AU01/AU02 merge:", merge)
    rep["merge"] = merge

    # S gate on chosen labelers (smile labeler = py-feat, the existing training label)
    chosen = {"smile": "smile_pyfeat", **{c: f"{c}_{choice[c]}" for c in ("raise", "lower") if choice[c]}}
    S = {f"smile~{c}": pooled_r(df, chosen["smile"], chosen[c])[0] for c in ("raise", "lower") if c in chosen}
    if "raise" in chosen and "lower" in chosen:
        S["raise~lower"] = pooled_r(df, chosen["raise"], chosen["lower"])[0]
    separable = [c for c in ("raise", "lower") if c in chosen and abs(S[f"smile~{c}"]) <= S_MAX_R]
    rep["S"] = {"r": S, "separable_from_smile": separable}
    print("== S (separability):", {k: round(v, 3) for k, v in S.items()}, "separable from smile:", separable)

    if len(separable) == 2 and abs(S["raise~lower"]) <= S_MAX_R:
        decision = "GO: 3 channels (smile, raise, lower)"
    elif separable:
        keep = max(separable, key=lambda c: crit[c][choice[c]]["A"]["mean"])
        decision = f"GO: 2 channels (smile + {keep})"
    elif chosen.keys() - {"smile"}:
        decision = "STOP: brow channels pass V and A but are collinear with smile (S)"
    else:
        decision = "STOP: no brow channel passes V and A"
    rep["decision"] = decision
    print("\n== DECISION:", decision)

    # ------------------------------------------------------------------ descriptive
    desc = {}
    neu = df.emotion == "neutral"
    desc["speech_confound_neutral"] = {f"{c}_{d}": pooled_r(df, f"{c}_{d}", "geo_AU25_26_inner_lip_gap", neu)[0]
                                       for c in ("raise", "lower") for d in DETECTORS}
    for c in ("smile", "raise", "lower"):
        df[f"{c}_scaled"] = actor_scaled(df, chosen[c]) if c in chosen else np.nan
    rng = {}
    for c in ("raise", "lower"):
        if c not in chosen:
            continue
        g = df[(df.emotion == TARGET[c]) & (df.level == 3)]
        per_clip = g.groupby("clip_key")[f"{c}_scaled"].agg(lambda s: s.quantile(0.9) - s.quantile(0.1))
        rng[c] = float((per_clip >= 0.3).mean())
    desc["within_clip_range_ge_0.3"] = rng
    if all(c in chosen for c in CHANNELS):
        bins = df[[f"{c}_scaled" for c in CHANNELS]].dropna().ge(0.5).astype(int)
        cells = bins.value_counts().sort_index()
        desc["joint_coverage"] = {"".join(map(str, k)): int(v) for k, v in cells.items()}
    # skin tone (luma of a central crop patch on neutral frames)
    try:
        import cv2
        lum = {}
        for a, g in df[neu].groupby("actor"):
            ys = []
            for p in g.abs_path.iloc[:: max(1, len(g) // args.luma_frames)][: args.luma_frames]:
                im = cv2.imread(p)
                h, w = im.shape[:2]
                patch = im[int(.35 * h):int(.65 * h), int(.35 * w):int(.65 * w)]
                ys.append(float(np.median(cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)[..., 0])))
            lum[a] = float(np.median(ys))
        sk = {}
        for c in CHANNELS:
            for d in DETECTORS:
                med = df[neu].groupby("actor")[f"{c}_{d}"].median()
                common = [a for a in lum if a in med and np.isfinite(med[a])]
                sk[f"{c}_{d}"] = float(stats.spearmanr([lum[a] for a in common], [med[a] for a in common]).correlation)
        desc["skin_tone_spearman_neutral_median_vs_luma"] = sk
        desc["actor_luma"] = lum
    except (AttributeError, ValueError) as e:  # unreadable crop
        print(f"skin-tone check skipped: {e!r}")
    # label continuity with the existing training root
    if args.existing:
        old = {}
        for name in ("train_happy_neutral.json", "validation_happy_neutral.json"):
            p = args.existing / "annotations" / name
            if p.exists():
                for rec in json.loads(p.read_text()):
                    for f in rec["frames"]:
                        old[f["path"]] = f["au12"]
        m = df[df.path.isin(old.keys())]
        if len(m):
            a, b = m.au12_field.to_numpy(), m.path.map(old).to_numpy()
            desc["label_continuity_au12"] = {"n": len(m), "r": float(np.corrcoef(a, b)[0, 1]),
                                             "mean_abs_diff": float(np.abs(a - b).mean()),
                                             "max_abs_diff": float(np.abs(a - b).max())}
            import cv2
            sample = m.sample(min(300, len(m)), random_state=0)
            diffs = []
            for p, ap in zip(sample.path, sample.abs_path):
                new = cv2.imread(ap).astype(float)
                diffs.append(float(np.abs(new - cv2.imread(str(args.existing / p)).astype(float)).mean()))
            desc["crop_pixel_mean_abs_diff"] = {"n": len(diffs), "max": max(diffs), "mean": float(np.mean(diffs))}
    rep["descriptive"] = desc
    print("\n== descriptive")
    print(json.dumps(desc, indent=1, default=float)[:4000])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=1, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
