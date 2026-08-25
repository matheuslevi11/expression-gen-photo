"""Analyze whole-face AU co-activation from ``results_allau.jsonl`` (re-scoring outputs).

Read-only aggregation. Computes, per AU:
  1. ramp correlation aggregated over clips (Fisher-z mean + BCa bootstrap CI), Spearman primary;
  2. per-commanded-level co-activation trajectories (mean +/- SEM);
  3. paired trained-vs-baseline Fisher-z difference (Wilcoxon + paired-t + bootstrap CI);
  4. Benjamini-Hochberg FDR across the 20 AUs;
  5. per-AU detection validity (coverage, range, frac_clips_estimable) -- gate before interpreting;
  6. AU x AU Spearman co-activation matrix (trained, baseline, and trained-baseline difference).

Writes ``au_coactivation_stats.json`` under the trained set dir. py-feat outputs are AU
*presence probabilities*, so claims are about monotonic association, not calibrated intensity.

Usage::
    conda run -n genphoto python scripts/analyze_allau.py \
        --trained inference_output/batch_eval/trained \
        --baseline inference_output/batch_eval/baseline
    # optional: --verify-au12   (assert aus['AU12'] reproduces existing au12)
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
AU_ORDER = ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU11", "AU12",
            "AU14", "AU15", "AU17", "AU20", "AU23", "AU24", "AU25", "AU26", "AU28", "AU43"]


GEO_CHANNELS = ["AU45_eye_aspect_ratio", "AU25_26_inner_lip_gap",
                "AU1_2_brow_raise", "AU4_inner_brow_gap"]


def load_scaled(set_dir: Path):
    p = set_dir / "results_allau.jsonl"
    recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return [r for r in recs if r.get("experiment") == "scaled"]


def load_scaled_file(set_dir: Path, fname: str):
    p = set_dir / fname
    if not p.exists():
        return None
    recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return [r for r in recs if r.get("experiment") == "scaled"]


def per_channel_ramp(recs, key, channels):
    """Aggregate per-clip ramp correlation for an explicit channel list (geometry/LibreFace)."""
    out = {}
    for ch in channels:
        rs = [r.get(key, {}).get(ch) for r in recs]
        rbar, n = fisher_mean(rs)
        lo, hi = bootstrap_ci_z(rs)
        out[ch] = dict(rbar=rbar, ci_low=lo, ci_high=hi, n_clips=n)
    return out


def _finite(xs):
    return np.asarray([x for x in xs if x is not None and np.isfinite(x)], float)


def fisher_mean(rs):
    z = np.arctanh(np.clip(_finite(rs), -0.999, 0.999))
    return (float(np.tanh(z.mean())), int(z.size)) if z.size else (float("nan"), 0)


def bootstrap_ci_z(rs, n=2000):
    """BCa CI for the Fisher-z mean, back-transformed to r. NaN if degenerate."""
    z = np.arctanh(np.clip(_finite(rs), -0.999, 0.999))
    if z.size < 3 or z.std() < 1e-9:
        return (float("nan"), float("nan"))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = stats.bootstrap((z,), np.mean, method="BCa", n_resamples=n, vectorized=True)
        return (float(np.tanh(res.confidence_interval.low)),
                float(np.tanh(res.confidence_interval.high)))
    except Exception:
        return (float("nan"), float("nan"))


def per_au_ramp(recs, key):
    """Aggregate per-clip ramp correlation per AU (key = 'au_ramp_spearman'|'au_ramp_pearson')."""
    out = {}
    for au in AU_ORDER:
        rs = [r.get(key, {}).get(au) for r in recs]
        rbar, n = fisher_mean(rs)
        lo, hi = bootstrap_ci_z(rs)
        out[au] = dict(rbar=rbar, ci_low=lo, ci_high=hi, n_clips=n)
    return out


def trajectories(recs):
    """Per AU, per frame index (commanded level): pooled mean +/- SEM of detected value."""
    targets = recs[0]["targets"] if recs else []
    out = {}
    for au in AU_ORDER:
        per_frame = []
        for t in range(len(targets)):
            vals = _finite([r["aus"][au][t] for r in recs if au in r.get("aus", {})])
            per_frame.append(dict(level=float(targets[t]),
                                  mean=float(vals.mean()) if vals.size else None,
                                  sem=float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else None,
                                  n=int(vals.size)))
        out[au] = per_frame
    return out


def paired_diff(trained, baseline, key):
    """Per AU: paired (by id) Fisher-z difference trained-baseline, with Wilcoxon + t + boot CI."""
    bl = {r["id"]: r for r in baseline}
    out, pvals = {}, {}
    for au in AU_ORDER:
        d = []
        for r in trained:
            b = bl.get(r["id"])
            if not b:
                continue
            rt, rbv = r.get(key, {}).get(au), b.get(key, {}).get(au)
            if rt is None or rbv is None or not (np.isfinite(rt) and np.isfinite(rbv)):
                continue
            d.append(np.arctanh(np.clip(rt, -0.999, 0.999)) - np.arctanh(np.clip(rbv, -0.999, 0.999)))
        d = np.asarray(d, float)
        rec = dict(delta_r=float(np.tanh(d.mean())) if d.size else float("nan"), n_pairs=int(d.size))
        # Wilcoxon signed-rank (primary) + paired-t on z (cross-check)
        p_w = float("nan")
        if d.size >= 5 and np.any(d != 0):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p_w = float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
            except Exception:
                p_w = float("nan")
        rec["p_wilcoxon"] = p_w
        rec["p_ttest"] = float(stats.ttest_1samp(d, 0.0).pvalue) if d.size >= 3 and d.std() > 1e-9 else float("nan")
        # paired bootstrap CI on tanh(mean(d))
        if d.size >= 3 and d.std() > 1e-9:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = stats.bootstrap((d,), np.mean, method="BCa", n_resamples=2000, vectorized=True)
                rec["ci_low"], rec["ci_high"] = float(np.tanh(res.confidence_interval.low)), float(np.tanh(res.confidence_interval.high))
            except Exception:
                rec["ci_low"] = rec["ci_high"] = float("nan")
        else:
            rec["ci_low"] = rec["ci_high"] = float("nan")
        out[au] = rec
        pvals[au] = p_w
    # BH-FDR across the AUs with a finite Wilcoxon p
    aus_fin = [au for au in AU_ORDER if np.isfinite(pvals[au])]
    if aus_fin:
        q = stats.false_discovery_control([pvals[au] for au in aus_fin], method="bh")
        for au, qv in zip(aus_fin, q):
            out[au]["q_bh"] = float(qv)
            out[au]["significant_q05"] = bool(qv < 0.05)
    for au in AU_ORDER:
        out[au].setdefault("q_bh", float("nan"))
        out[au].setdefault("significant_q05", False)
    return out


def validity(recs):
    out = {}
    for au in AU_ORDER:
        allvals = [v for r in recs for v in r.get("aus", {}).get(au, [])]
        fin = _finite(allvals)
        n_est = sum(1 for r in recs
                    if _finite(r.get("aus", {}).get(au, [])).size >= 2
                    and _finite(r.get("aus", {}).get(au, [])).std() > 1e-8)
        out[au] = dict(
            frac_finite=float(len(fin) / len(allvals)) if allvals else 0.0,
            mean=float(fin.mean()) if fin.size else None,
            sd=float(fin.std()) if fin.size else None,
            dyn_range=float(fin.max() - fin.min()) if fin.size else None,
            frac_clips_estimable=float(n_est / len(recs)) if recs else 0.0,
        )
    return out


def au_matrix(recs):
    """Pooled frame-level 20xN, then pairwise Spearman (NaN-masked). Returns 20x20 list."""
    cols = {au: [] for au in AU_ORDER}
    for r in recs:
        for au in AU_ORDER:
            cols[au].extend(r.get("aus", {}).get(au, []))
    M = np.full((20, 20), np.nan)
    rank = lambda x: np.argsort(np.argsort(x))
    for i, ai in enumerate(AU_ORDER):
        xi = np.asarray(cols[ai], float)
        for j, aj in enumerate(AU_ORDER):
            xj = np.asarray(cols[aj], float)
            m = np.isfinite(xi) & np.isfinite(xj)
            if m.sum() < 3 or xi[m].std() < 1e-9 or xj[m].std() < 1e-9:
                continue
            M[i, j] = float(np.corrcoef(rank(xi[m]), rank(xj[m]))[0, 1])
    return M.tolist()


def verify_au12(trained_dir):
    old = {json.loads(l)["id"]: json.loads(l).get("au12")
           for l in (trained_dir / "results.jsonl").read_text().splitlines() if l.strip()}
    new = load_scaled(trained_dir)
    maxdiff, n = 0.0, 0
    for r in new:
        o = old.get(r["id"])
        if o is None:
            continue
        d = np.nanmax(np.abs(np.asarray(o) - np.asarray(r["aus"]["AU12"])))
        maxdiff = max(maxdiff, float(d)); n += 1
    print(f"[verify-au12] compared {n} clips; max|AU12_new - au12_old| = {maxdiff:.3e}")
    return maxdiff


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trained", type=Path, default=REPO_ROOT / "inference_output/batch_eval/trained")
    ap.add_argument("--baseline", type=Path, default=REPO_ROOT / "inference_output/batch_eval/baseline")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dose-dir", type=Path, default=REPO_ROOT / "inference_output/batch_eval/dose_response")
    ap.add_argument("--fineface", type=Path, default=REPO_ROOT / "inference_output/batch_eval/fineface")
    ap.add_argument("--verify-au12", action="store_true")
    args = ap.parse_args()

    if args.verify_au12:
        verify_au12(args.trained)

    tr, bl = load_scaled(args.trained), load_scaled(args.baseline)
    print(f"loaded trained scaled n={len(tr)}, baseline scaled n={len(bl)}")

    stats_out = dict(
        n_trained=len(tr), n_baseline=len(bl), au_order=AU_ORDER,
        ramp_spearman_trained=per_au_ramp(tr, "au_ramp_spearman"),
        ramp_pearson_trained=per_au_ramp(tr, "au_ramp_pearson"),
        ramp_spearman_baseline=per_au_ramp(bl, "au_ramp_spearman"),
        trajectories_trained=trajectories(tr),
        trajectories_baseline=trajectories(bl),
        paired_diff_spearman=paired_diff(tr, bl, "au_ramp_spearman"),
        validity_trained=validity(tr),
        validity_baseline=validity(bl),
        au_matrix_trained=au_matrix(tr),
        au_matrix_baseline=au_matrix(bl),
    )
    # trained - baseline difference matrix
    Mt, Mb = np.array(stats_out["au_matrix_trained"]), np.array(stats_out["au_matrix_baseline"])
    stats_out["au_matrix_diff"] = (Mt - Mb).tolist()

    # independent cross-checks: LibreFace (different model family) + FaceMesh geometry
    lf_tr = load_scaled_file(args.trained, "results_libreface.jsonl")
    lf_bl = load_scaled_file(args.baseline, "results_libreface.jsonl")
    if lf_tr:
        stats_out["ramp_spearman_libreface_trained"] = per_au_ramp(lf_tr, "au_ramp_spearman")
    if lf_bl:
        stats_out["ramp_spearman_libreface_baseline"] = per_au_ramp(lf_bl, "au_ramp_spearman")
    stats_out["geo_ramp_spearman_trained"] = per_channel_ramp(tr, "geo_ramp_spearman", GEO_CHANNELS)
    stats_out["geo_ramp_spearman_baseline"] = per_channel_ramp(bl, "geo_ramp_spearman", GEO_CHANNELS)

    # H4: does co-activation emerge with training? per-checkpoint per-AU ramp Spearman
    if args.dose_dir and args.dose_dir.exists():
        dose = {}
        for step in (1000, 5000, 10000, 25000, 50000, 100000):
            recs = load_scaled_file(args.dose_dir / f"step_{step}", "results_allau.jsonl")
            if recs:
                dose[str(step)] = per_au_ramp(recs, "au_ramp_spearman")
        if dose:
            stats_out["dose_response"] = dose
            print("dose-response emergence (AU06 Spearman by step):",
                  {k: round(v["AU06"]["rbar"], 3) for k, v in dose.items()})

    # prior-art (FineFace) leakage contrast
    ff = load_scaled_file(args.fineface, "results_allau.jsonl") if args.fineface else None
    if ff:
        stats_out["fineface_ramp_spearman"] = per_au_ramp(ff, "au_ramp_spearman")
        print(f"fineface scaled n={len(ff)} scored for contrast")

    out_path = args.out or (args.trained / "au_coactivation_stats.json")
    out_path.write_text(json.dumps(stats_out, indent=2))
    print(f"wrote {out_path}")

    # readable summary: per-AU trained Spearman (sorted), significance vs baseline,
    # and the independent LibreFace r where available (py-feat vs different-family agreement)
    lf = stats_out.get("ramp_spearman_libreface_trained", {})
    print(f"\n{'AU':6}{'r_pyfeat':>10}{'95% CI':>17}{'r_libreface':>13}{'Δr(tr-bl)':>11}{'q_BH':>8}{'estimable':>11}")
    order = sorted(AU_ORDER, key=lambda a: -(stats_out["ramp_spearman_trained"][a]["rbar"]
                                             if np.isfinite(stats_out["ramp_spearman_trained"][a]["rbar"]) else -9))
    for au in order:
        s = stats_out["ramp_spearman_trained"][au]
        d = stats_out["paired_diff_spearman"][au]
        v = stats_out["validity_trained"][au]
        star = "*" if d["significant_q05"] else " "
        ci = f"[{s['ci_low']:+.2f},{s['ci_high']:+.2f}]" if np.isfinite(s["ci_low"]) else "[   n/a    ]"
        lfr = lf.get(au, {}).get("rbar", float("nan"))
        lfs = f"{lfr:>+13.3f}" if np.isfinite(lfr) else f"{'--':>13}"
        print(f"{au:6}{s['rbar']:>+10.3f}{ci:>17}{lfs}{d['delta_r']:>+11.3f}{d['q_bh']:>8.3f}{star}{v['frac_clips_estimable']:>10.2f}")


if __name__ == "__main__":
    main()
