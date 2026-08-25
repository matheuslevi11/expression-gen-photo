"""Figures for the whole-face AU co-activation analysis (reads au_coactivation_stats.json).

Fig A: per-AU ramp-correlation dumbbell (trained vs frozen baseline), sorted, bootstrap CI,
       q<0.05 marked, "not estimable" AUs greyed.
Fig B: 4x5 co-activation trajectory small-multiples (detected AU vs commanded level).
Fig C: AU x AU trained-minus-baseline Spearman heatmap (CVD-safe diverging map, centered 0).

Okabe-Ito palette, matching inference_output/batch_eval/dose_response/plot.py. Outputs
.pdf + .png (300 dpi) to docs/experiments/figures/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REPO_ROOT = Path(__file__).resolve().parent.parent
BLUE, VERM, INK, MUTED, GRID = "#0072B2", "#D55E00", "#1a1a1a", "#5f6b7a", "#d9dde3"
GREEN = "#009E73"  # Okabe-Ito bluish green (3rd categorical hue)
# CVD-safe diverging map: blue (neg) -> light neutral -> vermillion (pos)
DIVERGING = LinearSegmentedColormap.from_list("bo", [BLUE, "#f2f2f2", VERM])
ESTIMABLE_MIN = 0.5  # frac_clips_estimable below this -> "not estimable", de-emphasised

plt.rcParams.update({
    "font.size": 11, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
    "axes.linewidth": 0.8, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
})


def _rbar(d, au):
    v = d[au]["rbar"]
    return v if np.isfinite(v) else np.nan


def fig_profile(S, out):
    aus = S["au_order"]
    tr, bl = S["ramp_spearman_trained"], S["ramp_spearman_baseline"]
    diff, val = S["paired_diff_spearman"], S["validity_trained"]
    order = sorted(aus, key=lambda a: (_rbar(tr, a) if np.isfinite(_rbar(tr, a)) else -9), reverse=True)

    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=300)
    for i, au in enumerate(order):
        y = len(order) - i
        rt, rb = _rbar(tr, au), _rbar(bl, au)
        estimable = val[au]["frac_clips_estimable"] >= ESTIMABLE_MIN
        col_t = BLUE if estimable else GRID
        # dumbbell connector
        if np.isfinite(rt) and np.isfinite(rb):
            ax.plot([rb, rt], [y, y], color=MUTED if estimable else GRID, lw=1.2, zorder=1)
        # baseline marker
        if np.isfinite(rb):
            ax.plot(rb, y, "s", ms=6, mfc="white", mec=VERM if estimable else GRID, mew=1.6, zorder=3)
        # trained marker + CI
        lo, hi = tr[au]["ci_low"], tr[au]["ci_high"]
        if np.isfinite(rt):
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [y, y], color=col_t, lw=2.2, alpha=0.35, zorder=2)
            ax.plot(rt, y, "o", ms=8, mfc="white", mec=col_t, mew=1.8, zorder=4)
        # label; star if significant vs baseline (q<0.05)
        sig = diff[au].get("significant_q05", False)
        label = au + (" *" if sig else "")
        ax.text(-1.06, y, label, ha="right", va="center", fontsize=9.5,
                color=INK if estimable else MUTED,
                fontweight="bold" if sig else "normal")

    ax.axvline(0, color=MUTED, lw=0.8, ls=(0, (4, 4)))
    ax.set_xlim(-1.08, 1.05)
    ax.set_ylim(0.3, len(order) + 0.7)
    ax.set_yticks([])
    ax.set_xlabel("ramp correlation  (Spearman $r$ of detected AU vs commanded intensity)")
    ax.set_title("Whole-face AU response to the AU12 ramp", fontsize=12.5, fontweight="bold", pad=26)
    ax.plot([], [], "o", mfc="white", mec=BLUE, mew=1.8, ms=8, label="trained (100k)")
    ax.plot([], [], "s", mfc="white", mec=VERM, mew=1.6, ms=6, label="frozen baseline")
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, borderaxespad=0.4)
    ax.text(1.03, len(order) + 0.4, "* q<0.05 (BH-FDR) vs baseline · grey = not estimable",
            ha="right", va="bottom", fontsize=7.6, color=MUTED)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", f"{out}.pdf/.png")


def fig_trajectories(S, out):
    aus = S["au_order"]
    tt, tb = S["trajectories_trained"], S["trajectories_baseline"]
    fig, axes = plt.subplots(4, 5, figsize=(12, 9), dpi=300, sharex=True, sharey=True)
    for ax, au in zip(axes.flat, aus):
        for traj, col, mk in ((tb, VERM, "s"), (tt, BLUE, "o")):
            lv = [p["level"] for p in traj[au]]
            m = np.array([p["mean"] if p["mean"] is not None else np.nan for p in traj[au]])
            se = np.array([p["sem"] if p["sem"] is not None else 0.0 for p in traj[au]])
            ax.plot(lv, m, "-", color=col, marker=mk, ms=4, lw=1.5)
            ax.fill_between(lv, m - se, m + se, color=col, alpha=0.15, lw=0)
        ax.set_title(au, fontsize=9.5)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("commanded", fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("detected", fontsize=8.5)
    fig.suptitle("AU co-activation trajectories: detected presence vs commanded AU12 (blue=trained, orange=baseline)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", f"{out}.pdf/.png")


def fig_matrix(S, out):
    aus = S["au_order"]
    M = np.array(S["au_matrix_diff"], float)
    fig, ax = plt.subplots(figsize=(8.2, 7), dpi=300)
    im = ax.imshow(M, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(aus))); ax.set_xticklabels(aus, rotation=90, fontsize=7.5)
    ax.set_yticks(range(len(aus))); ax.set_yticklabels(aus, fontsize=7.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Spearman co-activation, trained − baseline", fontsize=9)
    ax.set_title("Model-induced AU×AU co-activation (baseline-subtracted)",
                 fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", f"{out}.pdf/.png")


def fig_dose(S, out):
    dose = S.get("dose_response")
    if not dose:
        print("no dose_response in stats; skipping Fig D")
        return
    steps = sorted(int(k) for k in dose)
    series = {"AU12": BLUE, "AU06": VERM, "AU25": GREEN}
    marks = {"AU12": "o", "AU06": "s", "AU25": "^"}
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=300)
    ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 4)))
    for au, col in series.items():
        rs = [dose[str(s)][au]["rbar"] for s in steps]
        ax.plot(steps, rs, "-", color=col, marker=marks[au], ms=8, mfc="white", mec=col,
                mew=1.8, label=au)
    ax.set_xscale("log")
    ax.set_xticks(steps)
    ax.set_xticklabels([f"{s // 1000}k" for s in steps])
    ax.minorticks_off()
    ax.set_xlim(steps[0] * 0.85, steps[-1] * 1.3)
    ax.set_ylim(-0.25, 1.0)
    ax.set_xlabel("training steps (log scale)")
    ax.set_ylabel("ramp correlation  (Spearman $r$)")
    ax.set_title("AU co-activation emergence over training", fontsize=12.5, fontweight="bold", pad=10)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", f"{out}.pdf/.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats", type=Path,
                    default=REPO_ROOT / "inference_output/batch_eval/trained/au_coactivation_stats.json")
    ap.add_argument("--outdir", type=Path, default=REPO_ROOT / "docs/experiments/figures")
    ap.add_argument("--date", type=str, default="2026-08-25")
    args = ap.parse_args()
    S = json.loads(args.stats.read_text())
    args.outdir.mkdir(parents=True, exist_ok=True)
    base = args.outdir / f"{args.date}-au-coactivation"
    fig_profile(S, f"{base}-profile")
    fig_trajectories(S, f"{base}-trajectories")
    fig_matrix(S, f"{base}-matrix")
    fig_dose(S, f"{base}-emergence")


if __name__ == "__main__":
    main()
