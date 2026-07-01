"""Expression-control accuracy metric.

Analog of the original ``comp_metrics/accuracy_by_CorrCoef/`` scripts: for each generated GIF,
detect AU12 (smile) intensity per frame and report the Pearson correlation between the detected
trajectory and the requested target intensities.

A perfect adaptor would yield correlation 1.0 (frames monotonically increase in smile when the
target list is sorted ascending).

Example::

    python comp_metrics/expression_au_accuracy.py \\
        --gifs-dir inference_output/expression/ \\
        --intensity-list "[0.0, 0.25, 0.5, 0.75, 1.0]"
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Sequence

import numpy as np
from PIL import Image, ImageSequence

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
log = logging.getLogger("expression_au_accuracy")


def _load_pyfeat():
    try:
        from feat import Detector  # type: ignore
        return Detector
    except ImportError as e:  # pragma: no cover
        raise ImportError("py-feat is required. Install with: pip install py-feat") from e


def extract_frames(gif_path: Path) -> List[np.ndarray]:
    with Image.open(gif_path) as gif:
        frames = [np.array(f.convert("RGB")) for f in ImageSequence.Iterator(gif)]
    return frames


def detect_au12(detector, frames: Sequence[np.ndarray]) -> List[float]:
    out: List[float] = []
    import tempfile
    import os
    for rgb in frames:
        try:
            # Save to a temporary file because this py-feat version expects a path
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            Image.fromarray(rgb).save(tmp_path)
            df = detector.detect_image(tmp_path)
            os.unlink(tmp_path)
            au12_cols = [c for c in df.columns if c.upper() == "AU12"]
            if not au12_cols:
                out.append(0.0)
                continue
            value = float(df[au12_cols[0]].iloc[0])
            if not np.isfinite(value):
                value = 0.0
            out.append(float(np.clip(value, 0.0, 1.0)))
        except Exception as e:  # noqa: BLE001
            log.debug("py-feat failed on a frame (%s); using 0.0", e)
            out.append(0.0)
    return out


def correlation(detected: Sequence[float], target: Sequence[float]) -> float:
    if len(detected) != len(target) or len(target) < 2:
        return float("nan")
    a = np.asarray(detected, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    if a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gifs-dir", type=Path, required=True,
                        help="Directory containing generated *.gif samples.")
    parser.add_argument("--intensity-list", type=str, required=True,
                        help="JSON list of target intensities used at inference.")
    parser.add_argument("--pattern", type=str, default="*.gif",
                        help="Glob pattern for sample files. Default: *.gif")
    args = parser.parse_args()

    targets = json.loads(args.intensity_list)
    if not isinstance(targets, list) or not targets:
        log.error("--intensity-list must be a non-empty JSON list")
        return 1

    Detector = _load_pyfeat()
    detector = Detector(au_model="xgb", emotion_model="resmasknet", identity_model=None,
                        face_model="retinaface", landmark_model="mobilefacenet",
                        facepose_model="img2pose")

    gifs = sorted(args.gifs_dir.glob(args.pattern))
    if not gifs:
        log.error("No GIFs matched %s under %s", args.pattern, args.gifs_dir)
        return 1

    correlations: List[float] = []
    for gif in gifs:
        frames = extract_frames(gif)
        if len(frames) != len(targets):
            log.warning("%s: frame count %d != target count %d; skipping",
                        gif.name, len(frames), len(targets))
            continue
        detected = detect_au12(detector, frames)
        corr = correlation(detected, targets)
        log.info("%s  detected=%s  corr=%.4f",
                 gif.name, [round(v, 3) for v in detected], corr)
        if not np.isnan(corr):
            correlations.append(corr)

    if correlations:
        log.info("Mean Pearson AU12-target correlation across %d clips: %.4f",
                 len(correlations), float(np.mean(correlations)))
    else:
        log.warning("No valid correlations computed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
