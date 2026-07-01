"""Preprocess raw MEAD video clips into face-cropped frames + per-frame intensity annotations.

The MEAD dataset is distributed with one ``video.tar`` archive per identity::

    MEAD_root/
        M003/
            video.tar    # contains video/front/happy/level_3/001.mp4, …
        W009/
            video.tar
        ...

Each archive is extracted on first run into a sibling ``video_extracted/`` folder.
If you have already extracted the archives yourself (so that
``M003/video/front/…`` exists directly), the script detects that and skips extraction.

This script walks that tree, extracts evenly-spaced frames per clip, crops a face-centred
square, runs a smile / AU12 estimator per frame, and writes:

    out_root/
        frames/M003/happy/level_3/001/frame_00000.jpg
        ...
        annotations/train.json
        annotations/validation.json

The resulting JSON files are consumed directly by ``ExpressionMEAD``.

Two AU backends are supported:

    --au-method pyfeat      (recommended) real FACS AU12 from the ``py-feat`` library.
    --au-method timeline    no-deps fallback. Assumes the actor goes neutral → peak → neutral
                            roughly linearly in time and uses a triangular ramp on the clip
                            timeline as a proxy for AU12 intensity. Useful for smoke-testing
                            the full pipeline before installing py-feat.

Face cropping uses ``mediapipe`` (lightweight, pip-installable) and falls back to a centre
crop if no face is detected.

Example::

    python scripts/preprocess_mead.py \\
        --mead-root /data/MEAD \\
        --out-root  /data/MEAD_processed \\
        --emotion happy \\
        --view front \\
        --frames-per-clip 24 \\
        --au-method pyfeat
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
log = logging.getLogger("preprocess_mead")


# Canonical MEAD emotion folder names (note the past-tense forms used in the dataset).
MEAD_EMOTIONS = ["angry", "contempt", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]


# ---------------------------------------------------------------------------
# Optional dependencies (loaded lazily so the script imports cleanly without them).
# ---------------------------------------------------------------------------

def _load_mediapipe():
    try:
        import mediapipe as mp  # type: ignore
        return mp
    except ImportError:  # pragma: no cover - exercised at runtime only
        log.warning("mediapipe not installed; face crops will fall back to centre crop. "
                    "Install with: pip install mediapipe")
        return None


def _load_pyfeat():
    try:
        from feat import Detector  # type: ignore
        return Detector
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "py-feat is required for --au-method pyfeat. Install with: pip install py-feat"
        ) from e


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ClipInfo:
    """Resolved metadata for one .mp4 file in the MEAD tree."""
    actor: str
    emotion: str
    intensity_level: int
    view: str
    clip_id: str
    video_path: Path
    out_dir: Path  # where extracted frames will be written


@dataclass
class FrameRecord:
    relative_path: str
    au12: float


# ---------------------------------------------------------------------------
# File-tree discovery
# ---------------------------------------------------------------------------

def _count_tar_video_members(tar_path: Path) -> int:
    """Number of regular-file ``.mp4`` entries in ``tar_path``.

    Used to detect stale ``.extracted`` sentinels left by a previous run that aborted
    midway: if the on-disk tree has noticeably fewer ``.mp4`` files than the archive,
    we re-extract instead of trusting the sentinel.
    """
    try:
        with tarfile.open(tar_path, "r:*") as tf:
            return sum(1 for m in tf.getmembers() if m.isfile() and m.name.endswith(".mp4"))
    except tarfile.TarError as exc:
        log.warning("Could not enumerate %s: %s", tar_path, exc)
        return 0


def _count_disk_video_files(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for _ in root.rglob("*.mp4"))


def _extract_tar_if_needed(tar_path: Path, dest_dir: Path) -> bool:
    """Extract ``tar_path`` into ``dest_dir`` if it has not already been extracted.

    The sentinel file ``dest_dir/.extracted`` is created after a **successful**
    extraction so subsequent runs skip the (slow) tar pass entirely. The sentinel
    is **validated** against the archive's member count so a half-extracted tree
    from an earlier crashed run (e.g. only ``down/`` present when the tar
    actually contains every view) is detected and re-extracted instead of being
    silently trusted.

    Returns ``True`` on success, ``False`` if the archive is corrupt/incomplete.
    Partial extractions are removed so a re-run after a fixed download starts clean.
    """
    import shutil

    sentinel = dest_dir / ".extracted"
    if sentinel.exists():
        on_disk = _count_disk_video_files(dest_dir)
        in_tar = _count_tar_video_members(tar_path)
        # Allow some slack -- if the tar listing failed (in_tar==0) we trust the
        # sentinel; otherwise demand at least 90% of members on disk.
        if in_tar == 0 or on_disk >= int(0.9 * in_tar):
            log.debug("Skipping extraction (already done): %s", tar_path)
            return True
        log.warning(
            "Stale .extracted sentinel for %s (disk has %d mp4 files, archive has %d). "
            "Removing partial tree and re-extracting.",
            tar_path, on_disk, in_tar,
        )
        shutil.rmtree(dest_dir, ignore_errors=True)
    dest_dir.mkdir(parents=True, exist_ok=True)
    log.info("Extracting %s → %s  (this may take a while) ...", tar_path.name, dest_dir)
    try:
        with tarfile.open(tar_path, "r:*") as tf:
            tf.extractall(path=dest_dir)
    except tarfile.TarError as exc:
        log.error(
            "Failed to extract %s: %s\n"
            "  The archive is likely truncated (incomplete download).\n"
            "  Please re-download %s and try again.\n"
            "  Partial extraction at %s will be removed so the next run starts clean.",
            tar_path, exc, tar_path, dest_dir,
        )
        shutil.rmtree(dest_dir, ignore_errors=True)
        return False
    sentinel.touch()
    log.info("Extraction complete: %s", tar_path.name)
    return True


def _video_tar_candidates(actor_dir: Path) -> List[Path]:
    """All ``video*.tar`` archives directly under ``actor_dir``.

    Covers three observed MEAD packagings:
    - single ``video.tar`` (most actors, e.g. M003).
    - split archives ``video_1.tar`` + ``video_2.tar`` (e.g. M042).
    - nested ``video/1.tar`` + ``video/2.tar`` (e.g. W021) -- handled by
      :func:`_resolve_video_roots` after a one-level recursion.
    """
    return sorted(p for p in actor_dir.glob("video*.tar") if p.is_file())


_KNOWN_VIEWS = {"front", "down", "top", "left_30", "left_60", "right_30", "right_60"}


def _is_view_dir(p: Path) -> bool:
    """A view dir is named like a MEAD view *and* has emotion-named children."""
    if not p.is_dir() or p.name not in _KNOWN_VIEWS:
        return False
    children = {c.name for c in p.iterdir() if c.is_dir()}
    return bool(children & set(MEAD_EMOTIONS))


def _is_view_parent(p: Path) -> bool:
    """``p`` is a video root iff it contains at least one view directory as a child."""
    if not p.is_dir():
        return False
    return any(_is_view_dir(c) for c in p.iterdir() if c.is_dir())


def _resolve_video_roots(actor_dir: Path) -> List[Path]:
    """Return one or more directories that each contain ``<view>/<emotion>/<level>/...``.

    Multiple roots are returned for split-archive actors (M042) and nested-tar
    actors (W021); :func:`discover_clips` then walks each one and unions the
    resulting clip list.

    Strategy, in order:

    1. ``actor_dir/video*.tar`` archives are extracted into a sibling
       ``video_extracted_<stem>/`` each. The script then descends to the first
       level that looks like a view directory (so ``video/<view>/...``,
       ``1/<view>/...``, and ``<view>/...`` layouts all flatten to one root).
    2. ``actor_dir/video/`` is used as-is when it contains view subdirs.
    3. Otherwise, if ``actor_dir/video/`` contains nested ``*.tar`` files
       (e.g. W021), each is extracted one level down and resolved recursively.

    Returns ``[]`` if nothing usable is found.
    """
    roots: List[Path] = []

    tars = _video_tar_candidates(actor_dir)
    for tar in tars:
        extract_dir = actor_dir / f"video_extracted_{tar.stem}" if len(tars) > 1 \
            else actor_dir / "video_extracted"
        if not _extract_tar_if_needed(tar, extract_dir):
            continue
        root = _descend_to_view_root(extract_dir)
        if root is not None:
            roots.append(root)

    plain_video = actor_dir / "video"
    if plain_video.is_dir():
        nested_root = _descend_to_view_root(plain_video)
        if nested_root is not None and nested_root not in roots:
            roots.append(nested_root)
        else:
            # W021-style: video/1.tar, video/2.tar -- extract each in place.
            nested_tars = sorted(p for p in plain_video.glob("*.tar") if p.is_file())
            for tar in nested_tars:
                extract_dir = plain_video / f"_extracted_{tar.stem}"
                if not _extract_tar_if_needed(tar, extract_dir):
                    continue
                root = _descend_to_view_root(extract_dir)
                if root is not None:
                    roots.append(root)

    return roots


def _descend_to_view_root(start: Path, max_depth: int = 3) -> Optional[Path]:
    """Walk down at most ``max_depth`` levels until we find a dir that contains views.

    The various MEAD packagings put the view layer at different depths:
    - ``video_extracted/video/<view>/...`` (single-tar actors → returns ``video_extracted/video``)
    - ``video_extracted_video_1/1/<view>/...`` (split-tar actors → returns the ``1/`` dir)
    - ``video/<view>/...`` (already-extracted → returns ``video/``)
    - ``video/_extracted_1/1/<view>/...`` (nested-tar actors → returns the ``1/`` dir)
    """
    if not start.is_dir():
        return None
    if _is_view_parent(start):
        return start
    if max_depth <= 0:
        return None
    for child in sorted(p for p in start.iterdir() if p.is_dir()):
        found = _descend_to_view_root(child, max_depth - 1)
        if found is not None:
            return found
    return None


def _is_actor_dir(path: Path) -> bool:
    """Heuristically identify an actor directory.

    True if it contains ``video/`` or any ``video*.tar`` archive directly under it.
    """
    return (path / "video").is_dir() or bool(_video_tar_candidates(path))


def discover_clips(
    mead_root: Path,
    out_root: Path,
    emotions: Sequence[str],
    view: str,
    actors: Optional[Sequence[str]] = None,
) -> List[ClipInfo]:
    """Walk ``mead_root`` and return one :class:`ClipInfo` per matching video clip.

    Two acceptable layouts:

    A. Multi-identity root::

        MEAD/
            M003/
                video.tar      # OR a pre-extracted video/ dir
            W009/
                video.tar
            video-001/
                video/...

    B. Single-identity root (``--mead-root`` points directly at one actor folder)::

        video-001/
            video/<view>/<emotion>/<level>/XXX.mp4

    In layout B the leaf directory name (``video-001``) is used as the actor id.
    """
    clips: List[ClipInfo] = []
    if not mead_root.exists():
        raise FileNotFoundError(f"MEAD root not found: {mead_root}")

    if _is_actor_dir(mead_root):
        actor_dirs = [mead_root]
        log.info("Detected single-identity root (%s); treating it as one actor.", mead_root.name)
    else:
        actor_dirs = sorted(p for p in mead_root.iterdir() if p.is_dir())

    for actor_dir in actor_dirs:
        if actors and actor_dir.name not in actors:
            continue

        video_roots = _resolve_video_roots(actor_dir)
        if not video_roots:
            log.warning("No video.tar / video*.tar / video/ tree found for actor %s; skipping",
                        actor_dir.name)
            continue

        actor_clips_before = len(clips)
        seen_paths: set = set()
        for video_root in video_roots:
            view_dir = video_root / view
            if not view_dir.is_dir():
                log.debug("View %r not found under %s", view, video_root)
                continue
            for emotion in emotions:
                emo_dir = view_dir / emotion
                if not emo_dir.is_dir():
                    continue
                for level_dir in sorted(p for p in emo_dir.iterdir() if p.is_dir()):
                    try:
                        level = int(level_dir.name.split("_")[-1])
                    except (ValueError, IndexError):
                        log.debug("Skipping unrecognised level dir: %s", level_dir)
                        continue
                    for video_path in sorted(level_dir.glob("*.mp4")):
                        # Split-archive actors can have the same logical clip in
                        # multiple roots; dedupe by (emotion, level, stem).
                        key = (emotion, level, video_path.stem)
                        if key in seen_paths:
                            continue
                        seen_paths.add(key)
                        clip_id = f"{actor_dir.name}_{emotion}_level_{level}_{video_path.stem}"
                        out_dir = (
                            out_root / "frames"
                            / actor_dir.name / emotion / f"level_{level}" / video_path.stem
                        )
                        clips.append(
                            ClipInfo(
                                actor=actor_dir.name,
                                emotion=emotion,
                                intensity_level=level,
                                view=view,
                                clip_id=clip_id,
                                video_path=video_path,
                                out_dir=out_dir,
                            )
                        )

        if len(clips) == actor_clips_before:
            roots_str = ", ".join(str(r) for r in video_roots)
            log.warning(
                "Actor %s resolved %d video root(s) (%s) but yielded 0 clips for "
                "view=%r emotions=%s. Check that the requested view exists in the archive.",
                actor_dir.name, len(video_roots), roots_str, view, list(emotions),
            )
    return clips


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def sample_frame_indices(total_frames: int, n: int) -> List[int]:
    if total_frames <= 0:
        return []
    if total_frames <= n:
        return list(range(total_frames))
    return list(np.linspace(0, total_frames - 1, n).round().astype(int))


def extract_frames(video_path: Path, n: int) -> Tuple[List[np.ndarray], List[int]]:
    """Decode ``n`` evenly-spaced BGR frames from ``video_path``."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    targets = sample_frame_indices(total, n)
    frames, kept_indices = [], []
    target_set = set(targets)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in target_set:
            frames.append(frame)
            kept_indices.append(idx)
        idx += 1
        if len(frames) >= len(targets):
            break
    cap.release()
    return frames, kept_indices


# ---------------------------------------------------------------------------
# Face cropping
# ---------------------------------------------------------------------------

class FaceCropper:
    """Mediapipe-based face cropper with a centre-crop fallback.

    Supports mediapipe ≤0.9 (``solutions`` API) and gracefully falls back to a
    plain centre-square crop when mediapipe is not installed or uses the newer
    Tasks API (≥0.10) where ``solutions.face_detection`` was removed.
    """

    def __init__(self, target_h: int, target_w: int, margin: float = 0.4) -> None:
        self.target_h = target_h
        self.target_w = target_w
        self.margin = margin
        self._detector = None

        mp = _load_mediapipe()
        if mp is not None:
            try:
                self._detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.3
                )
                self._mp = mp
            except AttributeError:
                log.warning(
                    "mediapipe %s does not expose the legacy 'solutions' API "
                    "(removed in ≥0.10). Falling back to centre-square crops. "
                    "Install mediapipe<0.10 if you need face-aligned crops: "
                    "pip install 'mediapipe<0.10'",
                    getattr(mp, "__version__", "unknown"),
                )

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        if self._detector is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = self._detector.process(rgb)
            if res.detections:
                box = res.detections[0].location_data.relative_bounding_box
                cx = (box.xmin + box.width / 2.0) * w
                cy = (box.ymin + box.height / 2.0) * h
                side = max(box.width * w, box.height * h) * (1.0 + self.margin)
                bgr = self._square_crop(bgr, cx, cy, side)
            else:
                bgr = self._centre_square(bgr)
        else:
            bgr = self._centre_square(bgr)
        return cv2.resize(bgr, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _square_crop(bgr: np.ndarray, cx: float, cy: float, side: float) -> np.ndarray:
        h, w = bgr.shape[:2]
        side = float(min(side, min(h, w)))
        half = side / 2.0
        x0 = int(max(0, cx - half))
        y0 = int(max(0, cy - half))
        x1 = int(min(w, x0 + side))
        y1 = int(min(h, y0 + side))
        x0 = max(0, x1 - int(side))
        y0 = max(0, y1 - int(side))
        return bgr[y0:y1, x0:x1]

    @staticmethod
    def _centre_square(bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        return bgr[y0:y0 + side, x0:x0 + side]


# ---------------------------------------------------------------------------
# AU estimation
# ---------------------------------------------------------------------------

class AUEstimator:
    """Pluggable AU12 estimator. Returns one float in roughly ``[0, 1]`` per frame.

    Backends:

    - ``timeline``: zero-deps triangular ramp on the clip timeline. Useful for
      smoke-testing the pipeline end-to-end before installing py-feat.
    - ``pyfeat``: real FACS AU12 from py-feat's XGBoost AU head. Runs the
      ``retinaface → mobilefacenet → xgb`` low-level pipeline on each frame
      directly (the high-level ``detect_image`` API in py-feat 0.6 only accepts
      file paths, so we sidestep it here).

    AU is intentionally estimated on the **uncropped** decoded video frame: the
    AU heads were trained on naturally-framed faces and can be biased by the
    tight, fixed-size crop we save to disk for training. ``process_clip`` calls
    :meth:`estimate` before applying :class:`FaceCropper`.
    """

    def __init__(self, method: str) -> None:
        self.method = method
        self._detector = None
        self._au12_index: Optional[int] = None
        if method == "pyfeat":
            Detector = _load_pyfeat()
            # py-feat 0.6 rejects None for these heads; pass the cheapest stand-ins
            # we don't actually consume (the full Fex columns will be discarded).
            self._detector = Detector(
                au_model="xgb",
                emotion_model="svm",
                identity_model="facenet",
                face_model="retinaface",
                landmark_model="mobilefacenet",
                facepose_model="img2pose",
                device="cuda" if _cuda_available() else "cpu",
            )
            au_cols = list(self._detector.info.get("au_presence_columns", []))
            for i, c in enumerate(au_cols):
                if c.upper() == "AU12":
                    self._au12_index = i
                    break
            if self._au12_index is None:
                log.warning("py-feat detector exposes no AU12 column (saw %s); "
                            "AU12 will be reported as 0.0.", au_cols)
        elif method == "timeline":
            pass
        else:
            raise ValueError(f"Unknown au-method: {method}")

    def estimate(self, frames_bgr: Sequence[np.ndarray], frame_indices: Sequence[int],
                 total_frames: int) -> List[float]:
        if self.method == "timeline":
            return self._timeline(frame_indices, total_frames)
        return self._pyfeat(frames_bgr)

    @staticmethod
    def _timeline(frame_indices: Sequence[int], total_frames: int) -> List[float]:
        """Triangular ramp 0 → 1 → 0 across the clip; AU12 ≈ how close to peak."""
        if total_frames <= 1:
            return [0.0 for _ in frame_indices]
        out = []
        for idx in frame_indices:
            t = idx / (total_frames - 1)
            out.append(float(1.0 - abs(2.0 * t - 1.0)))
        return out

    def _pyfeat(self, frames_bgr: Sequence[np.ndarray]) -> List[float]:
        if self._au12_index is None:
            return [0.0 for _ in frames_bgr]
        out = []
        for bgr in frames_bgr:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            try:
                faces = self._detector.detect_faces(rgb)
                # detect_faces returns [[ [x1,y1,x2,y2,conf], ... ]] for single image input.
                if not faces or not faces[0]:
                    out.append(0.0)
                    continue
                landmarks = self._detector.detect_landmarks(rgb, faces)
                aus = self._detector.detect_aus(rgb, landmarks)
                au_arr = np.asarray(aus)  # (1, n_faces, n_aus)
                value = float(au_arr[0, 0, self._au12_index])
                if not np.isfinite(value):
                    value = 0.0
                out.append(float(np.clip(value, 0.0, 1.0)))
            except Exception as e:  # noqa: BLE001 - py-feat raises various errors per frame
                log.debug("py-feat failed on a frame (%s); using 0.0", e)
                out.append(0.0)
        return out


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore
        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Caption strategy
# ---------------------------------------------------------------------------

DEFAULT_CAPTION_TEMPLATE = "A portrait photograph of a {gender}, frontal view, neutral background."


def actor_caption(actor: str, template: str = DEFAULT_CAPTION_TEMPLATE) -> str:
    """Format ``template`` with a heuristic gender field derived from the actor id.

    The original MEAD distribution prefixes male identities with ``M`` and female with ``W``.
    Other distributions (e.g. ``video-001``) get the generic ``person`` gender. Pass
    ``--caption-template`` on the CLI to bypass this heuristic entirely.
    """
    if actor.startswith("M"):
        gender = "man"
    elif actor.startswith("W"):
        gender = "woman"
    else:
        gender = "person"
    try:
        return template.format(gender=gender, actor=actor)
    except (KeyError, IndexError):
        return template


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_clip(
    clip: ClipInfo,
    cropper: FaceCropper,
    au_estimator: AUEstimator,
    frames_per_clip: int,
    out_root: Path,
    caption_template: str = DEFAULT_CAPTION_TEMPLATE,
    overwrite: bool = False,
) -> Optional[dict]:
    if clip.out_dir.exists() and not overwrite and any(clip.out_dir.iterdir()):
        log.debug("Skipping %s (already processed)", clip.clip_id)
    else:
        clip.out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(clip.video_path))
    if not cap.isOpened():
        log.warning("Could not open %s; skipping", clip.video_path)
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frames_bgr, frame_indices = extract_frames(clip.video_path, frames_per_clip)
    if not frames_bgr:
        log.warning("No frames decoded from %s; skipping", clip.video_path)
        return None

    # Estimate AUs on the **uncropped** frames first: the py-feat AU head was
    # trained on naturally-framed faces and is biased by the tight 256x256 crop
    # we save for training. Crop afterwards.
    au12_values = au_estimator.estimate(frames_bgr, frame_indices, total_frames)
    cropped = [cropper(f) for f in frames_bgr]

    rel_root = clip.out_dir.relative_to(out_root / "frames").as_posix()
    frame_records: List[dict] = []
    for i, (img, au12) in enumerate(zip(cropped, au12_values)):
        rel_path = f"frames/{rel_root}/frame_{i:05d}.jpg"
        abs_path = out_root / rel_path
        cv2.imwrite(str(abs_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_records.append({"path": rel_path, "au12": float(au12)})

    return {
        "clip_id": clip.clip_id,
        "actor": clip.actor,
        "emotion": clip.emotion,
        "intensity_level": clip.intensity_level,
        "view": clip.view,
        "caption": actor_caption(clip.actor, caption_template),
        "frames": frame_records,
    }


def split_train_val(
    records: List[dict],
    val_actors: Optional[Sequence[str]] = None,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """Split clips into train / val by actor (preferred) or randomly by clip."""
    if val_actors:
        val = [r for r in records if r["actor"] in val_actors]
        train = [r for r in records if r["actor"] not in val_actors]
        return train, val
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(val_fraction * len(shuffled))))
    return shuffled[n_val:], shuffled[:n_val]


def attach_validation_intensity_targets(records: List[dict], targets: Sequence[float]) -> None:
    for r in records:
        r["intensity_list"] = list(map(float, targets))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mead-root", type=Path, required=True,
                        help="Path to MEAD dataset root, OR to a single identity folder "
                             "(auto-detected if it contains video/ or video.tar).")
    parser.add_argument("--out-root", type=Path, required=True, help="Directory to write processed frames + JSON.")
    parser.add_argument("--emotion", type=str, default="happy",
                        help=f"Emotion to extract (or 'all'). Default: happy. Valid: {MEAD_EMOTIONS}.")
    parser.add_argument("--view", type=str, default="front",
                        help="MEAD view angle dir name. Default: front. "
                             "Other choices in MEAD: down, top, left_30, left_60, right_30, right_60.")
    parser.add_argument("--actors", type=str, nargs="*", default=None,
                        help="Optional whitelist of actor IDs (e.g. M003 W009 video-001).")
    parser.add_argument("--frames-per-clip", type=int, default=24,
                        help="How many evenly-spaced frames to keep per clip. Default: 24.")
    parser.add_argument("--target-size", type=int, nargs=2, default=(256, 256),
                        help="Cropped face size (H W). The training pipeline resizes to 256x384 itself.")
    parser.add_argument("--au-method", choices=["pyfeat", "timeline"], default="pyfeat",
                        help="AU12 estimation backend. 'timeline' is a no-deps fallback.")
    parser.add_argument("--caption-template", type=str, default=DEFAULT_CAPTION_TEMPLATE,
                        help="Caption format string. Available fields: {gender}, {actor}. "
                             "Default infers gender from M/W prefix; pass an explicit template to "
                             "override (e.g. for video-001 style identities).")
    parser.add_argument("--val-actors", type=str, nargs="*", default=None,
                        help="Hold these actors out for validation (preferred over random split).")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of clips for validation if --val-actors not given.")
    parser.add_argument("--val-intensity-targets", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 0.75, 1.0],
                        help="Fixed intensity targets to attach to validation clips.")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract frames even if dir is non-empty.")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N clips (debug).")
    args = parser.parse_args(argv)

    emotions = MEAD_EMOTIONS if args.emotion == "all" else [args.emotion]

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "annotations").mkdir(exist_ok=True)

    log.info("Discovering clips under %s ...", args.mead_root)
    clips = discover_clips(args.mead_root, args.out_root, emotions, args.view, args.actors)
    if args.limit is not None:
        clips = clips[: args.limit]
    log.info("Found %d clips matching filters.", len(clips))
    if not clips:
        log.error("No clips matched. Check --mead-root / --emotion / --view / --actors.")
        return 1

    cropper = FaceCropper(target_h=args.target_size[0], target_w=args.target_size[1])
    au_estimator = AUEstimator(args.au_method)

    records: List[dict] = []
    for i, clip in enumerate(clips, 1):
        try:
            rec = process_clip(
                clip,
                cropper,
                au_estimator,
                args.frames_per_clip,
                args.out_root,
                caption_template=args.caption_template,
                overwrite=args.overwrite,
            )
            if rec is not None:
                records.append(rec)
        except Exception as e:  # noqa: BLE001
            log.exception("Failed processing %s: %s", clip.clip_id, e)
        if i % 25 == 0:
            log.info("  processed %d / %d clips", i, len(clips))

    log.info("Successfully processed %d clips. Splitting train/val ...", len(records))
    train, val = split_train_val(records, args.val_actors, args.val_fraction)
    attach_validation_intensity_targets(val, args.val_intensity_targets)

    train_path = args.out_root / "annotations" / "train.json"
    val_path = args.out_root / "annotations" / "validation.json"
    with open(train_path, "w") as fh:
        json.dump(train, fh, indent=2)
    with open(val_path, "w") as fh:
        json.dump(val, fh, indent=2)
    log.info("Wrote %d train clips → %s", len(train), train_path)
    log.info("Wrote %d val   clips → %s", len(val), val_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
