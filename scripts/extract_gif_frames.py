#!/usr/bin/env python
"""Extract individual frames from GIF(s) as PNG images.

Each expression-sweep GIF holds N frames (one per AU12/smile intensity level).
This dumps every frame to a still image so they can be dropped into a paper,
slide, or side-by-side figure.

Usage:
    # single GIF -> frames next to it in <stem>_frames/
    python scripts/extract_gif_frames.py path/to/sample.gif

    # every GIF under a directory (recursively), mirroring the tree under --out
    python scripts/extract_gif_frames.py results_2026-07-09_141024/ --out frames/

    # custom naming / format
    python scripts/extract_gif_frames.py sample.gif --format jpg --prefix smile
"""
import argparse
from pathlib import Path

from PIL import Image, ImageSequence


def extract(gif_path: Path, out_dir: Path, fmt: str, prefix: str) -> int:
    im = Image.open(gif_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = prefix or gif_path.stem
    n = 0
    for i, frame in enumerate(ImageSequence.Iterator(im)):
        # convert from palette (mode "P") to RGB so JPG/PNG save cleanly
        frame = frame.convert("RGB")
        dst = out_dir / f"{stem}_{i:02d}.{fmt}"
        frame.save(dst)
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path,
                    help="a .gif file, or a directory to search recursively")
    ap.add_argument("--out", type=Path, default=None,
                    help="output root. Default: <gif>_frames/ beside each GIF")
    ap.add_argument("--format", default="png", choices=["png", "jpg", "jpeg"],
                    help="output image format (default: png)")
    ap.add_argument("--prefix", default=None,
                    help="filename prefix (default: the GIF's stem)")
    args = ap.parse_args()

    fmt = "jpg" if args.format == "jpeg" else args.format

    if args.path.is_dir():
        gifs = sorted(args.path.rglob("*.gif"))
        if not gifs:
            raise SystemExit(f"No .gif files found under {args.path}")
    elif args.path.suffix.lower() == ".gif":
        gifs = [args.path]
    else:
        raise SystemExit(f"Not a GIF or directory: {args.path}")

    total_frames = 0
    for gif in gifs:
        if args.out is not None and args.path.is_dir():
            # mirror the source tree under --out, one subdir per GIF
            rel = gif.relative_to(args.path).with_suffix("")
            out_dir = args.out / rel
        elif args.out is not None:
            out_dir = args.out
        else:
            out_dir = gif.with_name(f"{gif.stem}_frames")
        n = extract(gif, out_dir, fmt, args.prefix)
        total_frames += n
        print(f"{gif}  ->  {out_dir}  ({n} frames)")

    print(f"\nDone: {total_frames} frames from {len(gifs)} GIF(s).")


if __name__ == "__main__":
    main()
