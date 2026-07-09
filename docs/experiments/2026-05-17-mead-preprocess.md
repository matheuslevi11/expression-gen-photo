# Full MEAD preprocess (May 17, 2026)

`scripts/preprocess_mead.py` run over the full MEAD download (front-view happy clips, MediaPipe face crops, per-frame AU12 via py-feat). Log ends with: `Successfully processed 4204 clips` → split into train/val JSON.

Output (`MEAD_processed/`, gitignored):

- **4,028** train clips, **176** validation clips
- **24** frames per clip, **~1.9 GB** of cropped JPEGs
- Annotations include per-frame `au12` from py-feat; validation entries include `intensity_list` for fixed-target evaluation
