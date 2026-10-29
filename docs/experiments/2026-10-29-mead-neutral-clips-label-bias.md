# MEAD neutral clips: preprocessing, and a skin-tone-associated offset in py-feat AU12 (October 29, 2026)

Next step 2 of [2026-10-28 FineFace on both detectors](2026-10-28-fineface-both-detectors.md) / decision of [2026-10-28 checkpoint selection](2026-10-28-checkpoint-selection-confirmation.md): every checkpoint lacks a neutral bottom of the scale, and MEAD-happy has almost no low-AU12 frames. This note covers getting MEAD's neutral clips into the training root and what their labels show. No training yet.

## Data

- **Source:** front-view `neutral/level_1` clips of all 47 actors (1,956 files, 4.27 GB; M026 and M032 ship each clip twice in split archives → **1,876 unique**), plus M003's 90 front `happy` clips for a labeler cross-check.
- **Transfer:** copied from the workstation's extracted MEAD tree as one archive (`mead_neutral_subset.tar`, sha256 `7839c9d8…aff74470`), because streaming `ssh | ssh` stalled three times. Member bytes equal the workstation's count exactly.
- **Preprocessing:** `scripts/preprocess_mead.py --emotion neutral --view front --val-actors W009 W011` (same 24 evenly spaced frames, MediaPipe crop, py-feat AU12 on the uncropped frame as in May). Run sharded by actor over one H100 node (`scripts/slurm/preprocess_mead_sdumont2nd.srm`, job 747836, 8 shards, ~40 min, 0 failures).
- **Merge:** merged into the training root by `scripts/merge_mead_emotion.py`:
  - frames under `MEAD_processed/frames/<actor>/neutral/`;
  - `annotations/{train,validation}_happy_neutral.json` = 4,028 + 1,796 train and 176 + 80 validation clips;
  - `train.json` / `validation.json` are untouched. Every frame path was verified to resolve.
- **Labeler cross-check:** M003's 90 happy clips, relabelled on the cluster (job 747837), against the May workstation labels: r = **0.998**, mean |Δ| 0.006, bias −0.0002, max 0.09. Cluster-made labels are interchangeable with May's.

## Labels

AU12 per frame (py-feat):

| | frames | mean | median | p95 | < 0.1 | < 0.3 | > 0.5 |
|---|---|---|---|---|---|---|---|
| happy (train) | 96,672 | 0.79 | 0.88 | 0.99 | 0.8% | 6.4% | 86.5% |
| neutral (train) | 43,104 | 0.32 | 0.27 | 0.77 | 17.9% | 54.6% | **23.4%** |

- Neutral within-clip std is 0.13 (median), so the labels are noisy.
- Support for constant sequences (clips with 5 frames within ±0.1) at c = 0.0 / 0.1 / 0.2 / 0.3 rises from 41 / 200 / 366 / 382 (happy only) to **555 / 1,390 / 1,676 / 1,548**.
- **Per-actor neutral medians range from 0.05 to 0.73.** Inspected frames: the high ones are talking or closed-mouth neutral faces without a visible smile (M005 0.73, W029 0.72, W026 0.57), while the low ones are visibly neutral too (W036 0.06).

**Skin-tone check.** Per actor (47), 40 neutral + 40 happy frames. The proxy is the median luma (YCrCb Y) of a central face patch; it is crude and confounded with lighting. Spearman vs luma:

| | py-feat AU12 | MediaPipe mouth-corner proxy (geometric, label-independent) |
|---|---|---|
| neutral frames | **−0.64 (p = 1e-6)** | −0.18 (p = 0.24) |
| happy frames | **−0.40 (p = 0.005)** | +0.02 (p = 0.91) |

- Within actor, happy > neutral for 100% (py-feat) and 96% (MediaPipe) of actors.
- Across actors, py-feat's neutral level is unrelated to MediaPipe's (Spearman +0.13, p = 0.39).

## Reading

1. **py-feat AU12 carries an actor-level offset associated with skin tone: darker faces read higher at the same geometric expression**, on neutral and happy frames alike, while the geometric detector shows no such trend. Neither detector is ground truth, so this is strong correlational evidence, not proof. It is consistent with documented demographic biases of AU detectors trained on predominantly light-skinned data.
2. **The training labels inherit it.** For darker-skinned MEAD actors, happy frames saturate near 0.95–0.97, and their neutral frames would be labelled 0.5–0.7 if taken at face value. Conditioning on raw py-feat values therefore ties the intensity scale to identity. A "c = 0" anchor learned from raw neutral labels would come mostly from light-skinned actors.
3. **The evaluation inherits it too.** The eval prompt bank names ethnicities (`prompt_idx` 14–23, 0-based: Black, East Asian, South Asian, Latina/o, Middle Eastern). Those enter `scaled` and the held-out calibration units (prompts 10–29), but not the original calibration units (0–9). Absolute py-feat levels and pooled py-feat CLS may differ by prompt group for detector reasons. MediaPipe, already the circularity guard, is also the guard here, and results should be broken down by prompt group before any demographic claim.
4. Earlier "male baseline-smile bias" findings were measured with py-feat. Whether part of that bias is detector-driven is open.

## Open decision

How to label the neutral clips (and whether to relabel the happy ones) for the low-end-data retrain. To be agreed before training and recorded in the retrain's entry.

Artifacts: `/petrobr/crono/matheus.aidano/MEAD_raw_subset/` (raw neutral + M003 happy mp4), `MEAD_neutral_processed/shard_*/` (logs + per-shard annotations; frames moved into `MEAD_processed`), `MEAD_labelcheck_M003/`, `mead_neutral_subset.tar`.
