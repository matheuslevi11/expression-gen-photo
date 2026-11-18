# Multi-AU E0: are MEAD brow labels usable for a 3-channel model? (design, November 15, 2026)

Pre-registration. No training. This entry fixes the data, the labelers and the decision criteria before any label of the new emotions is analysed. Results go in a separate entry.

## Why

- [2026-08-11 AU-vector proposal](2026-08-11-au-vector-proposal.md) left multi-AU control data-gated: happy-only MEAD gives no non-smile variance.
- [2026-08-25 AU co-activation](2026-08-25-au-coactivation.md) showed that our scalar AU12 axis drags the whole MEAD-happy expression, including some brow movement (LibreFace AU01 r 0.63, AU04 0.37; geometry agrees; py-feat barely sees it). FineFace, which commands AUs independently, localizes (AU01 leakage 0.017 vs our 0.337).
- AU12 calibration has plateaued for the current recipe ([2026-11-14](2026-11-14-longer-training-results.md)), and training is now cheap (25k steps ≈ 2 h on 4 H100).

**Plan agreed on 2026-11-15:**
- **Three channels:** smile (AU12), brow raise (AU1+2) and brow lower (AU4).
- **Encoding:** one AU per physical channel of the existing 6-channel embedding, plus a per-AU CCL prompt (`cin: 384` unchanged).
- **Why brows and not mouth AUs:** brows are less confounded with speech than mouth AUs (MEAD clips are speech), and they target the leakage measured on 08-25.
- **Order:** E0 (this entry) decides whether the brow labels are usable and with which labeler. E1 then trains the 3-channel model against an AU12-only arm on the same data.
- **Later expansion:** if this works, expand toward FineFace's AU set.

## Data

- **Source.** The workstation that holds the full MEAD is unreachable, so E0 uses the Hugging Face mirror `NoahMartinezXiang/2020_MEAD`.
  - File used: `MEAD.zip` (sha256 `cb73db0b…71daad`, 14.27 GB).
  - Contents: 6,568 front-view clips, 1920×1080 at 30 fps, covering 10 actors × 8 emotions. Emotional levels 1–3 hold ~30 clips each; neutral holds 40.
  - The mirror's `apache-2.0` tag is the uploader's. MEAD's own terms apply, and the thesis cites MEAD.
- **Provenance check.**
  - **Byte identity:** 490 mirror clips are byte-identical (size + CRC32) to official files already on the cluster. These are all 400 neutral clips of these actors plus M003's 90 happy clips.
  - **Actor labels are permuted for 7 of 10 actors in the mirror.** The byte matches give the map:

    | mirror | M003 | M005 | M007 | M009 | M011 | W036 | W037 | W038 | W039 | W040 |
    |---|---|---|---|---|---|---|---|---|---|---|
    | official | M003 | **W040** | **M011** | M009 | **M007** | W036 | **W038** | **W035** | **W037** | **M005** |

  - **The official IDs are the right ones:** the M/W prefix of each official ID matches the person's gender.
  - **One person per label:** a contact sheet with one frame per mirror label and emotion shows the same person in all 8 emotions.
  - **Emotional clips, not just neutral:** in the smoke test, the W040 happy crops made from the mirror are pixel-identical to the existing `MEAD_processed` crops, which came from the official archives.
- **Extraction.** Clips are extracted under their official IDs into `MEAD_raw_subset/<actor>/video/front/<emotion>/level_<n>/`, next to the 47 actors' neutral clips. Artifacts in `/petrobr/crono/matheus.aidano/MEAD_hf/`: `extract_remap.py`, `manifest.json` (per-clip mirror name, official actor, size, CRC32), `identity_sheet.jpg`, and `MEAD.zip`, which is kept.
- **Actors:** M003, M005, M007, M009, M011, W035, W036, W037, W038, W040. All are in the happy training split; neither validation actor (W009, W011) is among them. Clips per emotion range from 77 to 90, except M009 surprised with 58.
- **Not available:** the other 37 actors' non-neutral clips, including their raw happy clips. They stay on the workstation (backlog in `status.md`).

## Labels

Every clip is reprocessed: the same 24 evenly spaced frames and the same MediaPipe crop as May. Three detectors score the frames:

| Detector | How | Output |
|---|---|---|
| **py-feat** `xgb` | `preprocess_mead.py --all-aus`: the same per-frame call that makes the AU12 training labels, on the uncropped frame, now keeping all 20 AU columns | presence probability [0, 1]; `aus = null` on no-face frames (the AU12 field keeps its 0.0 fallback) |
| **LibreFace** 0.2.0 | `score_mead_frames.py --scorer libreface` on the 256² crops (isolated `libreface` env, CPU) | 12 AU intensities, 0–5 |
| **Geometry** (FaceMesh) | `score_mead_frames.py --scorer geo` on the crops | brow-raise and inner-brow-gap ratios, lip gap, eye aspect ratio, mouth-corner smile proxy |

Candidate measurement of each channel:

| Channel | py-feat | LibreFace | geometry |
|---|---|---|---|
| smile | AU12 | AU12 | mouth-corner proxy |
| brow raise | mean(AU01, AU02) | mean(AU01, AU02) | `AU1_2_brow_raise` |
| brow lower | AU04 | AU04 | −`AU4_inner_brow_gap` (negated so that higher = more furrow) |

**Definitions:**
- **Clip value:** the mean over the clip's frames that have a face.
- **Actor-centered:** minus the actor's mean over all of that actor's frames. This removes identity offsets such as the skin-tone offset of [2026-10-29](2026-10-29-mead-neutral-clips-label-bias.md).
- **Actor-scaled:** (v − actor's neutral median) / (actor's p99 over all frames − actor's neutral median).

All criteria below are correlations or within-actor contrasts, so the detectors' different scales do not matter.

## Pre-registered criteria

Fixed before analysing any label of these actors. The only data seen are the smoke test: 2 W040 surprised clips at level 1 (py-feat clip means AU01 0.67, AU02 0.60, AU04 0.19) and 30 of their frames on each independent scorer.

**Per brow channel × labeler** (3 labelers):

- **V — validity.** Per actor: median clip value of the target emotion at level 3 minus median clip value of neutral. The target is surprised for brow raise and angry for brow lower. **Pass: the difference is > 0 in ≥ 9 of 10 actors.**
- **A — agreement.** Pearson r between the labeler and each of the other two detectors, on actor-centered frame values pooled over all 8 emotions. **Pass: r ≥ 0.4 with at least one of the two.**
  - Yardstick (reported, not a gate): the same statistic for smile, i.e. py-feat AU12 vs the mouth-corner proxy and vs LibreFace AU12.
- **D — dose (descriptive).** Per actor, the Spearman correlation between level (neutral = 0, then 1, 2, 3) and clip value within the target emotion. Reported as the median over actors.

**Choices:**

- **Labeler per channel:** among labelers that pass V and A, the one with the highest mean A, i.e. the most corroborated. If py-feat is within 0.05 of the best, py-feat is chosen, because one detector call then labels all channels. The other two detectors become that channel's independent judges in E1.
- **AU01 / AU02 merge:** keep mean(AU01, AU02) if their actor-centered frame r on the chosen labeler is ≥ 0.5. Otherwise the brow-raise channel uses whichever of AU01 and AU02 has the higher A; ties go to AU02, which is not part of sadness.

**Channel-level gate:**

- **S — separability.** Actor-centered frame |r|, over all frames and on the chosen labelers, must be ≤ 0.7:
  - between a brow channel and smile, for that channel to be **separable**;
  - between the two brow channels, for both to be kept.

  Collinear labels cannot be decoupled by conditioning.

**Decision:**

- **GO with 3 channels:** both brow channels pass V and A, both are separable from smile, and they are separable from each other.
- **GO with 2 channels** (smile + one brow channel): only one brow channel passes and is separable, or both are but they are collinear with each other. In that case the one with the higher mean A is kept. The E1 encoding of the unused physical channel is decided in E1's design.
- **STOP:** no brow channel passes V, A and separability. The negative result is written up before any training.

**Descriptive** (reported, no gate):

1. **Speech confound:** within neutral clips, the actor-centered frame r between each brow channel and the inner lip gap.
2. **Within-clip range:** fraction of target-emotion level-3 clips whose actor-scaled p90 − p10 is ≥ 0.3.
3. **Joint coverage:** frame counts in the 2×2×2 grid of actor-scaled smile, brow raise and brow lower, each split at 0.5.
4. **Detector failures:** no-face and failure rate per detector and emotion.
5. **Skin tone:** each actor's neutral median vs face luma, Spearman per channel × detector (n = 10, low power).
6. **Label continuity:**
   - the new py-feat AU12 vs the existing labels on these actors' happy and neutral frames (expected r ≈ 0.99, as for M003 on 10-29);
   - crop pixel equality between the new and existing happy/neutral crops.

**Fixed in the results entry, not here:**

- the normalization for E1 (the AU12 piecewise-linear map of `normalize_labels_per_actor.py` extended to each channel, or actor-scaled);
- the E1 emotion set;
- the E1 data plan: (a) these 10 actors for the new emotions plus all 47 for happy/neutral, or (b) these 10 only.

**Limits:**
- With 10 actors, statements are within-actor contrasts on a small panel. The ≥ 9/10 rule has a one-sided sign-test p of 0.011.
- The brow variation the model would learn comes from 10 identities. E1's identity guardrail has to watch for brow commands pulling faces toward these actors.

## Amendment (November 18, 2026, before any E0 label was analysed)

**New data.** The workstation came back, and the other 37 actors' front clips are now on the cluster:
- all 7 non-neutral emotions, 23,017 unique clips (one archive per emotion, sha256-verified);
- the transferred happy clip set is identical to the 3,307 clips of the May training annotations for these actors.

With the 47 actors' neutral clips already there, the cluster now holds all 8 emotions for all 47 actors.

**Primary analysis: unchanged.** The 10 mirror actors, with the criteria above.

**Added: a replication on the 37 workstation actors.**
- **Same pipeline:** `preprocess_mead.py --all-aus`, LibreFace and geometry, `analyze_multiau_e0.py`. Output: `MEAD_multiau_processed_37/`.
- **Same criteria:** V becomes ≥ 90% of actors (≥ 34 of 37); the A and S thresholds are unchanged.
- **Labeler:** the choice is made on the 10. The replication reports which labeler the 37 would choose.
- **Conflict rule:** if the two panels reach different decisions, the more conservative one holds. The results entry then has to explain the disagreement before E1 starts.

## Execution

- **Preprocess:** `scripts/slurm/preprocess_mead_sdumont2nd.srm` with `EXTRA_ARGS="--all-aus"`, one job per emotion (792737–792744), 10 actor shards each. Output: `MEAD_multiau_processed/<emotion>/shard_*/`, separate from the training root.
- **Scoring:** `scripts/slurm/score_mead_frames_sdumont2nd.srm` on a `cpu_amd` node, once for geo and once for LibreFace (weights cached in `/petrobr/crono/matheus.aidano/weights_libreface/`).
- **Analysis:** `scripts/analyze_multiau_e0.py` (written before the results; committed with them).
