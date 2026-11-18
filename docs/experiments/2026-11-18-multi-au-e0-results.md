# Multi-AU E0: results (November 18, 2026)

Results of the design pre-registered in [2026-11-15 multi-AU E0 design](2026-11-15-multi-au-e0-design.md), including its amendment of November 18 (replication on the 37 workstation actors). The criteria and thresholds were not changed after the results.

One code change was made before the first analysis: `analyze_multiau_e0.py` had V hard-coded at ≥ 9 actors; it now passes at ≥ 90% of the panel, which is 9/10 on the primary panel and 34/37 on the replication, as the amendment states.

**Outcome: GO with 3 channels on both panels.**
- **Channels and labels:** smile (py-feat AU12), brow raise (py-feat mean(AU01, AU02)) and brow lower (py-feat AU04).
- **The panels agree** on the decision, the labeler, the AU01/AU02 merge and every S gate, so the conflict rule does not apply.
- **Geometry's brow raise fails V on both panels.** All 7 of its failing actors are dark-skinned.
- **Two findings outside the pre-registered gates bear on E1's emotion set:** py-feat's smile label fires on negative emotions, and brow raise with brow lower together almost never occurs in MEAD.

## Runs

- **Preprocess, 10 mirror actors.** `preprocess_mead_sdumont2nd.srm` with `EXTRA_ARGS="--all-aus"`, one job per emotion, 10 actor shards each.
  - surprised 792737; happy, fear, disgusted, contempt 792760–792763. The first submissions of the last four (792741–792744) were cancelled while pending.
  - angry, sad, neutral needed three attempts; the first two landed on nodes sdumont2nd2049–2051. 792738–792740 hung in `conda info --base` and were cancelled. 792757–792759, with conda found through `command -v conda`, hung at the launcher's CUDA check (`torch.zeros(1, device='cuda')`) until the 2 h limit. 800087–800089 ran with those nodes excluded and the check under `timeout 180`.
  - Output `MEAD_multiau_processed/`: 6,568 clips, 157,632 frames, no errors.
- **Scoring, 10 actors.** `score_mead_frames_sdumont2nd.srm` on `cpu_amd`, 96 processes: surprised, happy, fear, disgusted and contempt in 800090 (geometry) and 800091 (LibreFace); angry, sad and neutral in 801463 / 801464. 157,632 lines per detector.
- **Preprocess, 37 workstation actors.** 801466–801473, 12 actor shards per emotion, 2 GPUs per job, about 1 h each. Output `MEAD_multiau_processed_37/`: 24,492 clips, 587,808 frames.
  - One transferred clip is missing: `W014/video/front/angry/level_1/017.mp4` is 0 bytes in the MEAD distribution itself (0 bytes, dated 2020-08-27, in the workstation archive listing).
- **Scoring, 37 actors.** First submitted to `cpu_amd` (801815 / 801816), which estimated a start three days later. Cancelled and re-run on the CPUs of H100 nodes (the H100 QoS requires one GPU per job; it stays unused): 801831 (LibreFace, 48 processes × 2 threads) and 801832 (geometry, 24 processes). 587,808 lines per detector; LibreFace took 1 h 16, geometry 10 min.
- **Analysis.** Files are in `/petrobr/crono/matheus.aidano/e0_multiau/`.
  - Pre-registered: `analyze_multiau_e0.py` → `e0_report.json` and `e0_report.log` (10 actors), `e0_report_37.json` and `e0_report_37.log` (37 actors).
  - Follow-ups marked *not pre-registered* below: `e0_posthoc.py` → `e0_report_posthoc.json` / `e0_report_37_posthoc.json`, printed output in `e0_posthoc.log` / `e0_posthoc_37.log`.

## Data checks

- **Detector failures** (frames; the analysis prints rates, which round to 0.000 in every emotion):

  | panel | frames | py-feat no-face | LibreFace failure | geometry with a NaN |
  |---|---|---|---|---|
  | 10 actors | 157,632 | 6 | 0 | 8 |
  | 37 actors | 587,808 | 14 | 0 | 44 |
- **Label continuity** with the training root, on happy and neutral frames:
  - new py-feat AU12 vs the existing labels: r 0.999 on both panels (10 actors: 31,128 frames, mean |Δ| 0.0065, max 0.27; 37 actors: 114,792 frames, mean |Δ| 0.0070, max 0.50);
  - 300 sampled crops per panel are pixel-identical to the existing ones.

  The `--all-aus` pipeline reproduces the training data.

## Primary panel (10 actors): pre-registered criteria

V: actors whose target-emotion level-3 median clip value exceeds their neutral median (need 9 of 10; one-sided sign-test p). A: actor-centered pooled frame r with each of the other two detectors (need ≥ 0.4 with one). D: median over actors of the within-actor Spearman ρ between level and clip value (descriptive).

| channel | labeler | V | A (r with the others) | mean A | D | pass |
|---|---|---|---|---|---|---|
| brow raise (target: surprised) | **py-feat** | 10/10 (p 0.001) | LibreFace 0.703, geometry 0.746 | **0.725** | 0.75 | yes |
| | LibreFace | 10/10 (p 0.001) | py-feat 0.703, geometry 0.519 | 0.611 | 0.84 | yes |
| | geometry | **8/10** (p 0.055) | py-feat 0.746, LibreFace 0.519 | 0.632 | 0.58 | **no** (V) |
| brow lower (target: angry) | **py-feat** | 10/10 (p 0.001) | LibreFace 0.654, geometry 0.553 | **0.604** | 0.79 | yes |
| | LibreFace | 10/10 (p 0.001) | py-feat 0.654, geometry 0.493 | 0.574 | 0.83 | yes |
| | geometry | 10/10 (p 0.001) | py-feat 0.553, LibreFace 0.493 | 0.523 | 0.69 | yes |

- **Geometry's brow raise fails V** on M011 (−0.008) and W038 (−0.000).
- **Smile yardstick** (A for smile, not a gate): py-feat vs LibreFace 0.601, py-feat vs mouth-corner proxy 0.372, LibreFace vs proxy 0.379. Both brow channels agree across detectors at least as well as the smile channel the main model is trained on.

**Choices.**
- **Labeler:** py-feat for both brow channels. It has the highest mean A on each, so the py-feat tie rule was not needed. One py-feat call labels all three channels, and LibreFace and geometry are the independent judges in E1.
- **AU01 / AU02:** py-feat AU01 vs AU02 r = 0.785 ≥ 0.5, so brow raise is mean(AU01, AU02).

**S (separability)**, actor-centered frame r on the chosen labelers (py-feat for all three):

| pair | r | gate \|r\| ≤ 0.7 |
|---|---|---|
| smile ~ brow raise | −0.054 | pass |
| smile ~ brow lower | −0.192 | pass |
| brow raise ~ brow lower | −0.642 | pass, by 0.058 |

**Decision (primary): GO with 3 channels** (smile, brow raise, brow lower).

**Descriptive** (pre-registered, no gate):

1. **Speech confound**, actor-centered frame r with the inner lip gap within neutral clips: py-feat raise 0.02, lower 0.15; LibreFace 0.08, 0.21; geometry −0.32, 0.23. The chosen labeler is the least confounded. The geometric brow raise moves with mouth opening.
2. **Within-clip range:** 62% of surprised level-3 clips (brow raise) and 72% of angry level-3 clips (brow lower) have an actor-scaled p90 − p10 ≥ 0.3. The brows move within a clip, as AU12 does in happy clips.
3. **Joint coverage**, frames per cell of actor-scaled (smile, raise, lower) split at 0.5:

   | cell | none | smile | raise | lower | smile + raise | smile + lower | raise + lower | all three |
   |---|---|---|---|---|---|---|---|---|
   | frames | 50,005 | 31,079 | 30,662 | 27,654 | 9,396 | 8,071 | **616** | **143** |

   Each channel alone is well covered. Brow raise and brow lower together are almost absent (0.5% of frames).
4. **Detector failures:** see Data checks.
5. **Skin tone**, Spearman between each actor's neutral median and face luma (n = 10):
   - py-feat smile −0.85: the actor offset of [2026-10-29](2026-10-29-mead-neutral-clips-label-bias.md) shows again (lower luma, higher neutral AU12).
   - py-feat brow raise −0.12 and brow lower +0.09: no comparable offset at n = 10. The replication does find one on brow raise.
   - LibreFace: smile −0.57, raise −0.33, lower +0.10. Geometry: smile +0.05, raise −0.43, lower +0.38.
6. **Label continuity:** see Data checks.

## Replication (37 actors)

Same pipeline and criteria. V needs 34 of 37 actors.

| channel | labeler | V | A (r with the others) | mean A | D | pass |
|---|---|---|---|---|---|---|
| brow raise (target: surprised) | **py-feat** | 37/37 | LibreFace 0.708, geometry 0.743 | **0.725** | 0.83 | yes |
| | LibreFace | 37/37 | py-feat 0.708, geometry 0.526 | 0.617 | 0.89 | yes |
| | geometry | **32/37** | py-feat 0.743, LibreFace 0.526 | 0.635 | 0.60 | **no** (V) |
| brow lower (target: angry) | **py-feat** | 37/37 | LibreFace 0.603, geometry 0.608 | **0.606** | 0.90 | yes |
| | LibreFace | 36/37 | py-feat 0.603, geometry 0.488 | 0.546 | 0.80 | yes |
| | geometry | 37/37 | py-feat 0.608, LibreFace 0.488 | 0.548 | 0.80 | yes |

- **V failures:**
  - geometry's brow raise fails on M013, W018, W025, W029 and W033;
  - LibreFace's brow lower fails on M029 (−0.25).
- **Smile yardstick:** py-feat vs LibreFace 0.599, py-feat vs proxy 0.450, LibreFace vs proxy 0.387.
- **Labeler the 37 would choose:** py-feat for both brow channels, as on the 10.
- **AU01 / AU02:** r = 0.827, so the merge holds.
- **S:** smile ~ raise −0.073, smile ~ lower −0.169, raise ~ lower −0.611. All three pass.

**Decision (replication): GO with 3 channels.** It agrees with the primary panel.

**Descriptive:**

1. **Speech confound**, neutral clips: py-feat raise 0.03, lower 0.11; LibreFace 0.22, 0.24; geometry −0.40, 0.28. py-feat is again the least confounded.
2. **Within-clip range:** 38% of surprised level-3 clips (brow raise) and 70% of angry level-3 clips (brow lower). Brow raise varies less within a clip than on the 10 (62%).
3. **Joint coverage:**

   | cell | none | smile | raise | lower | smile + raise | smile + lower | raise + lower | all three |
   |---|---|---|---|---|---|---|---|---|
   | frames | 193,013 | 95,339 | 133,974 | 101,383 | 36,099 | 24,438 | **3,151** | **397** |

   Raise and lower together: 0.6% of frames.
4. **Detector failures:** see Data checks.
5. **Skin tone**, neutral median vs luma (n = 37):
   - py-feat smile −0.55 (p < 0.001): the 2026-10-29 offset replicates on more actors;
   - the brow channels have actor offsets too: py-feat brow raise +0.37 (p 0.02), LibreFace brow lower −0.41 (p 0.01), geometry brow lower +0.43 (p 0.008);
   - every other channel × detector has |ρ| ≤ 0.28.
6. **Label continuity:** see Data checks.

## Follow-ups (not pre-registered)

From `e0_posthoc.py`, on both panels (10 / 37).

1. **The brow raise ~ brow lower correlation is mostly between emotions.**
   - **Within single emotions,** py-feat's r runs from −0.18 to −0.52 on the 10 and from −0.13 to −0.56 on the 37, against −0.64 / −0.61 pooled.
   - **py-feat rarely scores AU02 and AU04 together:** AU04 vs AU02 is −0.72 / −0.67, and AU04 vs AU01 is −0.48 / −0.50.
   - **The other detectors:** LibreFace's raise ~ lower is −0.28 / −0.33. Geometry's is −0.80 / −0.78, because its two ratios use the same brow landmarks.
   - **The missing raise + lower combination is in the data, not only in py-feat.** All three detectors read fear level 3 as raise with little or no lowering: actor-scaled brow lower is −0.32 / 0.06 on py-feat, 0.04 / 0.12 on LibreFace and −0.07 / 0.07 on geometry. The FACS prototype of fear includes AU4.
2. **py-feat's smile fires on negative emotions; LibreFace's does not.** The table gives the median over actors of the actor-scaled smile channel at level 3, as 10 / 37 actors:

   | detector | neutral | happy | contempt | disgusted | angry | fear | sad | surprised |
   |---|---|---|---|---|---|---|---|---|
   | py-feat AU12 | 0.02 / 0.03 | 0.77 / 0.83 | 0.45 / 0.39 | 0.37 / 0.23 | 0.22 / 0.26 | 0.16 / 0.18 | 0.31 / 0.09 | 0.06 / 0.04 |
   | LibreFace AU12 | 0.01 / 0.00 | 0.52 / 0.60 | 0.14 / 0.10 | 0.08 / 0.08 | 0.03 / 0.08 | 0.10 / 0.06 | 0.10 / 0.09 | 0.07 / 0.03 |

   - **Where:** on both panels, py-feat exceeds LibreFace by 0.15–0.31 on contempt, disgusted and angry. On sad it does so only on the 10.
   - **What py-feat's AU12 seems to follow:** cheek and upper-lip movement. py-feat AU06 / AU10 are high on the same emotions (37 actors: disgusted 0.36 / 0.78, angry 0.40 / 0.65).
   - **The mouth-corner proxy cannot referee:** it reads 0.44–0.46 on angry.
   - **Why the gates missed it:** the S gate measures linear dependence between channels, not whether the smile label means a smile outside happy. The current model never met the problem, since it trains on happy and neutral only.
3. **Skin tone and the V contrasts** (Spearman between an actor's V contrast and face luma):
   - **Geometry's brow raise:** all 7 actors that fail V are dark-skinned: M011 and W038 (luma 83, 79) of the 10; M013, W018, W025, W029 and W033 (luma 52–66) of the 37. ρ = +0.44 (p 0.20) / +0.43 (p 0.009). FaceMesh brow landmarks appear to under-register brow raise on dark faces.
   - **py-feat's brow raise:** on the 10, the three weakest contrasts were on dark faces (ρ +0.33, p 0.35). This does not replicate: on the 37, ρ = −0.34 (p 0.04), i.e. darker faces have larger contrasts.
   - **LibreFace's brow raise:** +0.72 (p 0.02) / +0.27 (p 0.11).
   - **Brow lower:** |ρ| ≤ 0.21 for all three detectors on the 37.
4. **Per-emotion brow profile**, py-feat, actor-scaled, level 3 (median over actors, 10 / 37):

   | | neutral | happy | surprised | fear | contempt | angry | disgusted | sad |
   |---|---|---|---|---|---|---|---|---|
   | brow raise | 0.00 / −0.00 | 0.11 / 0.11 | 0.61 / 0.86 | 0.72 / 0.72 | 0.11 / 0.45 | −0.39 / −0.75 | −0.37 / −0.74 | −0.13 / −0.27 |
   | brow lower | 0.02 / 0.02 | −0.19 / −0.01 | −0.34 / −0.03 | −0.32 / 0.06 | 0.09 / 0.13 | 0.71 / 0.71 | 0.69 / 0.65 | 0.65 / 0.60 |

   Brow raise comes from surprised and fear, and on the 37 also from contempt. Brow lower comes from angry, disgusted and sad, which also push brow raise below neutral.

## For the E1 design

The design left three E1 choices to this entry: the normalization, the emotion set and the data plan. Two inputs changed after it was written: all 47 actors now have all 8 emotions, and py-feat's smile fires outside happy. This entry therefore gives a recommendation for each choice, and the E1 design fixes them.

- **Data plan: all 47 actors, with W009 and W011 held out as in May.** Options (a) and (b) assumed that only the 10 mirror actors had non-neutral clips. The labels pass on the 37 as well, and 45 training identities instead of 10 addresses the identity limit noted in the design.
- **Normalization: the per-actor piecewise-linear map of `normalize_labels_per_actor.py`, applied to each channel** (the actor's neutral median → floor, 1 → 1).
  - The brow channels have actor offsets too (replication, descriptive 5), so they need per-actor normalization as AU12 does.
  - The same map leaves the smile labels identical to the main model's.
  - Actor-scaling (p99 → 1) would tie the top of each scale to the actor's single strongest frames.
- **Emotion set: neutral, happy, surprised, fear, angry and sad. Contempt and disgusted are left out.**
  - **Why those two:** py-feat's smile false positives are largest there on both panels (follow-up 2).
  - **What still comes in:**
    - brow raise from surprised and fear;
    - brow lower from angry and sad;
    - smile from happy.
  - **The remaining cost:** py-feat still reads 0.22 / 0.26 smile on angry, against 0.03 / 0.08 on LibreFace. Comparing the C arm (AU12-only, same data) with the current main model (happy + neutral) measures what the added emotions cost smile control.
- **Judges:**
  - **Brow raise:** LibreFace is the independent judge. Geometry's brow raise fails V on both panels, all on dark faces, so it is reported only by skin-tone group.
  - **Brow lower:** judged by LibreFace and geometry.
- **Commands to test:**
  - **Single-channel commands:** each brow channel alone, with the others at neutral, is the primary test.
  - **Combined commands:** raise + lower occurs in 0.5–0.6% of training frames, so a combined command is extrapolation and is reported as such.
- **Leakage:** E1 measures each command's leakage onto the other channels and AUs, as [2026-08-25](2026-08-25-au-coactivation.md) did for smile. Brow raise comes from surprised and fear clips, so its leakage onto the mouth has to be checked.
