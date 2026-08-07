# Pre-Registered Analysis Plan (v1.1)

## Single-Cell Kinematic Phenotyping of Human Sperm: Does Trajectory Structure Predict DNA Integrity Beyond WHO Motility Categories?

**Version:** 1.1 — revised for internal consistency with prior work (see §0).
**Author:** N. Barnard (UiT).
**Status:** To be frozen and timestamped (OSF/Zenodo) BEFORE any model is fit against DFI, HDS, vitality, or morphology.
**Dataset:** VISEM-Tracking (Zenodo 7293726), 20 videos with paired clinical semen analysis (the *orig20* cohort; hand-annotated tracks).
**Code freeze commit:** `<HASH>` — the commit that adds `experiments/features_track.py`, `features_sample.py`, `cluster_tracks.py`, `predict_clinical.py`, `figures_prereg.py`. Filled in at freeze time.

---

## 0. Relationship to prior work (transparency)

Two DFI analyses on this project already exist and are cited here so the sequence of
tests is fully transparent (guarding against an across-plans garden-of-forking-paths):

- **Dynamics/memory features vs DFI (already run, null).** `experiments/dfi_predict.py`
  tested whether *switching-dynamics* features (dwell probabilities, hysteresis, dwell
  CV) improve out-of-sample DFI prediction over CASA composition, n=77. Result was a
  clean null (Ridge Spearman: CASA 0.501, memory 0.427, CASA+memory 0.499; ΔMAE-equiv
  Δρ = −0.002, permutation p = 0.567). The mover–stayer→DFI association collapsed under
  composition control (`experiments/stayer_dfi.py`).
- **This plan is a DISTINCT question.** It uses *instantaneous single-cell kinematic
  distribution* features (quantiles of VCL/LIN/ALH/BCF, power, turning, heterogeneity),
  **not** temporal dynamics features, and it is restricted to the clean *orig20* cohort
  (hand-annotated tracks) to avoid the tracking-pipeline batch effect documented across
  the two cohorts. It asks whether the *shape* of the single-cell kinematic distribution
  adds information about DNA integrity beyond the three WHO percentages.

Because a prior DFI test was null, the confirmatory bar here is deliberately the
**incremental** value over WHO-3 (§2, H1), not a bare "kinematics correlate with DFI".

---

## 1. Motivation and framing

Manual WHO semen analysis compresses the motility of an entire ejaculate into three
percentages (progressive, non-progressive, immotile) because a human observer cannot do
more. An automated tracking pipeline measures the full kinematic trajectory of every
visible cell and therefore has access to the whole *distribution* of single-cell motion,
not just its trichotomised summary.

The scientific question is whether that distribution carries clinically meaningful
information the WHO summary destroys. "Clinically meaningful" is operationalised using
measurements that are orthogonal to motility in both biology and measurement modality:
DNA fragmentation index (DFI, SCSA/flow cytometry), high DNA stainability (HDS), vitality,
and normal morphology. Measured independently of any motility assessment, they cannot leak
into kinematic features by construction; any predictive signal is a genuine cross-modality
association.

This is a hypothesis-generating pilot at n = 20 whose deliverable is an honest,
leakage-free effect-size estimate that either justifies or kills a properly powered
follow-up. It is not expected to produce definitive clinical claims.

---

## 2. Hypotheses (locked)

**H1 (primary — INCREMENTAL).** Adding single-cell kinematic-distribution features to the
three WHO percentages lowers leave-one-out DFI error relative to WHO-3 alone; i.e.
MAE(Set C = WHO-3 + kinematics) < MAE(Set A = WHO-3). This *incremental* framing is the
primary test because a raw "kinematics beat WHO-3" comparison can succeed trivially by
re-deriving the WHO percentages at a better threshold (see §2a).

**H1b (primary — ORTHOGONALISED).** Kinematic features carry DFI information *beyond linear
composition*: a model on kinematic features residualised (within training folds) against
WHO-3 (denoted Set B⊥) beats the mean-predictor null for DFI.

**H2 (secondary).** H1 (incremental) holds for at least one of HDS, vitality, normal
morphology, under Holm–Bonferroni within the secondary family.

**H3 (structural).** The empirical distribution of single-cell kinematic features, pooled
across samples, is better described by a partition that does not coincide with the WHO
progressive/non-progressive/immotile trichotomy than by the WHO partition (BIC + cluster
stability + NMI to WHO, §7). **H3 is the lead deliverable**: it is descriptive, requires no
clinical target, and "either answer is a contribution."

**Directional expectation for H1/H1b:** vigour-tail features (upper quantiles of VCL,
PWR = ALH×BCF) and within-sample heterogeneity (dispersion measures) carry any DFI signal.
Recorded so post-hoc storytelling can be checked against it.

### 2a. Why the primary is incremental / orthogonalised (fix over v1.0)

Set B (v1.0) includes a "vigorous fraction (VCL>50)" and an "intermediate fraction
(VCL∈[5,25])" — these are motility composition under different thresholds and are
near-collinear with the WHO percentages. Hence "Set B beats Set A" is a weak claim (it can
be satisfied by re-binning motility). The two claims that survive this objection are the
**incremental** test (C vs A) and the **composition-orthogonalised** test (B⊥ vs null).
Both are pre-registered as primary. Raw "B vs A" is reported descriptively only.

---

## 3. Data and frozen pipeline configuration

All kinematic features come from ONE frozen configuration, removing the detector as a
confound:

| Item | Setting | Rationale |
|---|---|---|
| Detection | Ground-truth VISEM tracks in `outputs/tracks/{id}_tracks.csv` | Isolates biology from detector error |
| Tracker | As delivered in the frozen `outputs/tracks/` (hand-annotated) | Single configuration; no tracker shopping |
| Calibration | 1.422 px/µm, 50 fps (`config.py`) | Fixed constant |
| Quality filters | `events/detect_events.filter_tracks`: mean conf < 0.4; VCL > 200; jitter (VCL>20 & LIN<0.02); duration < 0.3 s; min 10 frames | Frozen; never re-tuned against a clinical target |
| Videos | All 20: 11,12,13,14,15,19,21,22,23,24,29,30,35,36,38,47,52,54,60,82 | No exclusions unless a video yields <10 usable tracks (then excluded and reported) |

The commit hash of the code used is recorded in the header before analysis. No listed
parameter may change after the first model is fit; any deviation is logged in §10.

**Forbidden:** re-running the WHO threshold sweep, changing quality filters, or altering
tracker settings after observing any result against a clinical target. The inherited WHO
thresholds (VCL≥25, STR≥0.5, VCL≤5) are used as-is for the WHO baseline and H3 geometry;
their prior optimisation against motility GT is acceptable *for the baseline* because it can
only strengthen the baseline, biasing the comparison against H1.

---

## 4. Feature definitions (locked)

### 4.1 Per-track (single-cell) features

Computed per retained track. VCL/VSL/VAP/LIN/STR/WOB/ALH/BCF are exactly
`events/detect_events.compute_track_metrics`. No per-track features may be added after
unblinding.

| # | Feature | Definition |
|---|---|---|
| 1 | VCL | Curvilinear velocity (µm/s) |
| 2 | VSL | Straight-line velocity (µm/s) |
| 3 | VAP | 5-frame smoothed-path velocity (µm/s) |
| 4 | LIN | VSL/VCL |
| 5 | STR | VSL/VAP |
| 6 | WOB | VAP/VCL |
| 7 | ALH | Mean lateral head amplitude about smoothed path (µm) |
| 8 | BCF | Beat-cross frequency (Hz) |
| 9 | PWR | ALH × BCF (µm/s), flagellar-power proxy |
| 10 | TAC | Circular SD of frame-to-frame turning angles (rad) |
| 11 | VAR_V | CV of instantaneous speed within the track |
| 12 | DUR | Track duration (s) — DIAGNOSTIC ONLY, never a predictor (§8.2) |

### 4.2 Sample-level feature sets (the models compared)

**Set A — WHO-3 (3 features).** Progressive %, non-progressive %, immotile %.
*Primary uses raw tracked-only fractions* (always defined; deviation from v1.0 "adjusted",
logged §10). The detection-JSON "untracked→immotile" adjusted percentages are a sensitivity
(§6.4).

**Set B — Kinematic quantile set (24 features).** For each of VCL, LIN, ALH, BCF: p10, p50,
p90, IQR (16). Plus PWR p90 & IQR (2); TAC median & IQR (2); VAR_V median (1); vigorous
fraction VCL>50 (1); intermediate fraction VCL∈[5,25] (1); log(#retained tracks) (1). = 24.
VSL/VAP/STR/WOB are omitted from the quantile backbone (near-deterministic functions of the
others → collinearity, not information).

**Set C — WHO-3 + kinematic (27 features).** Union of A and B; tests *incremental* value.

**Set B⊥ — composition-orthogonalised kinematics.** Set B features residualised on Set A
(WHO-3) using training-fold OLS; tests information *beyond* linear composition.

No other feature set may be evaluated against clinical targets. Later exploratory features
go in a clearly labelled exploratory section computed only after the confirmatory analysis.

### 4.3 Preprocessing

Features z-scored using training-fold statistics only (inside each LOO fold — never on the
full dataset). Targets untransformed; DFI additionally as log(DFI) in sensitivity (§6.2).

---

## 5. Confirmatory statistical protocol (H1, H1b, H2)

### 5.1 Targets

Primary: **DFI (%)**. Secondary: HDS (%), vitality (%), normal morphology (%). Samples
missing a target are dropped for that target only, with counts reported.

### 5.2 Model and validation

- **Estimator:** ridge regression (stable at n=20, p≤27; no estimator shopping). Elastic
  net & GP appear only in sensitivity (§6.1).
- **Validation:** leave-one-video-out (LOO). In VISEM, video = patient (re-verified and
  stated).
- **Hyperparameter (ridge α):** nested — inner LOO over the 19 training samples per outer
  fold, grid α ∈ logspace(−3, 6, 19). The upper end is high enough that a p=27 model at
  n≈18 can shrink toward the training mean (avoiding pathological overfit) rather than
  being capped at weak regularisation. α never selected using the held-out sample. (Inner
  LOO error uses the exact ridge LOO/hat-matrix closed form for speed.)
- **Metrics:** LOO MAE (primary), LOO R² = 1 − SSE/SST (SST from training-fold means, so a
  useless model scores ≤ 0), Spearman ρ(pred, obs).

### 5.3 Inference

- **H1 (incremental):** ΔMAE = MAE(Set A) − MAE(Set C). Paired permutation test: 10,000
  permutations of the target vector, recomputing the full nested-LOO ΔMAE each time. Both
  models share each permuted target (paired). One-sided p, α = 0.05.
- **H1b (orthogonalised):** MAE(Set B⊥) vs mean-predictor, same permutation machinery.
- **Also reported (descriptive):** MAE(Set B) vs MAE(Set A) (raw, non-primary, per §2a).
- **Rank co-primary:** the entire H1 test repeated with Spearman ρ(pred,obs) as the
  statistic (not rank-invariant under ridge; threshold fractions are calibration-sensitive,
  so a rank-based read is reported alongside MAE).
- **Multiplicity:** H1 on DFI is the single primary at α = 0.05. Secondary targets (H2) use
  Holm–Bonferroni within the secondary family. Everything else is descriptive.
- **Uncertainty:** 95% CIs on MAEs (BCa bootstrap over the per-sample LOO absolute-error
  vector, 10,000 resamples) and on Spearman (BCa over pred–obs pairs).

### 5.4 Power honesty (verbatim in the paper)

With n = 20 and LOO, the detectable association is roughly |r| ≳ 0.55–0.6 at 80% power. A
null does not establish absence of a weaker signal; it bounds it. This is a pilot whose
deliverable is an effect-size estimate with an honest interval. (Context: the existing n=77
CASA-only DFI model reaches ρ≈0.50, so H1 must find something *incremental* to composition.)

---

## 6. Pre-specified sensitivity analyses

Reported in the supplement regardless of outcome; none may be promoted to primary.

1. Elastic net (l1_ratio ∈ {0.1,0.5,0.9}) and GP (RBF+white) in place of ridge.
2. log(DFI) target transform.
3. Adjusted (detection-JSON untracked→immotile) WHO percentages in Set A.
4. Untracked-adjustment removed elsewhere (only affects Set A; stress-tests the baseline).
5. Track-count sensitivity: exclude the two lowest-track videos and re-run.
6. Rank-based everything (Spearman-only), immune to calibration and monotone error.

---

## 7. Structural analysis (H3): does the data recover the WHO trichotomy?

Pooled across all retained tracks, on per-track (VCL, LIN, ALH, BCF, PWR, TAC), z-scored
globally:

1. **GMM** k = 1…10, full covariance, 50 restarts; model selection by BIC. Report the BIC
   curve, not just the winner. **Honesty guard:** if BIC has no interior minimum (k* sits at
   the cap), the data are a *continuum*, not k* discrete modes — H3(a) still holds (the WHO
   3-way split is not the natural geometry) but no specific number of sperm "types" is claimed.
2. **Stability:** 500 bootstrap resamples of tracks; adjusted Rand index (ARI) between the
   full-data labelling and each bootstrap model's labelling of the full data. Median
   ARI < 0.6 → clustering reported as unstable, H3 inconclusive.
3. **Geometry vs WHO:** map each cell to its WHO class via the frozen thresholds; NMI
   between GMM components and WHO classes; visualise component boundaries against the
   VCL=25, VCL=5, STR=0.5 planes.
4. **Batch guard:** χ² of component membership vs video ID; per-video composition plot. A
   single video contributing >50% of any component is flagged and discussed.
5. **Interpretation rule (locked):** H3 supported only if (a) BIC-selected k ≠ 3 with stable
   clustering, or (b) k = 3 but NMI(GMM, WHO) < 0.5. Otherwise report the trichotomy as a
   reasonable compression.

---

## 8. Threats to validity

### 8.1 Circularity / leakage
Kinematics and the WHO baseline come from the same tracks — acceptable, since the comparison
is between *summaries* of identical raw data. Leakage would be tuning any pipeline parameter
against clinical targets (forbidden §3) or feature selection outside LOO folds (forbidden;
feature sets are fixed). The B⊥ residualisation is fit inside training folds only.

### 8.2 Survivorship bias
Track duration correlates with motility class (the "progressive % rises over 30 s" effect is
tracker attrition, not physiology). DUR is excluded from all predictors; quantile features
are per-track (each cell once), not per-frame. A supplement reports per-sample median DUR vs
each clinical target; strong correlation would indicate residual attrition confounding.

### 8.3 Field-of-view sampling error
One 30-s FOV per patient adds sampling noise to sample-level features, inflating all error
estimates equally across Sets A/B/C — the *comparison* stays fair, absolute MAEs understate
achievable performance. Motivates multi-FOV acquisition in the follow-up.

### 8.4 2D projection of 3D motion
All kinematics are projections of helical 3D swimming; constant across samples; a limitation
for mechanistic interpretation of ALH/BCF, not for prediction.

### 8.5 Multiplicity beyond this document
Any analysis not specified here is exploratory and labelled as such. The confirmatory claims
are exactly H1, H1b, H2, H3.

---

## 9. Implementation roadmap

| Step | Script | Output |
|---|---|---|
| 1 | `experiments/features_track.py` | `outputs/prereg/features_track.csv` (no clinical) |
| 2 | `experiments/features_sample.py` | `outputs/prereg/features_sample.csv` (Sets A/B, no clinical) |
| 3 | Freeze + timestamp this document; record commit hash | — |
| 4 | `experiments/cluster_tracks.py` (H3) | `outputs/prereg/cluster_tracks.json` |
| 5 | `experiments/predict_clinical.py` (nested LOO, perm, BCa) | `outputs/prereg/predict_clinical.json` |
| 6 | `experiments/figures_prereg.py` | `outputs/prereg/figures/*.png` |

Clinical columns are merged only inside step 5; steps 1–2 never see them. That ordering is
the blinding mechanism.

---

## 10. Deviations log

| Date | Deviation | Reason |
|---|---|---|
| 2026-08-07 | Set A primary uses raw tracked-only WHO fractions (v1.0 said "adjusted") | Adjusted needs per-video detection JSONs (not guaranteed present); raw is always defined and cleaner. Adjusted retained as sensitivity §6.3. |
| 2026-08-07 | H1 primary changed from "Set B beats Set A" to INCREMENTAL (C vs A) + ORTHOGONALISED (B⊥) | Set B contains motility composition in disguise (§2a); the incremental/orthogonalised claims are the ones that survive review. |
| 2026-08-07 | Ridge α grid widened to logspace(−3,6,19) (was (−3,3,13)); k range 1…10 (was 1…8) | Upper α lets a p=27 model shrink to the mean at n≈18 (else Set C overfits pathologically); wider k range exposes whether BIC has an interior minimum (discrete modes) or keeps improving (continuum). |
| 2026-08-07 | Effective n = 18 (videos 23, 54 excluded: <10 usable tracks after frozen filters) | Pre-specified exclusion rule in section 3. |
---

## 11. Outcome interpretation (decided in advance)

- **H1 supported (Set C beats A, p<0.05, CI excludes 0):** proceed to external cohort;
  claim sub-visible kinematic structure predictive of DNA integrity; begin clinic
  conversations for a prospective sample.
- **Signal present but ns (point estimate favours C, CI crosses 0):** report as effect-size
  estimate; power the follow-up from it; do not claim prediction.
- **Clean null (C ≈ A):** publishable disciplined negative — "at single-FOV resolution,
  kinematic distributions add no DNA-integrity information beyond WHO categories" — folded
  into the error-decomposition / non-Markovian paper. Pipeline work is not wasted; framing
  changes.
- **H3 supported regardless of H1:** the structural finding stands alone — the WHO trichotomy
  is or is not the natural geometry of sperm kinematics. Either answer is a contribution.

No outcome of this experiment produces nothing publishable. That is by design.
