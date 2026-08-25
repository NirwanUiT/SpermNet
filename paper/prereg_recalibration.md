# Pre-registered decision rules: null recalibration and un-retraction standards

**Written 2026-08-25, BEFORE examining any output of `experiments/null_calibration.py`
(first run launched, log unread), before implementing or running the homogeneous
continuum null, and before implementing the beat-matched null (T0.3) or P1 power test
(T0.1).** Commit hash of the manuscript state these rules protect: `38706fa`.

Purpose: the paper's credibility rests on retraction discipline. Recalibrating the
continuum null (referee items B2, plus the extensive-ΔAIC and homogeneous-null points)
could *weaken* the null and thereby re-open retracted claims. To prevent selective
un-retraction, the rules below fix, in advance, what evidence un-retracts what.

## Ground rules

- All comparisons use video-cluster bootstrap 95 % CIs. All outcomes are reported
  regardless of direction. No rule below may be revised after results are seen.
- A raw-velocity spectral peak at 5–20 Hz in GT is **declared non-evidence** for the
  beat mattering to state assignment: a 25-frame boxcar places a spectral null near
  10–13 Hz, so the beat is expected in raw velocity and expected to be attenuated at
  the classifier input. The calibration surfaces that count are (i) the **windowed-VCL
  series** (the actual classifier input) and (ii) the **threshold-crossing interval
  distribution**. T0.3 (beat-matched null) is built only if GT and null differ on
  those surfaces, not on raw velocity alone.

## Claim-by-claim rules

**E. Refractory / serial anti-correlation (retracted §3.4).** No outcome of T0.2,
T0.3, or the homogeneous null can un-retract this claim. Its retraction rests on the
trait-controlled estimator and injection power test applied to the *real* data (P3),
which are independent of continuum-null calibration. Only a demonstrated defect in the
P3 estimator itself could re-open it.

**B. Second-order memory g₂ (retracted §3.2).** Un-retraction to "evidence of
non-Markovian switching" requires ALL of:
1. the best-matched null (per-track parameters, beat/ACF-matched if T0.2 triggers
   T0.3, switch rate within ±10 % of GT) gives g₂_null < g₂_GT with the cluster CI of
   the difference excluding zero;
2. the same direction and exclusion at ≥ 2 of the 3 classifier windows (13/25/51);
3. the residual is NOT attributable to quenched heterogeneity: it must persist in the
   contrast against the per-track (heterogeneity-carrying) null, not only against a
   homogeneous null.
Anything less is reported as "unexplained residual memory, mechanism unresolved" —
not biology.

**A. Dwell-law forms (retracted §3.2/§3.3).** Un-retraction for a motile state
requires ALL of: (i) the best-matched null prefers a different law family than GT in
that state; (ii) per-episode ΔAIC(exp→best) of GT ≥ 2× the null's, at reported episode
counts; (iii) the ≥ 3×-window restricted competition agrees. The immotile law remains
unidentifiable regardless of any null outcome (censoring is a property of the data).

**C. EB decomposition and D. within-cell regularity.** Follow rule B (they are
derivative of the null's mimicry). Additionally for C: if the homogeneous null shows
the EB machinery is deceived only under heterogeneity, the EB indictment is *sharpened*
(the EB null was built to absorb aggregation and failed), not weakened.

## Homogeneous-null mechanism rules (attribution, not un-retraction)

Let g₂_hom be the homogeneous (pooled-parameter) continuum null's block g₂ at 25
frames, and g₂_het = +0.052 the per-track null's.
- g₂_hom < 25 % of g₂_het → layer-1 memory is predominantly **aggregation over
  quenched heterogeneity** (mover–stayer, ref 27); manuscript mechanism sentences and
  the layer-1 attribution are rewritten accordingly; the discretisation contribution
  at W/τ ≈ 7.6 is reported as minor.
- g₂_hom ≥ 50 % of g₂_het → **discretisation-intrinsic memory** confirmed in a regime
  where the timescale argument (τ ≈ 3.3 frames, blocks 25 frames ≈ 7.6 τ apart)
  predicts none; this requires a new mechanistic account before publication.
- Intermediate → both mechanisms reported with shares; U1 phase diagram gains a trait-
  dispersion axis.

## Extensive-statistic rule (applies before any interpretation)

ΔAIC is extensive in episode count. The §3.2 table is re-expressed as ΔAIC per episode
with episode counts for both cohorts. If per-episode normalisation reverses any
GT-vs-null overshoot, that reversal is reported as the primary resolution of B2 and
the beat investigation proceeds only for the calibration surfaces above.

## T0.1 design constraints (fixed now)

The injected semi-Markovian process must match GT on exactly what the OU null matches
(velocity marginal and lag-1 autocovariance), so that switching is the only difference.
The detection floor is reported in practitioner units: mode-dwell CV and mode-speed
separation as a fraction of the threshold gap, in addition to any internal parameter.
