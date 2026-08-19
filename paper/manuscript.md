# The measurement pipeline manufactures non-Markovian dynamics: discretisation and tracking artefacts in single-cell behaviour, and the one refractory signature in human sperm that survives

*Working manuscript draft — Work Package 4 (Event Detection and Tracking) of the doctoral
project "Shape representation and event detection of sub-cellular structures in
high-content/high-throughput microscopy" (UiT, BioAI group). Sperm motility is the primary
use case for the event-detection methodology; the same detect–track–represent–detect-events
pipeline targets sub-cellular structures such as mitochondria. Primary evidence: 1,138
hand-annotated single-cell trajectories (VISEM-Tracking ground truth, 20 videos), a
matched Markovian-continuum null passed through the identical classifier, and a
four-pipeline tracking-fidelity audit on the annotated videos.*

---

## Abstract

A standard way to quantify the behaviour of tracked biological objects — from swimming
cells to sub-cellular organelles — is to discretise a continuous phenotype into a small
state vocabulary and model the transitions. Reports of "non-Markovian" dynamics from such
pipelines are widespread: heavy-tailed state dwell times, memory beyond the current state,
and stable object-to-object heterogeneity. Using the largest hand-annotated single-cell
motility resource available (1,138 human-verified spermatozoon trajectories, 656,145
frame-states, VISEM-Tracking) we show that, for the canonical CASA motility states, this
entire phenomenology is manufactured by the measurement pipeline itself — with one sharply
defined exception. We construct a **memoryless (Markovian) continuum null**: per-track
Ornstein–Uhlenbeck velocity processes fit only to each trajectory's velocity marginal and
lag-1 autocovariance, passed through the identical windowed state classifier and scoring
code. This null — which contains no switching biology at all — reproduces or exceeds every
headline non-Markovian statistic: log-normal dwell laws in every state (ΔAIC over
exponential up to +4,490), a second-order memory gain of +0.052 per token (170 % of the
observed +0.030), window-robustness of that gain, apparent cell-to-cell heterogeneity
(ICC 0.13 vs 0.10 observed), sub-exponential within-cell dwell regularity, and it even
deceives a hierarchical empirical-Bayes decomposition into attributing 54 % "genuine
within-cell memory" to a memoryless process. Resolution and censoring audits further show
the motile dwell laws have barely one decade of dynamic range above the classifier window,
and the immotile dwell law is unidentifiable (64 % of immotile episodes touch a track
boundary; 41 % are whole tracks). **One statistic survives every control**: successive
dwell durations of the same cell anti-correlate far beyond the continuum null
(Δρ = −0.23, video-cluster bootstrap CI [−0.30, −0.18]; null −0.06 [−0.14, +0.01];
difference CI excludes zero), is robust to leave-one-video-out, to a per-cell estimator,
and to merging up to 38 % of episodes as suspected classifier flicker (which *strengthens*
it). Single human sperm are refractory switchers — an anti-bursty, resource-recovery-like
timing signature that is genuine biology, and that automated tracking (which inflates all
artefactual statistics a further ~1.9-fold) *erases* rather than inflates. Our results
re-open the interpretation of non-Markovian claims from track-then-discretise pipelines
across fields, and supply the two controls — a matched continuum null and
annotation-anchored tracking audits — that separate measurement artefact from biology.

---

## 1. Introduction

A recurring problem in high-content and high-throughput microscopy is to turn long image
sequences of many moving biological objects into a compact description of their *state* and
of the *events* — the changes of state — that they undergo over time. This couples two
sub-problems: a *representation* problem (choosing a small vocabulary of canonical states
for objects whose morphology or behaviour is in reality continuous and pleomorphic) and an
*event-detection* problem (modelling how a tracked object transitions between those states
along its trajectory). We study both in the setting that yields the richest single-object
statistics available to us — the motility of individual human spermatozoa imaged by video
microscopy — and use it to expose a general failure mode of the standard approach: a coarse
discrete state vocabulary combined with a memoryless transition model systematically
mis-describes the events. Sperm motility is thus the primary use case for a methodology
whose target domain is event detection in tracked biological structures.

Semen motility is clinically scored by CASA as the proportion of spermatozoa that are
progressively motile, non-progressively motile, or immotile at one instant. This reduces
a population of independently swimming cells, each following its own time-varying
trajectory, to three numbers — a discretisation of a continuous kinematic phenotype into
three canonical states, exactly the kind of state vocabulary the representation problem
must supply. Two modelling assumptions are built into this reduction and
into the Markov-chain models commonly fitted to CASA state sequences: (1) *memorylessness*
— a cell's next state depends only on its current state; and (2) *homogeneity* — all cells
are draws from one transition process. Neither has been tested at scale on single-cell
trajectories.

We test both directly — and then we test the tests. Hand-annotated tracking yields
thousands of verified state-switching events, enough to characterise the *distribution*
and *temporal structure* of motility switching rather than only its mean. On these
trajectories the standard analyses report exactly what the recent literature on tracked
behaviour would predict: heavy-tailed dwell laws, second-order memory, and stable
cell-to-cell heterogeneity. The central contribution of this paper is to show that these
findings do not survive two controls that track-then-discretise studies do not usually
run. The first is a *continuum null*: a memoryless (Markovian) continuous motion model,
fit per track to nothing but the velocity marginal and its lag-1 autocovariance, passed
through the identical windowed classifier — which turns out to manufacture the entire
non-Markovian phenomenology, and more of it than the data show. The second is an
*annotation-anchored tracking audit*, which shows automated tracking inflates the same
statistics a further two-fold while erasing genuine within-cell temporal structure. One
statistic survives both controls and every robustness test we could construct — a
refractory anti-correlation of successive dwells — and we propose it as the correct
minimal claim about single-sperm switching biology. Finally we ask whether anything
discarded by the population average carries clinical signal.

---

## 2. Data and methods

**Data tiers.** All headline dynamical statistics are computed on *ground-truth* (GT)
trajectories: the 20 VISEM-Tracking videos (50 fps) whose per-frame bounding boxes carry
human-verified persistent identities (`labels_ftid`), materialised into 1,138 single-cell
tracks with 656,145 frame-states (`experiments/gt_reanchor.py`). Two further tiers are
analysed *identically* but serve distinct, subordinate roles and are never pooled with GT:
(a) the *same 20 videos* re-tracked by automated pipelines — our fine-tuned detector with
BoT-SORT+ReID (the baseline), BoT-SORT without ReID, and ByteTrack — used exclusively for
the tracking-fidelity audit (§3.5); (b) *extra57* = 57 further VISEM participants with no
ground truth, tracked with the baseline pipeline (fine-tuned YOLOv8-l + BoT-SORT), used
only for pipeline-level replication and for per-participant clinical traits. Clinical
variables (including DFI) come from the VISEM clinical tables (n = 85); 77 participants
have both tracks and clinical data. Cross-participant statistics are computed within
cohort, never pooled (tracking pipeline is a batch effect).

**States.** For each track we assign a per-frame motility state
{Progressive, Non-progressive, Immotile} from kinematic parameters (VCL, STR, …) computed
over a 0.5 s (25-frame) sliding window, following standard CASA thresholds. A *dwell* is a
maximal run of one state along a track; a *switch* is a change of state.

**Dwell-time law (Fig. 1).** Pooling dwell episodes per state and cohort, we fit
exponential, gamma, Weibull and log-normal laws by maximum likelihood (location fixed at
0) and compare by AIC. Robustness is assessed by recomputing states at window sizes 13, 25
and 51 frames (`experiments/dwell_physics.py`, `dwell_physics_robust.py`).

**Order test (Fig. 2A).** To quantify memory we compare held-out per-token log-likelihood
of Markov models of order 0–3 by 5-fold cross-validation over tracks. Because the
sliding-window state series is ≈ 88 % autocorrelated between adjacent frames, the order
test is run on **decorrelated, non-overlapping 0.5 s blocks** (one state per block); the
2nd-over-1st gain g₂ = ℓ₂ − ℓ₁ is the memory statistic (`markov_property_test.py`,
`replicate_markov_extra.py`).

**Decomposition of the memory (Fig. 2).** We separate within-cell memory from population
heterogeneity with three parametric-bootstrap nulls that all preserve track lengths and
re-run the identical order test (`mover_stayer_null.py`, `mover_stayer_eb.py`):
- **HOM** — every track simulated from a single pooled first-order matrix (no heterogeneity);
- **HET (over-fit)** — every track simulated from its *own* fitted first-order matrix
  (maximal, noise-inflated heterogeneity; an upper bound on the aggregation artefact);
- **EB (fair)** — a hierarchical Dirichlet–multinomial: per context, per-cell transition
  rows are drawn from Dir(kₐ·mₐ) with concentration kₐ fit by maximum marginal likelihood,
  reproducing the *estimated true* between-cell heterogeneity without over-fitting.
The genuine-memory share is (g₂ᴿᴱᴬᴸ − g₂ᴱᴮ)/(g₂ᴿᴱᴬᴸ − g₂ᴴᴼᴹ); the aggregation share is its
complement. We corroborate with the ratio of within-track to pooled dwell CV
(`memory_decomposition.py`).

**Refractoriness test.** Within each cell we compute the lag-1 Spearman correlation of
successive state-controlled residual log-dwell durations and compare it with a
within-cell permutation null that preserves each cell's dwell multiset (so censoring and
heterogeneity cannot produce a spurious signal); the statistic is Δρ = ρ_obs − ρ_null.
All inference is by **video-level cluster bootstrap** (videos resampled with replacement),
with leave-one-video-out ranges and a per-cell Fisher-z estimator as corroboration
(`memory_decomposition.py`, `refractory_survivor.py`).

**Continuum null (the decisive control).** For every ground-truth track we fit a 2-D
Ornstein–Uhlenbeck (discrete AR(1)) velocity process — Markovian by construction — to
that track's velocity marginal (per-component mean, variance) and pooled lag-1
autocovariance, and nothing else; no memory statistic is ever fit. We simulate one
synthetic track per real track at the identical length (censoring reproduced by
construction; per-track parameters preserve quenched heterogeneity), then pass the
synthetic positions through the *identical* windowed classifier and the identical scoring
code for every statistic in this paper (`continuum_null.py`).

**Resolution and censoring audits.** Per state we report the fraction of dwell episodes
shorter than 1×/2×/3× the classifier window, re-fit the law competition restricted to
dwells ≥ 3× window, and repeat the full competition at windows of 5, 9 and 13 frames
(below the mean motile dwell). We report the fraction of episodes touching one or both
track boundaries, re-fit all laws by censored maximum likelihood (interior episodes
contribute density, boundary episodes survival), and redo the ground-truth-vs-automated
dwell contrast with ground-truth tracks truncated to the automated track-length
distribution (`dwell_resolution_audit.py`).

**Tracking-fidelity audit.** The 20 annotated videos are re-tracked by three automated
pipelines (BoT-SORT+ReID, BoT-SORT, ByteTrack over the same fine-tuned detector). For each
pipeline we compute (a) conventional association quality against GT (identity switches per
GT track, coverage) and (b) every downstream dynamical statistic above, scored by the
identical code. Fidelity is the agreement of (b) with GT (`experiments/tracker_fidelity.py`).

**Per-cell heterogeneity and clinical test (Fig. 3).** For each cell we compute a
length-normalised switch rate (switches per second) on tracks ≥ 1 s; per participant we
summarise the *distribution* of that rate across cells (mean, CV, Gini, and the fraction of
non-switching "stayers"). Reliability is the split-half Spearman correlation across
participants. The pre-registered clinical primary is the partial Spearman correlation of
switch-rate CV with DFI, controlling for mean switch rate and track count, computed per
cohort. Robustness of the secondary stayer-fraction signal is tested by additionally
controlling for static CASA composition and for age/BMI/abstinence
(`per_cell_kinetics.py`, `stayer_dfi.py`).

---

## 3. Results

### 3.1 The standard toolkit reports the full non-Markovian phenomenology

![Figure 1](figures/fig1_dwell_law.png)

**Figure 1. What a standard analysis concludes.** Empirical survival functions of state
dwell times on the 1,138 hand-annotated trajectories depart sharply from the exponential
(memoryless/Markov) law; a second-order Markov model beats a first-order one on held-out
data; a hierarchical decomposition attributes most of the gain to within-cell memory.
Sections 3.2–3.3 show that all of this is reproduced by a memoryless continuum passed
through the identical classifier. *(Figure to be regenerated with the continuum-null
overlay.)*

Applying the field's standard analyses to the ground-truth trajectories reproduces every
non-Markovian signature the tracked-behaviour literature would predict. Dwell times reject
the exponential law in every state (ΔAIC over exponential +1,195 progressive, n = 2,940
episodes; +1,343 non-progressive, n = 3,048; +176 immotile, n = 907), with log-normal the
best of four laws for the motile states and pooled dwell CV of 1.86–2.16 against 1 for a
memoryless process. On decorrelated 0.5 s blocks, a second-order Markov model improves
held-out per-token log-likelihood by **g₂ = +0.0304**, robustly across classifier windows
(+0.0202/+0.0304/+0.0416 at 13/25/51 frames); the durationless embedded chain shows a
second-order gain of +0.096; an explicit hidden-Markov model with up to four latent modes
is beaten by the raw second-order model (prior result, `hmm_vs_markov.json`). A
hierarchical empirical-Bayes decomposition attributes 71 % of g₂ to genuine within-cell
memory, with a between-cell ICC of 0.10; and within single cells, dwell durations appear
sub-exponentially regular (within-cell CV ≈ 0.77). By the standards of the discrete-state
literature this would constitute strong evidence for non-Markovian, history-dependent
switching with quenched heterogeneity. The automated baseline pipeline on the same videos
reports the same phenomenology with everything larger (g₂ = +0.0589; CV 3.2/3.2/2.4), and
an independent 57-participant automated cohort replicates it (g₂ = +0.0565).

The remainder of the paper subjects this phenomenology to two controls — a memoryless
continuum passed through the identical classifier (§3.2) and resolution/censoring audits
(§3.3) — and reports the single statistic that survives (§3.4).

### 3.2 A memoryless continuum manufactures all of it

The states are not observed; they are computed from continuous kinematics by a 0.5 s
sliding-window classifier. The correct null for "the switching is non-Markovian" is
therefore not a shuffled state sequence but a **Markovian continuous motion model passed
through the identical classifier**. For each ground-truth track we fit a 2-D
Ornstein–Uhlenbeck velocity process to that track's velocity marginal and lag-1
autocovariance only (median fitted velocity persistence a = 0.74; nothing dynamical beyond
lag 1 is fit, and no memory statistic is ever seen by the fit), simulated one synthetic
track per real track at the identical length, and scored the synthetic cohort with the
identical code (`continuum_null.py`).

The memoryless continuum reproduces — or exceeds — every statistic of §3.1:

| Statistic | Ground truth | Continuum null (memoryless) |
|---|---|---|
| Dwell law, best of 4 (all states) | log-normal (motile) | **log-normal, every state** |
| ΔAIC exp vs best (P / NP / I) | +1,195 / +1,343 / +176 | +959 / +4,490 / +1,237 |
| Pooled dwell CV (P / NP / I) | 1.86 / 2.16 / 1.05 | 1.59 / 3.49 / 1.39 |
| Block g₂ (13 / 25 / 51 frames) | +0.020 / +0.030 / +0.042 | **+0.050 / +0.052 / +0.053** |
| Frame-level g₂ | +0.0002 | +0.0019 |
| Between-cell ICC (log-dwell) | 0.103 | 0.126 |
| Within-cell dwell CV (median) | 0.77 | 0.80–0.93 |
| EB "genuine within-cell memory" share | 71 % | **54 %** |
| Serial Δρ (obs − permutation null) | **−0.23** | **−0.06** |

Three consequences. First, the **dwell-law and memory claims collapse as biology**: a
process with no switching dynamics at all — indeed no states at all — produces log-normal
dwell laws with large ΔAIC in every state, and *more* second-order block memory than the
real data (+0.052 vs +0.030, i.e. 170 %), with the same window-robustness. The windowed
discretisation of a continuous, memoryless trajectory is itself a memory-manufacturing
device: window overlap with state boundaries, threshold crossings of a smooth variable,
and mixture-of-kinematics within windows generate precisely the history-dependence the
order test detects. Second, the **decomposition machinery is equally deceived**: the
hierarchical EB null attributes 54 % "genuine within-cell memory" to the memoryless
continuum, so the 71 % split of §3.1 cannot be read as biology either; likewise the
apparent sub-exponential within-cell regularity (CV < 1) is largely reproduced (0.80–0.93)
by episode truncation alone. Third — and this is the pivot of the paper — exactly **one
statistic resists**: the serial anti-correlation of successive dwells, where the null
yields only a quarter of the observed effect. Section 3.4 stress-tests that survivor.

This result also retro-explains our own generative analyses. A zero-free-parameter
semi-Markov ladder (`refractory_model.py`) had shown that no timing ingredient
(heterogeneity, gamma shape, refractory coupling) produces block-scale g₂, while imposing
the empirical second-order embedded topology recovers 93 % of it; we had read that as
"sequence momentum". The continuum null shows the empirical second-order embedded topology
is itself largely a discretisation artefact — the ladder's durable lesson is about the
*statistic* (g₂ is blind to timing structure and sensitive to sequence context), not about
sperm.

### 3.3 Resolution and censoring audits: the dwell laws had little room to be laws

Two audits quantify how much dynamic range the dwell-law competition ever had
(`dwell_resolution_audit.py`).

**Resolution.** Mean motile dwells (0.85 s progressive, 0.54 s non-progressive) sit at
1–2× the 0.5 s classifier window: 62 % and 74 % of motile episodes are shorter than one
window, 86 % and 93 % shorter than three. Restricting the law competition to dwells
≥ 3× window leaves the log-normal ranking intact (ΔAIC over exponential +287 and +145 on
n = 422 and 209 episodes), and the full competition repeated at windows of 5, 9 and 13
frames — below the mean dwell — preserves it as well (e.g. +759/+1,022 at 5 frames). The
*form* is therefore not a smoothing artefact of one window choice; but with less than one
decade of usable range above any window, and with §3.2 showing a Markovian continuum
produces the same ranking at comparable ΔAIC, the law's rejection of the exponential
carries no biological information.

**Immotile censoring.** The immotile state is essentially unmeasurable at these track
lengths: 64 % of immotile episodes touch a track boundary and 41 % are whole tracks
(against 26 % and 11 % boundary-touching for the motile states), and the immotile episode
distribution tracks the track-length distribution over most of its range (medians 270 vs
445 frames; upper quartiles coincide). Censored maximum likelihood — interior episodes
contributing density, boundary episodes survival — *reverses* the complete-case verdict:
the censored best law for immotile is log-normal (ΔAIC +516 over exponential), not the
gamma found when truncated episodes are treated as complete. And the apparent
ground-truth-vs-automated contrast in the immotile law disappears once track length is
controlled: truncating ground-truth tracks to the automated track-length distribution
turns the immotile law log-normal with CV 2.06 — indistinguishable in form from the
automated pipelines. We therefore retract two claims from earlier drafts of this work: the
immotile dwell law is *unidentifiable* from these data (neither "gamma and light-tailed"
on ground truth nor "heavy tail manufactured by tracking" survives the length-matched
contrast), and no state's dwell-law *form* should be read as biology.

### 3.4 The survivor: single sperm are refractory switchers

![Figure 2](figures/fig2_decomposition.png)

**Figure 2. The one statistic the null cannot make.** Serial correlation of successive
state-controlled residual log-dwells within single cells: ground truth Δρ = −0.228
(video-cluster 95 % CI [−0.296, −0.182]) versus −0.060 [−0.142, +0.008] for the matched
memoryless continuum; the difference excludes zero, and aggressive flicker-merging
strengthens the effect. *(Figure to be regenerated: current panel shows the superseded
EB decomposition.)*

One statistic resists the continuum null, and it then survives every additional control we
could construct (`refractory_survivor.py`).

Successive dwell durations of the same cell — state-controlled residual log-dwells,
compared against a within-cell permutation null that preserves each cell's dwell multiset
— are anti-correlated: **Δρ = −0.228** on ground truth. Because dwell pairs are nested in
cells nested in 20 videos, we retire the naive pooled p-value and use video-level cluster
inference: the cluster-bootstrap 95 % CI is **[−0.296, −0.182]**, the leave-one-video-out
range is [−0.241, −0.214] (no single video carries the effect), and a per-cell estimator
(lag-1 Spearman per cell, Fisher-z averaged, no pooling) gives −0.26, agreeing with the
pooled value. The continuum null produces Δρ = −0.060 [−0.142, +0.008] — a small negative
bias from truncation and windowing — and the bootstrapped **GT-minus-null difference is
−0.168 [−0.263, −0.074]**, excluding zero.

The remaining artefactual explanation is classifier *flicker*: a brief misclassification
inside a long dwell creates a long–short–long triplet, which is negatively autocorrelated
by construction. Two facts rule it out. First, merging every 1–2-block episode flanked by
the same state on both sides — removing 34 % (≤ 25-frame threshold) or 38 % (≤ 50-frame)
of all episodes as potential flicker — makes the anti-correlation *stronger*, not weaker
(Δρ = −0.232 and −0.248). Second, flicker inflates the within-cell dwell CV, and the
observed within-cell CV is 0.77 — below the exponential value and at the bottom of the
continuum null's range — so a flicker-dominated record is inconsistent with the observed
regularity; the two statistics jointly over-constrain the artefact.

The surviving biological claim is deliberately minimal: *individual human sperm switch
motility states with a refractory, anti-bursty timing structure — a long dwell is followed
by a short one and vice versa, well beyond anything a memoryless continuum, censoring,
heterogeneity, or classifier flicker produces.* The natural mechanistic reading is a
resource-recovery or adaptation process in the switching machinery (e.g. Ca²⁺ or ATP
dynamics in the flagellar beat regulation), analogous to spike-frequency adaptation in
neurons — and notably *inverted* relative to bacterial run-and-tumble switching, where
behavioural variability manifests as positive serial dependence and burstiness. This
signature is invisible in automated tracking of the same videos (Δρ ≈ −0.03): identity
splices and fragmentation destroy within-cell serial structure — automated tracking
inflates every artefactual statistic while erasing the genuine one.

### 3.5 Tracking artefacts masquerade as dynamics — and MOT accuracy does not predict dynamical fidelity

The results above required hand-annotated trajectories; here we quantify what automated
tracking would have reported instead. We re-tracked the 20 annotated videos with three
pipelines sharing the same fine-tuned detector (BoT-SORT+ReID, BoT-SORT, ByteTrack) and
scored every dynamical statistic with the identical code (`tracker_fidelity.py`).

Every pipeline inflates the memory statistic, and by a similar factor: g₂ = 0.047–0.052
against 0.025 on ground truth with the same estimator — a **1.8–2.1× inflation** — while
inflating dwell CV in all states and shortening the mean immotile dwell twenty-fold (0.4 s
vs 9.9 s; a fragmentation effect, though §3.3 shows the immotile *law form* is not
identifiable in either data tier). The mechanism is identity error: fragmentation and ID
switches concatenate unrelated cells and truncate dwells, which *adds* apparent
history-dependence and heterogeneity on top of the discretisation artefact of §3.2 — the
two artefact layers compound. Meanwhile the one genuine signature is *destroyed*: the
refractory Δρ, −0.23 on ground truth, is ≈ −0.03 under automated tracking (§3.4).
A track-then-model analysis therefore overstates the artefactual dynamics roughly
two-fold while erasing the real one.

More surprising is the *dissociation* between conventional tracking quality and downstream
fidelity. ByteTrack has the best association accuracy of the three (0.47 identity switches
per GT track) yet the worst dynamical fidelity; across the three pipelines, dynamical error
correlates strongly with track coverage (r = +0.95) and *negatively* with identity switches
(r = −0.88). With n = 3 pipelines this is hypothesis-generating, not confirmatory, but the
direction is mechanistically sensible: a tracker that aggressively maintains coverage keeps
low-confidence, identity-impure segments whose splices manufacture dynamics, whereas a
conservative tracker that drops uncertain segments makes more nominal identity errors while
preserving the temporal statistics of what it keeps. The practical recommendation is
independent of the small n: **when tracks feed dynamical modelling, pipeline selection
should include downstream dynamical observables (dwell laws, memory statistics) scored
against annotation, not only MOT identity metrics**, because the two rank pipelines
differently.

### 3.6 The heterogeneity is a reliable per-man trait but not clinically incremental over CASA

![Figure 3](figures/fig3_heterogeneity.png)

**Figure 3. The mover–stayer axis is real but does not beat CASA for DFI.** (A) Per-man
heterogeneity features (CV and Gini of the single-cell switch rate, stayer fraction) are
highly reliable by split-half correlation, up to ρ ≈ 0.97 in the larger cohort. (B) The
apparent association between a man's stayer fraction and his DNA-fragmentation index
collapses to null once static CASA composition is controlled, in both cohorts. (C) Reason:
"stayers" are a mix of stable-progressive cells (associated with *lower* DFI) and
stable-immotile cells (associated with *higher* DFI); their net correlation merely
re-encodes the progressive/immotile percentages CASA already reports.

The per-participant heterogeneity summaries are highly reliable traits (split-half Spearman
0.95–0.97 in extra57; 0.56–0.79 in the smaller 20-video cohort, both under automated
tracking). Clinically, the pre-registered
primary — switch-rate CV versus DFI, controlling for mean rate and track count — is null in
both cohorts (ρ = −0.36, p = 0.18; ρ = −0.12, p = 0.38). A secondary stayer-fraction signal
appeared cross-cohort-consistent (ρ = −0.55 and −0.45) but vanished once static composition
was controlled (ρ = −0.21, p = 0.44; ρ = −0.03, p = 0.82), and further with age/BMI/
abstinence. State-resolved, stable-progressive fraction tracks lower DFI (ρ ≈ −0.44) and
stable-immotile fraction tracks higher DFI (ρ ≈ +0.42), i.e. the signal is the known
motility-composition–DFI relationship in disguise. Consistently, a pre-registered
prediction test found that memory/dynamics features add nothing to a CASA-composition model
for out-of-sample DFI (Ridge Spearman: CASA 0.501, memory 0.427, CASA+memory 0.499;
Δ = −0.002, permutation p = 0.567). Scope note: per-man traits are necessarily computed on
automated tracking (only 20 videos are annotated), so they are *pipeline-level* traits;
given §3.5, absolute dynamics values are inflated, but the reliability and null-association
analyses compare participants under one fixed pipeline and are unaffected in design — and a
null obtained on artefact-inflated features would, if anything, be more null on clean ones.

---

## 4. Discussion

We set out to characterise non-Markovian structure in sperm motility switching and ended
up with a different, more consequential result: **the standard track-then-discretise
pipeline manufactures nearly all of it**, in two compounding layers. The first layer is
discretisation. A windowed threshold classifier applied to a memoryless continuous motion
process generates heavy-tailed (log-normal) dwell laws in every state, a window-robust
second-order memory *larger* than the one observed, apparent cell-to-cell heterogeneity,
sub-exponential within-cell regularity — and it deceives the hierarchical null machinery
built to decompose such effects, which attributes half of the manufactured memory to
"genuine within-cell memory". The second layer is tracking: automated pipelines inflate
the same statistics a further ~1.9-fold through identity error, and conventional MOT
metrics do not rank pipelines by this downstream damage. Both layers were only visible
because hand-annotated trajectories and a matched continuum null existed; neither is
specific to sperm. Any study that discretises tracked continuous behaviour into states and
reports non-exponential dwells, memory, or individuality — a large literature spanning
animal-movement HMMs, single-particle tracking, and cell-behaviour phenotyping — is
exposed to the same two artefact layers unless it runs the corresponding controls, which
are cheap: simulate a Markovian continuum matched to each trajectory's marginal and lag-1
structure, push it through the identical classifier, and compare.

Against that null, one biological result stands, and it is sharper for having survived:
**single human sperm switch motility states with a refractory, anti-bursty timing
structure** (Δρ = −0.23, video-cluster CI [−0.30, −0.18]; continuum null −0.06;
difference excludes zero; robust to leave-one-video-out, per-cell estimation, and
aggressive flicker merging, which strengthens it). Successive dwells anti-correlate: the
cell behaves as if switching consumes a resource that must recover — an adaptation-like
mechanism analogous to spike-frequency adaptation in neurons, plausibly rooted in Ca²⁺ or
ATP dynamics of flagellar beat regulation. The comparison with bacterial run-and-tumble
switching is instructive and inverted: bacterial behavioural variability manifests as
positive serial dependence and burstiness driven by slow signalling-network fluctuations,
whereas sperm show *negative* serial dependence — a refractory clock rather than a
wandering rate. Testing whether this timescale shifts under progesterone/CatSper
modulation, viscosity, or temperature is the natural perturbation experiment, and a
first-passage model with a recovery variable is the natural quantitative target.

The clinical question we treat conservatively: per-man switching-heterogeneity traits are
highly reliable (split-half ρ ≈ 0.95) but do not improve prediction of DFI over standard
composition — the one apparently promising association proved to be composition in
disguise, and we report the pre-registered null as a null. DFI is in any case a surrogate;
the decisive test — whether switching dynamics predict fertilisation or live birth —
requires outcome-linked cohorts we do not have.

**Relation to event detection in sub-cellular structures.** The pipeline used here — detect
objects, track them, assign each a per-frame state from a small canonical vocabulary, then
model the sequence of state-changes — is the same pipeline required to detect morphological
events in sub-cellular organelles such as mitochondria, where a pleomorphic, continuously
deforming shape must likewise be reduced to a few canonical states whose transitions
(fission, fusion, elongation) are the events of interest. The two lessons drawn here transfer
directly to that domain, the central object of this doctoral project. First, the canonical
state vocabulary is a *discretisation of a continuum*: in a pre-registered single-cell
analysis the kinematic phenotype showed no discrete cluster structure (the model-selection
criterion improved monotonically to the search cap), so the three motility categories are a
coarse slice of a continuous manifold rather than its natural geometry — a caution for any
scheme that assigns organelle morphology to a fixed number of shape classes. Second — the
lesson this paper sharpens — *the discretisation itself manufactures event dynamics*:
thresholding a smooth morphological variable into shape classes will generate heavy-tailed
class dwell times and apparent transition memory even if the underlying morphodynamics are
Markovian, so any non-Markovian claim about organelle state transitions requires a
matched continuum null. Third, *tracking error compounds it*: identity splices manufacture
additional memory while erasing genuine within-object temporal structure — and for
organelles, where fission/fusion events *are* identity events, event-detection pipelines
must be validated on downstream dynamical observables against annotation, not on
detection/association metrics alone.

**A transparency note on the evolution of this work.** Earlier drafts, built on automated
tracking, claimed log-normal dwell laws as biology, a genuine-memory/heterogeneity
decomposition, quenched heterogeneity with no within-cell serial structure, and (after
re-anchoring on hand annotations) a light-tailed immotile law contrasting with a
tracking-manufactured heavy tail. Each of those claims fell to a control introduced later
— the annotation re-anchoring, the continuum null, the censoring audit — and is retracted
here, with the failing control reported in full. The refractory signature is the only
dynamical claim that has survived every control applied to it; all dynamical analyses in
this paper are exploratory (only the DFI test was pre-registered), and we label them so.

**Limitations.** (1) Ground truth is 20 videos / 1,138 cells from one dataset, one lab and
one imaging condition; the surviving effect is estimated with video-cluster CIs but its
generality needs a second annotated dataset. (2) The continuum null is Gaussian and
lag-1-matched; a continuum with heavier-tailed or longer-memory — but still Markovian —
velocity structure could in principle produce more of the observed Δρ than our null does,
though it would have to do so while remaining consistent with the flicker and CV
constraints. (3) The immotile state is unmeasurable at these track lengths (64 % boundary
episodes), so nothing about immotile dwell structure is claimed. (4) The tracker-fidelity
dissociation (MOT metrics vs dynamical fidelity) rests on three pipelines and is
hypothesis-generating. (5) Annotation is human and its own identity-error rate is not
zero; inter-annotator agreement on VISEM-Tracking is not documented, and a synthetic-video
calibration of annotation and tracking error is the right next control.

**Conclusion.** On annotation-grade single-cell trajectories with a matched continuum
null, the celebrated non-Markovian phenomenology of discretised motility states —
heavy-tailed dwell laws, second-order memory, decomposable heterogeneity — is manufactured
by the measurement pipeline, in two compounding layers (discretisation, then tracking).
What survives is minimal and specific: a refractory, anti-bursty switching clock in
individual human sperm. The constructive message is a pair of cheap, general controls —
simulate a Markovian continuum through your own classifier; audit your tracker on
downstream dynamical observables — without which non-Markovian claims from
track-then-discretise pipelines, in any domain, are uninterpretable.

---

## Reproducibility

| Result | Script | Output |
|---|---|---|
| **Continuum null (decisive control)** | `experiments/continuum_null.py` | `outputs/tracks_continuum_null/`, `outputs/markov/continuum_null.json` |
| **Resolution + censoring audits** | `experiments/dwell_resolution_audit.py` | `outputs/markov/dwell_resolution_audit.json` |
| **Refractory survivor test (flicker + cluster bootstrap)** | `experiments/refractory_survivor.py` | `outputs/markov/refractory_survivor.json` |
| Ground-truth track materialisation + standard-toolkit statistics | `experiments/gt_reanchor.py` | `outputs/tracks_gt/`, `outputs/markov/gt_reanchor.json` |
| Tracking-fidelity audit (GT vs 3 pipelines) | `experiments/tracker_fidelity.py` | `outputs/markov/tracker_fidelity.json` |
| Semi-Markov ladder (property of the g₂ statistic) | `experiments/refractory_model.py` | `outputs/markov/refractory_model.json` |
| Slow-latent generative mechanism on GT | `experiments/generative_model.py --fit-dir outputs/tracks_gt --fit-name gt` | `outputs/markov/generative_model_gt.json` |
| Dwell-time law + window invariance (automated cohorts) | `experiments/dwell_physics.py`, `dwell_physics_robust.py` | `outputs/markov/dwell_physics*.json` |
| Pipeline-level replication (extra57) | `experiments/replicate_markov_extra.py` | `outputs/markov/replication_extra.json` |
| Within- vs between-cell decomposition | `experiments/memory_decomposition.py` | `outputs/markov/memory_decomposition.json` |
| Censoring-aware dwell law + frailty (automated) | `experiments/dwell_censoring.py` | `outputs/markov/dwell_censoring.json` |
| Mover–stayer nulls (HOM/HET/EB) | `experiments/mover_stayer_null.py`, `mover_stayer_eb.py` | `outputs/markov/mover_stayer_*.json` |
| Per-cell heterogeneity + reliability | `experiments/per_cell_kinetics.py` | `outputs/markov/per_cell_kinetics.{json,csv}` |
| Stayer→DFI robustness | `experiments/stayer_dfi.py` | `outputs/markov/stayer_dfi.json` |
| DFI prediction (pre-registered) | `experiments/dfi_predict.py` | `outputs/markov/dfi_prediction.json` |
| Figures | `experiments/make_paper_figures.py` | `paper/figures/fig{1,2,3}_*.png` |

---

## References (core; to be completed)

1. Korobkova E, Emonet T, Vilar JMG, Shimizu TS, Cluzel P. From molecular noise to
   behavioural variability in a single bacterium. *Nature* 428, 574–578 (2004).
2. Auger-Méthé M, Field C, Albertsen CM, Derocher AE, Lewis MA, Jonsen ID, Mills
   Flemming J. State-space models' dirty little secrets: even simple linear Gaussian
   models can have estimation problems. *Scientific Reports* 6, 26677 (2016).
3. Pohle J, Langrock R, van Beest FM, Schmidt NM. Selecting the number of states in
   hidden Markov models: pragmatic solutions illustrated using animal movement.
   *JABES* 22, 270–293 (2017).
4. Metzler R, Jeon J-H, Cherstvy AG, Barkai E. Anomalous diffusion models and their
   properties: non-stationarity, non-ergodicity, and ageing at the centenary of single
   particle tracking. *Phys. Chem. Chem. Phys.* 16, 24128–24164 (2014).
5. Ulman V, et al. An objective comparison of cell-tracking algorithms. *Nature
   Methods* 14, 1141–1152 (2017).
6. Maška M, et al. The Cell Tracking Challenge: 10 years of objective benchmarking.
   *Nature Methods* 20, 1010–1020 (2023).
7. World Health Organization. *WHO laboratory manual for the examination and processing
   of human semen*, 6th edn (WHO, 2021).
8. Haugen TB, et al. VISEM-Tracking, a human spermatozoa tracking dataset.
   *Scientific Data* 10, 260 (2023).
