# Single human spermatozoa switch motility states with memory: hand-annotated trajectories separate a refractory, history-dependent switching process from tracking artefact

*Working manuscript draft — Work Package 4 (Event Detection and Tracking) of the doctoral
project "Shape representation and event detection of sub-cellular structures in
high-content/high-throughput microscopy" (UiT, BioAI group). Sperm motility is the primary
use case for the event-detection methodology; the same detect–track–represent–detect-events
pipeline targets sub-cellular structures such as mitochondria. Primary evidence: 1,138
hand-annotated single-cell trajectories (VISEM-Tracking ground truth, 20 videos).
Secondary: 57 further participants tracked automatically (pipeline-level replication) and a
four-pipeline tracking-fidelity audit on the annotated videos.*

---

## Abstract

Computer-Aided Sperm Analysis (CASA) summarises a semen sample by the instantaneous
fraction of cells in a small number of motility categories, implicitly treating each cell
as a memoryless (Markov) switcher drawn from one homogeneous population. We test both
assumptions on the largest hand-annotated single-cell resource available — 1,138
human-verified spermatozoon trajectories (656,145 frame-states, 6,895 dwell episodes; 20
VISEM-Tracking videos, 50 fps) — and, critically, on the *same videos* re-tracked by
four automated detector–tracker pipelines, so that biology and tracking artefact can be
separated for the first time. Three findings follow. (i) On ground truth, motility-state
**dwell times are heavy-tailed and non-exponential** — log-normal for the progressive and
non-progressive states (ΔAIC over exponential +1,195 and +1,343) — rejecting the Markov
assumption; the *immotile* state, by contrast, is gamma-distributed with a light tail
(CV ≈ 1.05), and the heavy immotile tail reported by automated pipelines is a tracking
artefact of fragmentation. (ii) Switching carries **genuine single-cell memory**: on
decorrelated 0.5 s blocks a second-order model beats a first-order one by g₂ = +0.030
per token (window-robust, +0.020/+0.030/+0.042 at 13/25/51 frames), and a hierarchical
empirical-Bayes null attributes **~71 % of it to within-cell memory** and ~29 % to stable
cell-to-cell rate heterogeneity (ICC = 0.10). (iii) The memory has a specific temporal
signature: within one cell, dwell durations are *less* dispersed than exponential
(CV ≈ 0.77) and successive dwells **anti-correlate** far below a within-cell permutation
null (Δρ = −0.24, p ≈ 10⁻²⁰) — single sperm are *refractory*, anti-bursty switchers, a
fast history-dependence that a slow doubly-stochastic (latent-modulation) model we fit
independently predicted as its missing ingredient. Methodologically, every automated
pipeline **inflates the memory statistic ~1.9-fold** and manufactures the immotile heavy
tail, while conventional multi-object-tracking accuracy fails to rank pipelines by this
downstream dynamical fidelity — tracker evaluation should include dynamical observables,
not only identity metrics. A reliable per-man switching-heterogeneity trait does **not**
improve prediction of the DNA-fragmentation index over standard CASA composition
(pre-registered null). The population-average snapshot is the wrong observable for sperm
motility: the dynamics are non-Markovian, refractory, and heterogeneous — and quantifying
them demands annotation-grade trajectories.

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

We test both directly. Hand-annotated tracking yields thousands of verified state-switching
events, enough to characterise the *distribution* and *temporal structure* of motility
switching rather than only its mean. We ask four questions: what law governs how long a
cell dwells in a state; whether switching carries memory beyond the current state; if so,
whether that memory is a property of single cells or an artefact of pooling a heterogeneous
population; and — a question the field has not asked — how much of the apparent dynamics is
manufactured by the *tracking pipeline itself*. The last question turns out to be decisive:
automated trackers fragment and re-join trajectories, and those identity errors masquerade
as biological memory and heavy-tailed dwell laws. We therefore anchor every headline
statistic on human-verified trajectories and use the automated pipelines as a controlled
perturbation, which yields a methodological result of independent interest for any
track-then-model pipeline. Finally we ask whether anything discarded by the population
average carries clinical signal.

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

**Refractoriness test.** Within each cell with ≥ 5 interior dwell episodes we compute the
lag-1 Spearman correlation of successive dwell durations and compare it with a
within-cell permutation null that preserves each cell's dwell multiset (so censoring and
heterogeneity cannot produce a spurious signal); the statistic is Δρ = ρ_obs − ρ_null.
The same episodes give the within-cell dwell CV (`memory_decomposition.py`,
`gt_reanchor.py`).

**Tracking-fidelity audit.** The 20 annotated videos are re-tracked by three automated
pipelines (BoT-SORT+ReID, BoT-SORT, ByteTrack over the same fine-tuned detector). For each
pipeline we compute (a) conventional association quality against GT (identity switches per
GT track, coverage) and (b) every downstream dynamical statistic above, scored by the
identical code. Fidelity is the agreement of (b) with GT (`experiments/tracker_fidelity.py`).

**Mechanistic budget.** To identify which physical ingredients are *sufficient* for the
observed memory we simulate a ladder of semi-Markov generative models over the empirical
embedded topology, adding one ingredient per rung — homogeneous exponential (null),
quenched rate heterogeneity (calibrated to the observed ICC), sub-exponential gamma dwell
shape (calibrated to the within-cell CV), refractory serial coupling (a Gaussian-copula
AR(1) on successive dwells calibrated to Δρ while preserving the gamma marginal exactly),
and finally the empirical second-order embedded topology. Each rung's block-level g₂ is a
zero-free-parameter prediction (`experiments/refractory_model.py`).

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

### 3.1 Dwell times are heavy-tailed for motile states — and the immotile heavy tail is a tracking artefact

![Figure 1](figures/fig1_dwell_law.png)

**Figure 1. The dwell-time law.** (A) Empirical survival functions of state dwell times on
log–log axes depart sharply from the exponential (memoryless/Markov) law, which
under-predicts the motile-state tails by orders of magnitude. (B) The exponential is
rejected in every state; on ground truth the best law is log-normal for the motile states
but *gamma with a light tail* for the immotile state. *(Figure to be regenerated on
ground-truth trajectories.)*

On the 1,138 ground-truth trajectories, dwell times reject the exponential (Markov) law in
every state: ΔAIC over exponential is +1,195 (progressive, n = 2,940 episodes), +1,343
(non-progressive, n = 3,048) and +176 (immotile, n = 907). But the *form* of the law is
state-dependent, and this matters. The two motile states are genuinely heavy-tailed and
best fit by a log-normal (dwell CV 1.86 and 2.16 against 1 for a memoryless process; the
empirical tail mass beyond 5× the mean exceeds the exponential prediction by 1.7× and
1.3×). The immotile state is *not*: its best law is a gamma (ΔAIC +40 over log-normal),
its CV is 1.05 and its tail carries **no excess mass** over the exponential prediction
(tail excess 0.0×). Mean dwells are 0.85 s (progressive), 0.54 s (non-progressive) and
9.9 s (immotile).

The contrast with automated tracking is diagnostic. On the *same videos*, the automated
baseline pipeline (19,685 tracks) reports a log-normal in every state including immotile,
with inflated dispersion throughout (CV 3.2/3.2/2.4) and a mean immotile dwell of ~0.4 s —
twenty-fold shorter than ground truth. The mechanism is fragmentation: an immotile cell
that is repeatedly lost and re-acquired contributes many short spurious "immotile dwells"
plus a scatter of track boundaries, which jointly manufacture a heavy tail on a light-tailed
state. The oft-convenient conclusion "all dwell laws are log-normal" is therefore partly an
artefact of the measurement pipeline; the defensible ground-truth statement is: *motile-state
dwells are heavy-tailed and non-exponential; immotile dwells are regular (near-gamma,
light-tailed) and long*. Two censoring caveats bound the immotile claim: mean immotile dwell
(494 frames) is comparable to the median track length (445 frames), so its absolute scale is
a lower bound; and the tail comparison uses interior and boundary episodes identically for
all laws, so the *ranking* of laws — not the millisecond scale — is the claim. On automated
data the rejection of the exponential survives censored maximum likelihood and classifier
windows of 13/25/51 frames (ΔAIC +1.1×10³ to +2.1×10⁵, `dwell_censoring.py`,
`dwell_physics_robust.py`); on ground truth we verified window-robustness for the memory
statistic (§3.2).

### 3.2 Switching carries memory beyond the current state

On decorrelated 0.5 s blocks of the ground-truth state sequences, a second-order Markov
model improves held-out per-token log-likelihood over a first-order model by
**g₂ = +0.0304**, and the result is robust to the classifier window (+0.0202, +0.0304,
+0.0416 at 13, 25 and 51 frames): the next state depends on where the cell came from. The
memory lives at the behavioural scale, not the frame scale — at full 20 ms resolution the
second-order gain is ≈ 0 (+0.0002), because the smooth sliding-window series makes the
current state at 20 ms nearly sufficient; it is only after decorrelating to 0.5 s blocks
that history beyond the current state becomes informative. The memory is even more visible
when time is removed entirely: on the *embedded chain* (the durationless sequence of
visited states), the second-order gain is +0.096 — knowing the previous state strongly
predicts the next transition. An explicit hidden-Markov model with up to four latent modes
is *beaten* by the raw second-order model at fewer parameters (prior result,
`hmm_vs_markov.json`), indicating history-dependence on the observed state rather than a
noisy read-out of a few discrete hidden modes.

Two subordinate replications: the automated baseline pipeline on the same videos gives
g₂ = +0.0589 — the qualitative signature survives, but the magnitude is inflated 1.9-fold
by tracking artefact (§3.5) — and the independent 57-participant automated cohort gives
g₂ = +0.0565, replicating the *pipeline-level* signature on disjoint participants.

### 3.3 Most of the memory is inside single cells — and it is refractory

![Figure 2](figures/fig2_decomposition.png)

**Figure 2. Decomposing the memory.** (A) Second-order gain g₂ for the observed
ground-truth data and for three parametric-bootstrap nulls: a homogeneous memoryless
population reproduces essentially none of the signal, a fair empirical-Bayes heterogeneous
memoryless population reproduces about a third, and the remainder is genuine within-cell
memory. (B) The within-cell temporal signature: successive dwell durations of one cell
anti-correlate far below the within-cell permutation null, and within-cell dwell CV sits
*below* the exponential value — single sperm are refractory, anti-bursty switchers.
*(Figure to be regenerated on ground-truth trajectories.)*

**Decomposition.** A homogeneous memoryless population (HOM null) yields g₂ = +0.004,
confirming the test does not manufacture memory. A heterogeneous memoryless population —
cells that are individually first-order but differ in their transition kinetics — can
reproduce part of the signal by aggregation (a Simpson/mover–stayer effect): the fair
empirical-Bayes null gives g₂ = +0.0115, and the deliberately over-fit HET null (each track
simulated from its own noisy matrix) gives +0.035. Against the fair null, the decomposition
attributes **71 % of the observed g₂ (+0.0302) to genuine within-cell memory and 29 % to
stable cell-to-cell heterogeneity**. Ground truth revises our earlier automated-pipeline
estimate of this split (53–60 % / 40–47 %): tracking errors masquerade as *heterogeneity*
(fragmented tracks look like distinct cells with distinct kinetics), so automated data
overstates the mover–stayer share. Consistently, between-cell dispersion on ground truth is
real but modest — ICC of log-dwell = 0.10, empirical-Bayes Dirichlet concentrations
k = 3.0/2.5/5.3 (progressive/non-progressive/immotile) — whereas the automated pipeline
gives ICC = 0.16 with a larger aggregation share (53 %). One honest retraction from our
automated-era analysis: on ground truth the observed g₂ (+0.0302) no longer exceeds the
over-fit HET upper bound (+0.035), so the "memory beyond *any* heterogeneous memoryless
model" argument rests on the fair EB null, not on the upper bound.

**The memory is refractory.** Two within-cell statistics, both immune to censoring and
heterogeneity by construction, characterise it. First, within a single cell the dwell-time
CV is **0.77** — *below* the exponential value of 1: individual cells are more regular than
Poisson switchers (sub-exponential; a gamma fit gives shape ≈ 1.7), even though the pooled
population is over-dispersed (CV 1.9–2.2). The population heavy tail is therefore a
mixture-of-regular-switchers arithmetic, not within-cell burstiness. Second, successive
dwell durations of the same cell are **anti-correlated**: lag-1 Spearman ρ = −0.125 against
a within-cell permutation null of +0.117 (the positive null reflects episode-count
selection), giving Δρ = **−0.242** (p = 2.5×10⁻²⁰). A long dwell is followed by a short
one and vice versa — the cell behaves as if switching consumes a resource that must
recover: a *refractory*, anti-bursty process. This signature is invisible in automated
tracking on the same videos (Δρ = −0.03): identity switches splice unrelated cells'
dwells and fragmentation truncates successive-dwell pairs, destroying the correlation
structure — which is why our own earlier automated-data analysis, and any analysis built
on automated tracking, concluded the heterogeneity was quenched with no within-cell
temporal structure. Ground truth reverses that conclusion.

### 3.4 What generates the memory? Sequence momentum, with refractory timing as a separate channel

The decomposition above *measures* the memory; here we identify what carries it, by two
complementary generative analyses on the ground-truth trajectories. Throughout, real and
simulated sequences are scored by identical self-contained code, all model timescales and
topologies are *calibrated* from empirical marginals (never fitted to the memory
statistic), and each simulated cell is cut to an empirical track length, so censoring is
reproduced by construction. On the internal block-modal estimator the ground-truth targets
are g₂ = +0.0252 and per-state dwell CV = (1.86, 2.16, 1.05) (the headline +0.0304 of §3.2
uses a slightly different block loader; both are 0.5 s-block statistics).

**A slow doubly-stochastic mechanism fails at the observed dispersion.** We first asked
whether slow internal modulation — each cell carrying Ornstein–Uhlenbeck "rate gear" and
"vigour" variables that drift its switching kinetics — can generate the memory
(`generative_model.py`). The memoryless mixture reproduces none of it (g₂ = −0.002),
confirming generatively that the signal is not an artefact of the calibrated marginals.
Slow modulation *can* generate memory, but only by over-dispersing dwell times: the
configuration matching the empirical dwell CV recovers just 9–34 % of the observed g₂,
while reaching 68–72 % forces dwell CV up to ≈ 4.4 — double the data. No parameterization
matches both. Real cells exhibit strong history-dependence at *modest* dwell dispersion,
so a mechanism whose only memory source is slowly varying rates is excluded; the model
predicts a distinct, faster ingredient in the transitions themselves.

**A zero-free-parameter ladder identifies the ingredient.** We then built a ladder of
semi-Markov models over the empirical embedded topology, each rung adding one calibrated
ingredient, with the block-level g₂ of each rung a parameter-free *prediction*
(`refractory_model.py`): homogeneous-exponential null, −15 % of the real g₂; + quenched
rate heterogeneity matched to the observed ICC = 0.10, −13 %; + gamma dwell shape matched
to the within-cell CV = 0.77, −22 %; + refractory serial coupling (Gaussian-copula AR(1),
φ = −0.30, reproducing the observed Δρ = −0.24 while preserving the gamma marginal
exactly), −20 %. **None of the timing ingredients produces block-scale memory.** The final
rung replaces the first-order embedded chain by the *empirical second-order embedded
topology* — which state the cell came from conditions which state it enters next — and
this single ingredient recovers **93 % of the observed g₂** (+0.0234 vs +0.0252), while
slightly overshooting the durationless embedded-chain gain (+0.144 vs +0.096), as expected
when the empirical context effect is imposed uniformly.

The memory therefore has an identified carrier and a clean anatomy. The block-scale
second-order gain is carried almost entirely by **sequence momentum in the transition
topology** — e.g. a cell that reached the non-progressive state from progressive is
disproportionately likely to return to progressive rather than decay to immotile — and not
by dwell-time structure of any kind: heterogeneity, sub-exponential regularity and
refractory anti-correlation each leave g₂ at zero. Conversely, the refractory timing
signature (Δρ = −0.24) and the sub-exponential regularity (CV = 0.77) are real,
independently measured properties of single cells that g₂ is blind to. Single sperm thus
carry **two dissociable memory channels**: a *sequence channel* (directional momentum in
which states are visited) and a *timing channel* (regular, refractory switching clocks) —
precisely the fast transition-level ingredient the slow-latent model predicted was
missing, now confirmed and localised.


### 3.5 Tracking artefacts masquerade as dynamics — and MOT accuracy does not predict dynamical fidelity

The results above required hand-annotated trajectories; here we quantify what automated
tracking would have reported instead. We re-tracked the 20 annotated videos with three
pipelines sharing the same fine-tuned detector (BoT-SORT+ReID, BoT-SORT, ByteTrack) and
scored every dynamical statistic with the identical code (`tracker_fidelity.py`).

Every pipeline inflates the memory statistic, and by a similar factor: g₂ = 0.047–0.052
against 0.025 on ground truth with the same estimator — a **1.8–2.1× inflation** —
and every pipeline converts the light-tailed immotile dwell law into a spurious heavy
tail (§3.1) while inflating dwell CV in all states. The mechanism is identity error:
fragmentation and ID switches concatenate unrelated cells and truncate dwells, which
*adds* apparent history-dependence and heterogeneity. A track-then-model analysis
therefore overstates non-Markovianity roughly two-fold even when its qualitative
conclusions survive.

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

Sperm motility switching is not the memoryless, homogeneous process implicit in CASA and
in standard Markov modelling — but the deviation is more structured, and smaller, than
automated tracking suggests. On human-verified trajectories the picture is: motile-state
dwells are heavy-tailed (log-normal), immotile dwells are long and regular; roughly 70 %
of the second-order memory is genuinely within-cell and it decomposes into two dissociable
channels — a *sequence channel* (directional momentum in the embedded transition topology,
which alone reproduces 93 % of the memory statistic with no free parameters) and a *timing
channel* (single cells switch with sub-Poisson regularity, CV ≈ 0.77, and refractory
anti-correlation of successive dwells, Δρ = −0.24). The remaining ~30 % is a stable but
modest cell-to-cell heterogeneity (ICC ≈ 0.10). The refractory timing is biologically
suggestive — a switching "clock" that consumes and recovers a resource, e.g. intracellular
Ca²⁺ or ATP dynamics in the flagellar beat machinery — and it was *predicted* before it was
measured: a slow doubly-stochastic model, fit independently, could not reproduce the memory
at the observed dwell dispersion and required a fast transition-level ingredient, which the
ground-truth serial statistics then confirmed. We offer the two-channel anatomy — sequence
momentum plus refractory clocks over a weakly heterogeneous rate mixture — as the minimal
phenomenology any mechanistic model of sperm motility regulation must reproduce.

The equally important result is methodological. Every automated tracking pipeline we
tested inflates the memory statistic ~1.9-fold, manufactures a heavy tail on the
light-tailed immotile dwell law, overstates cell-to-cell heterogeneity, and *erases* the
refractory signature — identity splices destroy within-cell serial structure while
creating spurious population-level history-dependence. Our own earlier analyses on
automated tracks drew exactly those wrong conclusions (quenched heterogeneity with no
within-cell temporal structure; log-normal dwells in every state), and we correct them
here. Because the biases are qualitative, not just quantitative — tracking error created a
*false negative* for refractoriness and a *false positive* for immotile heavy tails — no
amount of automated data substitutes for annotation when the object of study is dynamics.
And because conventional MOT identity metrics ranked our pipelines *opposite* to their
dynamical fidelity, tracker selection for track-then-model science should be validated on
downstream dynamical observables against annotation.

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
scheme that assigns organelle morphology to a fixed number of shape classes. Second, the
*events are not memoryless*: transitions carry sequence momentum and refractory timing,
both invisible to the population-average snapshot and to homogeneous Markov models. Third
— the lesson this paper adds — *tracking error masquerades as event dynamics*: identity
splices manufacture memory and heavy tails while erasing genuine within-object temporal
structure, so event-detection pipelines for organelles (where fission/fusion events are
precisely identity events) must be validated on downstream dynamical observables against
annotation, not on detection/association metrics alone.

**Limitations.** (1) Ground truth is 20 videos / 1,138 cells from one dataset; the
annotation-anchored statistics are precise (e.g. the refractory effect at p ≈ 10⁻²⁰) but
their generality across labs and imaging conditions rests on the automated pipeline-level
replication (57 further participants), which — as we show — preserves qualitative
signatures while inflating magnitudes. (2) Absolute dwell *scales* remain lower-bounded by
finite track length; the immotile mean dwell in particular is comparable to the median
track length, so its law is a censored comparison of forms, not scales. (3) States derive
from a 0.5 s sliding-window classifier; the memory statistic is robust across 13/25/51-frame
windows, but part of the block-scale autocorrelation may be classifier-induced, which is why
we emphasise the embedded-chain and dwell-level statistics, which do not share the window.
(4) The tracker-fidelity dissociation (MOT metrics vs dynamical fidelity) rests on three
pipelines and is hypothesis-generating. (5) The genuine-memory share is bracketed by fair
and over-fit nulls, not a point estimate, and on ground truth the observed g₂ no longer
exceeds the over-fit upper bound — the within-cell claim rests on the fair EB null together
with the direct within-cell statistics (CV, Δρ), which are null-immune by construction.

**Conclusion.** On annotation-grade single-cell trajectories, human sperm motility
switching is a non-Markovian process with an identified anatomy — sequence momentum plus
refractory, sub-Poisson switching clocks within a weakly heterogeneous population — and
the standard measurement stack fails it twice: the population-average snapshot discards
the dynamics, and automated tracking distorts them. Both failures are correctable, and the
same two corrections — model events with memory; validate trackers on dynamical fidelity —
transfer directly to event detection in sub-cellular structures.

---

## Reproducibility

| Result | Script | Output |
|---|---|---|
| Ground-truth track materialisation + all GT-anchored headline statistics | `experiments/gt_reanchor.py` | `outputs/tracks_gt/`, `outputs/markov/gt_reanchor.json` |
| Tracking-fidelity audit (GT vs 3 pipelines) | `experiments/tracker_fidelity.py` | `outputs/markov/tracker_fidelity.json` |
| Mechanistic ladder (heterogeneity/shape/refractory/2nd-order topology) | `experiments/refractory_model.py` | `outputs/markov/refractory_model.json` |
| Slow-latent generative mechanism on GT | `experiments/generative_model.py --fit-dir outputs/tracks_gt --fit-name gt` | `outputs/markov/generative_model_gt.json` |
| Dwell-time law + window invariance (automated cohorts) | `experiments/dwell_physics.py`, `dwell_physics_robust.py` | `outputs/markov/dwell_physics*.json` |
| Pipeline-level replication of memory (extra57) | `experiments/replicate_markov_extra.py` | `outputs/markov/replication_extra.json` |
| Within- vs between-cell decomposition | `experiments/memory_decomposition.py` | `outputs/markov/memory_decomposition.json` |
| Censoring-aware dwell law + frailty (automated) | `experiments/dwell_censoring.py` | `outputs/markov/dwell_censoring.json` |
| Mover–stayer nulls (HOM/HET/EB) | `experiments/mover_stayer_null.py`, `mover_stayer_eb.py` | `outputs/markov/mover_stayer_*.json` |
| Per-cell heterogeneity + reliability | `experiments/per_cell_kinetics.py` | `outputs/markov/per_cell_kinetics.{json,csv}` |
| Stayer→DFI robustness | `experiments/stayer_dfi.py` | `outputs/markov/stayer_dfi.json` |
| DFI prediction (pre-registered) | `experiments/dfi_predict.py` | `outputs/markov/dfi_prediction.json` |
| Figures | `experiments/make_paper_figures.py` | `paper/figures/fig{1,2,3}_*.png` |
