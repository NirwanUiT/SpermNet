# Human sperm motility switching is non-Markovian: single-cell memory and a stable mover–stayer population structure that population-average CASA discards

*Working manuscript draft — Work Package 4 (Event Detection and Tracking) of the doctoral
project "Shape representation and event detection of sub-cellular structures in
high-content/high-throughput microscopy" (UiT, BioAI group). Sperm motility is the primary
use case for the event-detection methodology; the same detect–track–represent–detect-events
pipeline targets sub-cellular structures such as mitochondria. Cohorts: VISEM &
VISEM-Tracking (n = 77 with paired tracks + clinical). All results cross-validated on two
independently tracked cohorts.*

---

## Abstract

Computer-Aided Sperm Analysis (CASA) summarises a semen sample by the instantaneous
fraction of cells in a small number of motility categories, implicitly treating each
cell as a memoryless (Markov) switcher drawn from one homogeneous population. Using
long single-cell trajectories from 77 men (20 hand-annotated VISEM-Tracking videos and
57 further participants tracked with an independent detector–tracker pipeline; ≈16 M
frame-level state observations per cohort), we show that both of these assumptions are
wrong, and we quantify how. (i) Motility-state **dwell times are log-normal, not
exponential** — the best of exponential, gamma, Weibull and log-normal laws in every
state and both cohorts, robust to the classifier window (13/25/51 frames) and rejecting
the exponential (Markov) law by ΔAIC up to ~10⁵. (ii) On decorrelated 0.5 s blocks a
second-order model beats a first-order one by ≈ +0.057 log-likelihood per token, i.e.
the next state depends on more than the current state. (iii) Using a hierarchical
(empirical-Bayes) parametric-bootstrap null we **decompose this "memory" into two
comparable, dissociable sources**: genuine within-cell memory (~53–60 %) and a stable
cell-to-cell heterogeneity in switching kinetics — a *mover–stayer* population structure
(~40–47 %); a homogeneous memoryless model reproduces none of it. The heterogeneity axis
is a highly reliable per-man trait (split-half ρ ≈ 0.95). It does **not**, however,
improve prediction of the clinical DNA-fragmentation index (DFI) over standard CASA
composition (pre-registered null), because the apparent "stayer" signal simply re-encodes
the progressive/immotile percentages CASA already reports. We conclude that the
population-average snapshot is the wrong observable for sperm motility: the dynamics are
non-Markovian with two distinct physical origins, both invisible to CASA and to standard
Markov/HMM models.

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

We test both directly. Modern tracking yields millions of state-switching events per
sample, enough to characterise the *distribution* and *temporal structure* of motility
switching rather than only its mean. We ask three questions: what law governs how long a
cell dwells in a state; whether switching carries memory beyond the current state; and, if
so, whether that memory is a property of single cells or an artefact of pooling a
heterogeneous population. Finally we ask whether anything discarded by the population
average carries clinical signal.

---

## 2. Data and methods

**Cohorts.** Two disjoint sets of participants are analysed identically and never pooled
for cross-participant statistics (they differ by tracking pipeline and sampling, a batch
effect we treat as a nuisance): *orig20* = 20 VISEM-Tracking participants with
hand-annotated tracks; *extra57* = 57 further VISEM participants tracked with our own
detector (fine-tuned YOLOv8-l) and BoT-SORT. Videos are 50 fps. Clinical variables
(including DFI) come from the VISEM clinical tables (n = 85). 77 participants have both
tracks and clinical data.

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

### 3.1 Dwell times are log-normal, not exponential

![Figure 1](figures/fig1_dwell_law.png)

**Figure 1. The dwell-time law is log-normal.** (A) Empirical survival functions of
state dwell times (solid, 57-participant cohort) on log–log axes lie on the fitted
log-normal law (dashed) and depart sharply from the exponential (memoryless/Markov) law
(dotted), which under-predicts the tail by orders of magnitude. (B) The exponential is
rejected relative to the log-normal by ΔAIC of 10⁴–10⁵ in every state and at every
classifier window (13/25/51 frames), so the heavy tail is not an artefact of the sliding
window.

Across all three states and both cohorts the log-normal is the best-fitting of the four
laws. It beats the exponential by ΔAIC = +1,067 to +103,593 and the gamma by +751 to
+53,576. The result is invariant to the classifier window: at windows of 13, 25 and 51
frames the log-normal remains best and the exponential is rejected by ΔAIC = +1.4×10⁴ to
+1.2×10⁵; the coefficient of variation of dwell times decreases with window (smoothing)
but the *form* does not change. The law is also invariant across the two independently
tracked cohorts despite a ~3× difference in absolute dwell scale — a signature of a
mechanism, not a pipeline artefact. Pooled dwell CV is 1.9–3.0 versus 1 for a memoryless
process.

### 3.2 Switching carries memory beyond the current state

On decorrelated 0.5 s blocks, a second-order Markov model improves held-out per-token
log-likelihood over a first-order model by **g₂ = +0.0587** (orig20) and **+0.0565**
(extra57) — the next state depends on where the cell came from, replicated on a fully
independent cohort. Consistently, an explicit hidden-Markov model with up to four latent
modes is *beaten* by the raw second-order model at fewer parameters (prior result,
`hmm_vs_markov.json`), indicating the memory is history-dependence on the observed state
rather than a noisy read-out of a few discrete hidden modes.

### 3.3 The memory has two comparable sources: single-cell memory and a mover–stayer population

![Figure 2](figures/fig2_decomposition.png)

**Figure 2. Decomposing the memory.** (A) Second-order gain g₂ for the observed data and
for three parametric-bootstrap nulls. A homogeneous memoryless population reproduces
essentially none of the signal; a heterogeneous memoryless population reproduces much of
it; the observed data exceeds even the (over-fit, upper-bound) heterogeneous null. (B) The
empirical-Bayes fair null splits the non-Markovian signal into ~53–60 % genuine
within-cell memory and ~40–47 % stable cell-to-cell (mover–stayer) heterogeneity,
consistently across cohorts. (C) Within a single cell, dwell times are near-memoryless
(median CV ≈ 1); the pooled population is strongly dispersed (CV ≈ 2), the arithmetic
signature of population heterogeneity.

A homogeneous memoryless population (HOM null) yields g₂ ≈ 0 (+0.0011 / −0.0012),
confirming the test does not manufacture memory. A heterogeneous memoryless population —
cells that are individually first-order but differ in their transition matrices —
reproduces most of the signal by aggregation (a Simpson/mover–stayer effect): the
over-fit HET null gives g₂ = +0.039 / +0.045, and the fair empirical-Bayes null gives
+0.029 / +0.022. Because fitting each short track its own matrix over-fits noise, the HET
null is an *upper bound* on the aggregation contribution. Crucially the **observed g₂ still
exceeds even this upper bound** in both cohorts, so a genuine within-cell memory component
is required. The fair decomposition attributes **53–60 % to single-cell memory and
40–47 % to stable mover–stayer heterogeneity**, with the immotile state the most
homogeneous across cells (Dirichlet concentration k ≈ 3.8–7.3) and the progressive state
the least (k ≈ 1.0–2.7). This is corroborated model-free: within a single cell dwell CV is
≈ 1 (near-memoryless) while the pooled CV is ≈ 2 (Fig. 2C), and successive dwell durations
along a track show no temporal drift beyond the population baseline (state-controlled
lag-1 ρ equals the shuffle null), so the heterogeneity is *quenched* (fixed per cell) rather
than a slow within-cell drift.

### 3.4 The heterogeneity is a reliable trait but not clinically incremental over CASA

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
0.95–0.97 in extra57; 0.56–0.79 in the smaller orig20). Clinically, the pre-registered
primary — switch-rate CV versus DFI, controlling for mean rate and track count — is null in
both cohorts (ρ = −0.36, p = 0.18; ρ = −0.12, p = 0.38). A secondary stayer-fraction signal
appeared cross-cohort-consistent (ρ = −0.55 and −0.45) but vanished once static composition
was controlled (ρ = −0.21, p = 0.44; ρ = −0.03, p = 0.82), and further with age/BMI/
abstinence. State-resolved, stable-progressive fraction tracks lower DFI (ρ ≈ −0.44) and
stable-immotile fraction tracks higher DFI (ρ ≈ +0.42), i.e. the signal is the known
motility-composition–DFI relationship in disguise. Consistently, a pre-registered
prediction test found that memory/dynamics features add nothing to a CASA-composition model
for out-of-sample DFI (Ridge Spearman: CASA 0.501, memory 0.427, CASA+memory 0.499;
Δ = −0.002, permutation p = 0.567).

---

## 4. Discussion

Sperm motility switching is not the memoryless, homogeneous process implicit in CASA and in
standard Markov modelling. Dwell times follow a log-normal law robust to window and cohort,
and the switching carries second-order memory that a homogeneous first-order model cannot
produce. Decomposing that memory shows it has two comparable and physically distinct
origins: a genuine single-cell memory (a cell's recent history biases its next transition)
and a stable cell-to-cell heterogeneity — some cells are persistent "stayers" and others
frequent "movers", each with its own fixed switching kinetics. A log-normal dwell law with
near-memoryless single cells but a dispersed population is exactly the arithmetic of such a
quenched mixture, and it explains why a global hidden-Markov model — which allows within-track
mode switching — is out-performed by raw history: the heterogeneity is *between* cells and
fixed, not *within* cells and dynamic.

The immediate methodological consequence is that the population average is the wrong
observable. Two samples with identical progressive percentages can have very different
mover–stayer structure, and that structure is a reliable per-man trait (split-half ρ ≈ 0.95).
Whether it is *clinically* useful is a separate question, and here we are deliberately
conservative: on this dataset the heterogeneity axis does not improve prediction of DFI over
standard composition, and the one apparently promising association proved to be composition
in disguise. We report this as a null. DFI is in any case a surrogate; the biologically
decisive test — whether switching dynamics or population structure predict fertilisation or
live birth — requires outcome-linked cohorts we do not have here.

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
*events are not memoryless*: transition dynamics carry history and reflect stable
object-to-object heterogeneity, both invisible to the population-average snapshot and to
homogeneous Markov models. An event-detection method for tracked biological structures
should therefore represent state as a position on a continuous manifold and model
transitions with memory, not as a homogeneous Markov chain over a handful of bins.

**Limitations.** States are derived from a sliding-window classifier; although the dwell law
and its rejection of the exponential are invariant across window sizes, finite track lengths
right-censor the longest dwells, so absolute dwell scales and the within-cell CV are
lower-bounded rather than exact. Cross-participant clinical correlations of dynamics features
are sensitive to the tracking pipeline (a cohort batch effect), which is why all clinical
statistics are computed within cohort and never pooled. The genuine-memory/heterogeneity
split is bracketed (fair EB estimate with an over-fit upper bound), not a point estimate.

**Conclusion.** Human sperm motility is a non-Markovian process with two dissociable sources
— single-cell memory and a stable mover–stayer population structure — both quantifiable from
routine video, both reliable, and both discarded by the population-average snapshot that
defines current practice.

---

## Reproducibility

| Result | Script | Output |
|---|---|---|
| Dwell-time law + window invariance | `experiments/dwell_physics.py`, `dwell_physics_robust.py` | `outputs/markov/dwell_physics*.json` |
| Independent replication of memory | `experiments/replicate_markov_extra.py` | `outputs/markov/replication_extra.json` |
| Within- vs between-cell decomposition | `experiments/memory_decomposition.py` | `outputs/markov/memory_decomposition.json` |
| Mover–stayer nulls (HOM/HET/EB) | `experiments/mover_stayer_null.py`, `mover_stayer_eb.py` | `outputs/markov/mover_stayer_*.json` |
| Per-cell heterogeneity + reliability | `experiments/per_cell_kinetics.py` | `outputs/markov/per_cell_kinetics.{json,csv}` |
| Stayer→DFI robustness | `experiments/stayer_dfi.py` | `outputs/markov/stayer_dfi.json` |
| DFI prediction (pre-registered) | `experiments/dfi_predict.py` | `outputs/markov/dfi_prediction.json` |
| Figures | `experiments/make_paper_figures.py` | `paper/figures/fig{1,2,3}_*.png` |
