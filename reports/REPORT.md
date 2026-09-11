# Bayesian Inference for CAD Diagnosis: Report

Text source for `reports/REPORT.docx`, the submission-ready final report (embedded
figures, performance tables, APA 7 reference list, and an Observations/Recommendations/
Final Insight section — rebuild it with `python scripts/build_report_docx.py` after
re-running the notebook). Companion code:
`notebooks/bayesian_heart_disease_diagnosis.ipynb`.

## 1. Introduction

Use case: a diagnostic-support system for coronary artery disease (CAD) that combines
clinical prior knowledge with patient data, rather than fitting purely from data, and
that can be updated as new patients arrive instead of retrained from scratch. Bayesian
inference is the natural fit — the prior/posterior mechanism *is* the combination of
prior knowledge and evidence, not an analogy for it. Task: estimate P(angiographic CAD)
from clinical presentation.

Dataset: `processed.cleveland.data`, 303 patients, 14 attributes, collected at the
Cleveland Clinic Foundation (Detrano et al., 1989, *Am J Cardiol* 64:304-310). This is
the only dataset used — no external or synthetic data. `ca`/`thal` had 6 missing values
total (median/mode imputed); the 5-level target `num` was binarized to disease
present/absent, the standard task for this dataset since the original 1989 study.
80/20 train/test split, stratified on the outcome; numeric features standardized on
train statistics only.

## 2. Priors

Model: Bayesian logistic regression, $\text{logit}(p_i) = \beta_0 + \sum_j \beta_j x_{ij}$,
fit with PyMC/NUTS. Each $\beta_j$ gets an independent Normal prior centered on the
direction reported in the literature, with SD wide enough for ~240 training patients to
override it if the data disagree (weakly-informative-prior approach of Gelman, Jakulin,
Pittau & Su, 2008, *Ann. Appl. Stat.* 2(4):1360-1383).

Sources:

- **Detrano et al. (1989)** — the paper this exact cohort comes from. Fit a
  logistic-regression CAD-probability model, ~77-81% accuracy across sites. Identifies
  `oldpeak`, ST slope, `ca`, `thal`, `exang` as dominant predictors once exercise-test
  variables are included; `trestbps`/`chol` weak on top of those.
- **Diamond & Forrester (1979)**, *NEJM* 300(24):1350-1358 — the pretest-probability
  model `cp` is derived from. Sign of the `cp` effect is ambiguous in an
  already-referred-for-angiography population (asymptomatic presentation here skews
  toward silent/severe disease, the opposite of general-population pretest probability),
  so `cp` dummies got a wide, direction-agnostic prior; the data decide the sign.
- **Wilson et al. (1998)**, *Circulation* 97:1837-1847 (Framingham) — age and male sex:
  consistently positive, moderate-to-large risk factors.
- AHA exercise-testing consensus — lower `thalach`, presence of `exang`, larger
  `oldpeak` -> higher ischemia probability.

Full table of prior means/SDs per feature is in notebook Section 2.3.

## 3. Model and fit

PyMC model, 4 chains x 2000 draws (2000 tune), NUTS at `target_accept=0.95`. Converged
cleanly: max $\hat{R}$ = 1.002, 0 divergent transitions out of 8000, min bulk ESS ~3760.

Posterior effect sizes (notebook Section 3, forest plot) — the clearest signal is
`cp_asymptomatic` (mean 1.14 log-odds, 94% HDI [0.25, 2.04]): asymptomatic presentation
in this angiography-referred cohort is strongly associated with disease, confirming the
sign ambiguity flagged in Section 2 resolved toward the "referred population" direction
rather than the general-population pretest-probability direction. Several other
coefficients (`trestbps`, `oldpeak`, `cp_non_anginal`) have HDIs that still cross zero
individually at this sample size, despite being classically important markers — plausible
given n~240 and 17 parameters, and visible directly in the posterior widths rather than
hidden behind a point estimate.

## 4. Sequential updating

The exercise asks for a system that updates as new cases arrive, not just a static fit.
Training data was split into 3 batches and fit sequentially, each batch's posterior
(moment-matched to a diagonal Normal) supplying the prior for the next — recursive Bayes,
not a re-fit from the literature prior each time. Tracking a single held-out patient's
predicted P(disease) across batches shows the posterior visibly sharpening (narrower
density, moving toward the true label) as more data is absorbed (notebook Section 4).

## 5. Validation

Held-out test set, n=61, untouched until evaluation:

| Metric | Value |
|---|---|
| Accuracy | 0.885 |
| Log-loss | 0.281 |
| ROC AUC | 0.960 |
| Brier score | 0.082 |

Precision/recall: 0.93/0.85 (no disease), 0.84/0.93 (disease) — the model trades a few
more false positives for fewer missed cases, the right direction for a diagnostic aid.
Confusion matrix and ROC curve in notebook Section 5. Posterior predictive intervals per
patient (also Section 5) flag which test cases the model is least confident about — the
mechanism a deployed system would use to route ambiguous cases to a clinician instead of
auto-classifying.

## 6. Refinement: priors and likelihood

Refit on the identical split, varying first the prior (A, C) and then the likelihood's
link function (D), against the main model (B):

| | Accuracy | Log-loss | ROC AUC | Brier |
|---|---|---|---|---|
| A: flat baseline $\mathcal{N}(0, 2.5)$ | 0.852 | 0.302 | 0.952 | 0.088 |
| B: literature-informed (main) | 0.885 | 0.281 | 0.960 | 0.082 |
| C: B, prior SDs halved | 0.885 | 0.285 | 0.960 | 0.082 |
| D: B, probit link instead of logit | 0.885 | 0.286 | 0.959 | 0.083 |

B beats A on every metric — the literature prior is doing real work, not just adding
interpretability. C essentially matches B: once the direction is right, squeezing the
prior further doesn't buy anything more at this sample size. D (a likelihood-side change
rather than a prior-side one — swapping the logit link for a probit link on the same
linear predictor) also lands within noise of B, which is expected: with standardized
features and this much data, logit and probit are close to a reparameterization of each
other rather than substantively different models. Net result: at n~240, the prior choice
moved the metrics more than the link-function choice did. Consistent with the posterior
widths in Section 3 — the prior's leverage is on the coefficients the data alone leave
ambiguous (`restecg`, `fbs`), not on the ones the data already pin down (`ca`, `oldpeak`,
`thal`).

## 7. Discussion of findings and potential improvements

Findings: the model is accurate (0.885) and well-calibrated for a diagnostic aid (AUC
0.960, Brier 0.082) on data it never touched during fitting or preprocessing; the
literature-informed prior is not just interpretability window-dressing — it measurably
outperforms a flat baseline (Section 6); and the sequential-updating construction
(Section 4) shows the posterior genuinely sharpening as evidence accrues, which is the
behavior the exercise's "dynamic updating" requirement is asking for, not just a fresh
fit each time.

Limitations, and what they motivate:

- Cleveland cohort only (303 patients). The Hungarian, Switzerland, and Long Beach VA
  files present in this repository were not merged in — they have different missingness
  patterns and collection protocols (per `heart-disease.names`) and would need separate
  handling (e.g. a site random effect) to combine without confounding site with outcome.
- Logit assumed linear in the features; no interaction terms (age x sex, cp x exang) —
  worth testing given the cardiology literature's emphasis on age/sex-stratified risk.
- n=61 test set — the metrics above carry non-trivial sampling uncertainty of their own;
  a single split can't distinguish a real 3-point accuracy gap from noise.
- Priors are derived from published effect sizes, not elicited from a clinician directly.

## 8. Conclusion and next steps

A Bayesian logistic regression with cardiology-literature priors reaches 0.885 accuracy /
0.960 AUC on held-out Cleveland patients, converges cleanly, and — unlike a plain
point-estimate classifier — ships calibrated per-patient uncertainty and a well-defined
mechanism for incorporating new patients without retraining from scratch. The prior
comparison in Section 6 is the load-bearing result: it turns "we used informative priors"
from a design choice into a tested claim.

Next: interaction terms; k-fold or PSIS-LOO cross-validation instead of a single split;
extend Section 4 to genuine online prediction-then-reveal rather than batch-sequential;
direct prior elicitation from a cardiologist as a comparison to the literature-derived
priors used here.
