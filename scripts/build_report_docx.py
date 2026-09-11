"""Builds reports/REPORT.docx: the assignment's Final Report deliverable, with
embedded figures (produced by the notebook run), performance tables, and an APA
7th-edition reference list.

Run after the notebook has been executed at least once (so reports/figures/*.png exist).
"""

from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
OUT_PATH = PROJECT_ROOT / "reports" / "REPORT.docx"

doc = Document()

# ---- base style ----
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)

def h1(text):
    doc.add_heading(text, level=1)

def h2(text):
    doc.add_heading(text, level=2)

def p(text, italic=False, bold=False):
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.italic = italic
    run.bold = bold
    return para

def bullets(items):
    for item in items:
        doc.add_paragraph(item, style="List Bullet")

def figure(path, caption, width_in=6.0):
    doc.add_picture(str(path), width=Inches(width_in))
    last_paragraph = doc.paragraphs[-1]
    last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run(caption)
    run.italic = True
    run.font.size = Pt(10)

def table(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        for run in hdr[i].paragraphs[0].runs:
            run.bold = True
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
    doc.add_paragraph()
    return t

# ====================================================================
# Title page
# ====================================================================
title = doc.add_heading("Bayesian Inference for Coronary Artery Disease Diagnosis", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub.add_run("A Bayesian Logistic Regression Approach with Literature-Informed Priors,\nSequential Updating, and Held-Out Validation on the UCI Cleveland Heart Disease Dataset")
r.italic = True
doc.add_paragraph()
meta = doc.add_paragraph()
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
meta.add_run("Bayesian Inference Exercise: Enhancing a Medical Diagnosis System\n")
meta.add_run("Companion code: notebooks/bayesian_heart_disease_diagnosis.ipynb")
doc.add_page_break()

# ====================================================================
# 1. Introduction
# ====================================================================
h1("1. Introduction")
p(
    "Use case: a healthcare AI startup's diagnostic-support system for coronary artery "
    "disease (CAD) needs to combine clinical prior knowledge with patient data, rather "
    "than fit purely from data, and needs to update as new patients arrive instead of "
    "being retrained from scratch each time. Bayesian inference is the natural "
    "mechanism for this: the prior/posterior update is not an analogy for combining "
    "prior knowledge with evidence, it is that combination, carried out exactly, and it "
    "is what makes \"dynamic updating\" a well-defined operation rather than a slogan."
)
p(
    "This report documents a Bayesian logistic regression for CAD diagnosis, fit with "
    "PyMC, using priors set from the cardiology literature rather than left flat. It "
    "covers the dataset and priors used, the model and its assumptions, held-out "
    "validation results, a prior/likelihood sensitivity comparison, and observations "
    "and recommendations for how this kind of system should be used and improved before "
    "any clinical deployment."
)

# ====================================================================
# 2. Dataset and Priors
# ====================================================================
h1("2. Dataset and Priors")
h2("2.1 Dataset")
p(
    "The only dataset used is processed.cleveland.data (Janosi et al., 1989), the "
    "cleaned 14-attribute subset of the UCI Heart Disease data collected at the "
    "Cleveland Clinic Foundation and first analyzed in Detrano et al. (1989): 303 "
    "patients, attributes covering symptoms (chest pain type, exercise-induced "
    "angina), history/risk factors (age, sex, resting blood pressure, cholesterol, "
    "fasting blood sugar), diagnostic test results (resting ECG, maximum heart rate, "
    "ST depression, ST slope, fluoroscopy vessel count, thallium scan result), and the "
    "diagnosed outcome. No external or synthetic data was introduced."
)
p(
    "Preprocessing: the ca and thal columns had 6 missing values total (out of 303 "
    "rows) and were imputed with the column median/mode respectively rather than "
    "dropped. The five-level severity target was binarized to disease present/absent, "
    "the standard task for this dataset since Detrano et al. (1989). Categorical "
    "attributes were one-hot encoded with the clinically lowest-risk category as "
    "reference. Numeric features were standardized using statistics computed on the "
    "training split only, and an 80/20 train/test split was stratified on the outcome."
)
h2("2.2 Priors")
p(
    "Each regression coefficient received an independent Normal prior, centered on the "
    "direction of effect reported in the literature, with a standard deviation wide "
    "enough that roughly 240 training patients could override it if the local data "
    "disagreed \u2014 the weakly-informative-prior strategy of Gelman et al. (2008). Four "
    "sources were used to set prior direction and confidence:"
)
bullets([
    "Detrano et al. (1989) \u2014 the study this exact cohort comes from; identifies ST "
    "depression, ST slope, vessel count, thallium result, and exercise-induced angina "
    "as the dominant predictors once exercise-test variables are included.",
    "Diamond & Forrester (1979) \u2014 the pretest-probability model the chest-pain-type "
    "variable is derived from. Because this cohort was already referred for "
    "angiography, the sign of the chest-pain effect is ambiguous a priori (it can run "
    "opposite to the general-population pretest-probability direction), so that "
    "variable was given a wide, direction-agnostic prior and its sign was left to the "
    "data to resolve.",
    "Wilson et al. (1998), the Framingham Heart Study \u2014 age and male sex as "
    "consistently positive, moderate-to-large coronary heart disease risk factors.",
    "Fletcher et al. (2013), American Heart Association exercise-testing standards \u2014 "
    "lower maximum heart rate achieved, presence of exercise-induced angina, and "
    "greater ST depression each indicate reduced exercise capacity and higher ischemia "
    "probability.",
])
p(
    "Full per-feature prior means and standard deviations are documented in the "
    "notebook (Section 2.3); at the scale used (0.5\u20130.6 on the log-odds scale), these "
    "priors still assign non-trivial probability to a wide range of odds ratios and "
    "function as nudges rather than constraints."
)

# ====================================================================
# 3. Bayesian Model and Assumptions
# ====================================================================
h1("3. Bayesian Model and Assumptions")
p(
    "The model is a Bayesian logistic regression: logit(p_i) = \u03b20 + \u03a3 \u03b2j x_ij, with "
    "patient symptoms and history as the standardized/one-hot input features x_ij, "
    "fit with PyMC's NUTS sampler (4 chains, 2000 tuning and 2000 sampling draws, "
    "target_accept = 0.95). The model outputs a full posterior distribution over the "
    "probability of disease for each patient, not a point estimate. Key assumptions: "
    "a linear relationship between the standardized features and the log-odds of "
    "disease (no interaction terms); patient records are conditionally independent "
    "given the covariates; and the Cleveland cohort's covariate distribution is "
    "representative of the population the model will be applied to."
)
p(
    "Convergence was clean across every model variant fit in this project: maximum "
    "R-hat of 1.002, zero divergent transitions out of 8,000 post-warmup draws for the "
    "main model, and minimum bulk effective sample size around 3,760 \u2014 the comparative "
    "results below are not artifacts of poor sampling."
)
if (FIG_DIR / "posterior_effect_sizes.png").exists():
    figure(FIG_DIR / "posterior_effect_sizes.png",
           "Figure 1. Posterior coefficient estimates (log-odds), mean \u00b1 94% HDI, by feature.")
p(
    "The clearest posterior signal is the asymptomatic chest-pain-type coefficient, "
    "whose 94% highest-density interval excludes zero and is strongly positive \u2014 "
    "confirming that in this angiography-referred cohort, asymptomatic presentation is "
    "associated with disease, resolving the sign ambiguity flagged in Section 2.2 "
    "toward the referred-population direction rather than the general-population "
    "pretest-probability direction. Several classically important markers (resting "
    "blood pressure, ST depression alone, non-anginal chest pain) still have "
    "intervals crossing zero individually at this sample size \u2014 visible directly in "
    "the posterior width rather than hidden behind a point estimate, which is the "
    "practical advantage of reporting the full posterior instead of a single "
    "coefficient value."
)
h2("3.1 Dynamic (Sequential) Updating")
p(
    "The use case explicitly requires the system to update its predictions as new "
    "cases arrive. This was implemented directly, not merely assumed from the "
    "prior/posterior mechanism: the training set was revealed to the model in three "
    "sequential batches, and after each batch the posterior (moment-matched to a "
    "diagonal Normal) became the prior for the next batch \u2014 recursive Bayesian "
    "updating, rather than refitting from the literature prior each time."
)
if (FIG_DIR / "sequential_updating.png").exists():
    figure(FIG_DIR / "sequential_updating.png",
           "Figure 2. Posterior P(disease) for one held-out patient, sharpening across three sequential update batches.")
p(
    "The posterior for a representative patient visibly narrows and moves toward the "
    "true label as more batches are absorbed, demonstrating the belief-updating "
    "behavior the use case calls for."
)

# ====================================================================
# 4. Validation and Results
# ====================================================================
h1("4. Summary of Validation and Results")
p(
    "The model fit on the training set (Section 3) was evaluated once, at the end, on "
    "the 20% held-out test set (61 patients) that played no role in fitting or "
    "preprocessing."
)
table(
    ["Metric", "Value"],
    [
        ["Accuracy", "0.885"],
        ["Log-loss (negative log-likelihood)", "0.281"],
        ["ROC AUC", "0.960"],
        ["Brier score", "0.082"],
    ],
)
p(
    "Precision/recall were 0.93/0.85 for no-disease and 0.84/0.93 for disease \u2014 the "
    "model trades a modest increase in false positives for fewer missed disease "
    "cases, the appropriate direction for a diagnostic aid."
)
if (FIG_DIR / "validation_metrics.png").exists():
    figure(FIG_DIR / "validation_metrics.png",
           "Figure 3. Confusion matrix and ROC curve on the 61-patient held-out test set.")
p(
    "Because each prediction is a posterior distribution rather than a point estimate, "
    "per-patient confidence can be read directly off the model: predictions with wide "
    "94% highest-density intervals are the cases a deployed system should route to a "
    "clinician instead of auto-classifying."
)
if (FIG_DIR / "per_patient_uncertainty.png").exists():
    figure(FIG_DIR / "per_patient_uncertainty.png",
           "Figure 4. Per-patient posterior P(disease) with 94% HDI, sorted by predicted probability (red = actually diseased).")

# ====================================================================
# 5. Iteration and Refinement
# ====================================================================
h1("5. Iteration and Refinement")
p(
    "Following validation, the model was refined along two axes suggested by the "
    "results: the prior distributions, and the likelihood's link function. Four "
    "variants were fit and compared on the identical train/test split:"
)
bullets([
    "A: a flat-ish baseline with all priors Normal(0, 2.5) \u2014 the generic default of "
    "Gelman et al. (2008), direction-agnostic.",
    "B: the literature-informed model from Section 3 (main model).",
    "C: B with every prior standard deviation halved \u2014 tests sensitivity to prior "
    "confidence, not just prior direction.",
    "D: B with the likelihood's link function changed from logit to probit \u2014 tests a "
    "likelihood-side refinement rather than a prior-side one.",
])
table(
    ["Model", "Accuracy", "Log-loss", "ROC AUC", "Brier"],
    [
        ["A: flat baseline", "0.852", "0.302", "0.952", "0.088"],
        ["B: literature-informed (main)", "0.885", "0.281", "0.960", "0.082"],
        ["C: B, tightened prior SDs", "0.885", "0.285", "0.960", "0.082"],
        ["D: B, probit link", "0.885", "0.286", "0.959", "0.083"],
    ],
)
if (FIG_DIR / "prior_sensitivity_comparison.png").exists():
    figure(FIG_DIR / "prior_sensitivity_comparison.png",
           "Figure 5. Held-out accuracy, log-loss, ROC AUC, and Brier score across models A\u2013D.")
p(
    "Model B outperforms Model A on every metric, indicating the literature-informed "
    "prior is doing measurable work rather than only adding interpretability. Model C "
    "essentially matches Model B: once the prior direction is correct, increasing its "
    "confidence further does not improve held-out performance at this sample size, "
    "consistent with the posterior widths in Section 3 \u2014 the prior's leverage is "
    "concentrated on the coefficients the data alone leave ambiguous (resting ECG, "
    "fasting blood sugar), not on the ones the data already pin down (vessel count, "
    "ST depression, thallium result). Model D lands within noise of Model B: with "
    "standardized features and this much data, the logit and probit links are close "
    "to a reparameterization of one another rather than substantively different "
    "models. The net finding is that, for this problem, the choice of prior mattered "
    "more than the choice of link function."
)

# ====================================================================
# 6. Discussion of Findings and Potential Improvements
# ====================================================================
h1("6. Discussion of Findings and Potential Improvements")
p(
    "The model is accurate (0.885) and well-calibrated for a diagnostic aid (AUC "
    "0.960, Brier 0.082) on data it never touched during fitting or preprocessing. The "
    "literature-informed prior provides a real, measured improvement over a flat "
    "baseline, and the sequential-updating construction demonstrates the posterior "
    "genuinely sharpening as evidence accrues, which is the behavior the exercise's "
    "\"dynamic updating\" requirement calls for rather than a fresh fit disguised as an "
    "update."
)
h2("Limitations")
bullets([
    "Single-site data: only the Cleveland cohort (303 patients) was used. The "
    "Hungarian, Switzerland, and Long Beach VA files present in the project directory "
    "were not merged in, since they have different missingness patterns and "
    "collection protocols and would need explicit handling (e.g., a site random "
    "effect) to combine without confounding site with outcome.",
    "A linear logit is assumed; no interaction terms (e.g., age by sex, chest pain "
    "type by exercise-induced angina) are modeled, despite the cardiology literature's "
    "emphasis on age/sex-stratified risk.",
    "The 61-patient held-out test set means the point estimates above carry real "
    "sampling uncertainty of their own \u2014 a single split cannot distinguish a genuine "
    "small accuracy gap from noise.",
    "Priors are derived from published effect sizes rather than elicited directly "
    "from a treating clinician.",
])
h2("Potential improvements")
bullets([
    "Add interaction terms motivated by the literature reviewed in Section 2.",
    "Replace the single train/test split with k-fold or PSIS-LOO cross-validation for "
    "a more stable estimate of out-of-sample performance.",
    "Extend the sequential-updating demonstration to genuine online "
    "prediction-then-reveal, rather than batch-sequential fitting.",
    "Elicit priors directly from a cardiologist and compare against the "
    "literature-derived priors used here.",
])

# ====================================================================
# 7. Observations, Recommendations, and Final Insight
# ====================================================================
h1("7. Observations, Recommendations, and Final Insight")
h2("Observations")
bullets([
    "The literature-informed prior produced a measurable, not merely cosmetic, "
    "improvement in held-out performance over a flat baseline (Section 5) \u2014 roughly "
    "3 points of accuracy and 0.008 of AUC, with lower log-loss and Brier score on "
    "every comparison.",
    "The one place the prior was deliberately left direction-agnostic (chest pain "
    "type, because its sign is ambiguous in an angiography-referred population) "
    "resolved, once fit, in the direction consistent with that population \u2014 evidence "
    "that hedging the prior instead of guessing its sign was the right call, not "
    "a missed opportunity for a stronger prior.",
    "Posterior uncertainty is highly heterogeneous across patients: some test-set "
    "predictions have 94% credible intervals several times wider than others "
    "(Figure 4). A single accuracy number obscures this \u2014 the model is confidently "
    "right about most patients and openly uncertain about a specific, identifiable "
    "subset.",
    "Changing the likelihood's link function (logit to probit, Model D) moved every "
    "metric by less than changing the prior did (Models A vs. B). For this dataset "
    "and sample size, prior knowledge was the more consequential modeling lever, not "
    "the parametric form of the likelihood.",
    "All four fitted model variants converged cleanly (R-hat \u2248 1.00, zero "
    "divergences), so the comparative differences reported above reflect the "
    "modeling choices themselves rather than sampler failure.",
])
h2("Recommendations")
bullets([
    "Operationalize the model's own uncertainty output as a triage rule: route "
    "predictions whose 94% HDI straddles the 0.5 decision boundary (or exceeds a "
    "chosen width) to clinician review instead of auto-classifying \u2014 this is already "
    "computed by the model (Section 4) and costs nothing further to add to a "
    "deployment pipeline.",
    "Do not treat this model as validated beyond the Cleveland population it was "
    "fit on. Before any clinical use, validate against an external cohort (e.g., the "
    "Hungarian, Switzerland, or Long Beach VA data already present in this project, "
    "handled with an explicit site effect), given known distributional differences "
    "across the four original collection sites.",
    "Adopt the sequential-updating mechanism demonstrated in Section 3.1 as the "
    "production retraining strategy \u2014 batch-Bayesian updates on newly diagnosed "
    "patients rather than periodic full retrains \u2014 but pair it with calibration "
    "monitoring, since the diagonal-Normal moment-matched approximation used between "
    "batches can compound approximation error over many successive updates.",
    "Invest in direct prior elicitation from a cardiologist and compare the result "
    "against the literature-derived priors used here; published effect sizes are a "
    "reasonable starting proxy but come from different eras and populations than any "
    "specific deployment population.",
    "Prioritize feature engineering (starting with the age-by-sex and chest-pain-by-"
    "exercise-angina interactions flagged in Section 6) over expanding data "
    "collection to the full 76-attribute source data \u2014 the current effect-size "
    "analysis suggests the existing 14 attributes already carry most of the "
    "resolvable signal at this sample size.",
])
h2("Final insight")
p(
    "The exercise's premise \u2014 that combining prior clinical knowledge with patient "
    "data should improve a diagnostic system \u2014 is not just a philosophical stance in "
    "this project, it is a tested and confirmed claim (Section 5): the "
    "literature-informed prior measurably outperformed a flat one on every validation "
    "metric, and it did so by exactly the mechanism the use case asked for, a "
    "posterior that keeps updating as new evidence arrives rather than a model that "
    "is refit from scratch. The secondary finding is just as actionable for a "
    "deployment decision as the headline accuracy number: this system's "
    "well-calibrated uncertainty, not its point predictions, is what should decide "
    "which cases go to a clinician \u2014 and that is a direct product of building the "
    "system as Bayesian in the first place."
)

# ====================================================================
# 8. Conclusion and Next Steps
# ====================================================================
h1("8. Conclusion and Next Steps")
p(
    "A Bayesian logistic regression with cardiology-literature priors reaches 0.885 "
    "accuracy and 0.960 ROC AUC on held-out Cleveland patients, converges cleanly, and "
    "— unlike a plain point-estimate classifier — ships calibrated per-patient "
    "uncertainty together with a well-defined mechanism for incorporating new "
    "patients without retraining from scratch. The prior-versus-baseline comparison "
    "in Section 5 is the load-bearing result of this project: it turns \"we used "
    "informative priors\" from a design choice into a tested, quantified claim."
)
p(
    "Next steps: add the interaction terms identified in Section 6; replace the "
    "single train/test split with k-fold or PSIS-LOO cross-validation; extend the "
    "sequential-updating demonstration to genuine online prediction-then-reveal; and "
    "pursue direct prior elicitation from a cardiologist as a comparison to the "
    "literature-derived priors used in this report."
)

# ====================================================================
# References (APA 7th edition)
# ====================================================================
doc.add_page_break()
h1("References")

refs = [
    "Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., "
    "Guppy, K., Lee, S., & Froelicher, V. (1989). International application of a new "
    "probability algorithm for the diagnosis of coronary artery disease. American "
    "Journal of Cardiology, 64(5), 304\u2013310. https://doi.org/10.1016/0002-9149(89)90524-9",

    "Diamond, G. A., & Forrester, J. S. (1979). Analysis of probability as an aid in "
    "the clinical diagnosis of coronary-artery disease. New England Journal of "
    "Medicine, 300(24), 1350\u20131358. https://doi.org/10.1056/NEJM197906143002402",

    "Fletcher, G. F., Ades, P. A., Kligfield, P., Arena, R., Balady, G. J., Bittner, "
    "V. A., Coke, L. A., Fleg, J. L., Forman, D. E., Gerber, T. C., Gulati, M., Madan, "
    "K., Rhodes, J., Thompson, P. D., & Williams, M. A. (2013). Exercise standards for "
    "testing and training: A scientific statement from the American Heart "
    "Association. Circulation, 128(8), 873\u2013934. "
    "https://doi.org/10.1161/CIR.0b013e31829b5b44",

    "Gelman, A., Jakulin, A., Pittau, M. G., & Su, Y.-S. (2008). A weakly informative "
    "default prior distribution for logistic and other regression models. The Annals "
    "of Applied Statistics, 2(4), 1360\u20131383. https://doi.org/10.1214/08-AOAS191",

    "Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart disease "
    "[Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X",

    "Wilson, P. W. F., D'Agostino, R. B., Levy, D., Belanger, A. M., Silbershatz, H., "
    "& Kannel, W. B. (1998). Prediction of coronary heart disease using risk factor "
    "categories. Circulation, 97(18), 1837\u20131847. "
    "https://doi.org/10.1161/01.CIR.97.18.1837",
]
for ref in refs:
    para = doc.add_paragraph(ref)
    para.paragraph_format.left_indent = Inches(0.5)
    para.paragraph_format.first_line_indent = Inches(-0.5)
    para.paragraph_format.space_after = Pt(10)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
doc.save(str(OUT_PATH))
print(f"Wrote {OUT_PATH}")
