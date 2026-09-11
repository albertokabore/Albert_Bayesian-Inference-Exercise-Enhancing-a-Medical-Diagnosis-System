# Bayesian Inference Exercise: Enhancing a Medical Diagnosis System

A Bayesian logistic regression model for coronary artery disease (CAD) diagnosis,
built with PyMC on the UCI/Cleveland Clinic heart-disease dataset. Prior distributions
are informed by the cardiology literature (see the notebook, Section 2) rather than
left flat, and the model is shown updating its beliefs sequentially as new patient
batches arrive.

## Dataset

`processed.cleveland.data`, already present in this repository — the cleaned,
14-attribute subset of the UCI Heart Disease dataset (303 patients, Cleveland Clinic
Foundation). No external or synthetic data is used. See `heart-disease.names` for the
full attribute documentation and `WARNING` / `ask-detrano` for known data-quality notes
from the original donors.

Source: Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S.,
Guppy, K., Lee, S., & Froelicher, V. (1989). International application of a new
probability algorithm for the diagnosis of coronary artery disease. *American Journal
of Cardiology*, 64, 304-310.

## Deliverables

- **Code**: `notebooks/bayesian_heart_disease_diagnosis.ipynb` — data processing, model
  development, training, and validation, modular and commented, runs end-to-end.
- **Final report**: `reports/REPORT.docx` — the submission-ready report (Introduction,
  Dataset & Priors, Model & Assumptions, Validation & Results, Iteration &
  Refinement, Discussion, Observations/Recommendations/Final Insight, Conclusion &
  Next Steps, APA 7 references), with embedded posterior-distribution, ROC-curve, and
  performance-table figures. `reports/REPORT.md` is its plain-text source.

## Project layout

```
src/data_preprocessing.py                          data loading, cleaning, encoding
notebooks/bayesian_heart_disease_diagnosis.ipynb    code deliverable: full analysis
scripts/build_notebook.py                           (re)generates the notebook
scripts/build_report_docx.py                        (re)generates reports/REPORT.docx
reports/REPORT.docx                                 final report deliverable (.docx)
reports/REPORT.md                                   plain-text source for the report
reports/figures/                                     figures embedded in the report
requirements.txt                                    Python dependencies
```

## Setup

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook notebooks\bayesian_heart_disease_diagnosis.ipynb
```

To regenerate the report after changing the analysis: re-run the notebook end-to-end
(refreshes `reports/figures/*.png`), then `python scripts\build_report_docx.py`.

## Contents of the notebook

1. Data collection & preprocessing
2. Prior knowledge integration (literature review, with citations)
3. Bayesian model development (PyMC logistic regression) + dynamic/sequential updating
4. Sequential updating demo
5. Model validation (accuracy, log-likelihood, confusion matrix, ROC/AUC)
6. Iteration and refinement (prior sensitivity + likelihood link-function comparison)
7. Discussion
8. References (APA)
