# AI-Based Detection and Forecasting of Seasonal Tourism Imbalance in Saudi Arabia

CS316 – Introduction to AI and Data Science, Prince Sultan University (Term 261)
Team: Fathima Mohamed · Aljoharah Albawardi · Rawand Alshoum — Instructor: Dr. Heyfa Ammar

## 1. Project overview
Tourism demand in Saudi Arabia differs between cities and months. Peaks cause crowding and strain on
services; low seasons waste capacity. This project:

1. **Measures seasonal concentration** of domestic and inbound tourism for 41 destinations
   (coefficient of variation, top-3-month share, peak/low ratio, seasonal index).
2. **Detects anomalies** (unusual months after removing trend and seasonality) with z-scores, and
   explains them (Hijri-calendar shifts, COVID-19, survey noise).
3. **Forecasts monthly tourists one month ahead** for each destination and compares a previous-month
   baseline, **Linear Regression**, **Linear Regression with PCA** and **Random Forest Regression**,
   using expanding-window time-series cross-validation and a final 2025 test year.

Sustainability link: SDG 8 (stable jobs), SDG 11 (sustainable cities), SDG 12 (responsible consumption).

## 2. Dataset
* **Source:** Ministry of Tourism, Data and Decision Support Deputy – *Tourism Demand Indicators in KSA*,
  open data: <https://mt.gov.sa/tic/open-data> (workbook last updated 12-04-2026).
* **Tables used:** `D5` Domestic tourists number & spending by destination (main cities/governorates) and
  `I6` Inbound tourists number & spending by destination. Monthly, January 2021 – December 2025.
* **File:** `data/DDS_MT_Master_Sheet_2021_2025.xlsx` (the original, unmodified workbook).
* After reshaping: 4,920 rows = 2 tourist types × 41 destinations × 60 months; variables: date, tourist
  type, province, destination, tourists (overnight visitors), spending (SAR).

## 3. Key results (produced by the notebooks, all seeds fixed)
| | Value |
|---|---|
| Selected model (chosen on 2023–2024 cross-validation only) | Random Forest, 300 trees, `min_samples_leaf=2`, all features |
| Cross-validated WAPE (2023–2024) | 22.91 % (Linear Regression 24.42 %, LR + PCA 23.44 %, previous-month baseline 26.81 %) |
| 2025 test WAPE, all destination-months | 20.50 % → forecast accuracy **79.50 %** |
| 2025 test accuracy, domestic destinations / inbound destinations | **83.36 %** / 66.33 % |
| 2025 test accuracy, national level (sum of 52 series) | **86.44 %** |
| 2025 test MAE / RMSE / R² | 39,522 tourists / 133,327 / 0.85 |

2025 test accuracy of the other models: Linear Regression 77.75 %, Linear Regression + PCA 76.72 %,
previous-month baseline 77.17 %.

*Forecast accuracy = 100 % − WAPE, where WAPE = Σ|actual − forecast| / Σ actual. It is the share of
tourists forecast correctly, **not** classification accuracy.* Full tables: `outputs/model_comparison.csv`.

## 4. Repository structure
```
CS316_Tourism_Project/
├── README.md                     this file
├── requirements.txt              Python packages
├── data/
│   ├── README.md                 where the dataset comes from / where to put it
│   ├── DDS_MT_Master_Sheet_2021_2025.xlsx   original Ministry of Tourism workbook
│   └── processed/                tidy CSVs written by notebook 01
├── src/
│   └── tourism.py                shared code: loading, memory optimisation, calendar & lag features, metrics
├── notebooks/
│   ├── 01_Data_Preparation_and_EDA.ipynb          loading, missing data, memory optimisation, EDA
│   ├── 02_Seasonality_and_Anomaly_Detection.ipynb seasonal concentration, peak months, z-score anomalies
│   └── 03_Tourism_Forecasting.ipynb               features, time-series CV, models, final 2025 test
├── outputs/
│   ├── figures/                  all figures (PNG, 200 dpi) used in report and slides
│   ├── model_comparison.csv      CV + train + 2025 test metrics of the best model of each family
│   ├── cv_results_all_models.csv per-fold validation results
│   ├── test_predictions_selected_model.csv
│   ├── seasonality_concentration.csv, anomalies.csv, memory_optimization.csv
├── report/                       IEEE-style report (draft) + PDF
└── presentation/                 10-slide presentation
```

## 5. How to run
### Option A – Google Colab (recommended)
1. Open <https://colab.research.google.com> → **File → Open notebook → GitHub**, paste this repository's URL
   and open `notebooks/01_Data_Preparation_and_EDA.ipynb`.
2. In the first code cell set `REPO_URL = "https://github.com/<user>/CS316_Tourism_Project.git"`
   (the cell clones the repository, so the data and `src/tourism.py` are available).
   *Without GitHub:* upload the whole `CS316_Tourism_Project` folder to Google Drive (`MyDrive`), leave
   `REPO_URL` unchanged and the cell mounts Drive instead (edit `DRIVE_PATH` if needed).
3. **Runtime → Run all.** Repeat for notebooks 02 and 03 (in this order; 03 took ~30 s on a 4-core CPU, expect longer on Colab).
4. Results appear in `outputs/`. In Colab, download them from the file browser (left panel) or
   copy them to Drive.

All required packages are pre-installed in Colab; no `pip install` is needed.

### Option B – local computer
```bash
git clone https://github.com/<user>/CS316_Tourism_Project.git
cd CS316_Tourism_Project
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/
```
Run the notebooks in order 01 → 02 → 03 (each one is independent of the others' outputs, because
they all load the data through `src/tourism.py`, but this order follows the story of the project).

## 6. Methods in brief
* **No data leakage:** forecasts for month *t* use only months ≤ *t−1*; Ramadan/Hajj dates are known in
  advance; scaler, encoder and PCA are fitted inside a scikit-learn `Pipeline` on the training fold only;
  the 2025 test year was used once, after the model was selected.
* **Missing data:** blanks are kept as missing (not zero); only fully observed series (52 of 82, covering
  99.7 % of domestic and 95.2 % of inbound tourists) are modelled; targets are never imputed.
* **Validation:** expanding-window CV with four half-year folds (2023 H1 – 2024 H2), selection by WAPE,
  ties within 0.5 pp resolved in favour of the simpler model.
* **Models:** previous-month baseline (no learning) · Linear Regression (least squares on log tourists;
  5 feature sets compared) · PCA + Linear Regression (80/90/95/99 % of variance kept) · Random Forest
  (300 averaged decision trees, `min_samples_leaf` ∈ {2, 5, 10}). Each is explained in notebook 03.
* **Memory optimisation:** category and down-cast numeric dtypes reduce the tidy table by ~87 %
  (`outputs/memory_optimization.csv`); see notebook 01 §5 and the report appendix.

## 7. Limitations
Short post-COVID history; city-level survey estimates have a larger margin of error; inbound
destination forecasts remain difficult (66 % accuracy); one-month-ahead setting only; the pre-Hajj
window is an approximation; Random Forest shows a gap between training (9.6 % WAPE) and test
(20.5 %) error. An earlier version of the project also evaluated Ridge, Gradient Boosting and an
ensemble on 2025 before the model set was reduced to course methods, so the 2025 test is not fully
untouched. Details: notebook 03 §13.

## 8. Course topics → implementation
| Course topic | Where | Evidence |
|---|---|---|
| Pandas import/reshape, grouping | `src/tourism.py`, NB01 §2 | tidy table, totals check |
| Summaries & visualisation | NB01 §6 | histograms, box plots, heatmaps, ECDF, scatter |
| Missing data | NB01 §4 | missingness tables/heatmap, exclusion rule |
| Mean, standard deviation, proportions | NB02 §1–2 | coefficient of variation, top-3 share, seasonal index |
| Standardisation / z-scores, outliers | NB02 §3 | residual z-scores, |z| > 3 anomalies |
| Normalisation | NB03 §5–6 | `StandardScaler` in the pipeline; PCA with vs. without scaling |
| Statistical learning (train/val/test, overfitting) | NB03 §2, §8–9 | expanding-window CV, 2025 test, train vs. test error |
| Predictive modelling, MAE/RMSE, gradient descent | NB03 §3, §12 | metrics table, GD = sklearn weights |
| Linear Regression | NB03 §5 | feature-group comparison, coefficients |
| One-hot encoding | NB03 §3 | `OneHotEncoder` for province |
| PCA | NB03 §6 | 80/90/95/99 % variance experiments, explained-variance plot |
| Extension (beyond lectures) | NB03 §7 | Random Forest Regression |
