# 🛩️ AeroRUL — Turbofan Engine Remaining Useful Life Prediction

**AeroRUL** is an end-to-end machine learning project that predicts the **Remaining Useful Life (RUL)** of turbofan jet engines from multivariate sensor data, and serves the trained model through an interactive **Streamlit** dashboard for fleet health monitoring and predictive maintenance.

> Built on NASA's C-MAPSS dataset · XGBoost Regression · Deployed with Streamlit

---

## 📌 Overview

Aircraft engines degrade gradually over their operating life. Being able to estimate how many operating cycles an engine has left — before it needs maintenance — is a classic **predictive maintenance** problem. This project:

1. Merges NASA's four C-MAPSS turbofan degradation sub-datasets (FD001–FD004) into a single fleet-level dataset.
2. Engineers rolling statistical features from raw sensor readings to capture degradation trends.
3. Trains and compares several regression models to predict RUL.
4. Deploys the best-performing model (**XGBoost**) in a production-style Streamlit web app with fleet monitoring, per-engine diagnostics, a what-if sensor simulator, batch scoring, and model-explainability views.

---

## 🗂️ Dataset

- **Source:** [NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) (also available directly from NASA's Prognostics Data Repository)
- **Subsets used:** FD001, FD002, FD003, FD004 — covering different combinations of operating conditions and fault modes
- **Structure:** each row is one engine at one operating cycle, with 3 operating-condition settings and 21 sensor readings
- **Target:** Remaining Useful Life (RUL), computed as `max_cycle_for_engine − current_cycle`, clipped at 125 cycles to reflect the standard piecewise-linear degradation assumption used in C-MAPSS literature

---

## 🔬 Methodology

| Stage | What was done |
|---|---|
| **Data merging** | All 4 sub-datasets combined into one fleet, with a unique `unit_id` per engine (dataset + engine number) |
| **Cleaning** | Missing-value check; near-constant (zero-variance) sensors dropped |
| **Feature engineering** | Rolling mean & standard deviation (window = 5 cycles) computed per sensor to capture degradation trends |
| **Feature selection** | Highly correlated features (Pearson r > 0.95) pruned to reduce redundancy and overfitting risk, while operating-condition settings are always kept |
| **Target construction** | RUL computed per engine and clipped at 125 cycles |
| **Train/test split** | Grouped split **by engine (`unit_id`)**, not by row, to prevent data leakage between an engine's own cycles |
| **Scaling** | StandardScaler applied for the linear model; tree-based models trained on unscaled features |
| **Model comparison** | Linear Regression, Decision Tree, Random Forest, and XGBoost trained and evaluated on the held-out engines |
| **Model selection** | **XGBoost** selected for deployment based on test-set accuracy |

### Reported test performance (XGBoost, held-out sample)

| Metric | Value |
|---|---|
| R² | 0.82 |
| RMSE | 17.8 cycles |
| MAE | 13.3 cycles |

---

## 🖥️ The App

The Streamlit app (`app.py`) is organized into 5 pages:

- **🏠 Fleet Overview** — KPIs, fleet-wide predicted RUL bar chart and distribution, health status breakdown (Healthy / Warning / Critical), downloadable fleet health report
- **🔧 Engine Diagnostics** — Deep dive into a single engine: RUL gauge, health status, sensor trend charts, raw feature values
- **🧪 Sensitivity Simulator** — Interactive sliders on the model's most influential features to explore "what-if" scenarios and see the predicted RUL update live
- **📊 Batch Prediction** — Upload a CSV to score many engines at once, with a downloadable results file
- **📈 Model Insights & Validation** — Feature importance, predicted-vs-actual scatter plot, and residual distribution

The app also includes built-in diagnostics that detect common deployment issues (missing files, Git LFS pointer stubs, mismatched feature lists between the model and the sample data) and surface a clear, actionable message instead of crashing silently.

---

## 🧰 Tech Stack

- **Data processing & modeling:** Python, Pandas, NumPy, Scikit-learn, XGBoost
- **Visualization (notebook):** Matplotlib, Seaborn
- **Visualization (app):** Plotly
- **Web app / deployment:** Streamlit
- **Model persistence:** native XGBoost JSON format + Joblib for the feature list

---

## 📁 Project Structure

```
.
├── app.py                         # Streamlit web application
├── notebook.ipynb                 # Data processing, feature engineering, model training & evaluation
├── xgboost_rul_model.json         # Trained XGBoost model (native, version-stable format)
├── model_features.pkl             # Exact list/order of feature columns used at training time
├── processed_test_sample.csv      # Pre-processed sample of test engines used by the app
└── requirements.txt                # Python dependencies
```

---

## 🚀 Running Locally

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app expects `xgboost_rul_model.json`, `model_features.pkl`, and `processed_test_sample.csv` to be present in the same directory as `app.py`.

---

## ☁️ Deployment Notes

- Deployed on **Streamlit Community Cloud**.
- ⚠️ `xgboost_rul_model.json`, `model_features.pkl`, and `processed_test_sample.csv` must always be regenerated **together, in a single top-to-bottom run of the notebook**, then committed together. Because the feature-selection step is data-driven (correlation-based pruning), regenerating them separately can produce a mismatched feature list between the model and the sample data.
- If large files are tracked with Git LFS, make sure the deployment platform actually pulls the LFS objects — otherwise only pointer stub files get deployed.

---

## 🎓 About

This project was built as a machine learning portfolio / graduation project, demonstrating a complete workflow from raw multivariate sensor data to a deployed, interactive predictive-maintenance tool.
