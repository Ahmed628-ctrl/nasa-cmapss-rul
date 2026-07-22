import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from xgboost import XGBRegressor

# --------------------------------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AeroRUL | Turbofan Prognostics & Health Management",
    page_icon="🛩️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# CUSTOM CSS — aerospace / avionics-inspired professional theme
# --------------------------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    .main { background-color: #060a12; }
    .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }

    /* ---- Header / title banner ---- */
    .app-header {
        background: linear-gradient(120deg, #071427 0%, #0a1c33 45%, #071427 100%);
        border: 1px solid #123252;
        border-radius: 16px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.45);
    }
    .app-header h1 {
        margin: 0;
        font-size: 1.7rem;
        font-weight: 800;
        color: #eaf2ff;
        letter-spacing: 0.01em;
    }
    .app-header p {
        margin: 0.35rem 0 0 0;
        color: #7fa8d9;
        font-size: 0.92rem;
    }

    /* ---- KPI cards ---- */
    .kpi-card {
        background: linear-gradient(150deg, #0d1c30, #081221);
        border: 1px solid #16324f;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        text-align: left;
        box-shadow: 0 4px 14px rgba(0,0,0,0.35);
        height: 100%;
    }
    .kpi-label {
        color: #7fa0c4;
        font-size: 0.74rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        color: #f2f6fc;
        font-size: 1.85rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .kpi-sub {
        color: #4f79a8;
        font-size: 0.75rem;
        margin-top: 0.2rem;
    }

    /* ---- Health status badges (per-engine RUL state) ---- */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.03em;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-critical { background-color: #3d1418; color: #ff6b6b; border: 1px solid #ff6b6b; }
    .badge-warning  { background-color: #3d2f10; color: #ffb84d; border: 1px solid #ffb84d; }
    .badge-healthy  { background-color: #10331b; color: #4ade80; border: 1px solid #4ade80; }

    /* ---- Model-confidence badges (model quality, distinct palette) ---- */
    .conf-badge {
        display: inline-block;
        padding: 0.3rem 0.85rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        font-family: 'JetBrains Mono', monospace;
    }
    .conf-high { background-color: #07272e; color: #22d3ee; border: 1px solid #22d3ee; }
    .conf-mid  { background-color: #332a0c; color: #ffcf5c; border: 1px solid #ffcf5c; }
    .conf-low  { background-color: #341015; color: #ff8080; border: 1px solid #ff8080; }

    section[data-testid="stSidebar"] {
        background-color: #050810;
        border-right: 1px solid #10192a;
    }
    h1, h2, h3 { color: #eaf2ff !important; }
    h3 { font-size: 1.15rem !important; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    hr { border-color: #10192a; }
    .stCaption, [data-testid="stCaptionContainer"] { color: #4f79a8 !important; }

    /* ---- Diagnostic panel (error/help box) ---- */
    .diag-box {
        background: #150a0c;
        border: 1px solid #5c2027;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #ffb3b3;
        line-height: 1.55;
    }
    .diag-box b { color: #ff8080; }

    /* ---- Methodology / info panel ---- */
    .info-box {
        background: linear-gradient(150deg, #0d1c30, #081221);
        border: 1px solid #16324f;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        font-size: 0.88rem;
        color: #c7d7ec;
        line-height: 1.65;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------
RUL_MAX = 125          # matches the piecewise-linear clipping used in training
CRITICAL_THRESHOLD = 30
WARNING_THRESHOLD = 70

MODEL_PATH = "xgboost_rul_model.json"
FEATURES_PATH = "model_features.pkl"
TEST_SAMPLE_PATH = "processed_test_sample.csv"

# Model-confidence thresholds, based on R^2 against the held-out sample
CONF_HIGH_R2 = 0.85
CONF_MID_R2 = 0.70

# --------------------------------------------------------------------------
# LOAD DIAGNOSTICS
# --------------------------------------------------------------------------
def _inspect_file(path):
    """Read the first bytes of a file to identify common deployment failures
    (missing file, Git LFS pointer stub, empty/truncated download, etc.)."""
    if not os.path.exists(path):
        return f"File not found at '{path}'. Confirm it was committed/pushed to the repo."

    size = os.path.getsize(path)
    if size == 0:
        return f"File '{path}' exists but is 0 bytes (empty/failed download)."

    try:
        with open(path, "rb") as f:
            head = f.read(200)
    except Exception as e:
        return f"Could not read '{path}': {e}"

    if head.startswith(b"version https://git-lfs"):
        return (
            f"'{path}' is a **Git LFS pointer file**, not the real binary model "
            f"(only {size} bytes). Streamlit Community Cloud does not fetch LFS "
            f"objects automatically. Fix: either stop tracking this file with LFS "
            f"(remove it from `.gitattributes`, `git lfs untrack`, then re-commit "
            f"the real binary), or add a build step that runs `git lfs pull`."
        )

    if path == MODEL_PATH and size < 500:
        return (
            f"'{path}' is only {size} bytes — far smaller than a real XGBoost "
            f"model artifact. It is likely a corrupted or placeholder file."
        )

    return None

def _diagnose_load_failure(exc: Exception) -> str:
    """Translate a raised exception + file inspection into an actionable message."""
    msg = str(exc)
    lines = []

    model_issue = _inspect_file(MODEL_PATH)
    feat_issue = _inspect_file(FEATURES_PATH)
    sample_issue = _inspect_file(TEST_SAMPLE_PATH)

    if model_issue:
        lines.append(f"• <b>{MODEL_PATH}</b>: {model_issue}")
    if feat_issue:
        lines.append(f"• <b>{FEATURES_PATH}</b>: {feat_issue}")
    if sample_issue:
        lines.append(f"• <b>{TEST_SAMPLE_PATH}</b>: {sample_issue}")

    if not lines:
        lines.append(f"• {msg}")

    return "<br>".join(lines)

# --------------------------------------------------------------------------
# CACHED LOADERS
# --------------------------------------------------------------------------
@st.cache_resource
def load_model_and_features():
    """Load the trained XGBoost model natively and the exact feature list."""
    model = XGBRegressor()
    model.load_model(MODEL_PATH)

    feature_cols = joblib.load(FEATURES_PATH)
    return model, feature_cols

@st.cache_data
def load_test_sample():
    """Load the pre-processed sample engines used for the live demo."""
    return pd.read_csv(TEST_SAMPLE_PATH)

@st.cache_data
def get_feature_importance(_model, feature_cols):
    """Extract and sort XGBoost feature importances."""
    importances = _model.feature_importances_
    return (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

@st.cache_data
def compute_model_confidence(_model, feature_cols, df, has_actual_rul, sample_size=2000):
    """Compute RMSE / MAE / R^2 once, used to drive the model-confidence badge."""
    if not has_actual_rul:
        return None
    eval_df = df.head(sample_size).copy()
    X = eval_df[feature_cols]
    preds = np.clip(_model.predict(X), 0, RUL_MAX)
    actual = eval_df["RUL"].values
    rmse = float(np.sqrt(np.mean((preds - actual) ** 2)))
    mae = float(np.mean(np.abs(preds - actual)))
    ss_res = np.sum((actual - preds) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}

# --------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------
def get_health_status(rul_value):
    """Map a predicted RUL value to a maintenance status label, color, and icon."""
    if rul_value <= CRITICAL_THRESHOLD:
        return "CRITICAL", "#ff6b6b", "badge-critical", "🔴"
    elif rul_value <= WARNING_THRESHOLD:
        return "WARNING", "#ffb84d", "badge-warning", "🟠"
    else:
        return "HEALTHY", "#4ade80", "badge-healthy", "🟢"

def get_confidence_badge(r2):
    """Map the model's R^2 on the held-out sample to a confidence label/color."""
    if r2 is None or np.isnan(r2):
        return "NOT EVALUATED", "conf-mid"
    if r2 >= CONF_HIGH_R2:
        return f"HIGH CONFIDENCE (R² {r2:.2f})", "conf-high"
    elif r2 >= CONF_MID_R2:
        return f"MODERATE CONFIDENCE (R² {r2:.2f})", "conf-mid"
    else:
        return f"LOW CONFIDENCE — REVIEW (R² {r2:.2f})", "conf-low"

def make_gauge(rul_value, max_value=RUL_MAX):
    """Build a professional gauge chart for the predicted RUL."""
    _, color, _, _ = get_health_status(rul_value)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rul_value,
        number={"suffix": " cycles", "font": {"size": 40, "color": "#f2f4f8", "family": "JetBrains Mono"}},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, max_value], "tickcolor": "#8b93a7", "tickfont": {"color": "#8b93a7"}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#0d1c30",
            "borderwidth": 0,
            "steps": [
                {"range": [0, CRITICAL_THRESHOLD], "color": "#3d1418"},
                {"range": [CRITICAL_THRESHOLD, WARNING_THRESHOLD], "color": "#3d2f10"},
                {"range": [WARNING_THRESHOLD, max_value], "color": "#10331b"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": rul_value,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f2f4f8"},
        height=320,
        margin=dict(l=30, r=30, t=30, b=10),
    )
    return fig

def predict_rul(model, feature_cols, row_df):
    """Run a single-row (or batch) prediction, aligning columns to the training feature order."""
    X = row_df[feature_cols]
    preds = model.predict(X)
    return np.clip(preds, 0, RUL_MAX)

def kpi_card(label, value, sub=None):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)

def app_header(title, subtitle):
    st.markdown(f"""
    <div class="app-header">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# LOAD ARTIFACTS
# --------------------------------------------------------------------------
try:
    model, feature_cols = load_model_and_features()
    test_df = load_test_sample()

    # Guard against the model/feature-list and the sample CSV coming from two
    # different notebook runs (e.g. correlation-based feature selection can
    # produce a slightly different feature set between runs). Fail loudly and
    # specifically instead of crashing deep inside a prediction call.
    missing_features = [c for c in feature_cols if c not in test_df.columns]
    if missing_features:
        raise ValueError(
            f"'{TEST_SAMPLE_PATH}' is missing {len(missing_features)} feature column(s) "
            f"that the model expects: {missing_features}. This happens when the CSV and "
            f"the model/feature-list ('{MODEL_PATH}', '{FEATURES_PATH}') were exported "
            f"from different runs of the notebook. Re-run the notebook top to bottom once, "
            f"then save all three artifacts at the very end of that same run before "
            f"re-uploading them together."
        )

    load_error = None
except Exception as e:
    model, feature_cols, test_df = None, None, None
    load_error = _diagnose_load_failure(e)

# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛩️ AeroRUL")
    st.markdown("**Turbofan Engine Prognostics & Health Management**")
    st.caption("NASA C-MAPSS · XGBoost Regression")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🏠 Fleet Overview",
            "🔧 Engine Diagnostics",
            "🧪 Sensitivity Simulator",
            "📊 Batch Prediction",
            "📈 Model Insights & Validation",
        ],
    )

    st.markdown("---")
    st.markdown("### Model Specification")
    st.markdown(
        """
        - **Algorithm:** XGBoost Regressor
        - **Target:** Remaining Useful Life (RUL)
        - **Training data:** Merged FD001–FD004
        - **RUL cap:** 125 cycles (piecewise-linear degradation)
        - **Selected over:** Linear Regression, Decision Tree, Random Forest
        """
    )

    with st.expander("ℹ️ Methodology"):
        st.markdown(
            """
            <div class="info-box">
            Sensor streams from all four C-MAPSS sub-datasets were merged into a single
            fleet, tagged with a unique engine identity, then enriched with rolling
            mean/std features per sensor to capture degradation trends. Near-constant
            sensors and highly correlated features (&gt; 0.95) were pruned before
            training. Four regressors were compared on a held-out, engine-level split
            (Linear Regression, Decision Tree, Random Forest, XGBoost); XGBoost was
            selected for deployment based on test-set accuracy.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    if st.button("🔄 Reload data & clear cache"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    st.caption("Built as part of a Machine Learning graduation project.")

if load_error:
    st.markdown(f"""
    <div class="diag-box">
    <b>⚠️ Model artifacts failed to load.</b><br><br>
    {load_error}
    </div>
    """, unsafe_allow_html=True)
    st.stop()

has_unit_id = "unit_id" in test_df.columns
has_cycle = "cycle" in test_df.columns
has_actual_rul = "RUL" in test_df.columns

confidence = compute_model_confidence(model, feature_cols, test_df, has_actual_rul)
conf_label, conf_class = get_confidence_badge(confidence["r2"] if confidence else None)

# Persistent model-confidence badge in the sidebar (visible on every page)
with st.sidebar:
    st.markdown("### Model Confidence")
    st.markdown(f'<span class="conf-badge {conf_class}">{conf_label}</span>', unsafe_allow_html=True)
    if confidence:
        st.caption(f"RMSE {confidence['rmse']:.1f} cycles · MAE {confidence['mae']:.1f} cycles")

# ==========================================================================
# PAGE: FLEET OVERVIEW
# ==========================================================================
if page == "🏠 Fleet Overview":
    app_header(
        "Turbofan Fleet — Remaining Useful Life Overview",
        "Estimated operating cycles remaining before maintenance is required, "
        "trained on NASA's C-MAPSS dataset (FD001–FD004, six operating conditions)."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Algorithm", "XGBoost", "Gradient-boosted trees")
    with c2:
        kpi_card("Input Features", f"{len(feature_cols)}", "sensor + rolling stats")
    with c3:
        n_engines = test_df["unit_id"].nunique() if has_unit_id else len(test_df)
        kpi_card("Engines in Sample", f"{n_engines}")
    with c4:
        kpi_card("RUL Cap", f"{RUL_MAX}", "cycles (piecewise-linear)")
    with c5:
        kpi_card("Model Confidence", conf_label.split(" (")[0], conf_label[conf_label.find("("):] if "(" in conf_label else "")

    st.markdown("###")
    st.subheader("Fleet Health — Predicted RUL by Engine")

    preview = test_df.head(1000).copy()  # subset kept for responsiveness
    preview["Predicted_RUL"] = predict_rul(model, feature_cols, preview).round(1)
    preview["Status"] = preview["Predicted_RUL"].apply(lambda v: get_health_status(v)[0])

    if has_unit_id:
        fleet = preview.groupby("unit_id", as_index=False)["Predicted_RUL"].min()
        fleet["Status"] = fleet["Predicted_RUL"].apply(lambda v: get_health_status(v)[0])
    else:
        fleet = preview[["Predicted_RUL", "Status"]].reset_index().rename(columns={"index": "unit_id"})

    color_map = {"CRITICAL": "#ff6b6b", "WARNING": "#ffb84d", "HEALTHY": "#4ade80"}

    col_bar, col_hist = st.columns([1.6, 1])
    with col_bar:
        fig = px.bar(
            fleet.sort_values("Predicted_RUL"),
            x="unit_id", y="Predicted_RUL", color="Status",
            color_discrete_map=color_map,
            labels={"unit_id": "Engine", "Predicted_RUL": "Predicted RUL (cycles)"},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f2f4f8"}, height=420,
            xaxis={"showticklabels": False},
            legend_title_text="",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_hist:
        hist_fig = px.histogram(
            fleet, x="Predicted_RUL", color="Status", nbins=25,
            color_discrete_map=color_map,
            labels={"Predicted_RUL": "Predicted RUL (cycles)"},
        )
        hist_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f2f4f8"}, height=420,
            legend_title_text="", showlegend=False,
            title="Distribution of Predicted RUL",
        )
        st.plotly_chart(hist_fig, use_container_width=True)

    n_crit = (fleet["Status"] == "CRITICAL").sum()
    n_warn = (fleet["Status"] == "WARNING").sum()
    n_healthy = (fleet["Status"] == "HEALTHY").sum()
    st.info(
        f"🔴 **{n_crit}** require immediate maintenance  |  🟠 **{n_warn}** need monitoring  |  "
        f"🟢 **{n_healthy}** healthy — out of **{len(fleet)}** engines in this sample."
    )

    csv_out = fleet.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Fleet Health Report (CSV)",
        data=csv_out,
        file_name="fleet_health_report.csv",
        mime="text/csv",
    )

# ==========================================================================
# PAGE: ENGINE DIAGNOSTICS
# ==========================================================================
elif page == "🔧 Engine Diagnostics":
    app_header("Single Engine Diagnostics", "Inspect predicted health for one engine from the fleet sample.")

    if has_unit_id:
        selected_unit = st.selectbox("Select Engine (unit_id)", sorted(test_df["unit_id"].unique()[:200]))
        unit_rows = test_df[test_df["unit_id"] == selected_unit].copy()
        if has_cycle:
            unit_rows = unit_rows.sort_values("cycle")
        current_row = unit_rows.iloc[[-1]]
    else:
        idx = st.selectbox("Select Row Index", test_df.head(200).index)
        unit_rows = test_df.loc[[idx]]
        current_row = unit_rows

    predicted_rul = float(predict_rul(model, feature_cols, current_row)[0])
    status_label, status_color, badge_class, icon = get_health_status(predicted_rul)

    col_gauge, col_info = st.columns([1.2, 1])

    with col_gauge:
        st.plotly_chart(make_gauge(predicted_rul), use_container_width=True)

    with col_info:
        st.markdown(f"""
        <div class="kpi-card" style="margin-bottom:1rem;">
            <div class="kpi-label">Predicted Remaining Useful Life</div>
            <div class="kpi-value">{predicted_rul:.1f} cycles</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f'<span class="status-badge {badge_class}">{icon} {status_label}</span>'
            f'&nbsp;&nbsp;<span class="conf-badge {conf_class}">{conf_label}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("###")

        if status_label == "CRITICAL":
            st.error("⚠️ Immediate maintenance recommended. Engine is close to end-of-life.")
        elif status_label == "WARNING":
            st.warning("🟠 Schedule maintenance soon. Degradation trend detected.")
        else:
            st.success("✅ Engine operating within normal parameters.")

        if has_actual_rul:
            actual_rul = float(current_row["RUL"].values[0])
            error = predicted_rul - actual_rul
            st.metric("Actual RUL (ground truth)", f"{actual_rul:.1f} cycles",
                       delta=f"{error:+.1f} prediction error")

    if has_cycle and len(unit_rows) > 1:
        st.markdown("---")
        st.subheader("Sensor Trend Over Observed Cycles")
        plottable = [c for c in feature_cols if c in unit_rows.columns and "roll" not in c][:4]
        if plottable:
            sensor_choice = st.multiselect("Sensors to display", plottable, default=plottable[:2])
            if sensor_choice:
                trend_fig = px.line(
                    unit_rows, x="cycle", y=sensor_choice,
                    labels={"cycle": "Cycle", "value": "Sensor Reading"},
                )
                trend_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#f2f4f8"}, height=380,
                )
                st.plotly_chart(trend_fig, use_container_width=True)

    with st.expander("View raw feature values used for this prediction"):
        st.dataframe(current_row[feature_cols].T.rename(columns={current_row.index[0]: "value"}))

# ==========================================================================
# PAGE: SENSITIVITY SIMULATOR
# ==========================================================================
elif page == "🧪 Sensitivity Simulator":
    app_header("What-If Sensor Simulator", "Adjust the most influential sensor readings and watch the predicted RUL update live. All other features are held at the dataset median.")

    importance_df = get_feature_importance(model, feature_cols)
    top_features = importance_df.head(8)["feature"].tolist()

    baseline = test_df[feature_cols].median(numeric_only=True)

    st.markdown("### Adjustable Parameters (Top Influential Features)")
    sim_values = baseline.copy()

    cols = st.columns(2)
    for i, feat in enumerate(top_features):
        col = cols[i % 2]
        f_min = float(test_df[feat].min())
        f_max = float(test_df[feat].max())
        f_default = float(baseline[feat])
        with col:
            sim_values[feat] = st.slider(
                feat, min_value=f_min, max_value=f_max, value=f_default,
                key=f"sim_{feat}",
            )

    sim_row = pd.DataFrame([sim_values])[feature_cols]
    sim_pred = float(predict_rul(model, feature_cols, sim_row)[0])
    status_label, status_color, badge_class, icon = get_health_status(sim_pred)

    st.markdown("---")
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.plotly_chart(make_gauge(sim_pred), use_container_width=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Simulated Predicted RUL</div>
            <div class="kpi-value">{sim_pred:.1f} cycles</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(
            f'<br><span class="status-badge {badge_class}">{icon} {status_label}</span>',
            unsafe_allow_html=True,
        )
        st.caption(
            "This simulator demonstrates the model's sensitivity to its most important "
            "features, as ranked by XGBoost's built-in feature importance."
        )
        if st.button("↺ Reset to dataset median"):
            for feat in top_features:
                st.session_state.pop(f"sim_{feat}", None)
            st.rerun()

# ==========================================================================
# PAGE: BATCH PREDICTION
# ==========================================================================
elif page == "📊 Batch Prediction":
    app_header("Batch Prediction", "Upload a CSV containing the required feature columns to score multiple engines at once.")

    with st.expander("Required columns"):
        st.code(", ".join(feature_cols))

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    use_demo = st.checkbox("Use the built-in test sample instead", value=not bool(uploaded_file))

    batch_df = None
    if uploaded_file is not None and not use_demo:
        batch_df = pd.read_csv(uploaded_file)
    elif use_demo:
        batch_df = test_df.head(1000).copy()

    if batch_df is not None:
        missing_cols = [c for c in feature_cols if c not in batch_df.columns]
        if missing_cols:
            st.error(f"❌ Uploaded file is missing required columns: {missing_cols}")
        else:
            batch_df["Predicted_RUL"] = predict_rul(model, feature_cols, batch_df).round(2)
            batch_df["Status"] = batch_df["Predicted_RUL"].apply(lambda v: get_health_status(v)[0])

            st.success(f"✅ Scored {len(batch_df)} rows successfully.")

            m1, m2, m3 = st.columns(3)
            with m1:
                kpi_card("Rows Scored", f"{len(batch_df)}")
            with m2:
                kpi_card("Avg Predicted RUL", f"{batch_df['Predicted_RUL'].mean():.1f}", "cycles")
            with m3:
                kpi_card("Critical Engines", f"{(batch_df['Status'] == 'CRITICAL').sum()}")

            id_cols = [c for c in ["unit_id", "cycle"] if c in batch_df.columns]
            display_cols = id_cols + ["Predicted_RUL", "Status"]

            def color_status(val):
                if val == 'CRITICAL': return 'color: #ff6b6b'
                elif val == 'WARNING': return 'color: #ffb84d'
                elif val == 'HEALTHY': return 'color: #4ade80'
                return ''

            st.dataframe(
                batch_df[display_cols].style.map(color_status, subset=['Status']),
                use_container_width=True,
                height=420,
            )

            csv_out = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions as CSV",
                data=csv_out,
                file_name="rul_predictions.csv",
                mime="text/csv",
            )

# ==========================================================================
# PAGE: MODEL INSIGHTS & VALIDATION
# ==========================================================================
elif page == "📈 Model Insights & Validation":
    app_header("Model Insights & Validation", "Explainability and performance evidence supporting the model-confidence rating shown throughout the app.")

    st.markdown(f'<span class="conf-badge {conf_class}" style="font-size:0.95rem;">{conf_label}</span>', unsafe_allow_html=True)
    st.markdown("###")

    importance_df = get_feature_importance(model, feature_cols)

    st.subheader("Top 15 Feature Importances (XGBoost)")
    top15 = importance_df.head(15)
    fig = px.bar(
        top15.sort_values("importance"),
        x="importance", y="feature", orientation="h",
        color="importance", color_continuous_scale="Teal",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f2f4f8"}, height=520, coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        **Why this matters:** operating condition context (`op_setting_1/2/3`) combined with raw
        and rolling sensor statistics allows XGBoost to learn condition-dependent degradation
        patterns without explicit regime clustering — the core scientific argument of this project.
        """
    )

    if has_actual_rul and confidence:
        st.markdown("---")
        st.subheader("Model Performance on Test Sample")

        m1, m2, m3 = st.columns(3)
        with m1:
            kpi_card("RMSE", f"{confidence['rmse']:.2f} cycles")
        with m2:
            kpi_card("MAE", f"{confidence['mae']:.2f} cycles")
        with m3:
            kpi_card("R² Score", f"{confidence['r2']:.3f}")

        eval_df = test_df.head(2000).copy()
        preds = predict_rul(model, feature_cols, eval_df)
        actual = eval_df["RUL"].values
        residuals = actual - preds

        col_scatter, col_resid = st.columns(2)

        with col_scatter:
            scatter_fig = px.scatter(
                x=actual, y=preds, opacity=0.4,
                labels={"x": "Actual RUL", "y": "Predicted RUL"},
                title="Predicted vs Actual RUL",
            )
            scatter_fig.add_shape(
                type="line", x0=0, y0=0, x1=RUL_MAX, y1=RUL_MAX,
                line=dict(color="white", dash="dash"),
            )
            scatter_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#f2f4f8"}, height=420,
            )
            st.plotly_chart(scatter_fig, use_container_width=True)

        with col_resid:
            resid_fig = px.histogram(
                x=residuals, nbins=40,
                labels={"x": "Residual (Actual − Predicted)"},
                title="Residual Distribution",
            )
            resid_fig.add_vline(x=0, line_dash="dash", line_color="white")
            resid_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#f2f4f8"}, height=420,
            )
            st.plotly_chart(resid_fig, use_container_width=True)
    else:
        st.info("Ground-truth RUL is not available in this sample, so performance metrics and residual plots are not shown.")

# --------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "NASA C-MAPSS Remaining Useful Life Prediction — Machine Learning Graduation Project | "
    "Model: XGBoost Regressor | Deployment: Streamlit"
)
