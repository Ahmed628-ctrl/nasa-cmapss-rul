

import joblib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------------------------------
st.set_page_config(
    page_title=RUL Predictive Maintenance  XGBoost,
    page_icon=✈️,
    layout=wide,
    initial_sidebar_state=expanded,
)

# --------------------------------------------------------------------------
# CUSTOM CSS — professional, industrial dashboard look
# --------------------------------------------------------------------------
st.markdown(
style
    .main { background-color #0e1117; }
    .block-container { padding-top 2rem; padding-bottom 2rem; }

    .kpi-card {
        background linear-gradient(145deg, #1a1f2b, #12151d);
        border 1px solid #2a2f3a;
        border-radius 14px;
        padding 1.2rem 1.4rem;
        text-align left;
        box-shadow 0 4px 14px rgba(0,0,0,0.35);
    }
    .kpi-label {
        color #8b93a7;
        font-size 0.8rem;
        text-transform uppercase;
        letter-spacing 0.06em;
        margin-bottom 0.3rem;
    }
    .kpi-value {
        color #ffffff;
        font-size 1.9rem;
        font-weight 700;
    }
    .status-badge {
        display inline-block;
        padding 0.35rem 0.9rem;
        border-radius 999px;
        font-weight 700;
        font-size 0.95rem;
        letter-spacing 0.03em;
    }
    .badge-critical { background-color #3d1418; color #ff6b6b; border 1px solid #ff6b6b; }
    .badge-warning  { background-color #3d2f10; color #ffb84d; border 1px solid #ffb84d; }
    .badge-healthy  { background-color #10331b; color #4ade80; border 1px solid #4ade80; }

    section[data-testid=stSidebar] {
        background-color #0b0d12;
        border-right 1px solid #21252f;
    }

    h1, h2, h3 { color #f2f4f8 !important; }
    .stTabs [data-baseweb=tab] { font-weight 600; }
style
, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------
RUL_MAX = 125          # matches the piecewise-linear clipping used in training
CRITICAL_THRESHOLD = 30
WARNING_THRESHOLD = 70

MODEL_PATH = xgboost_rul_model.pkl
FEATURES_PATH = model_features.pkl
TEST_SAMPLE_PATH = test_sample_for_streamlit.csv


# --------------------------------------------------------------------------
# CACHED LOADERS
# --------------------------------------------------------------------------
@st.cache_resource
def load_model_and_features()
    Load the trained XGBoost model and the exact feature list used during training.
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    return model, feature_cols


@st.cache_data
def load_test_sample()
    Load the pre-processed sample engines used for the live demo.
    return pd.read_csv(TEST_SAMPLE_PATH)


@st.cache_data
def get_feature_importance(_model, feature_cols)
    Extract and sort XGBoost feature importances.
    importances = _model.feature_importances_
    return (
        pd.DataFrame({feature feature_cols, importance importances})
        .sort_values(importance, ascending=False)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------
def get_health_status(rul_value)
    Map a predicted RUL value to a maintenance status label, color, and icon.
    if rul_value = CRITICAL_THRESHOLD
        return CRITICAL, #ff6b6b, badge-critical, 🔴
    elif rul_value = WARNING_THRESHOLD
        return WARNING, #ffb84d, badge-warning, 🟠
    else
        return HEALTHY, #4ade80, badge-healthy, 🟢


def make_gauge(rul_value, max_value=RUL_MAX)
    Build a professional gauge chart for the predicted RUL.
    _, color, _, _ = get_health_status(rul_value)

    fig = go.Figure(go.Indicator(
        mode=gauge+number,
        value=rul_value,
        number={suffix  cycles, font {size 40, color #f2f4f8}},
        domain={x [0, 1], y [0, 1]},
        gauge={
            axis {range [0, max_value], tickcolor #8b93a7, tickfont {color #8b93a7}},
            bar {color color, thickness 0.28},
            bgcolor #12151d,
            borderwidth 0,
            steps [
                {range [0, CRITICAL_THRESHOLD], color #3d1418},
                {range [CRITICAL_THRESHOLD, WARNING_THRESHOLD], color #3d2f10},
                {range [WARNING_THRESHOLD, max_value], color #10331b},
            ],
            threshold {
                line {color white, width 3},
                thickness 0.8,
                value rul_value,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor=rgba(0,0,0,0),
        plot_bgcolor=rgba(0,0,0,0),
        font={color #f2f4f8},
        height=320,
        margin=dict(l=30, r=30, t=30, b=10),
    )
    return fig


def predict_rul(model, feature_cols, row_df)
    Run a single-row (or batch) prediction, aligning columns to the training feature order.
    X = row_df[feature_cols]
    preds = model.predict(X)
    return np.clip(preds, 0, RUL_MAX)


def kpi_card(label, value)
    st.markdown(f
    div class=kpi-card
        div class=kpi-label{label}div
        div class=kpi-value{value}div
    div
    , unsafe_allow_html=True)


# --------------------------------------------------------------------------
# LOAD ARTIFACTS
# --------------------------------------------------------------------------
try
    model, feature_cols = load_model_and_features()
    test_df = load_test_sample()
    load_error = None
except Exception as e
    model, feature_cols, test_df = None, None, None
    load_error = str(e)

# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
with st.sidebar
    st.markdown(## ✈️ Predictive Maintenance)
    st.markdown(NASA C-MAPSS — Turbofan Engine RUL)
    st.markdown(---)

    page = st.radio(
        Navigate,
        [
            🏠 Overview,
            🔧 Engine Diagnostics,
            🧪 What-If Simulator,
            📊 Batch Prediction,
            📈 Model Insights,
        ],
    )

    st.markdown(---)
    st.markdown(### About the Model)
    st.markdown(
        
        - Algorithm XGBoost Regressor
        - Target Remaining Useful Life (RUL)
        - Training data Merged FD001–FD004
        - RUL cap 125 cycles (piecewise-linear degradation)
        
    )
    st.caption(Built as part of a Machine Learning graduation project.)

if load_error
    st.error(
        f⚠️ Could not load model artifacts. Make sure `xgboost_rul_model.pkl`, 
        f`model_features.pkl`, and `test_sample_for_streamlit.csv` are in the app directory.nn
        fDetails {load_error}
    )
    st.stop()

has_unit_id = unit_id in test_df.columns
has_cycle = cycle in test_df.columns
has_actual_rul = RUL in test_df.columns

# ==========================================================================
# PAGE OVERVIEW
# ==========================================================================
if page == 🏠 Overview
    st.title(Turbofan Engine — Remaining Useful Life Prediction)
    st.markdown(
        A predictive maintenance system estimating how many operating cycles remain 
        before an engine requires maintenance, trained on NASA's C-MAPSS dataset 
        (FD001–FD004, six operating conditions, XGBoost regression).
    )

    st.markdown(###)
    c1, c2, c3, c4 = st.columns(4)
    with c1
        kpi_card(Model, XGBoost)
    with c2
        kpi_card(Features, f{len(feature_cols)})
    with c3
        n_engines = test_df[unit_id].nunique() if has_unit_id else len(test_df)
        kpi_card(Engines in Sample, f{n_engines})
    with c4
        kpi_card(RUL Cap, f{RUL_MAX} cycles)

    st.markdown(###)
    st.subheader(Sample Engine Fleet — Predicted Health)

    preview = test_df.copy()
    preview[Predicted_RUL] = predict_rul(model, feature_cols, preview).round(1)
    preview[Status] = preview[Predicted_RUL].apply(lambda v get_health_status(v)[0])

    if has_unit_id
        fleet = preview.groupby(unit_id, as_index=False)[Predicted_RUL].min()
        fleet[Status] = fleet[Predicted_RUL].apply(lambda v get_health_status(v)[0])
    else
        fleet = preview[[Predicted_RUL, Status]].reset_index().rename(columns={index unit_id})

    color_map = {CRITICAL #ff6b6b, WARNING #ffb84d, HEALTHY #4ade80}
    fig = px.bar(
        fleet.sort_values(Predicted_RUL),
        x=unit_id, y=Predicted_RUL, color=Status,
        color_discrete_map=color_map,
        labels={unit_id Engine, Predicted_RUL Predicted RUL (cycles)},
    )
    fig.update_layout(
        paper_bgcolor=rgba(0,0,0,0), plot_bgcolor=rgba(0,0,0,0),
        font={color #f2f4f8}, height=420,
        xaxis={showticklabels False},
    )
    st.plotly_chart(fig, use_container_width=True)

    n_crit = (fleet[Status] == CRITICAL).sum()
    n_warn = (fleet[Status] == WARNING).sum()
    n_healthy = (fleet[Status] == HEALTHY).sum()
    st.info(
        f🔴 {n_crit} critical    🟠 {n_warn} need monitoring    
        f🟢 {n_healthy} healthy — out of {len(fleet)} engines in this sample.
    )

# ==========================================================================
# PAGE ENGINE DIAGNOSTICS
# ==========================================================================
elif page == 🔧 Engine Diagnostics
    st.title(🔧 Single Engine Diagnostics)
    st.markdown(Select an engine from the test sample to inspect its predicted health.)

    if has_unit_id
        selected_unit = st.selectbox(Select Engine (unit_id), sorted(test_df[unit_id].unique()))
        unit_rows = test_df[test_df[unit_id] == selected_unit].copy()
        if has_cycle
            unit_rows = unit_rows.sort_values(cycle)
        current_row = unit_rows.iloc[[-1]]
    else
        idx = st.selectbox(Select Row Index, test_df.index)
        unit_rows = test_df.loc[[idx]]
        current_row = unit_rows

    predicted_rul = float(predict_rul(model, feature_cols, current_row)[0])
    status_label, status_color, badge_class, icon = get_health_status(predicted_rul)

    col_gauge, col_info = st.columns([1.2, 1])

    with col_gauge
        st.plotly_chart(make_gauge(predicted_rul), use_container_width=True)

    with col_info
        st.markdown(f
        div class=kpi-card style=margin-bottom1rem;
            div class=kpi-labelPredicted Remaining Useful Lifediv
            div class=kpi-value{predicted_rul.1f} cyclesdiv
        div
        , unsafe_allow_html=True)

        st.markdown(
            f'span class=status-badge {badge_class}{icon} {status_label}span',
            unsafe_allow_html=True,
        )

        if status_label == CRITICAL
            st.error(⚠️ Immediate maintenance recommended. Engine is close to end-of-life.)
        elif status_label == WARNING
            st.warning(🟠 Schedule maintenance soon. Degradation trend detected.)
        else
            st.success(✅ Engine operating within normal parameters.)

        if has_actual_rul
            actual_rul = float(current_row[RUL].values[0])
            error = predicted_rul - actual_rul
            st.metric(Actual RUL (ground truth), f{actual_rul.1f} cycles,
                       delta=f{error+.1f} prediction error)

    # Sensor trend over the engine's observed cycles
    if has_cycle and len(unit_rows)  1
        st.markdown(---)
        st.subheader(Sensor Trend Over Observed Cycles)
        plottable = [c for c in feature_cols if c in unit_rows.columns and roll not in c][4]
        if plottable
            sensor_choice = st.multiselect(Sensors to display, plottable, default=plottable[2])
            if sensor_choice
                trend_fig = px.line(
                    unit_rows, x=cycle, y=sensor_choice,
                    labels={cycle Cycle, value Sensor Reading},
                )
                trend_fig.update_layout(
                    paper_bgcolor=rgba(0,0,0,0), plot_bgcolor=rgba(0,0,0,0),
                    font={color #f2f4f8}, height=380,
                )
                st.plotly_chart(trend_fig, use_container_width=True)

    with st.expander(View raw feature values used for this prediction)
        st.dataframe(current_row[feature_cols].T.rename(columns={current_row.index[0] value}))

# ==========================================================================
# PAGE WHAT-IF SIMULATOR
# ==========================================================================
elif page == 🧪 What-If Simulator
    st.title(🧪 What-If Sensor Simulator)
    st.markdown(
        Adjust the most influential sensor readings and watch the predicted RUL update live. 
        All other features are held at the dataset median.
    )

    importance_df = get_feature_importance(model, feature_cols)
    top_features = importance_df.head(8)[feature].tolist()

    baseline = test_df[feature_cols].median(numeric_only=True)

    st.markdown(### Adjustable Parameters (Top Influential Features))
    sim_values = baseline.copy()

    cols = st.columns(2)
    for i, feat in enumerate(top_features)
        col = cols[i % 2]
        f_min = float(test_df[feat].min())
        f_max = float(test_df[feat].max())
        f_default = float(baseline[feat])
        with col
            sim_values[feat] = st.slider(
                feat, min_value=f_min, max_value=f_max, value=f_default,
                key=fsim_{feat},
            )

    sim_row = pd.DataFrame([sim_values])[feature_cols]
    sim_pred = float(predict_rul(model, feature_cols, sim_row)[0])
    status_label, status_color, badge_class, icon = get_health_status(sim_pred)

    st.markdown(---)
    c1, c2 = st.columns([1, 1.3])
    with c1
        st.plotly_chart(make_gauge(sim_pred), use_container_width=True)
    with c2
        st.markdown(f
        div class=kpi-card
            div class=kpi-labelSimulated Predicted RULdiv
            div class=kpi-value{sim_pred.1f} cyclesdiv
        div
        , unsafe_allow_html=True)
        st.markdown(
            f'brspan class=status-badge {badge_class}{icon} {status_label}span',
            unsafe_allow_html=True,
        )
        st.caption(
            This simulator demonstrates the model's sensitivity to its most important 
            features, as ranked by XGBoost's built-in feature importance.
        )

# ==========================================================================
# PAGE BATCH PREDICTION
# ==========================================================================
elif page == 📊 Batch Prediction
    st.title(📊 Batch Prediction)
    st.markdown(
        Upload a CSV containing the required feature columns to score multiple engines at once.
    )

    with st.expander(Required columns)
        st.code(, .join(feature_cols))

    uploaded_file = st.file_uploader(Upload CSV file, type=[csv])

    use_demo = st.checkbox(Use the built-in test sample instead, value=not bool(uploaded_file))

    batch_df = None
    if uploaded_file is not None and not use_demo
        batch_df = pd.read_csv(uploaded_file)
    elif use_demo
        batch_df = test_df.copy()

    if batch_df is not None
        missing_cols = [c for c in feature_cols if c not in batch_df.columns]
        if missing_cols
            st.error(f❌ Uploaded file is missing required columns {missing_cols})
        else
            batch_df[Predicted_RUL] = predict_rul(model, feature_cols, batch_df).round(2)
            batch_df[Status] = batch_df[Predicted_RUL].apply(lambda v get_health_status(v)[0])

            st.success(f✅ Scored {len(batch_df)} rows successfully.)

            id_cols = [c for c in [unit_id, cycle] if c in batch_df.columns]
            display_cols = id_cols + [Predicted_RUL, Status]
            st.dataframe(
                batch_df[display_cols].style.applymap(
                    lambda v (
                        color #ff6b6b if v == CRITICAL else
                        color #ffb84d if v == WARNING else
                        color #4ade80 if v == HEALTHY else 
                    ),
                    subset=[Status],
                ),
                use_container_width=True,
                height=420,
            )

            csv_out = batch_df.to_csv(index=False).encode(utf-8)
            st.download_button(
                ⬇️ Download Predictions as CSV,
                data=csv_out,
                file_name=rul_predictions.csv,
                mime=textcsv,
            )

# ==========================================================================
# PAGE MODEL INSIGHTS
# ==========================================================================
elif page == 📈 Model Insights
    st.title(📈 Model Insights & Explainability)

    importance_df = get_feature_importance(model, feature_cols)

    st.subheader(Top 15 Feature Importances (XGBoost))
    top15 = importance_df.head(15)
    fig = px.bar(
        top15.sort_values(importance),
        x=importance, y=feature, orientation=h,
        color=importance, color_continuous_scale=Blues,
    )
    fig.update_layout(
        paper_bgcolor=rgba(0,0,0,0), plot_bgcolor=rgba(0,0,0,0),
        font={color #f2f4f8}, height=520, coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        
        Why this matters operating condition context (`op_setting_123`) combined with raw
        and rolling sensor statistics allows XGBoost to learn condition-dependent degradation
        patterns without explicit regime clustering — the core scientific argument of this project.
        
    )

    if has_actual_rul
        st.markdown(---)
        st.subheader(Model Performance on Test Sample)
        preds = predict_rul(model, feature_cols, test_df)
        actual = test_df[RUL].values
        rmse = float(np.sqrt(np.mean((preds - actual)  2)))
        mae = float(np.mean(np.abs(preds - actual)))
        r2 = 1 - np.sum((actual - preds)  2)  np.sum((actual - actual.mean())  2)

        m1, m2, m3 = st.columns(3)
        with m1
            kpi_card(RMSE, f{rmse.2f} cycles)
        with m2
            kpi_card(MAE, f{mae.2f} cycles)
        with m3
            kpi_card(R² Score, f{r2.3f})

        scatter_fig = px.scatter(
            x=actual, y=preds, opacity=0.4,
            labels={x Actual RUL, y Predicted RUL},
        )
        scatter_fig.add_shape(
            type=line, x0=0, y0=0, x1=RUL_MAX, y1=RUL_MAX,
            line=dict(color=white, dash=dash),
        )
        scatter_fig.update_layout(
            paper_bgcolor=rgba(0,0,0,0), plot_bgcolor=rgba(0,0,0,0),
            font={color #f2f4f8}, height=420,
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

# --------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------
st.markdown(---)
st.caption(
    NASA C-MAPSS Remaining Useful Life Prediction — Machine Learning Graduation Project  
    Model XGBoost Regressor  Deployment Streamlit
)
