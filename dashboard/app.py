import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# PAGE CONFIG

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    layout="wide"
)

st.title("🔧 Predictive Maintenance Dashboard")
st.subheader("Bearing Remaining Useful Life Prediction")

# LOAD DATA

BASE_DIR = Path(__file__).resolve().parent.parent

metrics = pd.read_csv(BASE_DIR / "model_metrics.csv")
latest = pd.read_csv(BASE_DIR / "Finalbest_test_predictions_final_v4.csv")
timeseries = pd.read_csv(BASE_DIR / "Finalbest_test_predictions_timeseries_v4.csv")

# ACTUAL RUL VALUES

actual_rul = {
    "Bearing1_3": 5730,
    "Bearing1_4": 339,
    "Bearing1_5": 1610,
    "Bearing1_6": 1460,
    "Bearing1_7": 7570,
    "Bearing2_3": 7530,
    "Bearing2_4": 1390,
    "Bearing2_5": 3090,
    "Bearing2_6": 1290,
    "Bearing2_7": 580,
    "Bearing3_3": 820
}

latest["Actual_RUL_s"] = latest["bearing"].map(actual_rul)

latest["Error_%"] = (
    abs(latest["Predicted_RUL_s"] - latest["Actual_RUL_s"])
    / latest["Actual_RUL_s"] * 100
)

latest["Health_Display"] = latest["Health_State"].replace({
    "Non-critical": "🟢 Non-critical",
    "Wear detectable": "🟡 Wear detectable",
    "Imminent failure": "🔴 Imminent failure"
})

# KPI SECTION

best_model = metrics.sort_values(
    "R2",
    ascending=False
).iloc[0]

avg_error = latest["Error_%"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Bearings",
    latest.shape[0]
)

col2.metric(
    "Average Error %",
    f"{avg_error:.1f}%"
)

col3.metric(
    "Health States",
    latest["Health_State"].nunique()
)

col4.metric(
    "Best Model",
    best_model["Model"]
)

st.divider()

# ALERT SECTION

critical = latest[
    latest["Health_State"] == "Imminent failure"
]

if len(critical) > 0:
    st.error(
        f"⚠ {len(critical)} bearing(s) require immediate maintenance!"
    )

# MODEL PERFORMANCE

st.header("Model Performance Comparison")

metrics = metrics.sort_values(
    "R2",
    ascending=False
)

fig_r2 = px.bar(
    metrics,
    x="Model",
    y="R2",
    title="R² Score Comparison",
    text_auto=".3f"
)

st.plotly_chart(fig_r2, use_container_width=True)

col1, col2 = st.columns(2)

with col1:

    fig_rmse = px.bar(
        metrics,
        x="Model",
        y="RMSE",
        title="RMSE Comparison",
        text_auto=".3f"
    )

    st.plotly_chart(
        fig_rmse,
        use_container_width=True
    )

with col2:

    fig_mae = px.bar(
        metrics,
        x="Model",
        y="MAE",
        title="MAE Comparison",
        text_auto=".3f"
    )

    st.plotly_chart(
        fig_mae,
        use_container_width=True
    )

# ACTUAL VS PREDICTED

st.header("Actual vs Predicted RUL")

comparison = latest.sort_values(
    "Actual_RUL_s",
    ascending=False
)

fig_compare = px.bar(
    comparison,
    x="bearing",
    y=["Actual_RUL_s", "Predicted_RUL_s"],
    barmode="group",
    title="Actual vs Predicted Remaining Useful Life"
)

st.plotly_chart(
    fig_compare,
    use_container_width=True
)

# TOP RISK BEARINGS

st.header("Highest Risk Bearings")

risk = latest.sort_values(
    "Predicted_RUL_s"
).head(5)

st.dataframe(
    risk[
        [
            "bearing",
            "Predicted_RUL_s",
            "Health_Display"
        ]
    ],
    use_container_width=True
)

# HEALTH STATUS

st.header("Bearing Health Status")

st.dataframe(
    latest[
        [
            "bearing",
            "Predicted_RUL_s",
            "Actual_RUL_s",
            "Error_%",
            "Uncertainty_s",
            "Health_Display"
        ]
    ],
    use_container_width=True
)

health_count = (
    latest["Health_State"]
    .value_counts()
    .reset_index()
)

health_count.columns = [
    "Health_State",
    "Count"
]

fig_health = px.pie(
    health_count,
    names="Health_State",
    values="Count",
    title="Health State Distribution"
)

st.plotly_chart(
    fig_health,
    use_container_width=True
)

# RUL TREND

st.header("RUL Degradation Trend")

bearing = st.selectbox(
    "Select Bearing",
    sorted(timeseries["bearing"].unique())
)

filtered = timeseries[
    timeseries["bearing"] == bearing
]

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=filtered["time_step"],
        y=filtered["Predicted_RUL_s"],
        mode="lines",
        name="Predicted RUL"
    )
)

fig.add_trace(
    go.Scatter(
        x=filtered["time_step"],
        y=filtered["Upper_RUL_s"],
        mode="lines",
        line=dict(width=0),
        showlegend=False
    )
)

fig.add_trace(
    go.Scatter(
        x=filtered["time_step"],
        y=filtered["Lower_RUL_s"],
        fill="tonexty",
        mode="lines",
        name="Uncertainty Band"
    )
)

fig.update_layout(
    title=f"{bearing} RUL Prediction",
    xaxis_title="Time Step",
    yaxis_title="Remaining Useful Life (s)"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# FINAL RESULTS TABLE

st.header("Final Bearing Predictions")

st.dataframe(
    latest,
    use_container_width=True
)

# DOWNLOAD

st.download_button(
    label="⬇ Download Results CSV",
    data=latest.to_csv(index=False),
    file_name="bearing_predictions.csv",
    mime="text/csv"
)