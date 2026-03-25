import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# configuration
st.set_page_config(
    page_title="Healthcare Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# configuration
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  /* Dark sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f1b2d 0%, #1a2e45 100%);
    color: #e2e8f0;
  }
  [data-testid="stSidebar"] .stMarkdown,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stMultiSelect label {
    color: #94a3b8 !important;
    font-size: 0.78rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
  }
  [data-testid="stSidebar"] h2 {
    color: #f1f5f9 !important;
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
  }

  /* Main area */
  .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* KPI cards */
  .kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  .kpi-label { font-size: 0.72rem; color: #64748b; letter-spacing: 0.07em; text-transform: uppercase; margin-bottom: 4px; }
  .kpi-value { font-size: 1.9rem; font-weight: 600; color: #0f172a; line-height: 1; }
  .kpi-delta { font-size: 0.78rem; color: #10b981; margin-top: 4px; }

  /* Section headers */
  .section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    color: #0f172a;
    margin-bottom: 0.5rem;
    padding-bottom: 6px;
    border-bottom: 2px solid #e2e8f0;
  }

  /* Chart containers */
  .chart-container {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 0.75rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }

  div[data-testid="stHorizontalBlock"] > div { gap: 0.75rem; }
</style>
""", unsafe_allow_html=True)


# loading
@st.cache_data
def load_data():
    """
    Loads the Kaggle Healthcare dataset (by prasad22).
    """
    try:
        df = pd.read_csv("healthcare_dataset.csv")
    except FileNotFoundError:
        rng = np.random.default_rng(42)
        n = 10_000
        conditions   = ["Diabetes", "Hypertension", "Asthma", "Obesity", "Arthritis", "Cancer"]
        blood_types  = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]
        adm_types    = ["Emergency", "Elective", "Urgent"]
        test_results = ["Normal", "Abnormal", "Inconclusive"]
        hospitals    = [f"Hospital {chr(65+i)}" for i in range(8)]
        medications  = ["Aspirin", "Ibuprofen", "Penicillin", "Paracetamol", "Lipitor"]

        admit_dates = pd.to_datetime("2020-01-01") + pd.to_timedelta(
            rng.integers(0, 365*4, n), unit="D"
        )
        los = rng.integers(1, 31, n)

        df = pd.DataFrame({
            "Name":               [f"Patient {i}" for i in range(n)],
            "Age":                rng.integers(18, 90, n),
            "Gender":             rng.choice(["Male", "Female"], n),
            "Blood Type":         rng.choice(blood_types, n),
            "Medical Condition":  rng.choice(conditions, n, p=[0.22,0.22,0.12,0.15,0.15,0.14]),
            "Date of Admission":  admit_dates,
            "Doctor":             [f"Dr. {c}" for c in rng.choice(list("ABCDEFGHIJ"), n)],
            "Hospital":           rng.choice(hospitals, n),
            "Insurance Provider": rng.choice(["Medicare","Aetna","UnitedHealth","Cigna","BlueCross"], n),
            "Billing Amount":     rng.uniform(1_000, 50_000, n).round(2),
            "Room Number":        rng.integers(100, 500, n),
            "Admission Type":     rng.choice(adm_types, n, p=[0.4, 0.35, 0.25]),
            "Discharge Date":     admit_dates + pd.to_timedelta(los, unit="D"),
            "Medication":         rng.choice(medications, n),
            "Test Results":       rng.choice(test_results, n, p=[0.45, 0.35, 0.20]),
        })

    # cleaning
    df.columns = df.columns.str.strip()
    df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], errors="coerce")
    df["Discharge Date"]    = pd.to_datetime(df["Discharge Date"],    errors="coerce")
    df.dropna(subset=["Date of Admission"], inplace=True)
    df["Age"]            = pd.to_numeric(df["Age"], errors="coerce").fillna(df["Age"].median() if "Age" in df else 45)
    df["Billing Amount"] = pd.to_numeric(df["Billing Amount"], errors="coerce").fillna(0)

    df["Length of Stay"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days.clip(lower=0)
    df["Admit Year"]     = df["Date of Admission"].dt.year
    df["Admit Month"]    = df["Date of Admission"].dt.to_period("M").astype(str)
    df["Age Group"] = pd.cut(
        df["Age"],
        bins=[0, 17, 34, 49, 64, 120],
        labels=["<18", "18-34", "35-49", "50-64", "65+"]
    )
    return df


df = load_data()

# color
PALETTE = px.colors.qualitative.Set2
COND_COLORS = {c: PALETTE[i % len(PALETTE)]
               for i, c in enumerate(df["Medical Condition"].unique())}


# sidebar filter
with st.sidebar:
    st.markdown("---")
    st.markdown("**Filters**")

    all_conditions = sorted(df["Medical Condition"].unique())
    sel_conditions = st.multiselect(
        "Medical Condition",
        all_conditions,
        default=all_conditions,
    )

    all_adm_types = sorted(df["Admission Type"].unique())
    sel_adm = st.multiselect(
        "Admission Type",
        all_adm_types,
        default=all_adm_types,
    )

    year_min, year_max = int(df["Admit Year"].min()), int(df["Admit Year"].max())
    sel_years = st.slider("Year Range", year_min, year_max, (year_min, year_max))

    age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
    sel_ages = st.slider("Age Range", age_min, age_max, (age_min, age_max))

    st.markdown("---")
    st.markdown(
        "<span style='color:#64748b;font-size:0.7rem'>Data: Kaggle Healthcare Dataset<br>"
        "prasad22/healthcare-dataset</span>",
        unsafe_allow_html=True,
    )

# filters
mask = (
    df["Medical Condition"].isin(sel_conditions) &
    df["Admission Type"].isin(sel_adm) &
    df["Admit Year"].between(*sel_years) &
    df["Age"].between(*sel_ages)
)
fdf = df[mask].copy()

if fdf.empty:
    st.warning("No data matches the current filters. Please widen your selection.")
    st.stop()


# header
st.markdown(
    "<h1 style='font-family:DM Serif Display,serif;color:#0f172a;margin-bottom:0'>Patient Analytics Dashboard</h1>"
    "<p style='color:#64748b;font-size:0.9rem;margin-top:4px'>Interactive overview of hospital admissions, billing & outcomes</p>",
    unsafe_allow_html=True,
)
st.markdown("---")


# kpi
k1, k2, k3 = st.columns(3)

def kpi(col, label, value, delta=None):
    delta_html = f"<div class='kpi-delta'>{delta}</div>" if delta else ""
    col.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'>{value}</div>"
        f"{delta_html}</div>",
        unsafe_allow_html=True,
    )

kpi(k1, "Total Patients",     f"{len(fdf):,}")
kpi(k2, "Avg Billing",        f"${fdf['Billing Amount'].mean():,.0f}")
kpi(k3, "Avg Length of Stay", f"{fdf['Length of Stay'].mean():.1f} days")


st.markdown("<br>", unsafe_allow_html=True)


# line+bar
r1a, r1b = st.columns([3, 2])

with r1a:
    st.markdown("<div class='section-header'>Monthly Admissions Over Time</div>", unsafe_allow_html=True)
    monthly = (
        fdf.groupby(["Admit Month", "Medical Condition"])
           .size()
           .reset_index(name="Admissions")
           .sort_values("Admit Month")
    )
    fig_line = px.line(
        monthly,
        x="Admit Month", y="Admissions",
        color="Medical Condition",
        color_discrete_map=COND_COLORS,
        markers=True,
        template="plotly_white",
    )
    fig_line.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=60),
        legend=dict(orientation="h", y=-0.25, x=0, font_size=11),
        xaxis=dict(tickangle=-45, tickfont_size=10),
        yaxis_title="Admissions",
        hovermode="x unified",
    )
    st.plotly_chart(fig_line, use_container_width=True)

with r1b:
    st.markdown("<div class='section-header'>Avg Billing by Condition</div>", unsafe_allow_html=True)
    billing_cond = (
        fdf.groupby("Medical Condition")["Billing Amount"]
           .mean()
           .reset_index()
           .sort_values("Billing Amount", ascending=True)
    )
    fig_bar = px.bar(
        billing_cond,
        x="Billing Amount", y="Medical Condition",
        orientation="h",
        color="Medical Condition",
        color_discrete_map=COND_COLORS,
        template="plotly_white",
        text=billing_cond["Billing Amount"].apply(lambda x: f"${x:,.0f}"),
    )
    fig_bar.update_traces(textposition="outside", textfont_size=10)
    fig_bar.update_layout(
        height=320,
        margin=dict(l=10, r=80, t=10, b=10),
        showlegend=False,
        xaxis_title="Avg Billing ($)",
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# scatter+pie
r2a, r2b = st.columns([3, 2])

with r2a:
    st.markdown("<div class='section-header'>Age vs. Billing Amount</div>", unsafe_allow_html=True)
    sample = fdf.sample(min(2000, len(fdf)), random_state=1)
    fig_scatter = px.scatter(
        sample,
        x="Age", y="Billing Amount",
        color="Medical Condition",
        symbol="Admission Type",
        color_discrete_map=COND_COLORS,
        opacity=0.65,
        template="plotly_white",
        hover_data=["Gender", "Length of Stay", "Test Results"],
    )
    fig_scatter.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=-0.18, x=0, font_size=10),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with r2b:
    st.markdown("<div class='section-header'>Admission Type Breakdown</div>", unsafe_allow_html=True)
    adm_counts = fdf["Admission Type"].value_counts().reset_index()
    adm_counts.columns = ["Admission Type", "Count"]
    fig_pie = px.pie(
        adm_counts,
        names="Admission Type", values="Count",
        color_discrete_sequence=["#3b82f6", "#10b981", "#f59e0b"],
        hole=0.45,
        template="plotly_white",
    )
    fig_pie.update_traces(textinfo="percent+label", textfont_size=12)
    fig_pie.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig_pie, use_container_width=True)


# heatmap+box
r3a, r3b = st.columns(2)

with r3a:
    st.markdown("<div class='section-header'>Length of Stay — Age Group × Condition</div>", unsafe_allow_html=True)
    heat_data = (
        fdf.groupby(["Age Group", "Medical Condition"])["Length of Stay"]
           .mean()
           .reset_index()
           .pivot(index="Age Group", columns="Medical Condition", values="Length of Stay")
    )
    fig_heat = go.Figure(go.Heatmap(
        z=heat_data.values,
        x=heat_data.columns.tolist(),
        y=heat_data.index.astype(str).tolist(),
        colorscale="Blues",
        text=np.round(heat_data.values, 1),
        texttemplate="%{text}d",
        showscale=True,
        colorbar=dict(title="Days", thickness=12),
    ))
    fig_heat.update_layout(
        height=300,
        margin=dict(l=60, r=10, t=10, b=10),
        xaxis_title="",
        yaxis_title="Age Group",
        template="plotly_white",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with r3b:
    st.markdown("<div class='section-header'>Billing Distribution by Test Result</div>", unsafe_allow_html=True)
    fig_box = px.box(
        fdf,
        x="Test Results", y="Billing Amount",
        color="Test Results",
        color_discrete_sequence=["#3b82f6", "#ef4444", "#f59e0b"],
        template="plotly_white",
        points="outliers",
        notched=True,
    )
    fig_box.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        yaxis_title="Billing Amount ($)",
    )
    st.plotly_chart(fig_box, use_container_width=True)


# stacked bar
st.markdown("<div class='section-header'>Patient Volume by Insurance Provider & Medical Condition</div>",
            unsafe_allow_html=True)
ins_cond = (
    fdf.groupby(["Insurance Provider", "Medical Condition"])
       .size()
       .reset_index(name="Count")
)
fig_stacked = px.bar(
    ins_cond,
    x="Insurance Provider", y="Count",
    color="Medical Condition",
    color_discrete_map=COND_COLORS,
    template="plotly_white",
    barmode="stack",
)
fig_stacked.update_layout(
    height=320,
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", y=-0.2, x=0, font_size=11),
    yaxis_title="Patient Count",
)
st.plotly_chart(fig_stacked, use_container_width=True)


# filtered data table
with st.expander("View Filtered Patient Records"):
    show_cols = ["Age", "Gender", "Medical Condition", "Admission Type",
                 "Hospital", "Billing Amount", "Length of Stay", "Test Results"]
    st.dataframe(
        fdf[show_cols].reset_index(drop=True),
        use_container_width=True,
        height=300,
    )
    st.caption(f"Showing {len(fdf):,} records matching current filters.")
