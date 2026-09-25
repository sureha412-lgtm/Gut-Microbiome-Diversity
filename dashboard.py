import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Optional statistical test for exploratory group comparison
try:
    from scipy.stats import mannwhitneyu
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

from sklearn.feature_selection import f_classif

import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# APP CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Gut Microbiome | Healthy vs IBD",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_XLSX = BASE_DIR / "ibd_datasets_combined.xlsx"
DEFAULT_PDF = BASE_DIR / "Project report.pdf"


# ============================================================
# REPORT-VALIDATED MODEL RESULTS
# IMPORTANT:
# These values are transcribed from Project report(1).pdf.
# They are NOT retrained/recomputed by this dashboard.
# ============================================================
MODEL_RESULTS = pd.DataFrame([
    {
        "Model": "Random Forest",
        "Accuracy": 0.9275,
        "Precision": 0.9310,
        "Recall": 0.7714,
        "F1-Score": 0.8438,
        "Mean CV Accuracy": 0.9102,
        "AUC": 0.9745,
        "TN": 101, "FP": 2, "FN": 8, "TP": 27,
    },
    {
        "Model": "SVM",
        "Accuracy": 0.9058,
        "Precision": 0.7895,
        "Recall": 0.8571,
        "F1-Score": 0.8219,
        "Mean CV Accuracy": 0.8721,
        "AUC": 0.9592,
        "TN": 95, "FP": 8, "FN": 5, "TP": 30,
    },
    {
        "Model": "XGBoost",
        "Accuracy": 0.9420,
        "Precision": 0.9355,
        "Recall": 0.8286,
        "F1-Score": 0.8788,
        "Mean CV Accuracy": 0.8980,
        "AUC": 0.9803,
        "TN": 101, "FP": 2, "FN": 6, "TP": 29,
    },
    {
        "Model": "Neural Network",
        "Accuracy": 0.8768,
        "Precision": 0.8214,
        "Recall": 0.6571,
        "F1-Score": 0.7302,
        "Mean CV Accuracy": 0.8500,
        "AUC": 0.9173,
        "TN": 98, "FP": 5, "FN": 12, "TP": 23,
    },
])

# Exact top-20 features and ANOVA F-scores reported in the PDF.
REPORT_TOP20 = pd.DataFrame([
    (1, "Gemmiger formicilis", 106.35),
    (2, "observed_richness", 106.08),
    (3, "Alistipes shahii", 86.17),
    (4, "Barnesiella intestinihominis", 75.91),
    (5, "shannon_diversity", 65.99),
    (6, "Firmicutes bacterium CAG 83", 60.80),
    (7, "Alistipes putredinis", 56.27),
    (8, "Ruminococcus bromii", 56.13),
    (9, "Oscillibacter sp 57 20", 55.23),
    (10, "Eubacterium siraeum", 46.65),
    (11, "fecalcal", 44.91),
    (12, "Eubacterium ventriosum", 43.83),
    (13, "Ruminococcus lactaris", 41.75),
    (14, "Akkermansia muciniphila", 40.81),
    (15, "Bacteroides fragilis", 36.37),
    (16, "Flavonifractor plautii", 33.57),
    (17, "Bifidobacterium adolescentis", 33.27),
    (18, "Agathobaculum butyriciproducens", 31.95),
    (19, "Ruminococcus bicirculans", 30.65),
    (20, "Clostridium symbiosum", 28.16),
], columns=["Rank", "Feature", "Report ANOVA F-score"])

REPORT_TOP20_NAMES = REPORT_TOP20["Feature"].tolist()

DIVERSITY_FEATURES = [
    "observed_richness",
    "shannon_diversity",
    "chao1_index",
    "microbial_load",
    "F_B_Ratio",
]

# Fields that are not microbiome predictors.
ID_COLUMNS = ["External ID"]
TARGET_COLUMNS = ["Disease", "Label"]


# ============================================================
# HELPERS
# ============================================================
def normalize_label(value):
    """Normalize workbook labels to 0=IBD, 1=Healthy."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"ibd", "disease", "0"}:
            return 0
        if v in {"healthy", "health", "1"}:
            return 1

    try:
        iv = int(float(value))
        if iv in (0, 1):
            return iv
    except Exception:
        pass

    return np.nan


def add_standardized_target(df):
    out = df.copy()

    # Prefer Label because it is consistently binary in the supplied workbook.
    if "Label" in out.columns:
        out["_Target"] = out["Label"].apply(normalize_label)
    elif "Disease" in out.columns:
        out["_Target"] = out["Disease"].apply(normalize_label)
    else:
        out["_Target"] = np.nan

    out["_Status"] = out["_Target"].map({0: "IBD", 1: "Healthy"})
    return out


def predictor_columns(df):
    """Return the 76 microbiome/diversity predictor columns."""
    excluded = set(ID_COLUMNS + TARGET_COLUMNS + ["_Target", "_Status"])
    cols = []
    for c in df.columns:
        if c in excluded:
            continue
        numeric = pd.to_numeric(df[c], errors="coerce")
        if numeric.notna().any():
            cols.append(c)
    return cols


@st.cache_data(show_spinner=False)
def load_workbook(path_string):
    path = Path(path_string)
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")

    xls = pd.ExcelFile(path)
    required = ["Full", "Train", "Test"]
    missing = [s for s in required if s not in xls.sheet_names]
    if missing:
        raise ValueError(
            "Required sheets are missing: " + ", ".join(missing)
        )

    full = add_standardized_target(pd.read_excel(path, sheet_name="Full"))
    train = add_standardized_target(pd.read_excel(path, sheet_name="Train"))
    test = add_standardized_target(pd.read_excel(path, sheet_name="Test"))

    return {"Full": full, "Train": train, "Test": test}


@st.cache_data(show_spinner=False)
def calculate_live_anova(train_df):
    """
    Recalculate ANOVA feature scores from the supplied Train sheet.
    This verifies the report's feature-selection results; it does not retrain
    the reported ML models.
    """
    df = train_df.copy()
    cols = predictor_columns(df)

    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # The report states that missing values were handled in preprocessing.
    # For dashboard verification only, median imputation is used here.
    X = X.fillna(X.median(numeric_only=True))

    valid_cols = X.columns[X.notna().all(axis=0)].tolist()
    X = X[valid_cols]

    y = df["_Target"]
    valid = y.notna()

    X = X.loc[valid]
    y = y.loc[valid].astype(int)

    f_values, p_values = f_classif(X, y)

    result = pd.DataFrame({
        "Feature": X.columns,
        "Live ANOVA F-score": f_values,
        "Live p-value": p_values,
    }).sort_values("Live ANOVA F-score", ascending=False)

    result["Live Rank"] = np.arange(1, len(result) + 1)
    return result


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


@st.cache_data(show_spinner=False)
def group_summary(df, feature):
    x = df[[feature, "_Status"]].copy()
    x[feature] = safe_numeric(x[feature])
    x = x.dropna(subset=[feature, "_Status"])

    rows = []
    for status in ["Healthy", "IBD"]:
        vals = x.loc[x["_Status"] == status, feature]
        rows.append({
            "Group": status,
            "N": int(vals.shape[0]),
            "Mean": vals.mean(),
            "Median": vals.median(),
            "Std Dev": vals.std(),
            "Min": vals.min(),
            "Max": vals.max(),
        })

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def exploratory_mann_whitney(df, feature):
    if not SCIPY_AVAILABLE:
        return None

    x = safe_numeric(df.loc[df["_Status"] == "Healthy", feature]).dropna()
    y = safe_numeric(df.loc[df["_Status"] == "IBD", feature]).dropna()

    if len(x) < 2 or len(y) < 2:
        return None

    result = mannwhitneyu(x, y, alternative="two-sided")

    return {
        "U": float(result.statistic),
        "p": float(result.pvalue),
        "n_healthy": len(x),
        "n_ibd": len(y),
    }


def format_p(value):
    if value is None or not np.isfinite(value):
        return "—"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def metric_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)


# ============================================================
# CSS
# ============================================================
st.markdown(
    """
<style>
.main-title {
    font-size: 2.25rem;
    font-weight: 750;
    margin-bottom: 0.1rem;
}
.subtitle {
    font-size: 1.05rem;
    color: #667085;
    margin-bottom: 1.1rem;
}
.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    margin-top: 0.8rem;
}
.small-note {
    color: #667085;
    font-size: 0.88rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================
st.markdown(
    '<div class="main-title">🧬 Gut Microbiome Research Dashboard</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Machine Learning-Based Analysis of Healthy vs IBD Gut Microbiome Profiles</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Research dashboard scope: sample-level and microbiome analyses are calculated from the supplied Excel workbook. "
    "Trained-model evaluation values are taken from the supplied project report and are not retrained by this application."
)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Dashboard Navigation")

    page = st.radio(
        "Go to",
        [
            "🏠 Overview",
            "👥 Sample Explorer",
            "🧬 Healthy vs IBD",
            "🔎 Top 20 Biomarkers",
            "🤖 Model Evaluation",
            "📊 Data Quality & Audit",
            "📖 Methods & Interpretation",
        ],
    )


# ============================================================
# LOAD DATA
# ============================================================
try:
    datasets = load_workbook(str(DEFAULT_XLSX))
except Exception as exc:
    st.write(f"Unable to load the Excel workbook: {exc}")
    st.stop()

full = datasets["Full"]
train = datasets["Train"]
test = datasets["Test"]

predictors = predictor_columns(full)
report_missing_features = [
    f for f in REPORT_TOP20_NAMES if f not in full.columns
]


# ============================================================
# GLOBAL DATA CHECKS
# ============================================================
full_n = len(full)
train_n = len(train)
test_n = len(test)

split_sum_matches = train_n + test_n == full_n

full_ids = set(full["External ID"].astype(str)) if "External ID" in full.columns else set()
train_ids = set(train["External ID"].astype(str)) if "External ID" in train.columns else set()
test_ids = set(test["External ID"].astype(str)) if "External ID" in test.columns else set()

id_overlap_train_test = len(train_ids & test_ids)
id_coverage = len(train_ids | test_ids) == len(full_ids) if full_ids else False

report_full_n = 1374
report_train_n = 1235
report_test_n = 138

report_count_discrepancy = (
    full_n != report_full_n
    or train_n != report_train_n
    or test_n != report_test_n
)

# Non-finite values in predictors
numeric_full = full[predictors].apply(pd.to_numeric, errors="coerce")
nonfinite_count = int(
    np.isinf(numeric_full.to_numpy(dtype=float)).sum()
)

missing_count = int(full[predictors].isna().sum().sum())


# ============================================================
# PAGE: OVERVIEW
# ============================================================
if page == "🏠 Overview":

    st.subheader("Project Objective")

    st.write(
        "The dashboard investigates differences in gut microbiome composition and "
        "diversity between Healthy individuals and IBD patients, while presenting "
        "the trained machine-learning evaluation reported in the project document."
    )

    st.markdown(
        """
        **Primary research questions**
        1. Which microbial and diversity features differ between Healthy and IBD samples?
        2. Which 20 features were selected by ANOVA feature selection?
        3. How do the Healthy and IBD groups differ for individual biomarkers?
        4. What model-performance results were obtained from the trained classifiers?
        5. How do the sample-level microbiome profiles look within the supplied dataset?
        """
    )

    st.subheader("Dataset at a glance")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Full samples", f"{full_n:,}")
    with c2:
        metric_card("Training samples", f"{train_n:,}")
    with c3:
        metric_card("Test samples", f"{test_n:,}")
    with c4:
        metric_card("Predictor features", f"{len(predictors):,}")
    with c5:
        metric_card("Report top features", "20")

    if report_count_discrepancy:
        st.caption(
            f"Data reconciliation: the supplied PDF reports Full = {report_full_n}, "
            f"Train = {report_train_n}, Test = {report_test_n}; the supplied Excel contains "
            f"Full = {full_n}, Train = {train_n}, Test = {test_n}. "
            f"Train + Test = {train_n + test_n}, matching the Excel Full count."
        )

    st.subheader("Class distribution")

    dist = (
        full["_Status"]
        .value_counts()
        .reindex(["Healthy", "IBD"])
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    dist.columns = ["Status", "Samples"]

    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            dist,
            x="Status",
            y="Samples",
            text="Samples",
            title="Full dataset class distribution",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.dataframe(
            dist.assign(
                Percentage=lambda d: (d["Samples"] / d["Samples"].sum() * 100).round(2)
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Workbook target mapping used throughout the dashboard: "
            "0 = IBD, 1 = Healthy."
        )

    st.subheader("Research workflow")

    workflow = [
        ("01", "Excel data", "Full / Train / Test"),
        ("02", "Data quality", "Missingness, non-finite values, IDs, class balance"),
        ("03", "Feature selection", "ANOVA SelectKBest / f_classif"),
        ("04", "Top 20", "Report-selected microbial and diversity biomarkers"),
        ("05", "Group analysis", "Healthy vs IBD distributions"),
        ("06", "Sample explorer", "Individual microbiome profile"),
        ("07", "Model evaluation", "PDF-reported trained-model results"),
        ("08", "Interpretation", "Diversity, taxa and methodological limitations"),
    ]

    cols = st.columns(4)
    for i, (num, title, detail) in enumerate(workflow):
        with cols[i % 4]:
            st.markdown(f"### {num} · {title}")
            st.caption(detail)


# ============================================================
# PAGE: SAMPLE EXPLORER
# ============================================================
elif page == "👥 Sample Explorer":

    st.subheader("Sample-Level Microbiome Explorer")

    st.write(
        "This section provides sample-level microbiome information from the Excel "
        "workbook. The supplied workbook contains an External ID and microbiome/"
        "diversity variables, but it does not contain demographic fields such as age "
        "or sex. Therefore, this dashboard does not invent or infer those patient details."
    )

    status_filter = st.multiselect(
        "Filter disease status",
        ["Healthy", "IBD"],
        default=["Healthy", "IBD"],
    )

    sample_pool = full[full["_Status"].isin(status_filter)].copy()

    if sample_pool.empty:
        st.write("No samples match the selected filter.")
        st.stop()

    if "External ID" in sample_pool.columns:
        sample_ids = sample_pool["External ID"].astype(str).tolist()
        selected_id = st.selectbox(
            "Select sample / External ID",
            sample_ids,
        )
        row = sample_pool[sample_pool["External ID"].astype(str) == selected_id].iloc[0]
    else:
        selected_idx = st.selectbox(
            "Select sample row",
            sample_pool.index.tolist(),
        )
        row = sample_pool.loc[selected_idx]

    status = row["_Status"]

    st.markdown(f"### Selected sample: `{row.get('External ID', 'N/A')}`")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Disease status", status)
    with c2:
        metric_card("Observed richness", f"{row.get('observed_richness', np.nan):.2f}")
    with c3:
        metric_card("Shannon diversity", f"{row.get('shannon_diversity', np.nan):.3f}")
    with c4:
        metric_card("Chao1 index", f"{row.get('chao1_index', np.nan):.2f}")

    st.subheader("Sample diversity / ecological profile")

    diversity_rows = []
    for feature in ["observed_richness", "shannon_diversity", "chao1_index", "microbial_load", "F_B_Ratio"]:
        if feature in row.index:
            value = safe_numeric(pd.Series([row[feature]])).iloc[0]
            diversity_rows.append({
                "Variable": feature,
                "Value": value,
            })

    st.dataframe(
        pd.DataFrame(diversity_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Top microbial abundances in this sample")

    microbial_features = [
        c for c in predictors
        if c not in DIVERSITY_FEATURES
    ]

    sample_values = pd.DataFrame({
        "Feature": microbial_features,
        "Abundance": [safe_numeric(pd.Series([row[c]])).iloc[0] for c in microbial_features],
    }).dropna()

    sample_values = sample_values.sort_values(
        "Abundance", ascending=False
    ).head(15)

    fig = px.bar(
        sample_values.sort_values("Abundance"),
        x="Abundance",
        y="Feature",
        orientation="h",
        title="Top 15 microbial features in selected sample",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Selected top-20 biomarker values")

    biomarker_values = []
    for feature in REPORT_TOP20_NAMES:
        if feature in row.index:
            biomarker_values.append({
                "Feature": feature,
                "Value": safe_numeric(pd.Series([row[feature]])).iloc[0],
            })

    st.dataframe(
        pd.DataFrame(biomarker_values),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PAGE: HEALTHY VS IBD
# ============================================================
elif page == "🧬 Healthy vs IBD":

    st.subheader("Healthy vs IBD Microbiome Difference Analysis")

    st.write(
        "This is the main analytical section of the dashboard. All values below "
        "are calculated directly from the supplied Excel workbook."
    )

    analysis_source = st.selectbox(
        "Analysis dataset",
        ["Full", "Train", "Test"],
        index=0,
    )

    df = datasets[analysis_source]

    available_diversity = [
        f for f in DIVERSITY_FEATURES if f in df.columns
    ]

    st.markdown("### Diversity overview")

    selected_diversity = st.multiselect(
        "Diversity / ecological variables",
        available_diversity,
        default=[
            f for f in ["observed_richness", "shannon_diversity", "chao1_index"]
            if f in available_diversity
        ],
    )

    if selected_diversity:
        long_parts = []
        for feature in selected_diversity:
            temp = df[["_Status", feature]].copy()
            temp["Feature"] = feature
            temp["Value"] = safe_numeric(temp[feature])
            temp = temp.rename(columns={"_Status": "Status"})
            long_parts.append(temp[["Status", "Feature", "Value"]])

        long_df = pd.concat(long_parts, ignore_index=True).dropna()

        fig = px.box(
            long_df,
            x="Feature",
            y="Value",
            color="Status",
            points=False,
            facet_col="Feature",
            facet_col_wrap=2,
            title="Healthy vs IBD distributions",
        )
        fig.update_layout(height=700, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        summary_parts = []
        for feature in selected_diversity:
            s = group_summary(df, feature)
            s.insert(0, "Feature", feature)
            summary_parts.append(s)

        summary = pd.concat(summary_parts, ignore_index=True)

        st.dataframe(
            summary.round(4),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Individual feature investigation")

    all_feature_options = [f for f in predictors if f in df.columns]

    selected_feature = st.selectbox(
        "Select a microbial feature or diversity variable",
        all_feature_options,
        index=(
            all_feature_options.index("observed_richness")
            if "observed_richness" in all_feature_options else 0
        ),
    )

    s = group_summary(df, selected_feature)

    c1, c2, c3, c4 = st.columns(4)
    healthy_row = s[s["Group"] == "Healthy"].iloc[0]
    ibd_row = s[s["Group"] == "IBD"].iloc[0]

    with c1:
        metric_card("Healthy median", f"{healthy_row['Median']:.4g}")
    with c2:
        metric_card("IBD median", f"{ibd_row['Median']:.4g}")
    with c3:
        metric_card(
            "Median difference",
            f"{healthy_row['Median'] - ibd_row['Median']:.4g}",
        )
    with c4:
        metric_card(
            "Healthy − IBD mean",
            f"{healthy_row['Mean'] - ibd_row['Mean']:.4g}",
        )

    plot_df = df[["_Status", selected_feature]].copy()
    plot_df["Value"] = safe_numeric(plot_df[selected_feature])
    plot_df = plot_df.dropna(subset=["Value"])

    fig = px.box(
        plot_df,
        x="_Status",
        y="Value",
        color="_Status",
        points="outliers",
        title=f"{selected_feature}: Healthy vs IBD",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    mw = exploratory_mann_whitney(df, selected_feature)

    if mw:
        st.caption(
            f"Exploratory Mann–Whitney U test: U = {mw['U']:.2f}, "
            f"two-sided p = {format_p(mw['p'])}. "
            "This is an exploratory group-comparison statistic, not a causal test."
        )
    else:
        st.caption("Mann–Whitney test unavailable because scipy is not installed.")

    st.markdown("### Top-20 biomarker group comparison")

    top20_available = [
        f for f in REPORT_TOP20_NAMES if f in df.columns
    ]

    comparison_rows = []
    for feature in top20_available:
        gs = group_summary(df, feature)
        h = gs[gs["Group"] == "Healthy"].iloc[0]
        i = gs[gs["Group"] == "IBD"].iloc[0]

        comparison_rows.append({
            "Feature": feature,
            "Healthy mean": h["Mean"],
            "IBD mean": i["Mean"],
            "Healthy median": h["Median"],
            "IBD median": i["Median"],
            "Mean difference (H−IBD)": h["Mean"] - i["Mean"],
            "Median difference (H−IBD)": h["Median"] - i["Median"],
        })

    comparison_df = pd.DataFrame(comparison_rows)

    st.dataframe(
        comparison_df.round(5),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "A positive Healthy−IBD difference means the value is higher in the Healthy group in this workbook; "
        "a negative value means it is higher in the IBD group. This is descriptive association in the supplied dataset and does not establish causation."
    )


# ============================================================
# PAGE: TOP 20 BIOMARKERS
# ============================================================
elif page == "🔎 Top 20 Biomarkers":

    st.subheader("Top 20 Selected Microbial Biomarkers")

    st.write(
        "The project report states that ANOVA SelectKBest (f_classif) was used "
        "to rank 76 predictor features and retain the top 20. The table below "
        "contains the exact report values and a live recalculation from the supplied "
        "Training sheet for verification."
    )

    live_anova = calculate_live_anova(train)

    merged = REPORT_TOP20.merge(
        live_anova[["Feature", "Live Rank", "Live ANOVA F-score", "Live p-value"]],
        on="Feature",
        how="left",
    )

    merged["F-score difference"] = (
        merged["Live ANOVA F-score"] - merged["Report ANOVA F-score"]
    )

    st.dataframe(
        merged.round({
            "Report ANOVA F-score": 2,
            "Live ANOVA F-score": 4,
            "Live p-value": 10,
            "F-score difference": 6,
        }),
        use_container_width=True,
        hide_index=True,
    )

    if report_missing_features:
        st.caption(
            "Report features missing from the Excel workbook: "
            + ", ".join(report_missing_features)
        )
    else:
        max_diff = np.nanmax(np.abs(merged["F-score difference"].to_numpy()))
        if max_diff < 0.05:
            st.caption(
                f"Live verification: maximum absolute F-score difference = {max_diff:.4f}."
            )
        else:
            st.caption(
                f"Live verification requires review: maximum absolute F-score difference = {max_diff:.4f}."
            )

    st.subheader("Interactive ANOVA ranking")

    top_n = st.slider(
        "Number of features to display",
        min_value=5,
        max_value=30,
        value=20,
    )

    plot_data = live_anova.head(top_n).sort_values("Live ANOVA F-score")

    fig = px.bar(
        plot_data,
        x="Live ANOVA F-score",
        y="Feature",
        orientation="h",
        hover_data=["Live p-value", "Live Rank"],
        title=f"Live ANOVA ranking — top {top_n}",
    )
    fig.update_layout(height=max(500, top_n * 28))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Feature-selection interpretation")

    st.write(
        "A higher ANOVA F-score indicates stronger statistical separation of the "
        "feature between the two disease-status groups under the ANOVA feature-selection "
        "procedure. It does not by itself establish biological causality."
    )

    st.subheader("Feature-space reduction")

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Original predictors", len(predictors))
    with c2:
        metric_card("Selected biomarkers", 20)
    with c3:
        metric_card("Reduction", f"{(1 - 20 / len(predictors)) * 100:.1f}%")

    st.caption(
        "The supplied report describes the reduction as approximately 74% (76 → 20)."
    )


# ============================================================
# PAGE: MODEL EVALUATION
# ============================================================
elif page == "🤖 Model Evaluation":

    st.subheader("Trained Model Evaluation — Project Report")

    st.caption(
        "These model metrics are taken from the supplied PDF report. "
        "This dashboard does not retrain Random Forest, SVM, XGBoost, or the Neural Network."
    )

    st.dataframe(
        MODEL_RESULTS[
            [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score",
                "Mean CV Accuracy",
                "AUC",
            ]
        ].style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-Score": "{:.4f}",
            "Mean CV Accuracy": "{:.4f}",
            "AUC": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Evaluation metric definitions")

    metric_definitions = {
        "Accuracy": "The proportion of all samples classified correctly. It gives an overall measure of prediction correctness.",
        "Precision": "Among samples predicted as a class, the proportion that truly belongs to that class. It reflects how reliable positive/class predictions are.",
        "Recall": "Among samples that truly belong to a class, the proportion correctly identified. It measures how completely the model detects that class.",
        "F1-Score": "The harmonic mean of precision and recall. It balances missed cases and incorrect positive predictions in one measure.",
        "Mean CV Accuracy": "The average accuracy across cross-validation folds reported in the project. It indicates how consistently the model performed across the validation splits.",
        "ROC-AUC": "The area under the ROC curve across classification thresholds. Higher values indicate better separation of the two classes in the reported evaluation.",
        "Confusion Matrix": "A count of true positives, true negatives, false positives and false negatives. It shows exactly how predictions were distributed between correct and incorrect classifications.",
    }
    for name, definition in metric_definitions.items():
        st.markdown(f"**{name}:** {definition}")

    st.subheader("Performance comparison")

    metric = st.selectbox(
        "Metric",
        ["Accuracy", "Precision", "Recall", "F1-Score", "Mean CV Accuracy", "AUC"],
    )

    fig = px.bar(
        MODEL_RESULTS,
        x="Model",
        y=metric,
        text=MODEL_RESULTS[metric].map(lambda x: f"{x:.4f}"),
        title=f"Reported {metric}",
    )
    fig.update_traces(textposition="outside")
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("ROC-AUC values reported in the PDF")

    auc_df = MODEL_RESULTS[["Model", "AUC"]].copy()

    fig = px.bar(
        auc_df,
        x="Model",
        y="AUC",
        text=auc_df["AUC"].map(lambda x: f"{x:.4f}"),
        title="Reported ROC-AUC",
    )
    fig.update_traces(textposition="outside")
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Confusion matrix")

    selected_model = st.selectbox(
        "Select model",
        MODEL_RESULTS["Model"].tolist(),
    )

    selected = MODEL_RESULTS[
        MODEL_RESULTS["Model"] == selected_model
    ].iloc[0]

    cm = np.array([
        [selected["TN"], selected["FP"]],
        [selected["FN"], selected["TP"]],
    ])

    fig = px.imshow(
        cm,
        text_auto=True,
        x=["IBD (0)", "Healthy (1)"],
        y=["IBD (0)", "Healthy (1)"],
        labels={
            "x": "Predicted label",
            "y": "True label",
            "color": "Count",
        },
        title=f"{selected_model} — reported confusion matrix",
        aspect="equal",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "The report's confusion matrices use 0 = IBD and 1 = Healthy. "
        "Therefore TP/FN in the report correspond to the Healthy class."
    )

    st.subheader("Model evaluation notes")

    st.markdown(
        """
        - Random Forest: report describes strong interpretability through feature importance.
        - SVM: report identifies the highest recall among the evaluated models.
        - XGBoost: report gives Accuracy = 94.20% and AUC = 0.9803.
        - Neural Network: included as the feed-forward/deep-learning benchmark.
        """
    )

    st.caption(
        "Methodology audit note: the report states that 10-fold cross-validation was applied to Random Forest, SVM and XGBoost, "
        "while its model-comparison table also contains a Neural Network CV value (0.8500). The dashboard preserves the reported value."
    )


# ============================================================
# PAGE: DATA QUALITY & AUDIT
# ============================================================
elif page == "📊 Data Quality & Audit":

    st.subheader("Dataset Quality and Reconciliation Audit")

    st.write(
        "This section intentionally checks the Excel workbook independently of the "
        "narrative values in the PDF."
    )

    checks = pd.DataFrame([
        {
            "Check": "Required sheets present",
            "Result": "PASS" if set(["Full", "Train", "Test"]).issubset(datasets.keys()) else "FAIL",
            "Detail": "Full, Train and Test",
        },
        {
            "Check": "Train + Test = Full",
            "Result": "PASS" if split_sum_matches else "FAIL",
            "Detail": f"{train_n} + {test_n} = {train_n + test_n}; Full = {full_n}",
        },
        {
            "Check": "Train/Test ID overlap",
            "Result": "PASS" if id_overlap_train_test == 0 else "FAIL",
            "Detail": f"{id_overlap_train_test} overlapping External IDs",
        },
        {
            "Check": "Full IDs covered by Train + Test",
            "Result": "PASS" if id_coverage else "REVIEW",
            "Detail": "Based on External ID values",
        },
        {
            "Check": "Missing predictor values",
            "Result": "PASS" if missing_count == 0 else "REVIEW",
            "Detail": f"{missing_count:,} missing cells",
        },
        {
            "Check": "Non-finite predictor values",
            "Result": "PASS" if nonfinite_count == 0 else "REVIEW",
            "Detail": f"{nonfinite_count:,} ±inf values",
        },
        {
            "Check": "Duplicate full rows",
            "Result": "PASS" if full.duplicated().sum() == 0 else "REVIEW",
            "Detail": f"{int(full.duplicated().sum())} duplicates",
        },
        {
            "Check": "Duplicate External IDs",
            "Result": (
                "PASS"
                if ("External ID" not in full.columns or full["External ID"].duplicated().sum() == 0)
                else "REVIEW"
            ),
            "Detail": (
                f"{int(full['External ID'].duplicated().sum())} duplicate IDs"
                if "External ID" in full.columns else "External ID not present"
            ),
        },
    ])

    st.dataframe(
        checks,
        use_container_width=True,
        hide_index=True,
    )

    if report_count_discrepancy:
        st.caption(
            "Report/Excel count discrepancy: the PDF says Full = 1374, "
            f"but the Excel contains {full_n}. The Excel's Train + Test equals "
            f"{train_n + test_n}, which exactly matches the Excel Full count."
        )

    if nonfinite_count:
        st.caption(
            f"The workbook contains {nonfinite_count} non-finite predictor values. "
            "They occur in the F_B_Ratio field because a ratio can become undefined when the denominator is zero. "
            "The dashboard excludes non-finite values from descriptive plots/statistics rather than replacing them with an invented biological value."
        )

    st.subheader("Sheet summaries")

    sheet_rows = []
    for name, df in datasets.items():
        sheet_rows.append({
            "Sheet": name,
            "Rows": len(df),
            "Columns": len(df.columns),
            "Predictors": len(predictor_columns(df)),
            "IBD": int((df["_Status"] == "IBD").sum()),
            "Healthy": int((df["_Status"] == "Healthy").sum()),
            "Missing cells": int(df.isna().sum().sum()),
        })

    st.dataframe(
        pd.DataFrame(sheet_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Target verification")

    target_cross = pd.crosstab(
        full["Disease"],
        full["Label"],
        dropna=False,
    )

    st.dataframe(
        target_cross,
        use_container_width=True,
    )

    st.caption(
        "The supplied workbook uses Label = 0 for IBD and Label = 1 for Healthy. "
        "The Full sheet's Disease field is text, whereas Train/Test contain numeric "
        "Disease values; the dashboard normalizes both representations to _Target."
    )


# ============================================================
# PAGE: METHODS & INTERPRETATION
# ============================================================
elif page == "📖 Methods & Interpretation":

    st.subheader("Project Methodology")

    st.markdown(
        """
        ### 1. Dataset
        The dashboard uses the supplied Excel workbook with three sheets:
        **Full**, **Train**, and **Test**.

        ### 2. Preprocessing documented in the report
        - Duplicate label columns removed.
        - Numerical microbiome abundance variables isolated.
        - Features with more than 90% zeros evaluated.
        - Missing values handled by the preprocessing pipeline.
        - Z-score standardization applied to microbial abundance features.
        - The same preprocessing workflow applied to training and testing data.

        ### 3. Feature selection
        The report specifies **ANOVA SelectKBest (`f_classif`)** and selection of
        the top 20 features from 76 predictor features.

        ### 4. Models
        The supplied report evaluates:
        - Random Forest
        - Support Vector Machine (SVM)
        - XGBoost
        - Feed-Forward Neural Network

        ### 5. Evaluation
        Reported metrics include:
        Accuracy, Precision, Recall, F1-score, Mean CV Accuracy, ROC-AUC and
        confusion matrices.

        ### 6. Biological interpretation
        The report identifies microbial diversity measures such as
        **observed richness** and **Shannon diversity**, together with specific
        microbial taxa, as important discriminatory features.
        """
    )

    st.subheader("How to interpret the Healthy vs IBD analysis")

    st.markdown(
        """
        **Descriptive difference:**  
        A difference in group mean/median describes how the supplied Healthy and
        IBD samples differ in this dataset.

        **Feature-selection evidence:**  
        A high ANOVA F-score indicates that the feature had strong statistical
        separation between the two classes under the feature-selection procedure.

        **Classification evidence:**  
        Model accuracy/AUC describe how the trained models performed on the
        evaluation reported in the project document.

        **Biological caution:**  
        Classification importance or statistical association does not establish
        that a microbe causes IBD. The dashboard therefore separates descriptive
        data analysis from biological interpretation.
        """
    )

    st.subheader("Report-supported findings")

    findings = [
        (
            "Reduced microbial diversity",
            "The report identifies observed richness and Shannon diversity as highly "
            "discriminative features and describes lower diversity in IBD."
        ),
        (
            "Multiple taxa contribute to the signature",
            "The report describes a community-level dysbiosis pattern rather than a "
            "single microbial organism as the complete signature."
        ),
        (
            "Selected biomarkers",
            "The report highlights Gemmiger formicilis, observed_richness, "
            "shannon_diversity, Alistipes shahii, Barnesiella intestinihominis, "
            "Akkermansia muciniphila and Bacteroides fragilis among important features."
        ),
        (
            "Model evaluation",
            "The report contains trained-model performance, ROC-AUC and confusion "
            "matrix results; those values are presented without retraining them here."
        ),
    ]

    for title, text in findings:
        with st.expander(title):
            st.write(text)

    st.subheader("Important scope limitation")

    st.caption(
        "Scope limitation: this is a research-analysis dashboard, not a clinical diagnostic tool. "
        "The supplied Excel file contains microbiome/sample-level variables and an External ID, but no age, sex, medication, "
        "disease-severity, endoscopic, laboratory or treatment fields; those patient attributes therefore cannot be displayed or inferred."
    )

    st.subheader("Source files used by this dashboard")

    st.code(
        f"Patient/sample data: {DEFAULT_XLSX.name}\n"
        f"Model-evaluation source: {DEFAULT_PDF.name}\n"
        f"Model training environment: Python-based machine-learning workflow\n"
        f"Dashboard framework: Streamlit {st.__version__}",
        language="text",
    )

    st.caption(
        "Model results are transcribed from the supplied project report; patient/sample analyses are calculated from the supplied Excel workbook. "
        "The dashboard itself does not retrain the reported models."
    )


# ============================================================
# FOOTER
# ============================================================
st.caption(
    "Gut Microbiome Research Dashboard • Healthy vs IBD • "
    "Excel-driven sample analysis + PDF-reported trained-model evaluation"
)
