import streamlit as st
import pandas as pd
import numpy as np

from recommendations import recommend_test, get_teaching_content
from validators import validate_meta, validate_numeric_list, validate_groups, validate_survival_inputs
from diagnostics import run_all_diagnostics
from plots import (
    plot_residuals, plot_regression, plot_kaplan_meier, plot_km_loglog,
    plot_boxplot, plot_bar, plot_histogram, plot_forest, plot_qq
)
from pdf_builder import build_report

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Public Health Biostatistics Lab", layout="wide")

st.markdown("""
<style>
    .test-badge {
        background: #1a6b3c; color: white;
        padding: 0.4rem 1rem; border-radius: 6px;
        font-size: 1.1rem; font-weight: 600;
        display: inline-block; margin-bottom: 0.5rem;
    }
    .assumption-item {
        padding: 0.25rem 0;
        border-left: 3px solid #2ecc71;
        padding-left: 0.75rem; margin: 0.3rem 0; font-size: 0.9rem;
    }
    .mistake-box {
        background: #fff3cd; border-left: 4px solid #ffc107;
        padding: 0.75rem 1rem; border-radius: 4px; font-size: 0.9rem;
    }
    .epi-box {
        background: #e8f4fd; border-left: 4px solid #3498db;
        padding: 0.75rem 1rem; border-radius: 4px; font-size: 0.9rem;
    }
    .decision-step { color: #555; font-size: 0.9rem; padding: 0.15rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Public Health Biostatistics Lab")
st.markdown("Decision-guided statistical analysis for epidemiology students.")
st.divider()

st.markdown("""
The app selects statistical tests based on:
• Study design  • Outcome type  • Exposure type
""")

teaching_mode = st.toggle("Teaching Mode (Show Learning Guidance)", value=True)
if teaching_mode:
    st.info("💡 Teaching mode is on. Explanations, assumptions, and epidemiology examples will appear alongside results.")

st.divider()

# ── 1. Study Configuration ────────────────────────────────────────────────────
st.header("1. Study Configuration")

col1, col2 = st.columns(2)
with col1:
    design = st.selectbox(
        "Study Design", ["independent", "paired"],
        help="**Paired**: same subjects measured twice or matched pairs. **Independent**: separate groups."
    )
    y_type = st.selectbox(
        "Outcome Type", ["binary", "continuous", "time-to-event"],
        help="**Binary**: yes/no. **Continuous**: measured value. **Time-to-event**: survival/failure time."
    )
with col2:
    x_type = st.selectbox(
        "Exposure Type", ["binary", "categorical", "continuous"],
        help="**Binary**: exposed/unexposed. **Categorical**: 3+ groups. **Continuous**: measured exposure."
    )
    groups_k = st.number_input(
        "Number of Groups (categorical exposure only)",
        min_value=2, value=3,
        disabled=(x_type != "categorical")
    )

normal = True
if y_type == "continuous" and x_type in ["binary", "categorical"]:
    normal = st.toggle(
        "Assume normality within groups", value=True,
        help="Turn off for small samples or skewed data — a non-parametric test will be recommended."
    )

meta = {"design": design, "y_type": y_type, "x_type": x_type, "normal": normal}
if x_type == "categorical":
    meta["groups_k"] = groups_k

# Validate meta
meta_check = validate_meta(meta)
if not meta_check["valid"]:
    for err in meta_check["errors"]:
        st.error(err)
    st.stop()

st.divider()

# ── 2. Statistical Decision Path ──────────────────────────────────────────────
st.header("2. Statistical Decision Path")

decision_steps = [
    f"Study design → **{design}**",
    f"Outcome type → **{y_type}**",
    f"Exposure type → **{x_type}**",
]
if x_type == "categorical":
    decision_steps.append(f"Number of groups → **{groups_k}**")
if y_type == "continuous" and x_type in ["binary", "categorical"]:
    decision_steps.append(f"Normality assumed → **{'Yes' if normal else 'No'}**")

for step in decision_steps:
    st.markdown(f"<div class='decision-step'>▸ {step}</div>", unsafe_allow_html=True)

st.markdown("")
recommended_test = recommend_test(meta)

if recommended_test and recommended_test != "Manual selection recommended":
    st.success(f"✅ Recommended Test: **{recommended_test}**")
else:
    st.warning("⚠️ No standard test matched your configuration. See guidance below.")

# ── 3. Teaching Content ───────────────────────────────────────────────────────
if teaching_mode and recommended_test:
    teaching_content = get_teaching_content(recommended_test)
    st.divider()
    st.header("3. About This Test")
    st.subheader(f"📚 {recommended_test}")
    st.markdown(f"**What it does:** {teaching_content['description']}")
    st.markdown(f"> **When to use:** {teaching_content['when_to_use']}")

    if teaching_content["assumptions"]:
        st.markdown("**Assumptions to check:**")
        for a in teaching_content["assumptions"]:
            st.markdown(f"<div class='assumption-item'>✓ {a}</div>", unsafe_allow_html=True)
        st.markdown("")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Effect size measure:**")
        st.code(teaching_content["effect_size"], language=None)
    with col_b:
        st.markdown("**Common mistake:**")
        st.markdown(f"<div class='mistake-box'>⚠️ {teaching_content['common_mistake']}</div>", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("**Epidemiology example:**")
    st.markdown(f"<div class='epi-box'>🔬 {teaching_content['epi_example']}</div>", unsafe_allow_html=True)
else:
    teaching_content = get_teaching_content(recommended_test)

st.divider()

# ── 4. Data Upload & Analysis ─────────────────────────────────────────────────
st.header("4. Data Upload & Analysis")
st.markdown("Upload a CSV to run the recommended test on your data.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
figures = []
test_results = None
diag_results = None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns")
        st.dataframe(df.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    st.markdown("**Map your columns:**")
    cols = ["— select —"] + df.columns.tolist()

    # ── Column selectors based on test type ───────────────────────────────────
    outcome_col = st.selectbox("Outcome column", cols)
    exposure_col = None
    duration_col = None
    event_col    = None

    if y_type == "time-to-event":
        duration_col = st.selectbox("Duration column (time to event)", cols)
        event_col    = st.selectbox("Event column (1=event, 0=censored)", cols)
        group_col    = st.selectbox("Group column (optional, for KM curves)", cols)
    else:
        exposure_col = st.selectbox("Exposure / group column", cols)

    run_btn = st.button("▶ Run Analysis", type="primary")

    if run_btn:
        # ── Survival / KM / Cox ───────────────────────────────────────────────
        if y_type == "time-to-event":
            if "— select —" in [duration_col, event_col]:
                st.error("Please select duration and event columns.")
                st.stop()

            durations = df[duration_col].dropna().tolist()
            events    = df[event_col].dropna().tolist()
            groups    = df[group_col].tolist() if group_col != "— select —" else None

            val = validate_survival_inputs(durations, events, groups)
            if not val["valid"]:
                for e in val["errors"]: st.error(e)
                st.stop()

            from tests.survival import run_kaplan_meier, run_cox_regression

            km_result  = run_kaplan_meier(durations, events, groups)
            test_results = km_result

            fig_km = plot_kaplan_meier(durations, events, groups)
            figures.append(fig_km)
            st.pyplot(fig_km)

            if groups:
                fig_ll = plot_km_loglog(durations, events, groups)
                figures.append(fig_ll)
                with st.expander("📐 Log-Log Plot (Proportional Hazards Check)"):
                    st.pyplot(fig_ll)

            st.info(km_result.get("interpretation", ""))

            # Cox if covariates available beyond duration/event
            other_cols = [c for c in df.columns if c not in [duration_col, event_col]]
            if other_cols:
                with st.expander("🔬 Cox Proportional Hazards Regression"):
                    cox_df = df[[duration_col, event_col] + other_cols].dropna()
                    try:
                        cox_result = run_cox_regression(cox_df, duration_col, event_col)
                        st.write("**Hazard Ratios:**")
                        hr_df = pd.DataFrame({
                            "HR":       cox_result["hazard_ratios"],
                            "CI Lower": cox_result["ci_lower_95"],
                            "CI Upper": cox_result["ci_upper_95"],
                            "p-value":  cox_result["p_values"],
                        })
                        st.dataframe(hr_df, use_container_width=True)
                        st.info(f"Concordance index: {cox_result['concordance_index']}")
                        st.info(cox_result["interpretation"])

                        fig_f = plot_forest(
                            labels    = list(cox_result["hazard_ratios"].keys()),
                            estimates = list(cox_result["hazard_ratios"].values()),
                            ci_lower  = list(cox_result["ci_lower_95"].values()),
                            ci_upper  = list(cox_result["ci_upper_95"].values()),
                            title     = "Cox Hazard Ratios",
                            xlabel    = "Hazard Ratio",
                        )
                        figures.append(fig_f)
                        st.pyplot(fig_f)
                    except Exception as e:
                        st.warning(f"Cox regression could not run: {e}")

        # ── All other tests ───────────────────────────────────────────────────
        else:
            if "— select —" in [outcome_col, exposure_col]:
                st.error("Please select outcome and exposure columns.")
                st.stop()

            outcome_data  = pd.to_numeric(df[outcome_col],  errors="coerce").dropna().tolist()
            exposure_data = df[exposure_col].dropna().tolist()

            # Split into groups
            group_labels = sorted(df[exposure_col].dropna().unique().tolist())
            groups_data  = [
                pd.to_numeric(df.loc[df[exposure_col] == g, outcome_col], errors="coerce")
                  .dropna().tolist()
                for g in group_labels
            ]

            # Diagnostics
            if y_type == "continuous":
                diag_results = run_all_diagnostics(groups_data, labels=[str(g) for g in group_labels])

                with st.expander("🔍 Assumption Diagnostics"):
                    for nr in diag_results["normality"]:
                        icon = "✅" if nr["is_normal"] else "⚠️"
                        st.markdown(f"{icon} **{nr['label']}** — {nr['interpretation']}")
                        fig_qq = plot_qq(
                            df.loc[df[exposure_col] == nr["label"], outcome_col]
                              .dropna().tolist(),
                            label=str(nr["label"])
                        )
                        figures.append(fig_qq)
                        st.pyplot(fig_qq)

                    ev = diag_results["equal_variance"]
                    icon = "✅" if ev.get("equal_variance") else "⚠️"
                    st.markdown(f"{icon} **Levene's test** — {ev['interpretation']}")
                    st.info(f"Recommendation: {diag_results['recommendation']}")

            # Run the appropriate test
            try:
                if recommended_test in ["Two-sample t-test (Welch)", "Paired t-test"]:
                    from scipy import stats
                    g0, g1 = groups_data[0], groups_data[1]
                    if recommended_test == "Paired t-test":
                        stat, p = stats.ttest_rel(g0, g1)
                    else:
                        stat, p = stats.ttest_ind(g0, g1, equal_var=False)
                    test_results = {
                        "statistic": round(float(stat), 4),
                        "p_value":   round(float(p), 4),
                        "interpretation": (
                            f"Significant difference between groups (p = {p:.4f})."
                            if p < 0.05 else
                            f"No significant difference between groups (p = {p:.4f})."
                        )
                    }

                elif recommended_test == "Mann-Whitney U test":
                    from scipy import stats
                    stat, p = stats.mannwhitneyu(groups_data[0], groups_data[1], alternative="two-sided")
                    test_results = {
                        "statistic": round(float(stat), 4),
                        "p_value":   round(float(p), 4),
                        "interpretation": (
                            f"Significant rank difference between groups (p = {p:.4f})."
                            if p < 0.05 else
                            f"No significant rank difference (p = {p:.4f})."
                        )
                    }

                elif recommended_test == "ANOVA":
                    from scipy import stats
                    stat, p = stats.f_oneway(*groups_data)
                    test_results = {
                        "F_statistic": round(float(stat), 4),
                        "p_value":     round(float(p), 4),
                        "interpretation": (
                            f"Significant difference across groups (p = {p:.4f}). Run post-hoc tests."
                            if p < 0.05 else
                            f"No significant difference across groups (p = {p:.4f})."
                        )
                    }

                elif recommended_test == "Kruskal-Wallis test":
                    from scipy import stats
                    stat, p = stats.kruskal(*groups_data)
                    test_results = {
                        "H_statistic": round(float(stat), 4),
                        "p_value":     round(float(p), 4),
                        "interpretation": (
                            f"Significant difference across groups (p = {p:.4f}). Run post-hoc tests."
                            if p < 0.05 else
                            f"No significant difference across groups (p = {p:.4f})."
                        )
                    }

                elif recommended_test in ["2×2 Chi-square", "Chi-square test"]:
                    from scipy import stats
                    ct = pd.crosstab(df[exposure_col], df[outcome_col])
                    chi2, p, dof, expected = stats.chi2_contingency(ct)
                    test_results = {
                        "chi2_statistic": round(float(chi2), 4),
                        "degrees_of_freedom": int(dof),
                        "p_value": round(float(p), 4),
                        "interpretation": (
                            f"Significant association between exposure and outcome (p = {p:.4f})."
                            if p < 0.05 else
                            f"No significant association (p = {p:.4f})."
                        )
                    }

                elif recommended_test == "McNemar's test":
                    from scipy import stats
                    ct = pd.crosstab(df[exposure_col], df[outcome_col])
                    result = stats.mcnemar(ct, exact=False)
                    test_results = {
                        "statistic": round(float(result.statistic), 4),
                        "p_value":   round(float(result.pvalue), 4),
                        "interpretation": (
                            f"Significant change in binary outcome (p = {result.pvalue:.4f})."
                            if result.pvalue < 0.05 else
                            f"No significant change in binary outcome (p = {result.pvalue:.4f})."
                        )
                    }

                elif recommended_test == "Linear regression":
                    from scipy import stats
                    x = pd.to_numeric(df[exposure_col], errors="coerce").dropna()
                    y = pd.to_numeric(df[outcome_col],  errors="coerce").dropna()
                    min_len = min(len(x), len(y))
                    x, y = np.array(x[:min_len]), np.array(y[:min_len])
                    slope, intercept, r, p, se = stats.linregress(x, y)
                    test_results = {
                        "slope":     round(float(slope), 4),
                        "intercept": round(float(intercept), 4),
                        "R²":        round(float(r**2), 4),
                        "p_value":   round(float(p), 4),
                        "std_error": round(float(se), 4),
                        "interpretation": (
                            f"Significant linear association (β = {slope:.4f}, p = {p:.4f})."
                            if p < 0.05 else
                            f"No significant linear association (p = {p:.4f})."
                        )
                    }
                    fig_reg = plot_regression(x, y, slope, intercept,
                                              xlabel=exposure_col, ylabel=outcome_col)
                    fig_res = plot_residuals(x, y, slope, intercept)
                    figures += [fig_reg, fig_res]
                    st.pyplot(fig_reg)
                    with st.expander("📐 Residual Plot"):
                        st.pyplot(fig_res)

                elif recommended_test == "Logistic regression":
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.preprocessing import LabelEncoder
                    import scipy.stats as sp

                    x = pd.to_numeric(df[exposure_col], errors="coerce")
                    y = pd.to_numeric(df[outcome_col],  errors="coerce")
                    mask = x.notna() & y.notna()
                    x_arr = x[mask].values.reshape(-1, 1)
                    y_arr = y[mask].values

                    clf = LogisticRegression()
                    clf.fit(x_arr, y_arr)
                    coef = clf.coef_[0][0]
                    test_results = {
                        "coefficient": round(float(coef), 4),
                        "odds_ratio":  round(float(np.exp(coef)), 4),
                        "interpretation": f"OR = {np.exp(coef):.4f} per unit increase in {exposure_col}."
                    }

            except Exception as e:
                st.error(f"Test could not run: {e}")
                st.stop()

            # ── Display results ───────────────────────────────────────────────
            if test_results:
                st.subheader("📊 Results")
                for k, v in test_results.items():
                    if k != "interpretation":
                        st.metric(k.replace("_", " ").capitalize(), v)
                st.info(test_results.get("interpretation", ""))

            # ── Visualisations ────────────────────────────────────────────────
            if y_type == "continuous" and groups_data:
                fig_box = plot_boxplot(groups_data,
                                       labels=[str(g) for g in group_labels],
                                       title=f"{outcome_col} by {exposure_col}",
                                       ylabel=outcome_col)
                figures.append(fig_box)
                st.pyplot(fig_box)

                with st.expander("📊 Histograms per group"):
                    for g, label in zip(groups_data, group_labels):
                        fig_h = plot_histogram(g, label=str(label))
                        figures.append(fig_h)
                        st.pyplot(fig_h)

            elif y_type == "binary":
                counts = df[outcome_col].value_counts()
                fig_bar = plot_bar(
                    categories=counts.index.astype(str).tolist(),
                    values=counts.values.tolist(),
                    title=f"Distribution of {outcome_col}",
                    xlabel=outcome_col
                )
                figures.append(fig_bar)
                st.pyplot(fig_bar)

st.divider()

# ── 5. PDF Export ─────────────────────────────────────────────────────────────
st.header("5. Export Report")

student_name = st.text_input("Your name (optional, appears on report)")

if st.button("📄 Generate PDF Report", type="secondary"):
    if not recommended_test:
        st.warning("Configure your study design first.")
    else:
        with st.spinner("Building report..."):
            try:
                pdf_bytes = build_report(
                    meta              = meta,
                    recommended_test  = recommended_test,
                    test_results      = test_results,
                    teaching_content  = get_teaching_content(recommended_test) if teaching_mode else None,
                    diagnostics       = diag_results,
                    figures           = figures if figures else None,
                    student_name      = student_name,
                )
                st.download_button(
                    label     = "⬇️ Download PDF",
                    data      = pdf_bytes,
                    file_name = "biostats_report.pdf",
                    mime      = "application/pdf",
                )
                st.success("Report ready!")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
