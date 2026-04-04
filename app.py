import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import math
import random

st.set_page_config(page_title="Epidemiology Decision Simulator", layout="wide")

# ==================================================
# LOGIN GATE
# ==================================================

def check_credentials(username, password):
    """
    Add or remove users here directly.
    Format:  "username": "password"
    """
    USERS = {
        "marymathis": "epilab2024",
        "student1":   "epilab2024",
        "student2":   "epilab2024",
        "guest":      "epilab2024",
    }
    # Also check Streamlit Cloud secrets if available
    try:
        cloud_users = st.secrets.get("users", {})
        if cloud_users:
            return username in cloud_users and cloud_users[username] == password
    except Exception:
        pass
    return username in USERS and USERS[username] == password

def login_screen():
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("## 🧭 Epidemiology Decision Simulator")
        st.markdown("*EpiLab Interactive — licensed access only*")
        st.divider()
        st.markdown("**Please log in to continue.**")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Log In", type="primary", use_container_width=True):
            if check_credentials(username, password):
                st.session_state["authenticated"] = True
                st.session_state["current_user"] = username
                st.rerun()
            else:
                st.error("Incorrect username or password.")
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("Access issues? Contact your course instructor.")


if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login_screen()
    st.stop()

# ==================================================
# HELPER FUNCTIONS
# ==================================================

def draw_ci(label, estimate, ci_low, ci_high):
    significant = not (ci_low <= 1 <= ci_high)
    color = "#2e7d32" if significant else "#c0392b"
    sig_text = "CI does not cross 1 → statistically significant" if significant else "CI crosses 1 → not statistically significant"
    all_vals = [ci_low, ci_high, estimate, 1.0]
    span = max(all_vals) - min(all_vals)
    pad = max(span * 0.35, 0.4)
    x_min = max(0.001, min(all_vals) - pad)
    x_max = max(all_vals) + pad
    def to_pct(val):
        return round((val - x_min) / (x_max - x_min) * 100, 2)
    pct_low = to_pct(ci_low)
    pct_high = to_pct(ci_high)
    pct_est = to_pct(estimate)
    pct_null = to_pct(1.0)
    html = f"""
    <div style="background:#f9f9f9; border-radius:6px; padding:16px 12px 8px 12px; margin:8px 0 16px 0;">
      <div style="position:relative; height:60px; margin: 0 20px;">
        <div style="position:absolute; top:28px; left:0; right:0; height:2px; background:#cccccc;"></div>
        <div style="position:absolute; top:24px; left:{pct_low}%; width:{pct_high - pct_low}%; height:10px; background:{color}; border-radius:5px;"></div>
        <div style="position:absolute; top:20px; left:calc({pct_est}% - 9px); width:18px; height:18px; background:{color}; border-radius:50%;"></div>
        <div style="position:absolute; top:8px; left:{pct_null}%; width:2px; height:44px; background:#333; border-left: 2px dashed #333;"></div>
        <div style="position:absolute; top:0px; left:{pct_null}%; transform:translateX(-50%); font-size:11px; color:#333; white-space:nowrap;">1 (null)</div>
        <div style="position:absolute; top:46px; left:{pct_low}%; transform:translateX(-50%); font-size:11px; color:{color};">{round(ci_low,2)}</div>
        <div style="position:absolute; top:46px; left:{pct_est}%; transform:translateX(-50%); font-size:12px; color:{color}; font-weight:bold; white-space:nowrap;">{label} = {round(estimate,2)}</div>
        <div style="position:absolute; top:46px; left:{pct_high}%; transform:translateX(-50%); font-size:11px; color:{color};">{round(ci_high,2)}</div>
      </div>
      <div style="text-align:center; font-size:12px; color:{color}; font-style:italic; margin-top:28px;">{sig_text}</div>
    </div>"""
    st.markdown(html, unsafe_allow_html=True)


def chi2_explanation_expander(chi2_val, p_val, dof, table_array, col_names, row_names, tail_note=""):
    from scipy.stats import chi2_contingency, chi2 as chi2_dist
    _, _, _, expected = chi2_contingency(table_array)
    num_rows, num_cols = table_array.shape
    contributions = []
    for i in range(num_rows):
        for j in range(num_cols):
            o = table_array[i][j]; e = expected[i][j]
            if e > 0:
                contrib = (o - e)**2 / e
                contributions.append((contrib, i, j, o, e))
    contributions.sort(reverse=True)
    top_contrib, top_i, top_j, top_o, top_e = contributions[0]
    top_cell = f"{row_names[top_i]} / {col_names[top_j]}"
    top_direction = "more" if top_o > top_e else "fewer"
    top_pct = round(top_contrib / chi2_val * 100, 0)
    crit_val = round(chi2_dist.ppf(0.95, dof), 3)
    ratio = chi2_val / crit_val
    if ratio < 0.7:
        magnitude = f"well below the critical value ({crit_val}) — consistent with chance"
        magnitude_plain = f"Your χ² of {round(chi2_val,3)} is well below the critical value of {crit_val} needed for significance at df={dof}."
    elif ratio < 1.0:
        magnitude = f"approaching but below the critical value ({crit_val})"
        magnitude_plain = f"Your χ² of {round(chi2_val,3)} is close to but still below the critical value of {crit_val} at df={dof}."
    elif ratio < 2.0:
        magnitude = f"above the critical value ({crit_val}) — statistically significant"
        magnitude_plain = f"Your χ² of {round(chi2_val,3)} exceeds the critical value of {crit_val} at df={dof}."
    elif ratio < 4.0:
        magnitude = f"substantially above the critical value ({crit_val}) — strong evidence against H₀"
        magnitude_plain = f"Your χ² of {round(chi2_val,3)} is {round(ratio,1)}× the critical value of {crit_val} at df={dof}."
    else:
        magnitude = f"far above the critical value ({crit_val}) — very strong evidence against H₀"
        magnitude_plain = f"Your χ² of {round(chi2_val,3)} is {round(ratio,1)}× the critical value of {crit_val} at df={dof}."
    p_str = "< 0.0001" if p_val < 0.0001 else str(round(p_val, 4))
    grand_total = int(table_array.sum())
    total_outcome = int(table_array[:, 0].sum())
    overall_pct = round(total_outcome / grand_total * 100, 1)
    with st.expander("🔢 Show me the math — Chi-Square"):
        st.markdown(f"#### What does χ²({dof}) = {round(chi2_val, 3)} actually mean?")
        st.info(f"""
**In plain language:** The chi-square statistic measures how different your actual table is from what you'd see if there were absolutely no association.

Think of it this way: if exposure had **no effect at all** on outcome, you'd expect roughly the same proportion of {col_names[0]} ({overall_pct}% overall) in every exposure group. The chi-square statistic measures how far your table deviates from that pattern.

**{magnitude_plain}**

The largest single discrepancy came from the **{top_cell}** cell, which had {int(top_o)} cases observed vs. {round(top_e, 1)} expected — {top_direction} than the null would predict. This one cell alone drove {int(top_pct)}% of the total χ² value.

The resulting p-value of {p_str}{tail_note} means: if there were truly no association, a chi-square this large or larger would occur **{p_str} of the time** by chance. {'That is rare enough to reject H₀.' if p_val < 0.05 else 'That is not rare enough to reject H₀ at α = 0.05.'}
        """)
        st.markdown("---")
        st.markdown("**Step 1: Your observed counts (O)**")
        obs_df = pd.DataFrame(table_array, columns=col_names, index=row_names)
        st.table(obs_df)
        st.markdown("**Step 2: Expected counts (E) — if exposure and outcome were independent**")
        st.caption("E = (Row Total × Column Total) ÷ Grand Total.")
        exp_df = pd.DataFrame([[round(expected[i][j], 2) for j in range(num_cols)] for i in range(num_rows)], columns=col_names, index=row_names)
        st.table(exp_df)
        st.markdown("**Step 3: Each cell's contribution — (O − E)² ÷ E**")
        contrib_data = {}
        for j in range(num_cols):
            contrib_data[col_names[j]] = [round((table_array[i][j] - expected[i][j])**2 / expected[i][j], 3) if expected[i][j] > 0 else 0 for i in range(num_rows)]
        contrib_df = pd.DataFrame(contrib_data, index=row_names)
        st.table(contrib_df)
        st.markdown(f"""
**Step 4: Sum all contributions → χ²({dof}) = {round(chi2_val, 3)}**

Critical value at α = 0.05 with df = {dof}: **{crit_val}**
Your χ²: **{round(chi2_val, 3)}** → {"✅ exceeds critical value — reject H₀" if chi2_val >= crit_val else "❌ below critical value — fail to reject H₀"}

**Largest contributor:** **{top_cell}** — {int(top_o)} observed vs. {round(top_e, 1)} expected. This cell drove {int(top_pct)}% of the total χ².

**Step 5: Interpret**

p = {p_str}{tail_note} → {'We **reject** the null hypothesis.' if p_val < 0.05 else 'We **fail to reject** the null hypothesis.'}
        """)


def rr_or_explanation_expander(a, b, c, d, row_names, col_names, rr, or_val,
                                ci_low_rr, ci_high_rr, ci_low_or, ci_high_or,
                                is_cross_sectional=False):
    pabbr = "PR" if is_cross_sectional else "RR"
    plabel = "Prevalence Ratio (PR)" if is_cross_sectional else "Risk Ratio (RR)"
    risk_word = "prevalence" if is_cross_sectional else "risk"
    cell_style = "border:1px solid #aaa; padding:10px 16px; text-align:center; font-size:14px;"
    label_style = "border:1px solid #aaa; padding:10px 16px; text-align:center; font-size:12px; color:#555; background:#f5f5f5; font-weight:bold;"
    total_style = "border:1px solid #ccc; padding:10px 16px; text-align:center; font-size:13px; color:#555; background:#f0f0f0;"
    table_html = f"""
<div style="margin-bottom:16px;">
  <p style="font-weight:bold; font-size:13px; margin-bottom:8px;">2×2 Table Cell Labels</p>
  <table style="border-collapse:collapse; width:100%;">
    <tr>
      <td style="{label_style} background:#fff; border:none;"></td>
      <td style="{label_style}">{col_names[0]}</td>
      <td style="{label_style}">{col_names[1]}</td>
      <td style="{label_style}">Row Total</td>
    </tr>
    <tr>
      <td style="{label_style}">{row_names[0]}</td>
      <td style="{cell_style} background:#e8f4e8;"><span style="font-size:10px;color:#888;font-style:italic;">a = </span><span style="font-size:20px;font-weight:bold;color:#2e7d32;">{int(a)}</span></td>
      <td style="{cell_style} background:#fdecea;"><span style="font-size:10px;color:#888;font-style:italic;">b = </span><span style="font-size:20px;font-weight:bold;color:#c0392b;">{int(b)}</span></td>
      <td style="{total_style}">{int(a+b)}</td>
    </tr>
    <tr>
      <td style="{label_style}">{row_names[1]}</td>
      <td style="{cell_style} background:#fdecea;"><span style="font-size:10px;color:#888;font-style:italic;">c = </span><span style="font-size:20px;font-weight:bold;color:#c0392b;">{int(c)}</span></td>
      <td style="{cell_style} background:#e8f4e8;"><span style="font-size:10px;color:#888;font-style:italic;">d = </span><span style="font-size:20px;font-weight:bold;color:#2e7d32;">{int(d)}</span></td>
      <td style="{total_style}">{int(c+d)}</td>
    </tr>
    <tr>
      <td style="{label_style}">Col Total</td>
      <td style="{total_style}">{int(a+c)}</td>
      <td style="{total_style}">{int(b+d)}</td>
      <td style="{total_style}">{int(a+b+c+d)}</td>
    </tr>
  </table>
</div>
<div style="display:flex; gap:24px; margin-top:8px;">
  <div style="flex:1; background:#eef4fb; border-radius:6px; padding:12px 16px; font-size:13px;">
    <strong style="color:#1a4a7a;">{plabel} ({pabbr})</strong><br>
    {pabbr} = [a ÷ (a+b)] ÷ [c ÷ (c+d)]<br>
    = [{int(a)} ÷ {int(a+b)}] ÷ [{int(c)} ÷ {int(c+d)}]<br>
    = {round(a/(a+b),4)} ÷ {round(c/(c+d),4)}<br>
    = <strong style="color:#1a4a7a;">{round(rr,3)}</strong><br>
    <span style="font-size:11px;color:#666;">95% CI: ({round(ci_low_rr,3)}, {round(ci_high_rr,3)})</span>
  </div>
  <div style="flex:1; background:#fdf3ee; border-radius:6px; padding:12px 16px; font-size:13px;">
    <strong style="color:#8a4a1a;">Odds Ratio (OR)</strong><br>
    OR = (a × d) ÷ (b × c)<br>
    = ({int(a)} × {int(d)}) ÷ ({int(b)} × {int(c)})<br>
    = {int(a*d)} ÷ {int(b*c)}<br>
    = <strong style="color:#8a4a1a;">{round(or_val,3)}</strong><br>
    <span style="font-size:11px;color:#666;">95% CI: ({round(ci_low_or,3)}, {round(ci_high_or,3)})</span>
  </div>
</div>
"""
    with st.expander(f"🔢 Show me the math — {plabel} & OR"):
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown(f"""
**What does each cell mean?**

| Cell | Meaning |
|------|---------|
| **a** | {row_names[0]} who **have** {col_names[0]} |
| **b** | {row_names[0]} who **do not have** {col_names[0]} |
| **c** | {row_names[1]} who **have** {col_names[0]} |
| **d** | {row_names[1]} who **do not have** {col_names[0]} |

**{plabel} ({pabbr})** compares the {risk_word} in each row using **row proportions** (a ÷ row total, c ÷ row total).

**Odds Ratio (OR)** uses the **cross-product** (a×d) ÷ (b×c).

**When do {pabbr} and OR agree?** When the outcome is rare (<10%), OR ≈ {pabbr}. As the outcome becomes more common, OR diverges further from 1.
        """)


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

NAV_STRUCTURE = [
    ("MODULE 1 — Study Design & Causation", [
        ("study_designs",       "🔬", "Study Designs",             "Cohort, case-control, RCT, crossover"),
        ("bias",                "⚠️", "Bias",                       "Selection, recall, misclassification"),
        ("confounding",         "🔀", "Confounding & Effect Mod.",  "Control methods, stratification, DAGs"),
        ("causal_inference",    "🔗", "Causal Inference",           "Bradford Hill criteria"),
    ]),
    ("MODULE 2 — Foundations", [
        ("disease_frequency",   "📊", "Disease Frequency",          "Incidence, prevalence, epidemic curves"),
        ("screening",           "🩺", "Screening & Diagnostics",    "Sensitivity, specificity, PPV, NPV"),
    ]),
    ("MODULE 3 — Measures & Analysis", [
        ("measures_association","📈", "Measures of Association",    "RR, OR, PR, IRR, chi-square"),
        ("advanced_measures",   "📉", "Advanced Epi Measures",      "PAR, SMR, AR%, NNT, HR"),
        ("standardization",     "📏", "Standardization",            "Direct, indirect, SMR, age-adjustment"),
        ("hypothesis_testing",  "🧪", "Hypothesis Testing & Power", "p-values, CI, Type I/II, sample size"),
    ]),
    ("MODULE 4 — Practice", [
        ("practice_design",     "🎯", "Study Design",               "Classify design, outcome, exposure"),
        ("practice_advanced",   "🎯", "Advanced Measures",          "Select the right measure"),
        ("practice_confounding","🎯", "Confounding & Bias",         "Identify and reason through bias"),
        ("practice_screening",  "🎯", "Screening & Frequency",      "PPV, attack rates, epi curves"),
    ]),
    ("Reference", [
        ("glossary",            "📖", "Glossary",                   "All key terms and formulas"),
    ]),
]


if "current_page" not in st.session_state:
    st.session_state["current_page"] = "home"

# ── NAVIGATION ────────────────────────────────────────────────
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "study_designs"

ALL_PAGES = [
    ("📐 Study Designs",                "study_designs"),
    ("⚠️ Bias",                          "bias"),
    ("🔀 Confounding & Effect Mod.",     "confounding"),
    ("🔗 Causal Inference",              "causal_inference"),
    ("📊 Disease Frequency",             "disease_frequency"),
    ("🔬 Screening & Diagnostics",       "screening"),
    ("📈 Measures of Association",       "measures_association"),
    ("📉 Advanced Epi Measures",         "advanced_measures"),
    ("📏 Standardization",               "standardization"),
    ("🧪 Hypothesis Testing & Power",    "hypothesis_testing"),
    ("🎯 Practice: Study Design",        "practice_design"),
    ("🎯 Practice: Advanced Measures",   "practice_advanced"),
    ("🎯 Practice: Confounding & Bias",  "practice_confounding"),
    ("🎯 Practice: Screening & Freq.",   "practice_screening"),
    ("📖 Glossary",                      "glossary"),
]
PAGE_LABELS = [p[0] for p in ALL_PAGES]
PAGE_KEYS   = [p[1] for p in ALL_PAGES]

with st.sidebar:
    st.markdown(f"### 🧭 EpiLab")
    st.caption(f"Logged in as **{st.session_state.get('current_user', '')}**")
    if st.button("↩ Log Out", key="logout_btn"):
        st.session_state["authenticated"] = False
        st.rerun()
    st.divider()
    cur = st.session_state.get("current_page", "study_designs")
    cur_idx = PAGE_KEYS.index(cur) if cur in PAGE_KEYS else 0
    chosen = st.radio("", PAGE_LABELS, index=cur_idx, label_visibility="collapsed")
    chosen_key = PAGE_KEYS[PAGE_LABELS.index(chosen)]
    if chosen_key != cur:
        st.session_state["current_page"] = chosen_key
        st.rerun()

current_page = st.session_state.get("current_page", "study_designs")



# ==================================================
# MODULE 1: STUDY DESIGNS
# ==================================================
if current_page == "study_designs":
    st.title("📐 Study Designs")
    st.markdown("Epidemiologic study design determines what measure of association you can calculate, what biases are possible, and how strong the evidence for causation can be.")

    section = st.radio("Section:", [
        "1️⃣ Design Overview",
        "2️⃣ Design Selector",
        "3️⃣ RCT & Evidence Hierarchy"
    ], horizontal=True)
    st.divider()

    if section == "1️⃣ Design Overview":
        st.subheader("The Core Question: How Did Sampling Begin?")
        st.info("The single most important question in identifying a study design is: **where did the researcher start sampling from?** Exposure? Outcome? Neither? Individual or group?")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("#### 🟩 Cohort")
            st.markdown("**Starts with:** Exposure status")
            st.markdown("**Logic:** Groups defined by exposure → followed to see who develops outcome")
            st.markdown("**Prospective:** Collect data going forward")
            st.markdown("**Retrospective:** Use historical records — exposure still defined *before* outcome")
            st.markdown("```\n① Exposure defined\n        ↓\n② Outcome measured\n```")
            st.success("Produces: **RR / IRR**")
            st.markdown("*Best for: common exposures, multiple outcomes*")

        with col_b:
            st.markdown("#### 🟦 Case-Control")
            st.markdown("**Starts with:** Outcome status")
            st.markdown("**Logic:** Cases (have disease) + Controls (don't) → look **backward** at past exposure")
            st.markdown("**Matched variant:** Each case paired with controls on confounders — controls confounding by design")
            st.markdown("```\n① Past exposure (recalled)\n        ↑\n② Start: disease yes/no\n```")
            st.info("Produces: **OR**")
            st.markdown("*Best for: rare diseases, long latency*")

        with col_c:
            st.markdown("#### 🟧 Cross-Sectional")
            st.markdown("**Starts with:** A random sample")
            st.markdown("**Logic:** Exposure and outcome measured **simultaneously** — a snapshot")
            st.markdown("**Cannot** establish temporal order (which came first?)")
            st.markdown("")
            st.markdown("```\nExposure ─┐\n          ├─ same moment\nOutcome  ─┘\n```")
            st.warning("Produces: **PR**")
            st.markdown("*Best for: prevalence estimates, hypothesis generation*")

        st.divider()

        # Ecological study
        st.markdown("#### 🌍 Ecological Study")
        ecol1, ecol2, ecol3 = st.columns(3)
        with ecol1:
            st.markdown("**Unit of analysis:** Groups or populations — not individuals")
            st.markdown("**Logic:** Compare average exposure and average outcome rates across groups (countries, cities, time periods)")
            st.markdown("**Examples:** Country-level fat intake vs. breast cancer rates; air pollution index vs. city-wide asthma hospitalizations")
        with ecol2:
            st.markdown("**Timeline:**")
            st.markdown("```\nGroup A: Exposure rate → Outcome rate\nGroup B: Exposure rate → Outcome rate\nGroup C: Exposure rate → Outcome rate\n         ↓\n   Compare across groups\n```")
            st.markdown("*Data come from registries, surveillance systems, or administrative records*")
        with ecol3:
            st.markdown("**Best for:** Hypothesis generation, policy surveillance, studying exposures that vary at the population level (water fluoridation, legislation)")
            st.markdown("**Not for:** Establishing individual-level causation")
            st.error("⚠️ **Ecological fallacy** — group-level associations may not hold at the individual level")
            st.markdown("Produces: **Correlation coefficient / Rate ratio**")

        st.warning("""
**⚠️ The Ecological Fallacy (Aggregation Bias)**

Just because countries with higher fat intake have higher breast cancer rates does NOT mean individuals who eat more fat have higher breast cancer risk. The association at the group level may be driven by confounding variables that also vary between countries (wealth, screening rates, reproductive factors), not by fat intake itself.

**Classic example:** Countries with more TVs per capita have lower infant mortality — TVs don't protect infants. Wealth causes both.
        """)

        st.divider()
        st.markdown("#### 🟪 Case-Crossover")
        ccol1, ccol2, ccol3 = st.columns(3)
        with ccol1:
            st.markdown("**Starts with:** Cases only (people who had the event)")
            st.markdown("**Logic:** Compare each person's exposure just before their event (hazard period) vs. at a matched control time for the same person (no event)")
        with ccol2:
            st.markdown("**Timeline:**")
            st.markdown("```\nControl period  →  Hazard period\n(same person,      (just before\n no event)          the event)\n```")
            st.markdown("*Eliminates between-person confounding — each person is their own control*")
        with ccol3:
            st.markdown("**Best for:** Transient exposures with acute effects (air pollution → MI, alcohol → injury)")
            st.markdown("**Not for:** Chronic, stable exposures")
            st.markdown("Produces: **OR**")
        st.markdown("*Key question: Was the person more exposed just before the event than during a typical period?*")

        st.divider()
        st.markdown("#### 🔵 Randomized Controlled Trial (RCT)")
        rcol1, rcol2 = st.columns(2)
        with rcol1:
            st.markdown("**Logic:** Participants **randomly assigned** to intervention or control. Randomization distributes known and unknown confounders equally across groups.")
            st.markdown("**Gold standard** for establishing causation.")
        with rcol2:
            st.markdown("**Limitations:** Expensive, ethical constraints (can't randomize harmful exposures), Hawthorne effect, may not generalize (external validity)")
            st.markdown("Produces: **RR / HR / RD**")

        with st.expander("📊 Study Design Comparison Table"):
            comparison_df = pd.DataFrame({
                "Design": ["Cohort","Case-Control","Cross-Sectional","Ecological","Case-Crossover","RCT"],
                "Unit of analysis": ["Individual","Individual","Individual","Group / Population","Individual (self-matched)","Individual"],
                "Sampling starts from": ["Exposure","Outcome","Population sample","Population aggregates","Cases only","Random assignment"],
                "Temporal direction": ["Forward (or records)","Backward","Simultaneous","Varies","Both (same person)","Forward"],
                "Measure produced": ["RR / IRR","OR","PR","Correlation / Rate ratio","OR","RR / HR / RD"],
                "Best for": ["Common exposures, multiple outcomes","Rare diseases, long latency","Prevalence, hypothesis generation","Policy surveillance, hypothesis generation","Transient exposures, acute effects","Causal evidence, interventions"],
                "Main weakness": ["Expensive; loss to follow-up","Recall bias; control selection","No temporality","Ecological fallacy — can't infer individual risk","Only transient exposures","Expensive; ethical limits; external validity"]
            })
            st.table(comparison_df)

    elif section == "2️⃣ Design Selector":
        st.subheader("Design Selector")
        st.markdown("Work through the decision tree to identify the correct study design.")
        q1 = st.radio("1. Is this an experimental study where the researcher assigns exposure?", ["— Select —","Yes — researcher assigns","No — observational"], key="ds_q1")
        if q1 == "Yes — researcher assigns":
            st.success("**Randomized Controlled Trial (RCT)** — the researcher controls exposure assignment. If randomized, this is an RCT (or quasi-experimental if not randomized).")
        elif q1 == "No — observational":
            q2 = st.radio("2. What is the unit of analysis?", ["— Select —","Individuals","Groups or populations (countries, cities, time periods)"], key="ds_q2")
            if q2 == "Groups or populations (countries, cities, time periods)":
                st.success("**Ecological Study** — exposure and outcome are measured at the group level, not for individuals. Useful for hypothesis generation and policy surveillance, but cannot establish individual-level causation. Beware the ecological fallacy.")
            elif q2 == "Individuals":
                q3 = st.radio("3. How were participants sampled?", ["— Select —","By exposure status","By outcome (disease) status","Neither — random sample or whole population at one time","Cases only (each compared to themselves at another time)"], key="ds_q3")
                if q3 == "By exposure status":
                    st.success("**Cohort Study** — grouped by exposure, then followed to outcome. Can be prospective (going forward) or retrospective (historical records), but always exposure → outcome in logic.")
                elif q3 == "By outcome (disease) status":
                    st.success("**Case-Control Study** — cases (have disease) and controls (don't) are identified, then past exposure is assessed. Always retrospective in logic.")
                elif q3 == "Neither — random sample or whole population at one time":
                    st.success("**Cross-Sectional Study** — exposure and outcome measured simultaneously. No temporal ordering possible.")
                elif q3 == "Cases only (each compared to themselves at another time)":
                    st.success("**Case-Crossover Study** — each case serves as their own control. Exposure during a hazard period is compared to exposure during a control period for the same person.")

    elif section == "3️⃣ RCT & Evidence Hierarchy":
        st.subheader("Evidence Hierarchy")
        st.markdown("Not all study designs provide equally strong evidence for causation. The hierarchy reflects **internal validity** — how confident can we be the exposure causes the outcome? Higher levels have stronger designs for ruling out alternative explanations.")

        import streamlit.components.v1 as _comp_eh

        LEVELS = [
            {
                "num": 1,
                "title": "Systematic Reviews & Meta-Analyses",
                "badge": "STRONGEST",
                "badge_color": "#16a34a",
                "desc": "Pool and synthesize results across multiple RCTs using statistical methods. When studies are consistent, this provides the most reliable overall estimate of an effect.",
                "strengths": ["Reduces impact of any single study's chance findings", "Quantifies heterogeneity across studies", "Most comprehensive evidence base"],
                "limits": ["Only as good as the studies included", "Publication bias can distort pooled estimates", "Heterogeneity can make pooling misleading"],
                "measure": "Pooled RR / OR / HR",
                "bar_pct": 100,
                "bar_color": "#1d4ed8",
                "num_bg": "#1e3a8a",
            },
            {
                "num": 2,
                "title": "Randomized Controlled Trials (RCTs)",
                "badge": "GOLD STANDARD",
                "badge_color": "#b45309",
                "desc": "Random assignment distributes both measured and unmeasured confounders equally across groups. The only design that can establish causation without residual confounding concern.",
                "strengths": ["Controls for known and unknown confounders", "Clear temporal order", "Blinding possible"],
                "limits": ["Expensive and time-consuming", "Ethical limits (can't randomize harmful exposures)", "May not reflect real-world populations"],
                "measure": "RR / HR / Risk Difference",
                "bar_pct": 86,
                "bar_color": "#1d4ed8",
                "num_bg": "#1e40af",
            },
            {
                "num": 3,
                "title": "Prospective Cohort Studies",
                "badge": "OBSERVATIONAL",
                "badge_color": "#0369a1",
                "desc": "Participants classified by exposure at baseline, then followed forward to measure new outcomes. Establishes temporal order — exposure clearly precedes outcome.",
                "strengths": ["Clear temporality", "Can study multiple outcomes", "Measures incidence directly"],
                "limits": ["Residual confounding possible", "Expensive for rare diseases", "Loss to follow-up bias"],
                "measure": "RR / IRR / HR",
                "bar_pct": 74,
                "bar_color": "#2563eb",
                "num_bg": "#1d4ed8",
            },
            {
                "num": 4,
                "title": "Retrospective Cohort / Case-Control",
                "badge": "OBSERVATIONAL",
                "badge_color": "#0369a1",
                "desc": "Retrospective cohort uses historical records; case-control recruits by outcome status and looks back at exposure. Both are efficient but introduce more opportunities for bias.",
                "strengths": ["Efficient for rare outcomes (case-control)", "Faster and cheaper than prospective", "Can use existing data"],
                "limits": ["Recall bias (case-control)", "Selection bias in control group", "Confounding harder to rule out"],
                "measure": "OR (case-control) / RR (retro cohort)",
                "bar_pct": 62,
                "bar_color": "#3b82f6",
                "num_bg": "#2563eb",
            },
            {
                "num": 5,
                "title": "Cross-Sectional Studies",
                "badge": "OBSERVATIONAL",
                "badge_color": "#0369a1",
                "desc": "Exposure and outcome measured simultaneously in a single snapshot. Cannot determine which came first — useful for estimating prevalence and generating hypotheses.",
                "strengths": ["Fast and inexpensive", "Good for prevalence estimates", "Useful for generating hypotheses"],
                "limits": ["Cannot establish temporality", "Prevalence-incidence bias", "Not suitable for rare conditions"],
                "measure": "Prevalence Ratio (PR)",
                "bar_pct": 48,
                "bar_color": "#60a5fa",
                "num_bg": "#3b82f6",
            },
            {
                "num": 6,
                "title": "Ecological Studies",
                "badge": "GROUP-LEVEL",
                "badge_color": "#6d28d9",
                "desc": "Exposure and outcome measured at the group or population level — not for individuals. Compares aggregate rates across countries, cities, or time periods. Cannot establish individual-level causation.",
                "strengths": ["Inexpensive — uses existing data", "Good for generating hypotheses", "Useful for studying population-level exposures (policy, water supply, legislation)", "Can study exposures with little individual variation"],
                "limits": ["Ecological fallacy — group association ≠ individual risk", "Cannot control for individual-level confounders", "Correlation does not imply causation at individual level", "Aggregation bias can mask or create spurious associations"],
                "measure": "Correlation coefficient / Rate ratio",
                "bar_pct": 34,
                "bar_color": "#818cf8",
                "num_bg": "#4f46e5",
            },
            {
                "num": 7,
                "title": "Case Reports & Expert Opinion",
                "badge": "LOWEST",
                "badge_color": "#9a3412",
                "desc": "Individual case descriptions or consensus opinions without a systematic comparison group. Essential for identifying new conditions and rare adverse events, but cannot establish causation.",
                "strengths": ["Critical for identifying new diseases", "Detects rare adverse drug reactions", "Generates hypotheses quickly"],
                "limits": ["No comparison group", "High potential for bias", "Cannot quantify associations"],
                "measure": "Descriptive only",
                "bar_pct": 20,
                "bar_color": "#93c5fd",
                "num_bg": "#60a5fa",
            },
        ]

        _is_dark = False  # theme system removed — always light
        _card_bg    = "#1e2130" if _is_dark else "#ffffff"
        _card_bdr   = "#2e3246" if _is_dark else "#e5e7eb"
        _body_txt   = "#d1d5db" if _is_dark else "#374151"
        _head_txt   = "#f1f5f9" if _is_dark else "#111827"
        _sub_bg     = "#252836" if _is_dark else "#f8fafc"
        _sub_bdr    = "#3a3f52" if _is_dark else "#e2e8f0"
        _str_head   = "#4ade80" if _is_dark else "#166534"
        _lim_head   = "#f87171" if _is_dark else "#991b1b"
        _meas_bg    = "#1a1d27" if _is_dark else "#f0f9ff"
        _meas_txt   = "#93c5fd" if _is_dark else "#1e40af"
        _meas_bdr   = "#2563eb" if _is_dark else "#bfdbfe"

        cards_html = ""
        for lv in LEVELS:
            strengths_li = "".join(f'<li>{s}</li>' for s in lv["strengths"])
            limits_li    = "".join(f'<li>{l}</li>' for l in lv["limits"])
            cards_html += f"""
<div class="card">
  <div class="card-header">
    <div class="num-badge" style="background:{lv['num_bg']};">{lv['num']}</div>
    <div class="header-mid">
      <div class="card-title">{lv['title']}</div>
      <div class="card-desc">{lv['desc']}</div>
    </div>
    <span class="badge" style="background:{lv['badge_color']}22;color:{lv['badge_color']};border:1px solid {lv['badge_color']}55;">{lv['badge']}</span>
  </div>
  <div class="bar-row">
    <div class="bar-track">
      <div class="bar-fill" style="width:{lv['bar_pct']}%;background:linear-gradient(90deg,{lv['bar_color']},{lv['bar_color']}99);"></div>
    </div>
    <span class="bar-label">Evidence strength</span>
  </div>
  <div class="card-body">
    <div class="sl-box">
      <div class="sl-head" style="color:{_str_head};">✓ Strengths</div>
      <ul>{strengths_li}</ul>
    </div>
    <div class="sl-box">
      <div class="sl-head" style="color:{_lim_head};">✗ Limitations</div>
      <ul>{limits_li}</ul>
    </div>
    <div class="measure-pill" style="background:{_meas_bg};color:{_meas_txt};border:1px solid {_meas_bdr};">
      <span style="font-size:10px;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;opacity:0.7;">Typical measure</span>
      <span style="font-weight:700;margin-left:8px;">{lv['measure']}</span>
    </div>
  </div>
</div>"""

        full_html = f"""<!DOCTYPE html><html><head><style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:transparent; padding:4px 0 12px 0; }}
.card {{
  background:{_card_bg};
  border:1px solid {_card_bdr};
  border-radius:12px;
  margin:0 0 10px 0;
  overflow:hidden;
  transition:box-shadow 0.2s;
}}
.card:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,{"0.4" if _is_dark else "0.08"}); }}
.card-header {{
  display:flex; align-items:flex-start; gap:14px;
  padding:14px 16px 10px 16px;
}}
.num-badge {{
  width:32px; height:32px; border-radius:8px;
  display:flex; align-items:center; justify-content:center;
  font-size:15px; font-weight:800; color:white; flex-shrink:0;
}}
.header-mid {{ flex:1; min-width:0; }}
.card-title {{ font-size:15px; font-weight:700; color:{_head_txt}; line-height:1.3; }}
.card-desc  {{ font-size:12.5px; color:{_body_txt}; margin-top:3px; line-height:1.5; }}
.badge {{
  flex-shrink:0; font-size:9.5px; font-weight:700;
  letter-spacing:0.08em; text-transform:uppercase;
  padding:3px 8px; border-radius:20px; white-space:nowrap; margin-top:2px;
}}
.bar-row {{
  display:flex; align-items:center; gap:10px;
  padding:0 16px 10px 62px;
}}
.bar-track {{
  flex:1; height:5px; background:{_sub_bdr}; border-radius:3px; overflow:hidden;
}}
.bar-fill {{ height:100%; border-radius:3px; }}
.bar-label {{ font-size:10px; color:{_body_txt}; opacity:0.6; white-space:nowrap; }}
.card-body {{
  background:{_sub_bg};
  border-top:1px solid {_sub_bdr};
  padding:12px 16px;
  display:flex; flex-wrap:wrap; gap:12px; align-items:flex-start;
}}
.sl-box {{ flex:1; min-width:180px; }}
.sl-head {{ font-size:11px; font-weight:700; margin-bottom:5px; }}
.sl-box ul {{ list-style:none; padding:0; }}
.sl-box li {{ font-size:12px; color:{_body_txt}; line-height:1.5; padding:1px 0; }}
.sl-box li::before {{ content:"· "; color:#94a3b8; }}
.measure-pill {{
  display:flex; align-items:center; border-radius:6px;
  padding:7px 12px; font-size:12.5px;
  align-self:flex-end; white-space:nowrap;
}}
</style></head><body>
{cards_html}
</body></html>"""

        _comp_eh.html(full_html, height=len(LEVELS) * 195 + 20, scrolling=True)

        st.divider()
        with st.expander("💡 Important caveats about the hierarchy"):
            st.markdown("""
**The hierarchy is about internal validity, not overall usefulness.** Higher levels give stronger evidence for causation, but this doesn't make lower levels unimportant:

- **Case reports** are often the first signal of a new disease or drug reaction — without them, we'd never know to design an RCT
- **Observational studies** are the only ethical option when the exposure is harmful (smoking, radiation, poverty)
- **A well-conducted cohort study** beats a poorly designed RCT with low adherence and massive dropout
- **External validity** (generalizability) often favors observational studies — RCT participants are highly selected

Think of the hierarchy as a guide for *how much confounding you need to worry about*, not as a ranking of which studies matter.
            """)


    st.markdown("---")
    st.markdown("*Strong epidemiologists think structurally before computing.*")


# ==================================================
# MODULE 1: BIAS
# ==================================================
elif current_page == "bias":
    st.title("⚠️ Bias")
    st.markdown("Bias is a **systematic error** that leads to an incorrect estimate of the association between exposure and outcome. Unlike random error, bias does not average out with larger sample sizes.")
    st.info("**Key principle:** Bias operates in one direction — it either inflates or deflates the true association. Recognizing the type of bias helps you predict whether your result is likely an over- or under-estimate.")

    bias_section = st.radio("Section:", ["1️⃣ Selection Bias", "2️⃣ Information Bias", "3️⃣ Bias Direction Exercise"], horizontal=True)
    st.divider()

    if bias_section == "1️⃣ Selection Bias":
        st.subheader("Selection Bias")
        st.markdown("Selection bias occurs when **who is included in the study** is related to both exposure and outcome, creating a distorted sample that doesn't represent the target population.")

        with st.expander("🏥 Berkson's Bias (Hospital Admission Bias)", expanded=True):
            st.markdown("""
**What it is:** When both exposure and disease independently increase the probability of hospital admission, hospitalized patients show a spurious negative association between exposure and disease — even if no true association exists in the general population.

**Classic example:** Studying whether respiratory disease and bone fractures are associated. In the general population, they're unrelated. But in a hospital, patients with *only* respiratory disease (no fracture) and patients with *only* a fracture (no respiratory disease) are both admitted. Patients with both conditions are also there — but so are those with neither, just less commonly. The hospital sample distorts the apparent association.

**Study designs most affected:** Case-control studies using hospital-based controls.

**How to minimize:** Use population-based controls rather than hospital controls.
            """)

        with st.expander("🏃 Healthy Worker Effect"):
            st.markdown("""
**What it is:** Employed workers are systematically healthier than the general population because people who are very ill, disabled, or near death are less likely to be employed. When occupational cohorts are compared to the general population, mortality appears lower — not because of any protective effect of the job, but because of who gets selected into employment.

**Effect on SMR:** Causes SMR < 1, potentially masking true occupational hazards.

**How to minimize:** Compare workers to other workers (internal comparison) rather than to the general population.
            """)

        with st.expander("📉 Loss to Follow-Up / Non-Response Bias"):
            st.markdown("""
**What it is:** Participants who drop out of a study or refuse to participate differ systematically from those who stay. If dropout is related to both exposure and outcome, the remaining sample is not representative.

**Example:** In a cohort studying smoking and cancer, heavy smokers who develop early symptoms may be more likely to drop out (due to illness) before the outcome is recorded. This could underestimate the true RR.

**Non-response bias:** Survey participants who respond tend to be more health-conscious, educated, or concerned about the topic than non-responders — biasing prevalence estimates.

**How to minimize:** Track reasons for dropout; compare baseline characteristics of dropouts vs. completers; maximize follow-up rates.
            """)

        with st.expander("🔄 Volunteer / Self-Selection Bias"):
            st.markdown("""
**What it is:** People who volunteer for studies tend to be healthier, more educated, and more health-conscious than those who don't. This limits generalizability (external validity) even if the internal results are valid.

**Example:** Clinical trial volunteers often have fewer comorbidities, better adherence, and more social support than typical patients — so trial results may overestimate effectiveness in real-world practice.
            """)

    elif bias_section == "2️⃣ Information Bias":
        st.subheader("Information Bias")
        st.markdown("Information bias (also called **misclassification bias**) occurs when exposure or outcome is measured incorrectly. The consequences depend on whether the error is the same in all groups (non-differential) or differs between groups (differential).")

        with st.expander("📊 Non-Differential Misclassification", expanded=True):
            st.markdown("""
**What it is:** Measurement error that is **the same** in exposed and unexposed groups (or cases and controls). The error is random with respect to the other variable.

**Effect on the result:** Biases the measure of association **toward the null** (toward RR = 1 or OR = 1). You underestimate the true association. A finding of "no association" could be real — or it could be a true association that's been attenuated by non-differential misclassification.

**Example:** A self-reported physical activity questionnaire that systematically under-reports activity in *both* high-risk and low-risk individuals. The resulting OR or RR will be closer to 1 than the truth.

**Key insight:** Even if measurement error is random, it still biases results — it just biases them in a predictable direction (toward null).
            """)

        with st.expander("⚠️ Differential Misclassification"):
            st.markdown("""
**What it is:** Measurement error that **differs** between the groups being compared. The most common form is **recall bias** in case-control studies.

**Effect on the result:** Can bias the measure of association in either direction — away from or toward the null. The direction depends on the specific pattern of error. This is more dangerous than non-differential misclassification because you can't predict which way it goes.

**Recall bias (classic example):** In a case-control study of birth defects, mothers of affected children (cases) may search their memories more thoroughly for exposures during pregnancy than mothers of healthy children (controls). Cases over-report past exposures compared to controls — inflating the OR even if no true association exists.

**Interviewer bias:** If interviewers know who is a case vs. control (or exposed vs. unexposed), they may probe more thoroughly for certain exposures — introducing systematic differential error.

**How to minimize:** Blind interviewers to case/control status; use objective biomarkers instead of self-report; use standardized instruments.
            """)

        with st.expander("🔬 Visual: Non-Differential vs. Differential"):
            st.markdown("""
| Feature | Non-Differential | Differential |
|---|---|---|
| Error equal across groups? | ✅ Yes | ❌ No |
| Direction of bias | Toward null (attenuates) | Either direction |
| Predictability | Predictable | Unpredictable |
| Most common in | Cohort studies (outcome misclassification) | Case-control (recall bias) |
| Can produce false positive? | No | Yes |
| Can cause false negative? | Yes | Yes |
            """)

    elif bias_section == "3️⃣ Bias Direction Exercise":
        st.subheader("Bias Direction Exercise")
        st.markdown("For each scenario, identify the type of bias and predict which direction it pushes the result.")

        BIAS_SCENARIOS = [
            {
                "id": "b1",
                "title": "Scenario A: Case-Control of Alcohol & Esophageal Cancer",
                "description": "A hospital-based case-control study examines alcohol and esophageal cancer. Cases are cancer patients; controls are other hospitalized patients. However, heavy drinkers are hospitalized at higher rates for many conditions. Controls are thus more likely to be heavy drinkers than the general population.",
                "correct_type": "Selection bias (Berkson's bias)",
                "correct_direction": "Toward null — OR underestimates the true association",
                "explanation": "Controls are enriched with heavy drinkers because alcohol-related conditions independently increase hospitalization. This makes the control group more exposed than the general population would be, shrinking the apparent difference between cases and controls → OR biased toward 1.",
                "types": ["Selection bias (Berkson's bias)", "Recall bias (differential misclassification)", "Non-differential misclassification", "Healthy worker effect"],
                "directions": ["Toward null — OR underestimates the true association", "Away from null — OR overestimates", "Cannot determine direction"]
            },
            {
                "id": "b2",
                "title": "Scenario B: Cohort Study of Diet & Heart Disease",
                "description": "A cohort study uses a food frequency questionnaire to classify participants as high or low saturated fat consumers. The questionnaire has known measurement error — it misclassifies about 20% of high consumers as low, and 20% of low consumers as high. This error rate is the same regardless of whether participants later develop heart disease.",
                "correct_type": "Non-differential misclassification",
                "correct_direction": "Toward null — RR underestimates the true association",
                "explanation": "When exposure misclassification rate is the same in cases and non-cases (non-differential), the result is bias toward the null. Some truly high-fat consumers are classified as low-fat, diluting the exposed group. The estimated RR will be closer to 1 than the truth.",
                "types": ["Selection bias (Berkson's bias)", "Recall bias (differential misclassification)", "Non-differential misclassification", "Healthy worker effect"],
                "directions": ["Toward null — RR underestimates the true association", "Away from null — RR overestimates", "Cannot determine direction"]
            },
            {
                "id": "b3",
                "title": "Scenario C: Case-Control of Pesticide Exposure & Parkinson's Disease",
                "description": "Cases (Parkinson's patients) and controls are interviewed about lifetime pesticide exposure. Cases have spent years wondering what caused their disease and have discussed it extensively with their neurologist. Controls have no reason to think carefully about past pesticide exposures.",
                "correct_type": "Recall bias (differential misclassification)",
                "correct_direction": "Away from null — OR overestimates",
                "explanation": "Cases over-report past pesticide exposures because they've reflected on their history more than controls. This inflates the apparent association — the OR is larger than the true association. This is classic recall bias: differential misclassification of exposure between cases and controls.",
                "types": ["Selection bias (Berkson's bias)", "Recall bias (differential misclassification)", "Non-differential misclassification", "Healthy worker effect"],
                "directions": ["Toward null — OR underestimates the true association", "Away from null — OR overestimates", "Cannot determine direction"]
            },
            {
                "id": "b4",
                "title": "Scenario D: Occupational Cohort of Benzene & Leukemia",
                "description": "A chemical plant cohort is followed for 20 years and mortality compared to the US general population. SMR = 0.78. The plant medical director concludes benzene exposure is protective.",
                "correct_type": "Healthy worker effect",
                "correct_direction": "Toward null — SMR underestimates true occupational risk",
                "explanation": "Workers are healthier than the general population (which includes the very ill, disabled, and those near death). SMR < 1 may simply reflect this selection, not any protective effect of benzene. The healthy worker effect biases mortality comparisons toward null — potentially masking true excess risk.",
                "types": ["Selection bias (Berkson's bias)", "Recall bias (differential misclassification)", "Non-differential misclassification", "Healthy worker effect"],
                "directions": ["Toward null — SMR underestimates true occupational risk", "Away from null — SMR overestimates risk", "Cannot determine direction"]
            },
        ]

        if "bias_rc" not in st.session_state:
            st.session_state["bias_rc"] = 0
        rc = st.session_state["bias_rc"]

        col_hdr, col_rst = st.columns([5,1])
        with col_rst:
            if st.button("🔄 Reset", key="reset_bias"):
                st.session_state["bias_rc"] += 1
                st.rerun()

        for sc in BIAS_SCENARIOS:
            st.divider()
            st.subheader(sc["title"])
            st.markdown(sc["description"])
            sid = sc["id"]
            submitted_key = f"bias_submitted_{sid}_{rc}"
            already_submitted = st.session_state.get(submitted_key, False)

            type_choice = st.selectbox("What type of bias is this?", ["— Select —"] + sc["types"], key=f"bias_type_{sid}_{rc}", disabled=already_submitted)
            dir_choice = st.selectbox("How does it bias the result?", ["— Select —"] + sc["directions"], key=f"bias_dir_{sid}_{rc}", disabled=already_submitted)

            all_selected = type_choice not in [None,"— Select —"] and dir_choice not in [None,"— Select —"]
            if not already_submitted and all_selected:
                if st.button("Submit", key=f"bias_submit_{sid}_{rc}", type="primary"):
                    st.session_state[submitted_key] = True; st.rerun()

            if already_submitted:
                tc = type_choice == sc["correct_type"]
                dc = dir_choice == sc["correct_direction"]
                if tc and dc:
                    st.success("✅ Both correct!")
                else:
                    st.error("📋 Review your answers:")
                    if not tc:
                        st.markdown(f"**Bias type** — You selected: *{type_choice}*")
                        st.markdown(f"✅ Correct: **{sc['correct_type']}**")
                    if not dc:
                        st.markdown(f"**Direction** — You selected: *{dir_choice}*")
                        st.markdown(f"✅ Correct: **{sc['correct_direction']}**")
                st.info(f"**Explanation:** {sc['explanation']}")
                if st.button("🔄 Try Again", key=f"bias_retry_{sid}_{rc}"):
                    for k in [f"bias_type_{sid}_{rc}", f"bias_dir_{sid}_{rc}", submitted_key]:
                        if k in st.session_state: del st.session_state[k]
                    st.rerun()

    st.markdown("---")
    st.markdown("*Strong epidemiologists think structurally before computing.*")


# ==================================================
# MODULE 1: CONFOUNDING & EFFECT MODIFICATION
# ==================================================
elif current_page == "confounding":
    st.title("🔀 Confounding & Effect Modification")

    conf_section = st.radio("Section:", [
        "1️⃣ Confounding",
        "2️⃣ Controlling Confounding",
        "3️⃣ Effect Modification",
        "4️⃣ Interactive: Stratified Analysis",
        "5️⃣ DAG Library"
    ], horizontal=True)
    st.divider()

    if conf_section == "1️⃣ Confounding":
        st.subheader("What Is Confounding?")
        st.markdown("""
A **confounder** is a variable that distorts the apparent association between an exposure and an outcome. Three conditions must all be true:

1. The confounder is **associated with the exposure** (in the source population)
2. The confounder is **associated with the outcome** (independent of the exposure)
3. The confounder is **not on the causal pathway** between exposure and outcome (it's not an intermediate step)
        """)
        st.info("**Intuition:** A confounder provides an alternative explanation for your finding. The apparent association between exposure and outcome might be entirely or partly due to both being linked to the confounder.")

        with st.expander("☕ Classic Example: Coffee & Heart Disease", expanded=True):
            st.markdown("""
Early studies found coffee drinkers had higher rates of heart disease. Was coffee the culprit?

**The confounder: Smoking**
- Coffee drinkers were more likely to smoke (association with *exposure*)
- Smoking causes heart disease (association with *outcome*)
- Smoking is not caused by coffee (not on the causal pathway)

After adjusting for smoking, the association between coffee and heart disease disappeared entirely. The apparent association was confounding by smoking.
            """)

        with st.expander("🏊 Another Example: Swimming Pools & Drowning"):
            st.markdown("""
Counties with more ice cream sales have higher drowning rates. Is ice cream dangerous?

**The confounder: Summer / hot weather**
- Hot weather increases ice cream sales (association with *exposure*)
- Hot weather increases swimming and thus drowning (association with *outcome*)  
- Hot weather is not caused by ice cream (not on the causal pathway)

After adjusting for season/temperature, the association disappears. Classic ecological confounding.
            """)

        with st.expander("🔍 How to Identify a Confounder — The 10% Rule"):
            st.markdown("""
A practical rule: if adjusting for a variable **changes your RR/OR by more than 10%**, it is a meaningful confounder and should be controlled.

**Formula:**
RR change = |crude RR − adjusted RR| ÷ adjusted RR × 100

If this exceeds 10%, the variable is a confounder worth controlling.

**Important:** This is a rule of thumb, not a law. Context matters — a 5% change could matter clinically even if it's below the threshold.
            """)

        # Visual DAG
        dag_html = """
<div style="background:#f8f9fa;border-radius:8px;padding:20px;margin:12px 0;">
  <p style="font-weight:bold;margin-bottom:8px;font-size:13px;">DAG (Directed Acyclic Graph): Smoking confounds Coffee → Heart Disease</p>
  <div style="display:flex;align-items:center;justify-content:center;gap:0;flex-wrap:wrap;">
    <div style="text-align:center;padding:12px 20px;background:#fff3e0;border-radius:8px;border:2px solid #ef6c00;font-weight:bold;font-size:13px;">Smoking<br><span style="font-size:10px;color:#888;">(Confounder)</span></div>
    <div style="display:flex;flex-direction:column;align-items:center;gap:4px;padding:0 8px;">
      <div style="font-size:20px;color:#ef6c00;">↙</div>
      <div style="font-size:20px;color:#ef6c00;">↘</div>
    </div>
    <div style="display:flex;flex-direction:column;align-items:center;gap:16px;">
      <div style="text-align:center;padding:12px 20px;background:#e3f2fd;border-radius:8px;border:2px solid #1565c0;font-weight:bold;font-size:13px;">Coffee<br><span style="font-size:10px;color:#888;">(Exposure)</span></div>
      <div style="font-size:18px;color:#999;">↓ (spurious)</div>
      <div style="text-align:center;padding:12px 20px;background:#fce4ec;border-radius:8px;border:2px solid #c62828;font-weight:bold;font-size:13px;">Heart Disease<br><span style="font-size:10px;color:#888;">(Outcome)</span></div>
    </div>
    <div style="font-size:20px;color:#c62828;padding:0 8px;">→</div>
  </div>
  <p style="font-size:12px;color:#666;text-align:center;margin-top:12px;">Smoking → Coffee AND Smoking → Heart Disease creates a backdoor path. Remove it by adjusting for smoking.</p>
</div>"""
        st.markdown(dag_html, unsafe_allow_html=True)

    elif conf_section == "2️⃣ Controlling Confounding":
        st.subheader("Methods to Control Confounding")
        st.markdown("Confounding can be controlled at the **design stage** or the **analysis stage**.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔧 Design Stage")
            with st.expander("Randomization", expanded=True):
                st.markdown("""
**When:** RCTs only

**How:** Random assignment distributes confounders — both measured and unmeasured — equally across groups. If the sample is large enough, groups are balanced on all characteristics.

**Strength:** Controls for confounders you don't even know about.

**Limitation:** Only available in experimental studies.
                """)
            with st.expander("Restriction"):
                st.markdown("""
**How:** Limit the study population to a single level of the confounder. E.g., study only non-smokers to eliminate smoking as a confounder.

**Strength:** Simple; prevents confounding completely for that variable.

**Limitation:** Reduces sample size; limits generalizability; can't control for confounders you haven't thought of.
                """)
            with st.expander("Matching"):
                st.markdown("""
**How:** Each case is paired with one or more controls who have the same value of the confounder (e.g., same age, same sex).

**Strength:** Controls confounding by design; increases efficiency in case-control studies.

**Limitation:** Can't match on many variables; matched design must be analyzed with matched methods (conditional logistic regression); can't study matched variables as exposures; overmatching possible.
                """)

        with col2:
            st.markdown("#### 📊 Analysis Stage")
            with st.expander("Stratification (Mantel-Haenszel)", expanded=True):
                st.markdown("""
**How:** Stratify the analysis by levels of the confounder. Calculate stratum-specific RRs/ORs. If they're similar across strata, pool them using the Mantel-Haenszel method to get an adjusted estimate.

**Strength:** Transparent; lets you see stratum-specific effects; reveals effect modification.

**Limitation:** Only practical with a few confounders; becomes unwieldy with many.
                """)
            with st.expander("Multivariable Regression"):
                st.markdown("""
**How:** Include confounders as covariates in a regression model (logistic, Poisson, Cox). The coefficient for the exposure is adjusted for all covariates simultaneously.

**Strength:** Can control many confounders at once; flexible.

**Limitation:** Requires assumptions (linearity, no multicollinearity); residual confounding if variables are measured poorly; more of a black box than stratification.
                """)
            with st.expander("Propensity Score Methods"):
                st.markdown("""
**How:** Estimate each subject's probability of being exposed given their confounders (propensity score). Match, weight, or stratify by propensity score.

**Strength:** Can handle many confounders; creates balance similar to randomization for measured variables.

**Limitation:** Cannot control for unmeasured confounders; complex; less intuitive.
                """)

        st.info("**Residual confounding:** Even after adjustment, if confounders are measured imperfectly, some confounding remains. This is almost always present in observational studies to some degree.")

    elif conf_section == "3️⃣ Effect Modification":
        st.subheader("Effect Modification (Interaction)")
        st.markdown("""
Effect modification occurs when the **magnitude of the association between exposure and outcome differs across levels of a third variable**. Unlike confounding, effect modification is a real biological or social phenomenon — not a bias to be removed.

**Key distinction:**
- **Confounding** = distortion to be controlled/removed
- **Effect modification** = a finding to be reported and understood
        """)

        with st.expander("Example: Aspirin & Heart Attack, Stratified by Sex", expanded=True):
            st.markdown("""
Suppose aspirin reduces the risk of heart attack:
- In men: RR = 0.6 (40% risk reduction)
- In women: RR = 0.95 (5% risk reduction — essentially no effect)

Sex **modifies** the effect of aspirin. The overall (crude) RR would be somewhere between these — misleading for both sexes. Reporting sex-stratified RRs is more informative than a single pooled estimate.

**Effect modification on what scale?** Effect modification can occur on the additive scale (risk differences differ) or the multiplicative scale (risk ratios differ). These don't always agree — a variable can modify on one scale but not the other. Epidemiologists generally assess both.
            """)

        with st.expander("How to Detect Effect Modification"):
            st.markdown("""
1. **Stratify** by the suspected effect modifier
2. **Calculate stratum-specific** measures of association (RR or OR in each stratum)
3. **Compare** stratum-specific estimates: if they differ substantially, effect modification is present
4. **Statistical test:** Wald test or likelihood ratio test for interaction term in regression (p < 0.05 suggests effect modification)

**Rule of thumb:** If stratum-specific estimates differ by >1.5-fold (for multiplicative measures) or by a clinically meaningful amount, report them separately rather than pooling.
            """)

        with st.expander("Confounding vs. Effect Modification — Quick Reference"):
            st.markdown("""
| Feature | Confounding | Effect Modification |
|---|---|---|
| What is it? | Distortion of the true association | Real variation in the association |
| Goal? | Remove it | Report it |
| Stratum-specific measures | Similar across strata (after control) | Differ across strata |
| Pooled estimate appropriate? | Yes, after adjustment | No — report separately |
| Example | Age confounds occupation-mortality | Sex modifies aspirin effect |
            """)

    elif conf_section == "4️⃣ Interactive: Stratified Analysis":
        st.subheader("Interactive Stratified Analysis")
        st.markdown("Explore how stratification by a third variable reveals confounding or effect modification. Each preset names the exposure, outcome, and stratifying variable so you can see exactly what the numbers represent.")

        STRAT_PRESETS = {
            "None — I'll enter my own data": None,
            "Coffee & MI, stratified by Smoking": {
                "description": "**Scenario:** A cohort study finds coffee drinkers have higher rates of MI. Smoking is suspected as a confounder — it's associated with both coffee drinking and MI risk. We stratify by smoking status to see whether the coffee-MI association holds within each stratum.",
                "exposure": "Coffee drinking", "outcome": "Myocardial infarction (MI)",
                "stratifier": "Smoking status",
                "exposed_label": "Coffee drinker", "unexposed_label": "Non-coffee drinker",
                "outcome_label": "MI", "no_outcome_label": "No MI",
                "strata": [
                    {"name": "Smokers", "cells": (40, 160, 10, 190)},
                    {"name": "Non-smokers", "cells": (30, 120, 20, 280)},
                ],
                "teaching_point": "If the stratum-specific RRs are both close to 1 but the crude RR is elevated, smoking is confounding the coffee-MI relationship — coffee drinkers happen to smoke more, and smoking causes MI."
            },
            "Aspirin & Bleeding, stratified by Sex": {
                "description": "**Scenario:** A cohort study examines aspirin use and GI bleeding. We suspect the effect differs by sex — aspirin may be more harmful in women due to differences in gastric mucosa physiology.",
                "exposure": "Daily aspirin use", "outcome": "GI bleeding",
                "stratifier": "Sex",
                "exposed_label": "Aspirin user", "unexposed_label": "Non-user",
                "outcome_label": "GI bleed", "no_outcome_label": "No bleed",
                "strata": [
                    {"name": "Women", "cells": (45, 255, 12, 288)},
                    {"name": "Men", "cells": (18, 282, 14, 286)},
                ],
                "teaching_point": "If the RR in women is substantially higher than in men, sex is an effect modifier — the harm from aspirin differs by sex. Report stratum-specific RRs rather than a pooled estimate."
            },
            "Physical Activity & T2D, stratified by BMI": {
                "description": "**Scenario:** A prospective cohort examines whether physical activity protects against Type 2 Diabetes. BMI may confound this (active people tend to have lower BMI; BMI is a strong T2D predictor) — or BMI may modify the effect.",
                "exposure": "High physical activity", "outcome": "Type 2 Diabetes (T2D)",
                "stratifier": "BMI category",
                "exposed_label": "Active", "unexposed_label": "Inactive",
                "outcome_label": "T2D", "no_outcome_label": "No T2D",
                "strata": [
                    {"name": "Normal/Overweight BMI (<30)", "cells": (22, 278, 38, 262)},
                    {"name": "Obese BMI (≥30)", "cells": (55, 245, 90, 210)},
                ],
                "teaching_point": "If stratum-specific RRs are similar to each other but both differ from the crude RR, BMI is confounding. If the RRs differ substantially across strata, BMI modifies the protective effect of activity."
            },
            "None — I'll enter my own data": None,
        }

        if "strat_preset_choice" not in st.session_state:
            st.session_state["strat_preset_choice"] = "Coffee & MI, stratified by Smoking"

        preset_choice = st.selectbox(
            "Select a scenario:",
            [k for k in STRAT_PRESETS.keys() if k != "None — I'll enter my own data"] + ["None — I'll enter my own data"],
            key="strat_preset_choice"
        )
        preset = STRAT_PRESETS.get(preset_choice)

        if preset:
            st.info(preset["description"])
            st.divider()
            n_strata = len(preset["strata"])
            exposure = preset["exposure"]
            outcome = preset["outcome"]
            stratifier = preset["stratifier"]
            exp_lbl = preset["exposed_label"]
            unexp_lbl = preset["unexposed_label"]
            out_lbl = preset["outcome_label"]
            no_out_lbl = preset["no_outcome_label"]

            st.markdown(f"**Exposure:** {exposure} &nbsp;|&nbsp; **Outcome:** {outcome} &nbsp;|&nbsp; **Stratifying variable:** {stratifier}")
            st.divider()

            strata_data = []
            for s, stratum in enumerate(preset["strata"]):
                st.markdown(f"**Stratum {s+1}: {stratum['name']}**")
                default_a, default_b, default_c, default_d = stratum["cells"]

                # Show a labelled 2x2 with number inputs
                header_cols = st.columns([2, 3, 3])
                header_cols[1].markdown(f"<div style='text-align:center;font-size:12px;color:#c62828;font-weight:bold;'>✚ {out_lbl}</div>", unsafe_allow_html=True)
                header_cols[2].markdown(f"<div style='text-align:center;font-size:12px;color:#555;font-weight:bold;'>✕ {no_out_lbl}</div>", unsafe_allow_html=True)

                row1 = st.columns([2, 3, 3])
                row1[0].markdown(f"<div style='padding-top:8px;font-size:13px;font-weight:bold;color:#1565c0;'>{exp_lbl}</div>", unsafe_allow_html=True)
                a = row1[1].number_input(f"{exp_lbl} / {out_lbl}", min_value=0, value=default_a, key=f"sa_{s}_{preset_choice}", label_visibility="collapsed")
                b = row1[2].number_input(f"{exp_lbl} / {no_out_lbl}", min_value=0, value=default_b, key=f"sb_{s}_{preset_choice}", label_visibility="collapsed")

                row2 = st.columns([2, 3, 3])
                row2[0].markdown(f"<div style='padding-top:8px;font-size:13px;font-weight:bold;color:#555;'>{unexp_lbl}</div>", unsafe_allow_html=True)
                c = row2[1].number_input(f"{unexp_lbl} / {out_lbl}", min_value=0, value=default_c, key=f"sc_{s}_{preset_choice}", label_visibility="collapsed")
                d = row2[2].number_input(f"{unexp_lbl} / {no_out_lbl}", min_value=0, value=default_d, key=f"sd_{s}_{preset_choice}", label_visibility="collapsed")

                strata_data.append((a, b, c, d))
                st.markdown("")

        else:
            # Custom data entry
            st.divider()
            col1, col2 = st.columns(2)
            exposure   = col1.text_input("Exposure name", "Exposure", key="strat_exp")
            outcome    = col2.text_input("Outcome name", "Outcome", key="strat_out")
            exp_lbl    = col1.text_input("Exposed group label", "Exposed", key="strat_explbl")
            unexp_lbl  = col2.text_input("Unexposed group label", "Unexposed", key="strat_unexplbl")
            out_lbl    = col1.text_input("Outcome positive label", "Disease", key="strat_outlbl")
            no_out_lbl = col2.text_input("Outcome negative label", "No Disease", key="strat_nooutlbl")
            stratifier = col1.text_input("Stratifying variable", "Stratifier", key="strat_stratvar")
            n_strata   = st.radio("Number of strata:", [2, 3], horizontal=True, key="strat_n")

            strata_data = []
            for s in range(n_strata):
                strat_name = st.text_input(f"Stratum {s+1} name", f"Stratum {s+1}", key=f"strat_name_{s}")
                st.markdown(f"**{strat_name}**")

                header_cols = st.columns([2, 3, 3])
                header_cols[1].markdown(f"<div style='text-align:center;font-size:12px;color:#c62828;font-weight:bold;'>✚ {out_lbl}</div>", unsafe_allow_html=True)
                header_cols[2].markdown(f"<div style='text-align:center;font-size:12px;color:#555;font-weight:bold;'>✕ {no_out_lbl}</div>", unsafe_allow_html=True)

                row1 = st.columns([2, 3, 3])
                row1[0].markdown(f"<div style='padding-top:8px;font-size:13px;font-weight:bold;color:#1565c0;'>{exp_lbl}</div>", unsafe_allow_html=True)
                a = row1[1].number_input("a", min_value=0, value=40, key=f"sa_{s}_custom", label_visibility="collapsed")
                b = row1[2].number_input("b", min_value=0, value=160, key=f"sb_{s}_custom", label_visibility="collapsed")

                row2 = st.columns([2, 3, 3])
                row2[0].markdown(f"<div style='padding-top:8px;font-size:13px;font-weight:bold;color:#555;'>{unexp_lbl}</div>", unsafe_allow_html=True)
                c = row2[1].number_input("c", min_value=0, value=10, key=f"sc_{s}_custom", label_visibility="collapsed")
                d = row2[2].number_input("d", min_value=0, value=190, key=f"sd_{s}_custom", label_visibility="collapsed")

                strata_data.append((a, b, c, d))
                st.markdown("")

        if st.button("Run Stratified Analysis"):
            crude_a = sum(x[0] for x in strata_data)
            crude_b = sum(x[1] for x in strata_data)
            crude_c = sum(x[2] for x in strata_data)
            crude_d = sum(x[3] for x in strata_data)
            crude_rr = (crude_a/(crude_a+crude_b)) / (crude_c/(crude_c+crude_d)) if (crude_c+crude_d) > 0 and crude_c > 0 else None

            st.subheader("Results")

            # Stratum-specific RRs as named metrics
            results = []
            strata_names = [p["name"] for p in preset["strata"]] if preset else [f"Stratum {s+1}" for s in range(n_strata)]
            metric_cols = st.columns(n_strata)
            for s, (a, b, c, d) in enumerate(strata_data):
                if (a+b) > 0 and (c+d) > 0 and c > 0:
                    rr_s = (a/(a+b)) / (c/(c+d))
                    results.append(round(rr_s, 2))
                    metric_cols[s].metric(f"RR — {strata_names[s]}", round(rr_s, 2),
                        help=f"{exp_lbl} risk: {round(a/(a+b)*100,1)}% | {unexp_lbl} risk: {round(c/(c+d)*100,1)}%")
                else:
                    results.append(None)
                    metric_cols[s].metric(f"RR — {strata_names[s]}", "N/A")

            # Mantel-Haenszel pooled RR
            n_list = [(a+b+c+d) for a,b,c,d in strata_data]
            mh_num = sum(strata_data[s][0] * (strata_data[s][2]+strata_data[s][3]) / n_list[s] for s in range(n_strata))
            mh_den = sum(strata_data[s][2] * (strata_data[s][0]+strata_data[s][1]) / n_list[s] for s in range(n_strata))
            mh_rr = round(mh_num/mh_den, 2) if mh_den > 0 else None

            st.divider()
            col1, col2 = st.columns(2)
            col1.metric(f"Crude (Unstratified) RR", round(crude_rr,2) if crude_rr else "N/A",
                help=f"Combines all strata — ignores {stratifier}")
            col2.metric("Mantel-Haenszel Adjusted RR", mh_rr if mh_rr else "N/A",
                help=f"Pooled RR after adjusting for {stratifier}")

            # Interpretation
            if crude_rr and mh_rr:
                pct_change = abs(crude_rr - mh_rr) / mh_rr * 100
                st.divider()
                valid_results = [r for r in results if r is not None]
                em_ratio = max(valid_results)/min(valid_results) if len(valid_results) >= 2 and min(valid_results) > 0 else 1

                if pct_change > 10 and em_ratio < 1.5:
                    st.error(f"""
⚠️ **Confounding by {stratifier} detected.**

The crude RR ({round(crude_rr,2)}) differs from the adjusted RR ({mh_rr}) by {round(pct_change,1)}%, but the stratum-specific RRs ({', '.join(str(r) for r in results)}) are similar to each other — meaning the true {exposure}–{outcome} association is consistent across strata of {stratifier}.

The crude estimate was misleading because {stratifier} was distributed differently across {exp_lbl} and {unexp_lbl} groups. **Use the Mantel-Haenszel adjusted RR ({mh_rr}).**
                    """)
                elif em_ratio >= 1.5:
                    st.warning(f"""
⚠️ **Effect modification by {stratifier}.**

The stratum-specific RRs differ substantially ({', '.join(str(r) for r in results)}), suggesting the association between {exposure} and {outcome} is not the same across levels of {stratifier}. A single pooled RR would obscure this difference.

**Report stratum-specific RRs separately** rather than a single adjusted estimate.
                    """)
                else:
                    st.success(f"""
✅ **No meaningful confounding or effect modification by {stratifier}.**

Crude RR ({round(crude_rr,2)}) and adjusted RR ({mh_rr}) differ by only {round(pct_change,1)}%, and stratum-specific RRs ({', '.join(str(r) for r in results)}) are consistent. {stratifier} does not appear to distort or modify the {exposure}–{outcome} association.
                    """)

            # Teaching point if preset
            if preset and "teaching_point" in preset:
                with st.expander("💡 What to look for in this scenario"):
                    st.markdown(preset["teaching_point"])

            # Per-stratum 2x2 math
            cell_style  = "border:1px solid #aaa; padding:8px 14px; text-align:center; font-size:14px;"
            label_style = "border:1px solid #aaa; padding:8px 14px; text-align:center; font-size:12px; color:#555; background:#f5f5f5; font-weight:bold;"
            total_style = "border:1px solid #ccc; padding:8px 14px; text-align:center; font-size:13px; color:#555; background:#f0f0f0;"

            with st.expander("🔢 Show me the math — Stratum-Specific RRs"):
                st.markdown(f"**Cell key:** a = {exp_lbl} with {out_lbl} &nbsp;|&nbsp; b = {exp_lbl} without &nbsp;|&nbsp; c = {unexp_lbl} with {out_lbl} &nbsp;|&nbsp; d = {unexp_lbl} without")
                st.markdown("---")
                for s, (a, b, c, d) in enumerate(strata_data):
                    rr_s = round((a/(a+b)) / (c/(c+d)), 3) if (a+b) > 0 and (c+d) > 0 and c > 0 else None
                    risk_exp   = round(a/(a+b), 4) if (a+b) > 0 else 0
                    risk_unexp = round(c/(c+d), 4) if (c+d) > 0 else 0
                    table_html = f"""
<p style="font-weight:bold; margin-bottom:6px;">{strata_names[s]}</p>
<table style="border-collapse:collapse; width:100%; max-width:560px; margin-bottom:4px;">
  <tr>
    <td style="{label_style} background:#fff; border:none;"></td>
    <td style="{label_style}">✚ {out_lbl}</td>
    <td style="{label_style}">✕ {no_out_lbl}</td>
    <td style="{label_style}">Row Total</td>
  </tr>
  <tr>
    <td style="{label_style}">{exp_lbl}</td>
    <td style="{cell_style} background:#e8f4e8;"><span style="font-size:10px;color:#888;font-style:italic;">a = </span><span style="font-size:18px;font-weight:bold;color:#2e7d32;">{int(a)}</span></td>
    <td style="{cell_style} background:#fdecea;"><span style="font-size:10px;color:#888;font-style:italic;">b = </span><span style="font-size:18px;font-weight:bold;color:#c0392b;">{int(b)}</span></td>
    <td style="{total_style}">{int(a+b)}</td>
  </tr>
  <tr>
    <td style="{label_style}">{unexp_lbl}</td>
    <td style="{cell_style} background:#fdecea;"><span style="font-size:10px;color:#888;font-style:italic;">c = </span><span style="font-size:18px;font-weight:bold;color:#c0392b;">{int(c)}</span></td>
    <td style="{cell_style} background:#e8f4e8;"><span style="font-size:10px;color:#888;font-style:italic;">d = </span><span style="font-size:18px;font-weight:bold;color:#2e7d32;">{int(d)}</span></td>
    <td style="{total_style}">{int(c+d)}</td>
  </tr>
  <tr>
    <td style="{label_style}">Col Total</td>
    <td style="{total_style}">{int(a+c)}</td>
    <td style="{total_style}">{int(b+d)}</td>
    <td style="{total_style}">{int(a+b+c+d)}</td>
  </tr>
</table>"""
                    st.markdown(table_html, unsafe_allow_html=True)
                    if rr_s:
                        st.markdown(f"""
**RR ({strata_names[s]})** = [a ÷ (a+b)] ÷ [c ÷ (c+d)]
= [{int(a)} ÷ {int(a+b)}] ÷ [{int(c)} ÷ {int(c+d)}]
= {risk_exp} ÷ {risk_unexp}
= **{rr_s}**
                        """)
                    st.markdown("---")

                # Crude (combined) table
                st.markdown(f"**Combined (Crude) Table — ignoring {stratifier}**")
                ca, cb, cc, cd = crude_a, crude_b, crude_c, crude_d
                crude_risk_exp   = round(ca/(ca+cb), 4) if (ca+cb) > 0 else 0
                crude_risk_unexp = round(cc/(cc+cd), 4) if (cc+cd) > 0 else 0
                crude_table_html = f"""
<table style="border-collapse:collapse; width:100%; max-width:560px; margin-bottom:4px;">
  <tr>
    <td style="{label_style} background:#fff; border:none;"></td>
    <td style="{label_style}">✚ {out_lbl}</td>
    <td style="{label_style}">✕ {no_out_lbl}</td>
    <td style="{label_style}">Row Total</td>
  </tr>
  <tr>
    <td style="{label_style}">{exp_lbl}</td>
    <td style="{cell_style} background:#e8f4e8;"><span style="font-size:10px;color:#888;font-style:italic;">a = </span><span style="font-size:18px;font-weight:bold;color:#2e7d32;">{int(ca)}</span></td>
    <td style="{cell_style} background:#fdecea;"><span style="font-size:10px;color:#888;font-style:italic;">b = </span><span style="font-size:18px;font-weight:bold;color:#c0392b;">{int(cb)}</span></td>
    <td style="{total_style}">{int(ca+cb)}</td>
  </tr>
  <tr>
    <td style="{label_style}">{unexp_lbl}</td>
    <td style="{cell_style} background:#fdecea;"><span style="font-size:10px;color:#888;font-style:italic;">c = </span><span style="font-size:18px;font-weight:bold;color:#c0392b;">{int(cc)}</span></td>
    <td style="{cell_style} background:#e8f4e8;"><span style="font-size:10px;color:#888;font-style:italic;">d = </span><span style="font-size:18px;font-weight:bold;color:#2e7d32;">{int(cd)}</span></td>
    <td style="{total_style}">{int(cc+cd)}</td>
  </tr>
  <tr>
    <td style="{label_style}">Col Total</td>
    <td style="{total_style}">{int(ca+cc)}</td>
    <td style="{total_style}">{int(cb+cd)}</td>
    <td style="{total_style}">{int(ca+cb+cc+cd)}</td>
  </tr>
</table>"""
                st.markdown(crude_table_html, unsafe_allow_html=True)
                st.markdown(f"""
**Crude RR** = [a ÷ (a+b)] ÷ [c ÷ (c+d)]
= [{int(ca)} ÷ {int(ca+cb)}] ÷ [{int(cc)} ÷ {int(cc+cd)}]
= {crude_risk_exp} ÷ {crude_risk_unexp}
= **{round(crude_rr,3) if crude_rr else 'N/A'}**
                """)

            # Mantel-Haenszel expander
            with st.expander("🔢 Show me the math — Mantel-Haenszel Adjusted RR"):
                st.markdown(f"""
The **Mantel-Haenszel method** pools stratum-specific RRs into a single adjusted estimate, weighting each stratum proportionally to its size.

**Formula:** RR_MH = Σ[a_s × (c_s + d_s) / n_s] ÷ Σ[c_s × (a_s + b_s) / n_s]

Where: **a** = {exp_lbl} with {out_lbl}, **b** = {exp_lbl} without, **c** = {unexp_lbl} with {out_lbl}, **d** = {unexp_lbl} without, **n** = stratum total
                """)
                mh_num_check = 0; mh_den_check = 0
                for s, (a, b, c, d) in enumerate(strata_data):
                    n = a+b+c+d
                    num_s = round(a*(c+d)/n, 3)
                    den_s = round(c*(a+b)/n, 3)
                    mh_num_check += num_s; mh_den_check += den_s
                    st.markdown(f"""
**{strata_names[s]}** (n = {n})
- Numerator: a×(c+d)/n = {int(a)}×{int(c+d)}/{n} = **{num_s}**
- Denominator: c×(a+b)/n = {int(c)}×{int(a+b)}/{n} = **{den_s}**
                    """)
                st.markdown(f"""
---
**Sum of numerator terms:** {round(mh_num_check,3)}
**Sum of denominator terms:** {round(mh_den_check,3)}
**RR_MH = {round(mh_num_check,3)} ÷ {round(mh_den_check,3)} = {mh_rr}**
                """)

    elif conf_section == "5️⃣ DAG Library":
        st.subheader("DAG Library — Causal Structures in Epidemiology")
        st.markdown("""
A **Directed Acyclic Graph (DAG)** is a visual tool for representing causal assumptions. Nodes are variables, arrows show causal directions, and the structure determines what you should — and should **not** — adjust for in your analysis.

Understanding DAG structures is essential for deciding which variables to control for and which to leave alone.
        """)

        DAG_TYPES = [
            "Confounder",
            "Mediator",
            "Collider",
            "Moderator / Effect Modifier",
            "M-Bias",
            "Proxy / Surrogate",
        ]
        dag_choice = st.selectbox("Select a DAG structure:", DAG_TYPES, key="dag_choice")
        st.divider()

        def dag_box(label, sublabel, color_bg, color_border):
            return f"""<div style="display:inline-flex;flex-direction:column;align-items:center;
                justify-content:center;padding:12px 18px;background:{color_bg};
                border:2px solid {color_border};border-radius:10px;min-width:110px;
                text-align:center;font-size:13px;font-weight:700;line-height:1.3;">
                {label}<br><span style="font-size:10px;font-weight:400;color:#666;">{sublabel}</span>
                </div>"""

        def dag_arrow(label="", color="#555", vertical=False):
            if vertical:
                return f"""<div style="display:flex;flex-direction:column;align-items:center;
                    padding:2px 0;color:{color};font-size:11px;">
                    <div style="width:2px;height:30px;background:{color};"></div>
                    <div style="font-size:16px;line-height:1;color:{color};">▼</div>
                    {"<div style='font-size:10px;color:#888;'>"+label+"</div>" if label else ""}
                    </div>"""
            else:
                return f"""<div style="display:flex;align-items:center;gap:2px;padding:0 6px;color:{color};font-size:11px;">
                    <div style="height:2px;width:40px;background:{color};"></div>
                    <div style="font-size:16px;line-height:1;">▶</div>
                    {"<div style='font-size:10px;color:#888;'>"+label+"</div>" if label else ""}
                    </div>"""

        if dag_choice == "Confounder":
            st.markdown("#### Confounder Structure")
            st.markdown("""
A **confounder** is a common cause of both the exposure and the outcome. It creates a **backdoor path** — an indirect, non-causal route connecting exposure to outcome that biases the association.

**The three criteria for confounding:**
1. Associated with the exposure (in the source population)
2. Independently associated with the outcome
3. NOT on the causal pathway between exposure and outcome
            """)

            dag_html = f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:24px;margin:16px 0;text-align:center;">
  <div style="font-weight:700;font-size:13px;margin-bottom:20px;color:#1a202c;">Confounder DAG: Smoking confounds Coffee → Heart Disease</div>
  <div style="display:flex;flex-direction:column;align-items:center;gap:0;">
    <div>{dag_box("Smoking","Confounder","#fff3e0","#ef6c00")}</div>
    <div style="display:flex;gap:80px;margin-top:4px;">
      <div style="display:flex;flex-direction:column;align-items:flex-end;">
        <div style="width:2px;height:24px;background:#ef6c00;margin-right:55px;"></div>
        <div style="font-size:14px;color:#ef6c00;margin-right:50px;">↙</div>
      </div>
      <div style="display:flex;flex-direction:column;align-items:flex-start;">
        <div style="width:2px;height:24px;background:#ef6c00;margin-left:55px;"></div>
        <div style="font-size:14px;color:#ef6c00;margin-left:50px;">↘</div>
      </div>
    </div>
    <div style="display:flex;align-items:center;gap:16px;margin-top:4px;">
      {dag_box("Coffee","Exposure","#e3f2fd","#1565c0")}
      <div style="display:flex;flex-direction:column;align-items:center;">
        <div style="height:2px;width:60px;background:#dc2626;"></div>
        <div style="font-size:14px;color:#dc2626;">▶</div>
        <div style="font-size:10px;color:#dc2626;font-style:italic;">spurious</div>
      </div>
      {dag_box("Heart Disease","Outcome","#fce4ec","#c62828")}
    </div>
  </div>
  <div style="margin-top:16px;font-size:11px;color:#718096;background:#fff;border-radius:6px;padding:8px 12px;display:inline-block;">
    🔴 <b>Backdoor path:</b> Coffee ← Smoking → Heart Disease &nbsp;|&nbsp; ✅ <b>Fix:</b> Adjust for Smoking to block this path
  </div>
</div>"""
            st.markdown(dag_html, unsafe_allow_html=True)

            st.success("✅ **What to do:** Adjust for the confounder (stratify, regression, matching). This blocks the backdoor path and isolates the true causal effect.")
            st.error("❌ **What NOT to do:** Adjust for a variable on the causal pathway — that would be over-adjustment (see Mediator).")

            with st.expander("🔢 How backdoor paths work"):
                st.markdown("""
A **backdoor path** is any path from exposure to outcome that begins with an arrow **into** the exposure. These paths create spurious associations.

**In this example:**
- Coffee ← Smoking → Heart Disease is a backdoor path
- Even if coffee has NO effect on heart disease, this path creates a correlation
- Conditioning (adjusting) for Smoking **blocks** this path

**D-separation:** Two variables are d-separated (conditionally independent) if all paths between them are blocked. Adjusting for the confounder d-separates exposure from outcome via that path.
                """)

        elif dag_choice == "Mediator":
            st.markdown("#### Mediator Structure")
            st.markdown("""
A **mediator** (or intermediate variable) lies **on the causal pathway** between exposure and outcome. It is how (the mechanism by which) the exposure causes the outcome.

**Example:** Physical activity → Lower blood pressure → Reduced CVD risk
- Lower blood pressure is the mediator — it's *how* exercise reduces CVD risk
            """)

            dag_html = f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:24px;margin:16px 0;text-align:center;">
  <div style="font-weight:700;font-size:13px;margin-bottom:20px;color:#1a202c;">Mediator DAG: Physical Activity → Blood Pressure → CVD</div>
  <div style="display:flex;align-items:center;justify-content:center;gap:0;">
    {dag_box("Physical Activity","Exposure","#e3f2fd","#1565c0")}
    <div style="display:flex;flex-direction:column;align-items:center;padding:0 8px;">
      <div style="height:2px;width:50px;background:#2e7d32;"></div>
      <div style="font-size:14px;color:#2e7d32;">▶</div>
      <div style="font-size:10px;color:#2e7d32;">causes</div>
    </div>
    {dag_box("↓ Blood Pressure","Mediator","#e8f5e9","#2e7d32")}
    <div style="display:flex;flex-direction:column;align-items:center;padding:0 8px;">
      <div style="height:2px;width:50px;background:#c62828;"></div>
      <div style="font-size:14px;color:#c62828;">▶</div>
      <div style="font-size:10px;color:#c62828;">causes</div>
    </div>
    {dag_box("CVD Risk","Outcome","#fce4ec","#c62828")}
  </div>
  <div style="margin-top:12px;">
    <div style="font-size:11px;color:#718096;font-style:italic;">Physical Activity also has a direct path to CVD (dashed line below)</div>
    <div style="display:flex;align-items:center;justify-content:center;gap:0;margin-top:8px;">
      {dag_box("Physical Activity","Exposure","#e3f2fd","#1565c0")}
      <div style="height:2px;width:200px;background:#1565c0;border-top:2px dashed #1565c0;"></div>
      <div style="font-size:14px;color:#1565c0;">▶</div>
      {dag_box("CVD Risk","Outcome","#fce4ec","#c62828")}
    </div>
  </div>
  <div style="margin-top:16px;font-size:11px;color:#718096;background:#fff;border-radius:6px;padding:8px 12px;display:inline-block;">
    🟢 <b>Total effect</b> = Direct effect + Indirect effect (through blood pressure)
  </div>
</div>"""
            st.markdown(dag_html, unsafe_allow_html=True)

            st.error("❌ **Critical mistake:** If you adjust for blood pressure when studying physical activity → CVD, you block the main causal pathway. Your estimate of the physical activity effect becomes biased (over-adjustment bias).")
            st.success("✅ **What to do:** Decide your question first. Total effect? Don't adjust for mediator. Direct effect only? Use mediation analysis methods, not simple regression adjustment.")

            with st.expander("📊 Total vs. Direct vs. Indirect Effects"):
                st.markdown("""
**Total effect:** Effect of exposure on outcome through ALL pathways (direct + indirect). Estimated by NOT adjusting for mediator.

**Direct effect:** Effect of exposure on outcome NOT through the mediator. Requires special mediation analysis methods.

**Indirect effect (mediated effect):** The portion of the total effect that operates through the mediator.

**Why it matters:** If you want to know "does exercise reduce CVD risk?", adjust for nothing on the pathway. If you want to know "does exercise reduce CVD risk through mechanisms other than blood pressure?", you need formal mediation analysis — not simply adding blood pressure to a regression.
                """)

        elif dag_choice == "Collider":
            st.markdown("#### Collider Structure")
            st.markdown("""
A **collider** is a variable that is caused by **both** the exposure and the outcome (arrows collide into it). Colliders are the most misunderstood structure in DAGs.

**The key rule:** Colliders naturally **block** paths. But if you condition on (adjust for, stratify by, or select on) a collider, you **open** a spurious path between exposure and outcome — creating bias where none existed before.
            """)

            dag_html = f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:24px;margin:16px 0;text-align:center;">
  <div style="font-weight:700;font-size:13px;margin-bottom:20px;color:#1a202c;">Collider DAG: Talent and Hard Work both cause Success</div>
  <div style="display:flex;align-items:flex-end;justify-content:center;gap:0;">
    <div style="display:flex;flex-direction:column;align-items:center;">
      {dag_box("Talent","Exposure","#e3f2fd","#1565c0")}
      <div style="display:flex;align-items:center;margin-top:4px;">
        <div style="width:60px;height:2px;background:#7b1fa2;"></div>
        <div style="font-size:14px;color:#7b1fa2;">↘</div>
      </div>
    </div>
    <div style="margin-bottom:8px;">{dag_box("Success","Collider","#f3e5f5","#7b1fa2")}</div>
    <div style="display:flex;flex-direction:column;align-items:center;">
      {dag_box("Hard Work","Exposure","#e8f5e9","#2e7d32")}
      <div style="display:flex;align-items:center;margin-top:4px;">
        <div style="font-size:14px;color:#7b1fa2;">↙</div>
        <div style="width:60px;height:2px;background:#7b1fa2;"></div>
      </div>
    </div>
  </div>
  <div style="margin-top:16px;display:flex;gap:16px;justify-content:center;font-size:11px;">
    <div style="background:#e8f5e9;border-radius:6px;padding:8px 12px;color:#2e7d32;">
      ✅ <b>Unadjusted:</b> Talent and Hard Work are independent (no association)
    </div>
    <div style="background:#fce4ec;border-radius:6px;padding:8px 12px;color:#c62828;">
      ❌ <b>After conditioning on Success</b> (e.g., studying only successful people): Talent and Hard Work appear <i>negatively</i> correlated — the "talented people don't work hard" fallacy
    </div>
  </div>
</div>"""
            st.markdown(dag_html, unsafe_allow_html=True)

            st.error("❌ **Collider bias:** Conditioning on a collider opens a spurious path. This happens when you: restrict your sample by outcome (selection bias), adjust for a variable caused by both exposure and outcome, or use a mediator that is also a collider.")
            st.success("✅ **What to do:** Do NOT adjust for colliders. Identify them in your DAG before analysis. Berkson's bias and healthy worker effect are real-world examples of collider bias.")

            with st.expander("🏥 Real-World Example: Berkson's Bias"):
                st.markdown("""
**Scenario:** Hospital study of smoking → lung cancer.

**The collider:** Hospitalization — caused by both smoking (smoking-related illness) and lung cancer (cancer requiring treatment).

When you study only hospitalized patients, you condition on hospitalization. This opens a spurious path between smoking and lung cancer that distorts the true association.

**Result:** The OR estimated in hospital patients is different from (usually biased toward null compared to) the OR in the general population.

**Fix:** Use population-based controls, not hospital-based controls, in case-control studies.
                """)

            with st.expander("📉 Example: The 'Attractive People Are Dumb' Fallacy"):
                st.markdown("""
Study participants are recruited from a modeling agency (conditioning on Success/Fame — a collider caused by both Attractiveness and Talent/Skills).

In the general population: Attractiveness and Intelligence are uncorrelated.
In this sample: They appear negatively correlated — because to get selected, you needed either one to compensate for lacking the other.

This is collider bias. The agency is a collider (caused by both). Studying only agency models opens a spurious negative path between attractiveness and intelligence.
                """)

        elif dag_choice == "Moderator / Effect Modifier":
            st.markdown("#### Moderator / Effect Modifier Structure")
            st.markdown("""
A **moderator** (or effect modifier) is a variable that changes the **magnitude or direction** of the association between exposure and outcome across its levels. Unlike confounding (which distorts), effect modification is a real biological or social phenomenon.

**Key distinction:**
- **Confounder** = bias to be removed
- **Effect modifier** = finding to be reported
            """)

            dag_html = f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:24px;margin:16px 0;text-align:center;">
  <div style="font-weight:700;font-size:13px;margin-bottom:20px;color:#1a202c;">Moderator DAG: Sex modifies the Aspirin → Heart Attack association</div>
  <div style="display:flex;align-items:center;justify-content:center;gap:0;">
    {dag_box("Aspirin Use","Exposure","#e3f2fd","#1565c0")}
    <div style="display:flex;flex-direction:column;align-items:center;padding:0 12px;position:relative;">
      <div style="height:2px;width:80px;background:#c62828;"></div>
      <div style="font-size:14px;color:#c62828;">▶</div>
      <div style="font-size:10px;color:#555;font-style:italic;">effect size varies</div>
    </div>
    {dag_box("Heart Attack","Outcome","#fce4ec","#c62828")}
  </div>
  <div style="margin-top:12px;display:flex;justify-content:center;">
    <div style="display:flex;flex-direction:column;align-items:center;">
      {dag_box("Sex","Moderator","#fff8e1","#f9a825")}
      <div style="font-size:20px;color:#f9a825;margin-top:4px;">↕</div>
      <div style="font-size:10px;color:#f9a825;font-weight:600;">modifies the arrow strength</div>
    </div>
  </div>
  <div style="margin-top:16px;display:flex;gap:16px;justify-content:center;font-size:12px;">
    <div style="background:#e3f2fd;border-radius:6px;padding:8px 12px;">
      <b>Men:</b> RR = 0.60 (40% risk reduction) ✅
    </div>
    <div style="background:#fce4ec;border-radius:6px;padding:8px 12px;">
      <b>Women:</b> RR = 0.95 (no meaningful effect) ❌
    </div>
  </div>
  <div style="margin-top:10px;font-size:11px;color:#718096;">A single pooled RR would be misleading for both sexes. Report stratum-specific estimates.</div>
</div>"""
            st.markdown(dag_html, unsafe_allow_html=True)

            st.warning("⚠️ **Important:** The moderator arrow points TO the exposure-outcome path (it modifies the relationship), not directly to the outcome. This distinguishes it from a confounder (which is a common cause).")
            st.success("✅ **What to do:** Stratify by the modifier and report separate estimates. A single adjusted estimate obscures a clinically important difference.")

            with st.expander("📊 Additive vs. Multiplicative Effect Modification"):
                st.markdown("""
Effect modification can occur on different scales:

**Multiplicative scale (ratio measures):** RR or OR differs across strata
- RR in men = 0.6, RR in women = 0.95 → effect modification on multiplicative scale

**Additive scale (difference measures):** Risk difference differs across strata
- RD in men = -15%, RD in women = -2% → effect modification on additive scale

**They don't always agree!** A variable can modify on one scale but not the other.

**Which scale matters?** 
- Public health decisions (who benefits most from an intervention?) → additive scale
- Biological mechanism questions → multiplicative scale
- Report both when possible
                """)

        elif dag_choice == "M-Bias":
            st.markdown("#### M-Bias Structure")
            st.markdown("""
**M-bias** occurs when you adjust for a variable that is a collider on a path between two unmeasured common causes of the exposure and outcome. Adjusting for this collider opens a backdoor path that wasn't there before — introducing bias in a previously unbiased estimate.

It's called M-bias because the DAG has an M shape.
            """)

            dag_html = f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:24px;margin:16px 0;text-align:center;">
  <div style="font-weight:700;font-size:13px;margin-bottom:20px;color:#1a202c;">M-Bias DAG Structure</div>
  <div style="display:flex;justify-content:center;gap:40px;align-items:flex-start;">
    {dag_box("U₁","Unmeasured cause of E","#f5f5f5","#9e9e9e")}
    <div style="width:60px;"></div>
    {dag_box("U₂","Unmeasured cause of Y","#f5f5f5","#9e9e9e")}
  </div>
  <div style="display:flex;justify-content:center;gap:0;margin:4px 0;font-size:18px;color:#9e9e9e;">
    <div style="margin-right:20px;">↓ &nbsp; ↘</div>
    <div style="margin-left:20px;">↙ &nbsp; ↓</div>
  </div>
  <div style="display:flex;justify-content:center;gap:80px;align-items:center;">
    {dag_box("Exposure (E)","","#e3f2fd","#1565c0")}
    {dag_box("M","Pre-treatment variable","#fff3e0","#e65100")}
    {dag_box("Outcome (Y)","","#fce4ec","#c62828")}
  </div>
  <div style="margin-top:10px;display:flex;flex-direction:column;align-items:center;gap:4px;">
    <div style="height:2px;width:120px;background:#c62828;"></div>
    <div style="font-size:14px;color:#c62828;">▶</div>
    <div style="font-size:10px;color:#c62828;">E → Y (true causal effect)</div>
  </div>
  <div style="margin-top:16px;display:flex;gap:16px;justify-content:center;font-size:11px;">
    <div style="background:#e8f5e9;border-radius:6px;padding:8px 12px;color:#2e7d32;">
      ✅ <b>Without adjusting for M:</b> E and Y association is unbiased (M blocks the U₁-U₂ path)
    </div>
    <div style="background:#fce4ec;border-radius:6px;padding:8px 12px;color:#c62828;">
      ❌ <b>After adjusting for M:</b> Opens backdoor path E ← U₁ → M ← U₂ → Y, introducing bias
    </div>
  </div>
</div>"""
            st.markdown(dag_html, unsafe_allow_html=True)

            st.error("❌ **M-bias trap:** Including a pre-treatment variable that looks 'harmless' can actually introduce bias if it's a collider on a path between unmeasured common causes. This is why you need a DAG — you can't detect this from the data alone.")
            st.info("💡 **Practical implication:** Not all pre-treatment variables should be adjusted for. Draw your DAG first. If a variable is a collider on any path, do not adjust for it.")

        elif dag_choice == "Proxy / Surrogate":
            st.markdown("#### Proxy / Surrogate Variable Structure")
            st.markdown("""
A **proxy** (or surrogate) is a measured variable that stands in for an unmeasured variable of interest. Proxies are used when the true variable can't be directly measured.

**Examples:**
- Years of education as a proxy for socioeconomic status
- BMI as a proxy for body fat
- C-reactive protein as a proxy for systemic inflammation
            """)

            dag_html = f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:24px;margin:16px 0;text-align:center;">
  <div style="font-weight:700;font-size:13px;margin-bottom:20px;color:#1a202c;">Proxy DAG: Education proxies for SES</div>
  <div style="display:flex;align-items:center;justify-content:center;gap:12px;">
    {dag_box("SES (U)","Unmeasured","#f5f5f5","#9e9e9e")}
    <div style="display:flex;flex-direction:column;align-items:center;gap:4px;">
      <div style="font-size:12px;color:#555;">causes</div>
      <div style="height:2px;width:40px;background:#555;"></div>
      <div style="font-size:14px;color:#555;">▶</div>
    </div>
    {dag_box("Education","Proxy (measured)","#fff3e0","#e65100")}
  </div>
  <div style="margin-top:16px;display:flex;align-items:center;justify-content:center;gap:12px;">
    {dag_box("SES (U)","Unmeasured","#f5f5f5","#9e9e9e")}
    <div style="display:flex;flex-direction:column;align-items:center;gap:4px;">
      <div style="font-size:12px;color:#c62828;">also causes</div>
      <div style="height:2px;width:40px;background:#c62828;"></div>
      <div style="font-size:14px;color:#c62828;">▶</div>
    </div>
    {dag_box("Health Outcome","Outcome","#fce4ec","#c62828")}
  </div>
  <div style="margin-top:16px;font-size:11px;color:#718096;background:#fff;border-radius:6px;padding:8px 12px;display:inline-block;">
    When you adjust for Education (proxy), you partially adjust for SES — but imperfect measurement means residual confounding remains
  </div>
</div>"""
            st.markdown(dag_html, unsafe_allow_html=True)

            st.warning("⚠️ **Proxy limitations:** Adjusting for a proxy only partially controls for the underlying variable. The weaker the proxy-variable relationship, the more residual confounding remains. This is why residual confounding is almost always present in observational studies.")

            with st.expander("📊 Implications for interpretation"):
                st.markdown("""
**When a proxy is used as a confounder:**
- You get partial adjustment, not full adjustment
- The residual confounding biases your estimate toward the crude (unadjusted) estimate
- Report this as a limitation: "We adjusted for education as a proxy for SES; residual confounding by unmeasured aspects of SES may remain"

**When a proxy is used as an exposure:**
- Measurement error in the exposure → non-differential misclassification → bias toward null
- A stronger proxy (higher correlation with true variable) → less attenuation

**Common proxies in epidemiology:**
| True variable | Common proxy |
|---|---|
| Socioeconomic status | Education, income, occupation |
| Diet quality | Food frequency questionnaire |
| Physical activity | Step count, self-report |
| Stress | Cortisol, self-report scale |
| Inflammation | CRP, IL-6 |
                """)

        with st.expander("📋 Quick Reference: All DAG Structures"):
            st.markdown("""
| Structure | Definition | Arrows | Should you adjust? | Effect of adjusting |
|---|---|---|---|---|
| **Confounder** | Common cause of E and Y | C→E, C→Y | ✅ Yes | Removes bias |
| **Mediator** | E causes M causes Y | E→M→Y | ❌ No (for total effect) | Over-adjustment bias |
| **Collider** | Both E and Y cause C | E→C←Y | ❌ Never | Opens spurious path |
| **Moderator** | Modifies strength of E→Y | M modifies E→Y | Report separately | Hides effect modification |
| **M-Bias node** | Collider between unmeasured causes | U₁→M←U₂ | ❌ No | Introduces new bias |
| **Proxy** | Measured substitute for unmeasured variable | U→Proxy | Partial ⚠️ | Partial confounding control |

**The golden rule:** Draw your DAG **before** your analysis. Your adjustment set should block all backdoor paths without conditioning on colliders or mediators.
            """)

    st.markdown("---")
    st.markdown("*Strong epidemiologists think structurally before computing.*")


# ==================================================
# MODULE 1: CAUSAL INFERENCE
# ==================================================
elif current_page == "causal_inference":
    st.title("🔗 Causal Inference")
    st.markdown("Association does not equal causation. Causal inference is the process of evaluating whether an observed statistical association reflects a true cause-and-effect relationship.")

    ci_section = st.radio("Section:", ["1️⃣ Bradford Hill Criteria", "2️⃣ Criteria Application Exercise"], horizontal=True)
    st.divider()

    if ci_section == "1️⃣ Bradford Hill Criteria":
        st.subheader("Bradford Hill Criteria (1965)")
        st.info("Sir Austin Bradford Hill proposed nine criteria for evaluating evidence of causation from observational data. **No single criterion is necessary or sufficient** — they are considered collectively as a weight-of-evidence framework.")

        criteria = [
            ("1. Strength of Association", "A strong association (high RR or OR) is less likely to be entirely explained by undetected bias or confounding than a weak one. Example: RR = 10 for smoking and lung cancer — difficult to explain away by confounding alone.","🔴"),
            ("2. Consistency", "The association has been observed repeatedly by different investigators, in different populations, using different methods. Consistent replication reduces the likelihood that a single finding is due to chance or study-specific bias.","🟠"),
            ("3. Specificity", "The exposure is associated with a specific disease, not a general increase in all diseases. **Weakest criterion** — many exposures cause multiple outcomes (smoking causes lung cancer, heart disease, and more). Absence of specificity does not argue against causation.","🟡"),
            ("4. Temporality", "The exposure must precede the outcome. **The only truly required criterion.** If the disease appeared before the exposure, causation in that direction is impossible.","🟢"),
            ("5. Biological Gradient (Dose-Response)", "Greater exposure leads to greater incidence of the outcome. A dose-response relationship strengthens causal inference — but absence of it doesn't rule out causation (some effects have a threshold).","🔵"),
            ("6. Plausibility", "The association makes biological sense given existing knowledge. Plausibility is limited by current understanding — historically plausible mechanisms were missing for H. pylori → ulcers until the mechanism was discovered.","🟣"),
            ("7. Coherence", "The causal interpretation should not conflict with known facts about the natural history and biology of the disease. The evidence should 'cohere' with other established knowledge.","🟤"),
            ("8. Experiment", "Experimental evidence — especially RCTs — provides the strongest support. If removing the exposure reduces disease (e.g., smoking cessation reduces cancer risk), this supports causation.","⚫"),
            ("9. Analogy", "If we accept that one exposure in a similar class causes disease, we may more readily accept that a similar exposure does too. Weakest criterion — prone to overextension.","⚪"),
        ]

        for emoji, (title, desc, _) in zip([c[2] for c in criteria], criteria):
            with st.expander(f"{emoji} {title}"):
                st.markdown(desc)

        st.divider()
        st.markdown("""
**How to use these criteria in practice:**

1. **Temporality** is the only mandatory criterion — if exposure doesn't precede disease, stop.
2. **Strength, consistency, and dose-response** carry the most weight.
3. **Specificity and analogy** are the weakest.
4. Present the full weight of evidence across all applicable criteria.
5. Counter-arguments matter — consider alternative explanations (bias, confounding, chance) for each criterion.
        """)

    elif ci_section == "2️⃣ Criteria Application Exercise":
        st.subheader("Criteria Application Exercise")
        st.markdown("Read the evidence summary and identify which Bradford Hill criteria are supported.")

        CAUSAL_SCENARIOS = [
            {
                "id": "c1",
                "title": "Scenario: Physical Inactivity & Type 2 Diabetes",
                "evidence": """
A large prospective cohort (n = 68,000, 10-year follow-up) finds physically inactive adults have 1.9× the risk of T2D compared to active adults (RR = 1.9, 95% CI: 1.6–2.3). Five prior cohort studies across the US, Europe, and Asia reported similar findings (RR range 1.5–2.2). Risk of T2D increases with each quartile of inactivity (dose-response). Physiologically, exercise increases insulin sensitivity and glucose uptake in muscle tissue (well-established mechanism). RCTs of exercise interventions show reduced T2D incidence among high-risk individuals. People who become inactive after previously being active show increased T2D risk (temporality confirmed). No studies report T2D preceding physical inactivity.
                """,
                "supported_criteria": ["Strength of Association","Consistency","Temporality","Biological Gradient (Dose-Response)","Plausibility","Experiment"],
                "not_supported": ["Specificity","Coherence","Analogy"],
                "explanation": "Strong RR (1.9), replicated across multiple countries (consistency), dose-response confirmed, well-established insulin-sensitivity mechanism (plausibility), RCT evidence (experiment), and temporality confirmed. Specificity is weak because physical inactivity affects many conditions beyond T2D."
            },
        ]

        all_criteria = ["Strength of Association","Consistency","Specificity","Temporality","Biological Gradient (Dose-Response)","Plausibility","Coherence","Experiment","Analogy"]

        for sc in CAUSAL_SCENARIOS:
            st.subheader(sc["title"])
            st.info(sc["evidence"])
            selected = st.multiselect("Which criteria are supported by this evidence?", all_criteria, key=f"causal_{sc['id']}")
            if st.button("Check My Answer", key=f"causal_check_{sc['id']}"):
                correct_set = set(sc["supported_criteria"])
                selected_set = set(selected)
                missed = correct_set - selected_set
                extra = selected_set - correct_set
                if not missed and not extra:
                    st.success("✅ Perfect — all supported criteria identified correctly!")
                else:
                    if missed:
                        st.warning(f"**Missed:** {', '.join(missed)}")
                    if extra:
                        st.error(f"**Not clearly supported by the evidence given:** {', '.join(extra)}")
                st.info(f"**Explanation:** {sc['explanation']}")

    st.markdown("---")
    st.markdown("*Strong epidemiologists think structurally before computing.*")


# ==================================================
# MODULE 2: DISEASE FREQUENCY
# ==================================================
elif current_page == "disease_frequency":
    st.title("📊 Disease Frequency")
    st.markdown("Before comparing rates across groups, you need to be able to measure disease frequency accurately in a single population.")

    df_section = st.radio("Section:", ["1️⃣ Core Measures", "2️⃣ Interactive Calculator", "3️⃣ Prevalence-Incidence Relationship", "4️⃣ Epidemic Curves"], horizontal=True)
    st.divider()

    if df_section == "1️⃣ Core Measures":
        st.subheader("Core Measures of Disease Frequency")

        with st.expander("📌 Prevalence", expanded=True):
            st.markdown("""
**Definition:** The proportion of a population that has a condition at a specific point in time (point prevalence) or during a specific period (period prevalence).

**Formula:** Prevalence = Number with condition ÷ Total population at risk

**Key features:**
- Numerator: existing cases (new + old)
- Denominator: total population at risk (including those who already have the disease)
- Range: 0 to 1 (or 0% to 100%)
- No time unit — it's a proportion, not a rate
- Useful for: planning healthcare capacity, estimating burden, cross-sectional studies

**Example:** 1,200 of 10,000 adults have diabetes at a health fair → Prevalence = 12%
            """)

        with st.expander("📌 Cumulative Incidence (Attack Rate)"):
            st.markdown("""
**Definition:** The proportion of a disease-free population that develops the disease during a specified time period.

**Formula:** Cumulative Incidence = New cases during period ÷ Population at risk at start of period

**Key features:**
- Numerator: *new* cases only (excludes pre-existing disease)
- Denominator: people who were disease-free at the *start*
- Always has a time period attached (e.g., "5-year risk")
- Risk: probability that a disease-free individual develops disease in that period
- Assumes everyone in the denominator is followed the full period

**Example:** 300 of 5,000 disease-free adults develop hypertension over 5 years → 5-year CI = 6%

**Attack rate:** Cumulative incidence in the context of an outbreak (same formula, shorter time frame). E.g., 42 of 200 attendees at a catered event developed food poisoning → Attack rate = 21%
            """)

        with st.expander("📌 Incidence Rate (Incidence Density)"):
            st.markdown("""
**Definition:** The rate of new disease occurrence per unit of person-time at risk.

**Formula:** Incidence Rate = New cases ÷ Total person-time at risk

**Key features:**
- Numerator: new cases
- Denominator: person-time (e.g., person-years) — accounts for varying follow-up
- Units: cases per person-year, per 100,000 person-years, etc.
- Used when: follow-up varies (people enter/leave, die, are censored)
- More precise than cumulative incidence for cohort studies with variable follow-up

**Example:** 87 new cancer cases in a cohort contributing 24,500 person-years → IR = 87/24,500 = 3.55 per 1,000 person-years

**Person-time calculation:** A subject followed for 3.5 years contributes 3.5 person-years. 100 people followed for 1 year = 100 person-years = 10 people followed for 10 years = 100 person-years.
            """)

        with st.expander("📌 Mortality Rate & Case Fatality Rate"):
            st.markdown("""
**Mortality Rate:** Incidence of death in a population per unit time.
- Formula: Deaths ÷ Population × time unit
- Includes *all* causes of death unless specified (cause-specific mortality rate)
- Used for: population health surveillance, comparing mortality across populations

**Case Fatality Rate (CFR):** The proportion of *cases* (people with disease) who die from it.
- Formula: Deaths from disease ÷ Total cases × 100
- Reflects severity of disease given diagnosis
- Higher CFR = more fatal disease
- NOT a true rate (no time unit) — it's a proportion

**Distinguishing CFR from mortality rate:**
- Mortality rate: deaths per 1,000 population per year (denominator = whole population)
- CFR: deaths per 100 cases (denominator = people with disease)
- A rare but very fatal disease can have low mortality rate but high CFR

**Example:** COVID-19 in early 2020 — high CFR in elderly (~10–20%), but overall population mortality rate was low because prevalence was still low.
            """)

        with st.expander("📌 Epidemic Curves (Point-Source vs. Propagated)"):
            st.markdown("""
**Epidemic curve:** A histogram of case counts by time of symptom onset. Shape reveals transmission pattern.

**Point-source epidemic:**
- All cases exposed to the same source at approximately the same time
- Curve rises and falls sharply; width ≈ one incubation period
- Example: foodborne illness at a single event

**Propagated (person-to-person) epidemic:**
- Cases spread from person to person
- Curve shows multiple waves; each wave ≈ one incubation period apart
- Example: measles, COVID-19 spread through a community

**Mixed pattern:** Point-source exposure followed by secondary person-to-person transmission (e.g., SARS superspreader event in a hospital).

**Incubation period:** Time from exposure to symptom onset. For a point-source outbreak, the range of onset times tells you the plausible incubation period for that pathogen.
            """)

    elif df_section == "2️⃣ Interactive Calculator":
        st.subheader("Disease Frequency Calculator")

        calc_type = st.selectbox("What do you want to calculate?", [
            "Prevalence",
            "Cumulative Incidence (Attack Rate)",
            "Incidence Rate (Person-Time)",
            "Case Fatality Rate (CFR)",
        ])

        if calc_type == "Prevalence":
            cases = st.number_input("Number with condition", min_value=0, value=1200)
            pop = st.number_input("Total population", min_value=1, value=10000)
            if st.button("Calculate Prevalence"):
                prev = cases/pop
                st.metric("Prevalence", f"{round(prev*100,2)}%")
                st.metric("Prevalence (per 1,000)", round(prev*1000,1))
                st.success(f"{cases} existing cases in a population of {pop:,} → Prevalence = {round(prev*100,2)}%")

        elif calc_type == "Cumulative Incidence (Attack Rate)":
            new_cases = st.number_input("New cases during period", min_value=0, value=300)
            pop_at_risk = st.number_input("Disease-free population at start", min_value=1, value=5000)
            time_label = st.text_input("Time period (for labeling)", "5-year")
            if st.button("Calculate Cumulative Incidence"):
                ci = new_cases / pop_at_risk
                st.metric(f"{time_label} Cumulative Incidence", f"{round(ci*100,2)}%")
                st.success(f"{new_cases} new cases among {pop_at_risk:,} at-risk people over {time_label} → {round(ci*100,2)}% risk")

        elif calc_type == "Incidence Rate (Person-Time)":
            st.markdown("""
**Why person-time?** When participants are followed for different lengths of time — some drop out, some die of unrelated causes, some join late — simply counting who got sick is unfair. Someone followed for 10 years had far more *opportunity* to develop disease than someone followed for 6 months. Person-time puts everyone on equal footing by counting the time each person was actually at risk.
            """)
            st.divider()

            pt_mode = st.radio("Data entry mode:", [
                "📋 Enter summary totals",
                "👤 Build a cohort person-by-person (visual)"
            ], horizontal=True, key="pt_mode")

            if pt_mode == "📋 Enter summary totals":
                new_cases   = st.number_input("New cases", min_value=0, value=87)
                person_time = st.number_input("Total person-years at risk", min_value=0.1, value=24500.0, step=100.0)
                multiplier  = st.selectbox("Express rate per:", [1000, 10000, 100000])
                if st.button("Calculate Incidence Rate", key="ir_summary"):
                    ir = new_cases / person_time * multiplier
                    st.metric(f"Incidence Rate (per {multiplier:,} person-years)", round(ir, 2))
                    st.success(f"{new_cases} cases ÷ {person_time:,.0f} person-years × {multiplier:,} = **{round(ir,2)} per {multiplier:,} person-years**")

            else:
                st.markdown("Add up to 12 participants. Set each person's follow-up duration and whether they developed the outcome. The diagram will show each person's timeline.")

                # Default cohort
                PRESETS_PT = {
                    "Custom": None,
                    "Simple: 5 people, varied follow-up": [
                        {"label":"Person 1","years":5.0,"event":False,"reason":"Completed study"},
                        {"label":"Person 2","years":3.0,"event":True, "reason":"Developed disease"},
                        {"label":"Person 3","years":5.0,"event":False,"reason":"Completed study"},
                        {"label":"Person 4","years":1.5,"event":False,"reason":"Lost to follow-up"},
                        {"label":"Person 5","years":5.0,"event":True, "reason":"Developed disease"},
                    ],
                    "Realistic: 8-person cohort": [
                        {"label":"Person 1","years":4.0,"event":False,"reason":"Completed study"},
                        {"label":"Person 2","years":2.0,"event":True, "reason":"Developed disease"},
                        {"label":"Person 3","years":4.0,"event":False,"reason":"Completed study"},
                        {"label":"Person 4","years":1.0,"event":False,"reason":"Lost to follow-up"},
                        {"label":"Person 5","years":3.5,"event":True, "reason":"Developed disease"},
                        {"label":"Person 6","years":4.0,"event":False,"reason":"Completed study"},
                        {"label":"Person 7","years":0.5,"event":False,"reason":"Withdrew"},
                        {"label":"Person 8","years":4.0,"event":False,"reason":"Completed study"},
                    ],
                }

                preset_pt = st.selectbox("Load a preset:", list(PRESETS_PT.keys()), key="pt_preset")
                use_preset = PRESETS_PT[preset_pt]

                if use_preset:
                    n_people = len(use_preset)
                else:
                    n_people = st.slider("Number of participants", min_value=2, max_value=12, value=5, key="pt_n")

                study_duration = st.number_input("Maximum study duration (years)", min_value=1.0, max_value=20.0, value=5.0, step=0.5, key="pt_maxdur")
                multiplier = st.selectbox("Express rate per:", [100, 1000, 10000], key="pt_mult")

                people = []
                st.markdown("---")
                st.markdown("**Enter each participant's follow-up:**")
                header_cols = st.columns([3, 3, 2, 3])
                header_cols[0].markdown("**Participant**")
                header_cols[1].markdown("**Years followed**")
                header_cols[2].markdown("**Outcome?**")
                header_cols[3].markdown("**Reason left / ended**")

                for i in range(n_people):
                    if use_preset and i < len(use_preset):
                        default_label  = use_preset[i]["label"]
                        default_years  = use_preset[i]["years"]
                        default_event  = use_preset[i]["event"]
                        default_reason = use_preset[i]["reason"]
                    else:
                        default_label  = f"Person {i+1}"
                        default_years  = study_duration
                        default_event  = False
                        default_reason = "Completed study"

                    row = st.columns([3, 3, 2, 3])
                    label  = row[0].text_input("", value=default_label,  key=f"pt_lbl_{i}",  label_visibility="collapsed")
                    years  = row[1].number_input("", min_value=0.1, max_value=float(study_duration), value=min(default_years, study_duration), step=0.5, key=f"pt_yrs_{i}", label_visibility="collapsed")
                    event  = row[2].checkbox("",   value=default_event,  key=f"pt_evt_{i}",  label_visibility="collapsed")
                    reason = row[3].text_input("", value=default_reason, key=f"pt_rsn_{i}",  label_visibility="collapsed")
                    people.append({"label": label, "years": years, "event": event, "reason": reason})

                if st.button("Generate Person-Time Diagram", key="ir_visual", type="primary"):
                    total_pt    = sum(p["years"] for p in people)
                    total_cases = sum(1 for p in people if p["event"])
                    ir = total_cases / total_pt * multiplier if total_pt > 0 else 0

                    # ---- SVG timeline diagram ----
                    row_h    = 36
                    pad_top  = 50
                    pad_left = 110
                    pad_right = 80
                    axis_w   = 420
                    svg_h    = pad_top + row_h * len(people) + 40
                    svg_w    = pad_left + axis_w + pad_right

                    # axis ticks
                    tick_interval = max(1, int(study_duration // 5))
                    ticks = list(range(0, int(study_duration) + 1, tick_interval))

                    def x_pos(yr):
                        return pad_left + (yr / study_duration) * axis_w

                    tick_svg = ""
                    for t in ticks:
                        xp = x_pos(t)
                        tick_svg += f'<line x1="{xp}" y1="{pad_top - 8}" x2="{xp}" y2="{pad_top + row_h*len(people)}" stroke="#ddd" stroke-width="1"/>'
                        tick_svg += f'<text x="{xp}" y="{pad_top - 12}" text-anchor="middle" font-size="11" fill="#888">Yr {t}</text>'

                    bars_svg = ""
                    legend_svg = ""
                    for i, p in enumerate(people):
                        y_center = pad_top + i * row_h + row_h // 2
                        bar_w    = (p["years"] / study_duration) * axis_w
                        color    = "#e53935" if p["event"] else "#1e88e5"
                        end_x    = x_pos(p["years"])

                        # name label
                        bars_svg += f'<text x="{pad_left - 8}" y="{y_center + 4}" text-anchor="end" font-size="12" fill="#333">{p["label"]}</text>'
                        # bar
                        bars_svg += f'<rect x="{pad_left}" y="{y_center - 9}" width="{bar_w}" height="18" rx="4" fill="{color}" opacity="0.75"/>'
                        # person-years label inside or just after bar
                        pt_label = f"{p['years']}y"
                        label_x  = pad_left + bar_w + 4
                        bars_svg += f'<text x="{label_x}" y="{y_center + 4}" font-size="11" fill="{color}" font-weight="bold">{pt_label}</text>'

                        # end marker: X for event, circle for censored
                        if p["event"]:
                            bars_svg += f'<text x="{end_x}" y="{y_center + 5}" text-anchor="middle" font-size="16" fill="#e53935" font-weight="bold">✕</text>'
                        else:
                            bars_svg += f'<circle cx="{end_x}" cy="{y_center}" r="6" fill="none" stroke="#1e88e5" stroke-width="2"/>'

                    # axis line
                    axis_svg = f'<line x1="{pad_left}" y1="{pad_top + row_h*len(people)}" x2="{pad_left + axis_w}" y2="{pad_top + row_h*len(people)}" stroke="#aaa" stroke-width="1.5"/>'

                    # legend
                    leg_y = pad_top + row_h * len(people) + 18
                    legend_svg = f'''
<rect x="{pad_left}" y="{leg_y}" width="14" height="14" rx="3" fill="#e53935" opacity="0.75"/>
<text x="{pad_left + 18}" y="{leg_y + 11}" font-size="11" fill="#555">✕ Outcome event (contributes years up to event)</text>
<rect x="{pad_left + 260}" y="{leg_y}" width="14" height="14" rx="3" fill="#1e88e5" opacity="0.75"/>
<circle cx="{pad_left + 296}" cy="{leg_y + 7}" r="5" fill="none" stroke="#1e88e5" stroke-width="2"/>
<text x="{pad_left + 304}" y="{leg_y + 11}" font-size="11" fill="#555">○ Censored (completed, lost, withdrew)</text>'''

                    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h + 30}" style="font-family:sans-serif; display:block; max-width:100%;">
  <rect width="{svg_w}" height="{svg_h + 30}" fill="#fafafa" rx="8"/>
  <text x="{pad_left + axis_w//2}" y="18" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Person-Time Follow-Up Diagram</text>
  {tick_svg}
  {axis_svg}
  {bars_svg}
  {legend_svg}
</svg>"""
                    st.markdown(svg, unsafe_allow_html=True)

                    # ---- Summary table ----
                    st.divider()
                    rows = []
                    for p in people:
                        rows.append({
                            "Participant": p["label"],
                            "Years followed": p["years"],
                            "Outcome": "✕ Yes" if p["event"] else "○ No",
                            "Reason": p["reason"],
                        })
                    rows.append({
                        "Participant": "**TOTAL**",
                        "Years followed": round(total_pt, 1),
                        "Outcome": f"**{total_cases} events**",
                        "Reason": "",
                    })
                    st.table(pd.DataFrame(rows))

                    # ---- Calculation breakdown ----
                    st.divider()
                    st.markdown("#### Person-Time Calculation")
                    st.markdown("Each row below shows one participant's contribution. The bars above represent these exact values.")
                    calc_rows = []
                    for p in people:
                        calc_rows.append(f"- **{p['label']}:** {p['years']} years {'*(outcome event — contributes up to event date)*' if p['event'] else '*(censored — contributes full follow-up)*'}")
                    st.markdown("\n".join(calc_rows))
                    contrib_str = " + ".join(str(p["years"]) for p in people)
                    st.markdown(f"\n**Total person-time = {contrib_str} = {round(total_pt,1)} person-years**")

                    # ---- Rate ----
                    st.divider()
                    st.markdown("#### Incidence Rate")
                    st.markdown(f"""
**IR = New cases ÷ Total person-time × {multiplier:,}**

= {total_cases} ÷ {round(total_pt,1)} × {multiplier:,}

= **{round(ir,2)} per {multiplier:,} person-years**
                    """)
                    st.metric(f"Incidence Rate (per {multiplier:,} person-years)", round(ir, 2))

                    # ---- Why not just use n? ----
                    naive_rate = total_cases / len(people) * multiplier
                    with st.expander("🤔 What if we had ignored follow-up time?"):
                        st.markdown(f"""
If we had simply divided cases by the number of participants (ignoring how long each was followed):

**Naive rate = {total_cases} ÷ {len(people)} × {multiplier:,} = {round(naive_rate,2)} per {multiplier:,}**

vs. the correct person-time rate: **{round(ir,2)} per {multiplier:,} person-years**

{"These are similar here because follow-up times happen to be close." if abs(naive_rate - ir)/max(ir,0.001) < 0.15 else f"These differ because participants had **very different follow-up durations** — treating them all equally would be misleading. Person-time accounts for the fact that someone followed for {max(p['years'] for p in people)} years had far more opportunity to develop the outcome than someone followed for {min(p['years'] for p in people)} years."}

**The key principle:** Person-time is the correct denominator whenever follow-up duration varies. Each person contributes time at risk proportional to how long they were actually observed — no more, no less.
                        """)

        elif calc_type == "Case Fatality Rate (CFR)":
            deaths = st.number_input("Deaths from disease", min_value=0, value=142)
            total_cases = st.number_input("Total diagnosed cases", min_value=1, value=1800)
            if st.button("Calculate CFR"):
                cfr = deaths / total_cases
                st.metric("Case Fatality Rate", f"{round(cfr*100,2)}%")
                st.info(f"{deaths} deaths among {total_cases:,} cases → CFR = {round(cfr*100,2)}%")
                st.markdown("**Note:** CFR is a proportion, not a rate — it has no time unit. It describes disease severity (lethality), not how common the disease is.")

    elif df_section == "3️⃣ Prevalence-Incidence Relationship":
        st.subheader("The Prevalence-Incidence Relationship")
        st.markdown("""
Prevalence, incidence, and disease duration are linked by a simple but important relationship:

**P ≈ I × D**

Where:
- **P** = Prevalence
- **I** = Incidence rate
- **D** = Average disease duration

This relationship holds when prevalence is low (<10%) and the disease is at steady state.
        """)

        st.info("""
**What this means:**

Prevalence is high when:
- **Incidence is high** (many new cases per year), OR
- **Duration is long** (people live with the condition for many years), OR both

Prevalence can be low even for serious diseases if they are rapidly fatal (short duration) or rapidly cured.
        """)

        with st.expander("🔢 Interactive: Explore the Relationship"):
            inc = st.slider("Incidence rate (per 1,000/year)", 1, 50, 10)
            dur = st.slider("Average disease duration (years)", 1, 30, 5)
            prev_est = (inc/1000) * dur
            st.metric("Estimated Prevalence", f"{round(prev_est*100,1)}%")
            st.markdown(f"P ≈ I × D = {inc/1000} × {dur} = {round(prev_est,4)} ≈ {round(prev_est*100,1)}%")

        with st.expander("Examples"):
            st.markdown("""
| Disease | Incidence | Duration | Prevalence |
|---|---|---|---|
| Flu | High (each winter) | Short (~1 week) | Low at any point |
| HIV (pre-treatment era) | Lower | Long (years) | High relative to incidence |
| HIV (modern treatment) | Lower | Very long (decades) | Even higher — people live longer |
| Ebola (outbreak) | High during outbreak | Short (fatal quickly) | Low |

**Treatment and prevalence:** Effective treatment that extends life (but doesn't cure) **increases prevalence** — more people live longer with the disease. This is why diabetes and HIV prevalence have risen even as incidence has stabilized or fallen.
            """)

    elif df_section == "4️⃣ Epidemic Curves":
        st.subheader("Epidemic Curves")
        st.markdown("""
An **epidemic curve (epi curve)** is a histogram of case counts by time of symptom onset. Before any lab results come back, the shape of the curve tells you the likely transmission pattern, the probable incubation period, and whether an outbreak is still growing.
        """)
        st.divider()

        import math as _math
        import streamlit.components.v1 as _components

        EPI_PRESETS = {
            "— Select a scenario —": None,
            # --- POINT SOURCE ---
            "☢️ Point Source: Staph Toxin at a Catered Dinner": {
                "type": "point",
                "description": "**Scenario:** 180 attendees at a catered company dinner. 63 develop vomiting and diarrhea within 2–6 hours of the meal. No household contacts become ill afterward.",
                "peak": 4, "spread": 1.5, "total": 63,
                "x_label": "Hours after dinner",
                "color": "#e53935",
                "key_features": [
                    "Single sharp peak — nearly all cases within a 4-hour window",
                    "Incubation period of 2–6 hours → preformed bacterial toxin (*Staphylococcus aureus* or *Bacillus cereus*), not a live infection requiring replication",
                    "No secondary cases in households — exposure ended with the meal",
                    "Width of curve ≈ incubation period range (2–6 h)",
                ],
                "next_step": "Calculate **food-specific attack rates** for every dish served. The food with the highest RR and lowest p-value is the likely vehicle.",
                "contrast": "A propagated outbreak would show multiple waves, each ~one incubation period apart, with household contacts becoming ill days later.",
            },
            "☢️ Point Source: E. coli O157 at a County Fair": {
                "type": "point",
                "description": "**Scenario:** Health department investigates 38 cases of bloody diarrhea linked to a county fair petting zoo. Onset times cluster 48–96 hours after the fair date.",
                "peak": 72, "spread": 12, "total": 38,
                "x_label": "Hours after fair visit",
                "color": "#e53935",
                "key_features": [
                    "Single peak centered ~72 hours post-exposure",
                    "Incubation period of 48–96 hours consistent with **E. coli O157:H7** (vs. 2–6h for toxin)",
                    "Longer, flatter peak than toxin-mediated outbreaks — wider incubation range",
                    "No ongoing source — cases plateau and decline after removing fair exposure",
                ],
                "next_step": "Identify specific petting zoo animal contacts or shared water sources. Use **case-control study** with fair attendees as controls.",
                "contrast": "The longer incubation (days, not hours) distinguishes bacterial infection from preformed toxin — same point-source shape, different time scale.",
            },
            # --- PROPAGATED ---
            "🔗 Propagated: Norovirus on a Cruise Ship": {
                "type": "propagated",
                "description": "**Scenario:** A cruise ship with 2,000 passengers departs port. Day 2: 12 cases of vomiting/diarrhea reported. Over the following days, cases spread rapidly among passengers and crew.",
                "incubation": 2, "waves": 4, "index": 12, "r0": 2.5,
                "x_label": "Days into cruise",
                "color": "#1e88e5",
                "key_features": [
                    "Multiple successive waves, each ~2 days apart (norovirus incubation = 12–48h)",
                    "R₀ ≈ 2.5 on a ship — confined space, shared surfaces, aerosol spread amplify transmission",
                    "Wave sizes grow until herd immunity or intervention reduces R₀ below 1",
                    "Classic propagated pattern: each generation of cases infects the next",
                ],
                "next_step": "Implement enhanced hand hygiene, surface disinfection, isolation of symptomatic passengers. Goal: reduce R₀ below 1.",
                "contrast": "A point source would have caused all cases to cluster on day 2–3 and then stop. The ongoing waves confirm person-to-person spread.",
            },
            "🔗 Propagated: Measles in an Under-Vaccinated School": {
                "type": "propagated",
                "description": "**Scenario:** An unvaccinated child returns from international travel and attends school while infectious. Over the following weeks, measles spreads through the 30% of students who are unvaccinated.",
                "incubation": 10, "waves": 3, "index": 1, "r0": 3.2,
                "x_label": "Days from index case",
                "color": "#1e88e5",
                "key_features": [
                    "Waves ~10 days apart — measles incubation period is 8–12 days",
                    "R₀ = 12–18 in fully susceptible populations; R₀ ≈ 3.2 here because ~70% are vaccinated (partial herd immunity)",
                    "Outbreak self-limits when remaining susceptibles are exhausted or vaccinated",
                    "Herd immunity threshold for measles: ~94% (1 − 1/R₀ at true R₀ = 15)",
                ],
                "next_step": "Emergency vaccination campaign targeting unvaccinated students. Exclude symptomatic/exposed unvaccinated children until immune status confirmed.",
                "contrast": "The 10-day spacing between waves is the epidemiological signature of measles. Flu waves would be 2–3 days apart.",
            },
            # --- MIXED ---
            "🔀 Mixed: SARS Superspreader Event → Hospital Spread": {
                "type": "mixed",
                "description": "**Scenario:** A single hospitalized SARS patient (index case) is exposed to 30 healthcare workers and visitors in a poorly ventilated ward over 48 hours. Secondary spread then occurs among hospital staff.",
                "incubation": 5, "ps_cases": 28, "r0": 1.8, "waves": 2,
                "x_label": "Days from superspreader event",
                "color": "#7b1fa2",
                "key_features": [
                    "Sharp first wave at days 4–6 — the original superspreader exposure (point source)",
                    "Broader secondary waves — person-to-person spread among hospital staff contacts",
                    "Signature of mixed transmission: narrow first peak, progressively wider subsequent waves",
                    "High first-wave count (28 cases) reflects superspreader dynamics in confined space",
                ],
                "next_step": "Both interventions needed simultaneously: (1) identify and isolate the point-source contact, (2) implement droplet/contact precautions to interrupt propagated spread.",
                "contrast": "A pure point source would end after the first wave. A pure propagated curve would have a gradual initial rise. The sharp first peak + subsequent waves = mixed.",
            },
            "🔀 Mixed: Contaminated Water + Secondary Transmission (Cholera)": {
                "type": "mixed",
                "description": "**Scenario:** A broken water main contaminates drinking water in a neighborhood for 3 days, causing an initial cluster of cholera cases. Household transmission then continues after the water source is fixed.",
                "incubation": 3, "ps_cases": 40, "r0": 1.4, "waves": 3,
                "x_label": "Days from water contamination",
                "color": "#7b1fa2",
                "key_features": [
                    "First wave: sharp peak from contaminated water (point source, days 1–5)",
                    "Subsequent waves: household fecal-oral transmission even after water is fixed",
                    "Lower R₀ (1.4) than norovirus — cholera spreads less efficiently person-to-person",
                    "Outbreak ends when both source is removed AND household chains are interrupted",
                ],
                "next_step": "Fix the water supply AND implement oral rehydration therapy + hygiene education for household contacts. Source removal alone insufficient.",
                "contrast": "If only the water were fixed with no household intervention, you would see the first wave decline but secondary waves continue — a common outbreak investigation mistake.",
            },
            # --- ENDEMIC ---
            "📊 Endemic: Tuberculosis Notifications (High-Burden Setting)": {
                "type": "endemic",
                "description": "**Scenario:** A public health team reviews weekly TB case notifications in a high-burden urban district over 6 months. No outbreak is declared.",
                "baseline": 14, "noise": 4, "weeks": 26,
                "x_label": "Week",
                "color": "#43a047",
                "key_features": [
                    "Flat baseline ~14 cases/week — ongoing transmission from established reservoir",
                    "Week-to-week variation (±4) reflects reporting delays, not true transmission spikes",
                    "No explosive peak — disease is endemic, not epidemic",
                    "Sustained burden requires long-term control, not reactive outbreak response",
                ],
                "next_step": "Use **epidemic thresholds** (mean ± 2 SD of historical counts) to define alert zones. A true outbreak would be declared when counts exceed the upper threshold for 2+ consecutive weeks.",
                "contrast": "An outbreak superimposed on endemic TB would appear as a sustained rise *above* baseline — not just normal week-to-week noise.",
            },
            "📊 Endemic: Salmonella Background Level (US Surveillance)": {
                "type": "endemic",
                "description": "**Scenario:** National surveillance tracks weekly Salmonella laboratory-confirmed cases. No specific outbreak event identified — this is routine background incidence.",
                "baseline": 22, "noise": 6, "weeks": 52,
                "x_label": "Week of year",
                "color": "#43a047",
                "key_features": [
                    "Stable mean ~22 cases/week with natural variation",
                    "Salmonella is always circulating — endemic from multiple animal reservoirs",
                    "Seasonal peaks (summer) expected in real data but not shown here for clarity",
                    "Purpose of surveillance: detect clusters above background that signal an outbreak",
                ],
                "next_step": "Cluster detection algorithms (e.g., PFGE molecular typing, whole genome sequencing) identify when cases share a common source above background noise.",
                "contrast": "When a contaminated product enters the market (e.g., peanut butter Salmonella outbreak), the curve rises sharply above this stable baseline — detectable only because the baseline is well characterized.",
            },
        }

        if "epi_preset" not in st.session_state:
            st.session_state["epi_preset"] = "— Select a scenario —"

        preset_choice = st.selectbox(
            "Select a scenario:",
            list(EPI_PRESETS.keys()),
            key="epi_preset"
        )
        preset = EPI_PRESETS[preset_choice]

        if preset is None:
            st.info("Select a scenario above to see an epidemic curve, key features, and epidemiological interpretation.")
        else:
            st.info(preset["description"])
            st.divider()

            # ---- Generate counts ----
            import random as _random
            _random.seed(99)

            ptype = preset["type"]
            if ptype == "point":
                peak    = preset["peak"]
                spread  = preset["spread"]
                total   = preset["total"]
                n_steps = peak * 3 + 1
                steps   = list(range(n_steps))
                raw     = [_math.exp(-0.5 * ((h - peak) / spread) ** 2) for h in steps]
                scale   = total / max(sum(raw), 0.001)
                counts  = [max(0, round(r * scale)) for r in raw]
                x_vals  = steps

            elif ptype == "propagated":
                incubation = preset["incubation"]
                n_waves    = preset["waves"]
                index      = preset["index"]
                r0         = preset["r0"]
                n_steps    = incubation * (n_waves + 2)
                steps      = list(range(n_steps))
                counts     = [0.0] * n_steps
                wave_sizes = [index]
                for w in range(1, n_waves):
                    wave_sizes.append(round(wave_sizes[-1] * r0))
                for w, size in enumerate(wave_sizes):
                    peak_day   = w * incubation + incubation // 2
                    wave_spd   = max(1, incubation // 3)
                    for d in steps:
                        counts[d] += size * _math.exp(-0.5 * ((d - peak_day) / wave_spd) ** 2)
                counts = [max(0, round(c)) for c in counts]
                x_vals = steps

            elif ptype == "mixed":
                incubation = preset["incubation"]
                ps_cases   = preset["ps_cases"]
                r0         = preset["r0"]
                n_waves    = preset["waves"]
                n_steps    = incubation * (n_waves + 3)
                steps      = list(range(n_steps))
                counts     = [0.0] * n_steps
                # point source wave
                ps_peak    = incubation // 2 + 1
                ps_spd     = max(1, incubation // 4)
                for d in steps:
                    counts[d] += ps_cases * _math.exp(-0.5 * ((d - ps_peak) / ps_spd) ** 2)
                # secondary waves
                wave_sizes = [round(ps_cases * r0 * 0.25)]
                for w in range(1, n_waves):
                    wave_sizes.append(round(wave_sizes[-1] * r0 * 0.6))
                for w, size in enumerate(wave_sizes):
                    peak_day  = (w + 1) * incubation + incubation // 2
                    wave_spd  = max(2, incubation // 2)
                    for d in steps:
                        counts[d] += size * _math.exp(-0.5 * ((d - peak_day) / wave_spd) ** 2)
                counts = [max(0, round(c)) for c in counts]
                x_vals = steps

            else:  # endemic
                baseline = preset["baseline"]
                noise    = preset["noise"]
                n_weeks  = preset["weeks"]
                x_vals   = list(range(1, n_weeks + 1))
                counts   = [max(0, baseline + _random.randint(-noise, noise)) for _ in x_vals]

            color     = preset["color"]
            x_label   = preset["x_label"]
            max_count = max(counts) if counts else 1

            # ---- Build SVG via components.html so it actually renders ----
            chart_w   = 700
            chart_h   = 240
            pad_l     = 52
            pad_b     = 46
            pad_t     = 32
            pad_r     = 20
            plot_w    = chart_w - pad_l - pad_r
            plot_h    = chart_h - pad_b - pad_t
            n_bars    = len(counts)
            bar_gap   = 1
            bar_w     = max(2.0, (plot_w - bar_gap * (n_bars - 1)) / n_bars)

            def xp(i):
                return pad_l + i * (bar_w + bar_gap)
            def yp(c):
                return pad_t + plot_h - (c / max_count) * plot_h if max_count > 0 else pad_t + plot_h

            bars = ""
            for i, c in enumerate(counts):
                bh = (c / max_count) * plot_h if max_count > 0 else 0
                bars += f'<rect x="{round(xp(i),1)}" y="{round(yp(c),1)}" width="{round(bar_w,1)}" height="{round(bh,1)}" fill="{color}" rx="1" opacity="0.82"/>'

            # y ticks
            y_ticks = ""
            for frac in [0, 0.25, 0.5, 0.75, 1.0]:
                val = round(max_count * frac)
                ty  = round(pad_t + plot_h - frac * plot_h, 1)
                y_ticks += f'<line x1="{pad_l}" y1="{ty}" x2="{pad_l + plot_w}" y2="{ty}" stroke="#ececec" stroke-width="1"/>'
                y_ticks += f'<text x="{pad_l - 6}" y="{ty + 4}" text-anchor="end" font-size="11" fill="#999">{val}</text>'

            # x labels
            step   = max(1, n_bars // 10)
            x_lbls = ""
            for i in range(0, n_bars, step):
                lx = round(xp(i) + bar_w / 2, 1)
                x_lbls += f'<text x="{lx}" y="{chart_h - 4}" text-anchor="middle" font-size="10" fill="#999">{x_vals[i]}</text>'

            # axis lines
            axes = (
                f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+plot_h}" stroke="#bbb" stroke-width="1.5"/>'
                f'<line x1="{pad_l}" y1="{pad_t+plot_h}" x2="{pad_l+plot_w}" y2="{pad_t+plot_h}" stroke="#bbb" stroke-width="1.5"/>'
                f'<text x="{pad_l - 36}" y="{pad_t + plot_h//2}" text-anchor="middle" font-size="11" fill="#666" '
                f'transform="rotate(-90,{pad_l-36},{pad_t + plot_h//2})">Cases</text>'
                f'<text x="{pad_l + plot_w//2}" y="{chart_h + 14}" text-anchor="middle" font-size="11" fill="#666">{x_label}</text>'
            )

            # title
            title_txt = preset_choice.split(": ", 1)[-1] if ": " in preset_choice else preset_choice
            title_svg = f'<text x="{pad_l + plot_w//2}" y="18" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">{title_txt}</text>'

            svg_html = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;background:transparent;">
<svg xmlns="http://www.w3.org/2000/svg" width="{chart_w}" height="{chart_h + 20}"
     style="font-family:sans-serif;display:block;background:#fafafa;border-radius:8px;">
  {title_svg}
  {y_ticks}
  {bars}
  {axes}
  {x_lbls}
</svg></body></html>"""

            _components.html(svg_html, height=chart_h + 40)

            # ---- Key features ----
            st.divider()
            col_feat, col_interp = st.columns([1, 1])

            with col_feat:
                st.markdown("**Key features of this curve:**")
                for feat in preset["key_features"]:
                    st.markdown(f"✅ {feat}")

            with col_interp:
                st.markdown("**Next investigative step:**")
                st.info(preset["next_step"])
                st.markdown("**Contrast with other patterns:**")
                st.markdown(preset["contrast"])

            # ---- Comparison reference ----
            with st.expander("📊 Compare All Four Patterns Side by Side"):
                st.markdown("""
| Feature | ☢️ Point Source | 🔗 Propagated | 🔀 Mixed | 📊 Endemic |
|---|---|---|---|---|
| **Shape** | Single sharp peak | Multiple waves | Sharp peak + broader waves | Flat, stable baseline |
| **Time span** | One incubation period | Multiple incubation periods | Both | Ongoing |
| **Secondary cases?** | No | Yes | Yes (after initial wave) | Sustained background |
| **Curve width** | Narrow (≈ incubation range) | Wide, multi-peak | Narrow then wide | Flat |
| **R₀ relevant?** | No | Yes (>1 sustains spread) | Yes for secondary waves | ≈1 (stable) |
| **Intervention target** | Remove source | Interrupt transmission | Both simultaneously | Long-term control |
| **Classic example** | Food poisoning at event | Measles, norovirus on ship | SARS superspreader | Endemic TB, Salmonella |
                """)

            with st.expander("🔢 Reading the Incubation Period from the Curve"):
                st.markdown(f"""
**Point-source outbreak:**
The *range* of onset times after a known exposure = the plausible incubation period.
- 2–6 hours → preformed toxin (*Staph aureus*, *B. cereus*)
- 18–36 hours → bacterial infection (*Salmonella*, *Campylobacter*)
- 48–96 hours → *E. coli* O157, some viruses

**Propagated outbreak:**
The *gap between wave peaks* = one incubation period.
- 2-day waves → norovirus (12–48h incubation)
- 10-day waves → measles (8–12 days)
- 5-day waves → COVID-19 (4–6 days, early variant estimates)

**Why it matters:** Knowing the incubation period narrows the list of causative agents *before* lab results return, and defines the exposure window — what did cases eat, touch, or contact in the {int(preset.get("peak", preset.get("incubation", 5)))} {"hours" if ptype == "point" else "days"} before symptom onset?
                """)
    st.markdown("---")
    st.markdown("*Strong epidemiologists think structurally before computing.*")


# ==================================================
# MODULE 2: SCREENING & DIAGNOSTIC TESTS
# ==================================================
elif current_page == "screening":
    st.title("🔬 Screening & Diagnostic Tests")
    st.markdown("Evaluating the performance of a test requires understanding how sensitivity, specificity, and the prevalence of disease in the population interact.")

    screen_section = st.radio("Section:", ["1️⃣ Core Concepts", "2️⃣ Interactive 2×2 Calculator", "3️⃣ Prevalence Effect on PPV"], horizontal=True)
    st.divider()

    if screen_section == "1️⃣ Core Concepts":
        st.subheader("The Screening 2×2 Table")
        st.markdown("All screening test metrics come from a single 2×2 table comparing test result to true disease status.")

        table_html = """
<div style="margin:16px 0;">
<table style="border-collapse:collapse;width:100%;max-width:600px;">
  <tr>
    <td style="border:none;padding:8px;"></td>
    <td style="border:1px solid #aaa;padding:10px;text-align:center;background:#fce4ec;font-weight:bold;color:#c62828;">Disease Present</td>
    <td style="border:1px solid #aaa;padding:10px;text-align:center;background:#e8f5e9;font-weight:bold;color:#2e7d32;">Disease Absent</td>
    <td style="border:1px solid #eee;padding:10px;text-align:center;font-size:12px;color:#666;">Row Total</td>
  </tr>
  <tr>
    <td style="border:1px solid #aaa;padding:10px;background:#fff3e0;font-weight:bold;color:#e65100;">Test Positive</td>
    <td style="border:1px solid #aaa;padding:12px;text-align:center;font-size:18px;font-weight:bold;color:#2e7d32;">a<br><span style="font-size:11px;color:#888;">True Positive (TP)</span></td>
    <td style="border:1px solid #aaa;padding:12px;text-align:center;font-size:18px;font-weight:bold;color:#c62828;">b<br><span style="font-size:11px;color:#888;">False Positive (FP)</span></td>
    <td style="border:1px solid #eee;padding:10px;text-align:center;font-size:13px;">a+b</td>
  </tr>
  <tr>
    <td style="border:1px solid #aaa;padding:10px;background:#fff3e0;font-weight:bold;color:#e65100;">Test Negative</td>
    <td style="border:1px solid #aaa;padding:12px;text-align:center;font-size:18px;font-weight:bold;color:#c62828;">c<br><span style="font-size:11px;color:#888;">False Negative (FN)</span></td>
    <td style="border:1px solid #aaa;padding:12px;text-align:center;font-size:18px;font-weight:bold;color:#2e7d32;">d<br><span style="font-size:11px;color:#888;">True Negative (TN)</span></td>
    <td style="border:1px solid #eee;padding:10px;text-align:center;font-size:13px;">c+d</td>
  </tr>
  <tr>
    <td style="border:none;padding:8px;font-size:12px;color:#666;">Col Total</td>
    <td style="border:1px solid #eee;padding:10px;text-align:center;font-size:13px;">a+c</td>
    <td style="border:1px solid #eee;padding:10px;text-align:center;font-size:13px;">b+d</td>
    <td style="border:1px solid #eee;padding:10px;text-align:center;font-size:13px;">N</td>
  </tr>
</table>
</div>"""
        st.markdown(table_html, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Properties of the Test (Fixed)")
            st.markdown("""
**Sensitivity** = a ÷ (a+c)
- Proportion of true cases that test *positive*
- "If you have the disease, how likely is the test to catch it?"
- High sensitivity → few false negatives → good for ruling OUT disease
- Mnemonic: **SnNout** — Sensitive test, Negative result, rules Out

**Specificity** = d ÷ (b+d)
- Proportion of true non-cases that test *negative*
- "If you don't have the disease, how likely is the test to be negative?"
- High specificity → few false positives → good for ruling IN disease
- Mnemonic: **SpPin** — Specific test, Positive result, rules In
            """)

        with col2:
            st.markdown("#### Clinical Usefulness (Depend on Prevalence)")
            st.markdown("""
**Positive Predictive Value (PPV)** = a ÷ (a+b)
- Probability that a *positive test* means disease is truly present
- Changes with prevalence — even a specific test has low PPV in low-prevalence populations

**Negative Predictive Value (NPV)** = d ÷ (c+d)
- Probability that a *negative test* means disease is truly absent
- Increases as prevalence decreases (negatives more reliable when disease is rare)

**Accuracy** = (a+d) ÷ N
- Proportion of all tests that are correct
- Misleading in low-prevalence settings (always predicting negative = high accuracy but useless)
            """)

        st.divider()
        st.markdown("#### Sensitivity-Specificity Tradeoff")
        st.info("For any test, you can shift the cutpoint: lowering it increases sensitivity but decreases specificity (more positives, more false positives). Raising it increases specificity but decreases sensitivity. The ROC curve plots this tradeoff.")

    elif screen_section == "2️⃣ Interactive 2×2 Calculator":
        st.subheader("Screening Test Calculator")

        SCREEN_PRESETS = {
            "None — enter my own": None,
            "Mammography & Breast Cancer (50-year-olds)": {"a":10,"b":89,"c":1,"d":900,"desc":"Approximate values based on typical screening mammography performance in average-risk 50-year-old women. Prevalence ~1%."},
            "Rapid Strep Test": {"a":75,"b":10,"c":15,"d":900,"desc":"Rapid antigen test for Group A strep. Sensitivity ~83%, Specificity ~98%."},
            "PSA Screening (>4 ng/mL) & Prostate Cancer": {"a":70,"b":300,"c":30,"d":600,"desc":"PSA threshold of 4 ng/mL. Relatively low specificity leads to many false positives."},
        }

        preset_choice = st.selectbox("Load a preset:", list(SCREEN_PRESETS.keys()), key="screen_preset")
        preset = SCREEN_PRESETS[preset_choice]
        if preset and preset.get("desc"):
            st.info(preset["desc"])

        col1, col2, col3, col4 = st.columns(4)
        defaults = preset if preset else {"a":90,"b":10,"c":10,"d":890}
        a = col1.number_input("a (TP)", min_value=0, value=defaults["a"], key="sc_a")
        b = col2.number_input("b (FP)", min_value=0, value=defaults["b"], key="sc_b")
        c = col3.number_input("c (FN)", min_value=0, value=defaults["c"], key="sc_c")
        d = col4.number_input("d (TN)", min_value=0, value=defaults["d"], key="sc_d")

        if st.button("Calculate Test Performance"):
            total_disease = a + c
            total_no_disease = b + d
            total_pos = a + b
            total_neg = c + d
            N = a + b + c + d
            sens = a / total_disease if total_disease > 0 else 0
            spec = d / total_no_disease if total_no_disease > 0 else 0
            ppv = a / total_pos if total_pos > 0 else 0
            npv = d / total_neg if total_neg > 0 else 0
            acc = (a + d) / N if N > 0 else 0
            prev = total_disease / N

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Sensitivity", f"{round(sens*100,1)}%")
            col2.metric("Specificity", f"{round(spec*100,1)}%")
            col3.metric("PPV", f"{round(ppv*100,1)}%")
            col4.metric("NPV", f"{round(npv*100,1)}%")
            col5.metric("Prevalence", f"{round(prev*100,1)}%")

            st.divider()
            if ppv < 0.5:
                st.warning(f"⚠️ **Low PPV ({round(ppv*100,1)}%):** More than half of positive tests are false positives. In this population (prevalence = {round(prev*100,1)}%), a positive result is more likely to be a false alarm than a true case. Consider confirmatory testing.")
            else:
                st.success(f"✅ **PPV = {round(ppv*100,1)}%:** Most positive tests reflect true disease in this population.")

            with st.expander("🔢 Show me the math"):
                st.markdown(f"""
| Metric | Formula | Calculation | Result |
|---|---|---|---|
| Sensitivity | a ÷ (a+c) | {a} ÷ {total_disease} | **{round(sens*100,1)}%** |
| Specificity | d ÷ (b+d) | {d} ÷ {total_no_disease} | **{round(spec*100,1)}%** |
| PPV | a ÷ (a+b) | {a} ÷ {total_pos} | **{round(ppv*100,1)}%** |
| NPV | d ÷ (c+d) | {d} ÷ {total_neg} | **{round(npv*100,1)}%** |
| Accuracy | (a+d) ÷ N | {a+d} ÷ {N} | **{round(acc*100,1)}%** |
                """)

    elif screen_section == "3️⃣ Prevalence Effect on PPV":
        st.subheader("How Prevalence Changes PPV")
        st.markdown("One of the most important — and counterintuitive — facts in screening: **a test with excellent sensitivity and specificity can still have very poor PPV when disease prevalence is low.**")

        sens_fixed = st.slider("Test Sensitivity (fixed)", 50, 99, 90, 1, format="%d%%", key="sens_slider")
        spec_fixed = st.slider("Test Specificity (fixed)", 50, 99, 95, 1, format="%d%%", key="spec_slider")
        sens_fixed = sens_fixed / 100
        spec_fixed = spec_fixed / 100

        prevalences = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
        ppv_vals = []
        npv_vals = []
        for p in prevalences:
            tp = sens_fixed * p
            fp = (1 - spec_fixed) * (1 - p)
            fn = (1 - sens_fixed) * p
            tn = spec_fixed * (1 - p)
            ppv_v = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv_v = tn / (tn + fn) if (tn + fn) > 0 else 0
            ppv_vals.append(round(ppv_v * 100, 1))
            npv_vals.append(round(npv_v * 100, 1))

        result_df = pd.DataFrame({
            "Prevalence": [f"{round(p*100,1)}%" for p in prevalences],
            "PPV": [f"{v}%" for v in ppv_vals],
            "NPV": [f"{v}%" for v in npv_vals],
        })
        st.table(result_df)

        st.warning(f"""
**Key insight:** With sensitivity = {round(sens_fixed*100,0):.0f}% and specificity = {round(spec_fixed*100,0):.0f}%:
- At 0.1% prevalence (mass population screening): PPV = {ppv_vals[0]}% — most positives are false alarms
- At 10% prevalence (high-risk clinic): PPV = {ppv_vals[5]}%
- At 50% prevalence (symptomatic patients): PPV = {ppv_vals[8]}%

The test hasn't changed — only the population it's applied to.
        """)

        st.markdown("""
**Why does this happen?** In a low-prevalence population, there are very few true cases but an enormous number of disease-free people. Even a test with high specificity — meaning its false positive *rate* is low — will produce a large absolute *number* of false positives when applied to millions of non-cases. A 95% specific test still incorrectly flags 5% of non-cases as positive; in a population of 1,000,000 non-cases, that's 50,000 false positives regardless of how few true cases exist. The problem is not the test's performance — it's the mismatch between the test's false positive rate and the vast size of the non-case pool. This is the mathematical basis for why mass screening of low-risk populations often produces more harm (unnecessary follow-up, anxiety, procedures) than benefit.
        """)

    st.markdown("---")
    st.markdown("*Strong epidemiologists think structurally before computing.*")


# ==================================================
# MODULE 3: MEASURES OF ASSOCIATION (existing tab1)
# ==================================================
elif current_page == "measures_association":
    st.title("📈 Measures of Association")

    PRESETS = {
        "None — I'll enter my own data": None,
        "Cohort: Smoking & Lung Cancer": {
            "design": "Cohort", "outcome_type": "Binary", "exposure_type": "Binary (2 groups)",
            "row_0": "Smoker", "row_1": "Non-smoker", "col_0": "Lung Cancer", "col_1": "No Lung Cancer",
            "cell_0_0": 84, "cell_0_1": 2916, "cell_1_0": 14, "cell_1_1": 2986,
            "description": "**Scenario:** Prospective cohort of 6,000 adults over 10 years. *Doll & Hill (1950).*"
        },
        "Case-Control: H. pylori & Gastric Ulcer": {
            "design": "Case-Control", "outcome_type": "Binary", "exposure_type": "Binary (2 groups)",
            "row_0": "H. pylori positive", "row_1": "H. pylori negative",
            "col_0": "Gastric Ulcer (Case)", "col_1": "No Ulcer (Control)",
            "cell_0_0": 118, "cell_0_1": 62, "cell_1_0": 32, "cell_1_1": 138,
            "description": "**Scenario:** Hospital-based case-control study. *Marshall & Warren (1984).*"
        },
        "Cross-sectional: Obesity & Hypertension": {
            "design": "Cross-sectional", "outcome_type": "Binary", "exposure_type": "Binary (2 groups)",
            "row_0": "Obese (BMI ≥ 30)", "row_1": "Non-obese (BMI < 30)",
            "col_0": "Hypertension", "col_1": "No Hypertension",
            "cell_0_0": 210, "cell_0_1": 290, "cell_1_0": 120, "cell_1_1": 880,
            "description": "**Scenario:** One-time cross-sectional health survey. *NHANES.*"
        },
    }

    if "last_preset" not in st.session_state:
        st.session_state["last_preset"] = None

    col_title, col_reset = st.columns([5, 1])
    with col_reset:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reset", key="reset_moa"):
            for k in ["preset_choice_moa","last_preset","design","outcome_type","exposure_type",
                      "row_0","row_1","col_0","col_1","cell_0_0","cell_0_1","cell_1_0","cell_1_1"]:
                if k in st.session_state: del st.session_state[k]
            st.rerun()

    preset_choice = st.selectbox("Select a scenario:", list(PRESETS.keys()), key="preset_choice_moa")
    preset = PRESETS[preset_choice]

    if preset_choice != st.session_state["last_preset"]:
        st.session_state["last_preset"] = preset_choice
        if preset:
            for key in ["design","outcome_type","exposure_type","row_0","row_1","col_0","col_1",
                        "cell_0_0","cell_0_1","cell_1_0","cell_1_1"]:
                if key in preset: st.session_state[key] = preset[key]
        else:
            defaults = {"row_0":"Group 1","row_1":"Group 2","col_0":"Level 1","col_1":"Level 2",
                       "cell_0_0":0,"cell_0_1":0,"cell_1_0":0,"cell_1_1":0,
                       "design":"Cohort","outcome_type":"Binary","exposure_type":"Binary (2 groups)"}
            for k,v in defaults.items(): st.session_state[k] = v
        st.rerun()

    if preset: st.info(preset["description"])
    st.divider()

    design_options = ["Cohort","Case-Control","Cross-sectional"]
    if "design" not in st.session_state: st.session_state["design"] = "Cohort"
    design = st.selectbox("Select study design:", design_options,
                          index=design_options.index(st.session_state.get("design","Cohort")))

    outcome_options = ["Binary","Categorical (Nominal >2 levels)","Ordinal","Rate (person-time)"]
    if "outcome_type" not in st.session_state: st.session_state["outcome_type"] = "Binary"
    outcome_type = st.selectbox("Select outcome type:", outcome_options,
                                index=outcome_options.index(st.session_state.get("outcome_type","Binary")))

    exposure_options = ["Binary (2 groups)","Categorical (>2 groups)"]
    if "exposure_type" not in st.session_state: st.session_state["exposure_type"] = "Binary (2 groups)"
    exposure_type = st.selectbox("Select exposure type:", exposure_options,
                                 index=exposure_options.index(st.session_state.get("exposure_type","Binary (2 groups)")))

    st.divider()

    if outcome_type in ["Binary","Categorical (Nominal >2 levels)","Ordinal"]:
        num_rows = 2 if exposure_type == "Binary (2 groups)" else st.number_input("Number of Exposure Groups", min_value=2, value=3)
        num_cols = 2 if outcome_type == "Binary" else st.number_input("Number of Outcome Levels", min_value=2, value=3)

        st.subheader("Label Exposure Groups")
        row_names = [st.text_input(f"Exposure Group {i+1}", key=f"row_{i}") for i in range(num_rows)]
        st.subheader("Label Outcome Levels")
        col_names = [st.text_input(f"Outcome Level {j+1}", key=f"col_{j}") for j in range(num_cols)]
        st.subheader("Enter Cell Counts")
        table = []
        for i in range(num_rows):
            row = []
            cols_ui = st.columns(num_cols)
            for j in range(num_cols):
                key = f"cell_{i}_{j}"
                default_val = st.session_state.get(key, 0)
                val = cols_ui[j].number_input(f"{row_names[i]} / {col_names[j]}", min_value=0, value=int(default_val), key=key)
                row.append(val)
            table.append(row)

        if st.button("Run Analysis", key="run_moa"):
            table_array = np.array(table)
            if table_array.sum() == 0:
                st.warning("Please enter data.")
            else:
                df_display = pd.DataFrame(table_array, columns=col_names, index=row_names)
                df_display["Row Total"] = df_display.sum(axis=1)
                tr = df_display.sum(); tr.name = "Column Total"
                df_display = pd.concat([df_display, tr.to_frame().T])
                st.table(df_display)

                chi2_val, p_val, dof, _ = chi2_contingency(table_array)
                st.write(f"χ²({dof}) = {round(chi2_val,3)}, p = {round(p_val,4) if p_val >= 0.0001 else '< 0.0001'}")
                if p_val < 0.05: st.success("Statistically significant. Reject H₀.")
                else: st.warning("Insufficient evidence to reject H₀.")

                chi2_explanation_expander(chi2_val, p_val, dof, table_array, col_names, row_names)

                if outcome_type == "Binary" and exposure_type == "Binary (2 groups)":
                    a, b = table[0]; c, d = table[1]
                    if all(v > 0 for v in [a, b, c, d]):
                        is_cs = design == "Cross-sectional"
                        pabbr = "PR" if is_cs else "RR"
                        rr = (a/(a+b))/(c/(c+d))
                        se_log_rr = math.sqrt((1/a)-(1/(a+b))+(1/c)-(1/(c+d)))
                        ci_low_rr = math.exp(math.log(rr)-1.96*se_log_rr)
                        ci_high_rr = math.exp(math.log(rr)+1.96*se_log_rr)
                        or_val = (a*d)/(b*c)
                        se_log_or = math.sqrt(1/a+1/b+1/c+1/d)
                        ci_low_or = math.exp(math.log(or_val)-1.96*se_log_or)
                        ci_high_or = math.exp(math.log(or_val)+1.96*se_log_or)

                        st.subheader(f"{'Prevalence Ratio (PR)' if is_cs else 'Risk Ratio (RR)'}")
                        if ci_low_rr <= 1 <= ci_high_rr: st.warning(f"{pabbr} = {round(rr,2)} — CI includes 1. Not significant.")
                        else:
                            direction = "higher" if rr > 1 else "lower"
                            st.success(f"{pabbr} = {round(rr,2)} (95% CI: {round(ci_low_rr,2)}–{round(ci_high_rr,2)}). {round(rr,2)}× {direction}.")
                        draw_ci(pabbr, rr, ci_low_rr, ci_high_rr)

                        if design != "Cross-sectional":
                            st.subheader("Odds Ratio (OR)")
                            if ci_low_or <= 1 <= ci_high_or: st.warning(f"OR = {round(or_val,2)} — Not significant.")
                            else:
                                direction = "higher" if or_val > 1 else "lower"
                                st.success(f"OR = {round(or_val,2)} (95% CI: {round(ci_low_or,2)}–{round(ci_high_or,2)}). {round(or_val,2)}× {direction}.")
                            draw_ci("OR", or_val, ci_low_or, ci_high_or)

                        rr_or_explanation_expander(a, b, c, d, row_names, col_names,
                            rr, or_val, ci_low_rr, ci_high_rr, ci_low_or, ci_high_or,
                            is_cross_sectional=is_cs)
                else:
                    st.info("With 3+ categories or rate outcomes, chi-square is the appropriate test shown above.")

    else:  # Rate (person-time)
        st.subheader("Rate Outcome (Person-Time)")
        num_groups = 2 if exposure_type == "Binary (2 groups)" else st.number_input("Number of Groups", min_value=2, value=3)
        group_names = [st.text_input(f"Group {i+1}", key=f"grp_{i}") for i in range(int(num_groups))]
        cases_list = [st.number_input(f"Cases ({group_names[i]})", min_value=0, key=f"cases_{i}") for i in range(int(num_groups))]
        pt_list = [st.number_input(f"Person-Time ({group_names[i]})", min_value=0.1, value=1000.0, key=f"pt_{i}") for i in range(int(num_groups))]

        if st.button("Run Rate Analysis", key="run_rate_moa"):
            if num_groups == 2 and all(c > 0 for c in cases_list):
                r1 = cases_list[0]/pt_list[0]; r2 = cases_list[1]/pt_list[1]
                irr = r1/r2
                se_log_irr = math.sqrt(1/cases_list[0] + 1/cases_list[1])
                ci_low_irr = math.exp(math.log(irr)-1.96*se_log_irr)
                ci_high_irr = math.exp(math.log(irr)+1.96*se_log_irr)
                col1,col2,col3 = st.columns(3)
                col1.metric(f"Rate ({group_names[0]})", f"{round(r1*100000,1)}/100k person-time")
                col2.metric(f"Rate ({group_names[1]})", f"{round(r2*100000,1)}/100k person-time")
                col3.metric("IRR", round(irr,3))
                st.write(f"95% CI: ({round(ci_low_irr,3)}, {round(ci_high_irr,3)})")
                if ci_low_irr <= 1 <= ci_high_irr: st.warning("CI includes 1. Not significant.")
                else:
                    direction = "higher" if irr > 1 else "lower"
                    st.success(f"IRR = {round(irr,2)} — Rate in {group_names[0]} is {round(irr,2)}× {direction}.")
                draw_ci("IRR", irr, ci_low_irr, ci_high_irr)

    st.markdown("---")
    st.markdown("*Strong epidemiologists think structurally before computing.*")


# ==================================================
# MODULE 3: ADVANCED EPI MEASURES (existing tab2)
# ==================================================
elif current_page == "advanced_measures":
    st.title("📉 Advanced Epi Measures")

    col_t2, col_r2 = st.columns([5,1])
    with col_r2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reset", key="reset_adv"):
            for k in ["adv_measure_select"]:
                if k in st.session_state: del st.session_state[k]
            st.rerun()

    measure = st.selectbox("Select a measure:", [
        "Population Attributable Risk (PAR)",
        "Standardized Mortality Ratio (SMR)",
        "Attributable Risk & AR%",
        "Number Needed to Harm / Treat (NNH/NNT)",
        "Hazard Ratio (HR)"
    ], key="adv_measure_select")

    st.divider()

    if measure == "Population Attributable Risk (PAR)":
        st.subheader("Population Attributable Risk (PAR)")
        st.info("PAR estimates the proportion of disease in the **total population** attributable to a specific exposure.")
        data_mode = st.radio("Data entry", ["Use preset scenario","Enter my own data"], horizontal=True)
        if data_mode == "Use preset scenario":
            scenario = st.selectbox("Scenario", ["Smoking & Lung Cancer","Physical Inactivity & T2D","Obesity & CVD"])
            if scenario == "Smoking & Lung Cancer": Pe, RR = 0.14, 15.0
            elif scenario == "Physical Inactivity & T2D": Pe, RR = 0.46, 1.5
            else: Pe, RR = 0.42, 2.0
        else:
            Pe = st.number_input("Exposure prevalence (Pe)", min_value=0.001, max_value=0.999, value=0.30, step=0.01)
            RR = st.number_input("Risk Ratio (RR)", min_value=0.01, value=2.0, step=0.1)
        if st.button("Calculate PAR"):
            PAR_pct = (Pe * (RR - 1)) / (1 + Pe * (RR - 1)) * 100
            col1,col2,col3 = st.columns(3)
            col1.metric("Pe", f"{round(Pe*100,1)}%"); col2.metric("RR", round(RR,2)); col3.metric("PAR%", f"{round(PAR_pct,1)}%")
            st.success(f"{round(PAR_pct,1)}% of all cases in the population are attributable to this exposure.")

    elif measure == "Standardized Mortality Ratio (SMR)":
        st.subheader("Standardized Mortality Ratio (SMR)")
        st.info("SMR = Observed Deaths / Expected Deaths. Compares a study group to a reference population.")
        data_mode = st.radio("Data entry", ["Use preset scenario","Enter my own data"], horizontal=True, key="smr_mode")
        if data_mode == "Use preset scenario":
            scenario = st.selectbox("Scenario", ["Coal Miners & Respiratory Disease","Nuclear Workers & All-Cause Mortality","Firefighters & Cancer"], key="smr_scenario")
            if scenario == "Coal Miners & Respiratory Disease":
                age_groups=["20–34","35–44","45–54","55–64","65–74"]; observed=[2,8,22,41,35]
                ref_rates=[0.0003,0.0010,0.0038,0.0092,0.0198]; pop_sizes=[1200,1800,2100,1600,900]
            elif scenario == "Nuclear Workers & All-Cause Mortality":
                age_groups=["20–34","35–44","45–54","55–64","65–74"]; observed=[1,4,7,11,9]
                ref_rates=[0.0008,0.0018,0.0045,0.0110,0.0240]; pop_sizes=[2000,2500,1800,1200,600]
            else:
                age_groups=["20–34","35–44","45–54","55–64","65–74"]; observed=[1,6,19,38,31]
                ref_rates=[0.0001,0.0006,0.0024,0.0068,0.0160]; pop_sizes=[1500,2000,1900,1400,800]
            smr_df = pd.DataFrame({"Age Group":age_groups,"Pop Size":pop_sizes,"Observed":observed,
                "Ref Rate":ref_rates,"Expected":[round(pop_sizes[i]*ref_rates[i],2) for i in range(5)]})
            st.table(smr_df)
            total_observed = sum(observed); total_expected = sum([pop_sizes[i]*ref_rates[i] for i in range(5)])
        else:
            n_groups = st.number_input("Number of age groups", min_value=1, max_value=10, value=3)
            observed=[]; expected_list=[]
            for i in range(n_groups):
                c1,c2 = st.columns(2)
                with c1: obs = st.number_input(f"Observed {i+1}", min_value=0, key=f"smr_obs_{i}")
                with c2: exp = st.number_input(f"Expected {i+1}", min_value=0.0, step=0.1, key=f"smr_exp_{i}")
                observed.append(obs); expected_list.append(exp)
            total_observed = sum(observed); total_expected = sum(expected_list)
        if st.button("Calculate SMR"):
            if total_expected > 0:
                smr = total_observed / total_expected
                ci_low_s = max(0, smr - 1.96*(smr/math.sqrt(total_observed))) if total_observed > 0 else 0
                ci_high_s = smr + 1.96*(smr/math.sqrt(total_observed)) if total_observed > 0 else 0
                col1,col2,col3 = st.columns(3)
                col1.metric("Observed", int(total_observed)); col2.metric("Expected", round(total_expected,2)); col3.metric("SMR", round(smr,3))
                st.write(f"95% CI: ({round(ci_low_s,3)}, {round(ci_high_s,3)})")
                if ci_low_s <= 1 <= ci_high_s: st.warning("CI includes 1 — not significantly different from reference.")
                elif smr > 1: st.error(f"SMR = {round(smr,2)} — Excess mortality vs. reference population.")
                else: st.success(f"SMR = {round(smr,2)} — Lower mortality. May reflect healthy worker effect.")
                draw_ci("SMR", smr, ci_low_s, ci_high_s)

    elif measure == "Attributable Risk & AR%":
        st.subheader("Attributable Risk (AR) & AR%")
        data_mode = st.radio("Data entry", ["Use preset scenario","Enter my own data"], horizontal=True, key="ar_mode")
        if data_mode == "Use preset scenario":
            scenario = st.selectbox("Scenario", ["Hypertension & CVD","Unvaccinated & Measles","High Sodium & Stroke"], key="ar_scenario")
            if scenario == "Hypertension & CVD": r_exposed, r_unexposed = 0.12, 0.04
            elif scenario == "Unvaccinated & Measles": r_exposed, r_unexposed = 0.90, 0.02
            else: r_exposed, r_unexposed = 0.08, 0.03
        else:
            r_exposed = st.number_input("Risk in exposed", min_value=0.001, max_value=1.0, value=0.12, step=0.01)
            r_unexposed = st.number_input("Risk in unexposed", min_value=0.001, max_value=1.0, value=0.04, step=0.01)
        if st.button("Calculate AR & AR%"):
            ar = r_exposed - r_unexposed; ar_pct = (ar / r_exposed) * 100
            col1,col2,col3,col4 = st.columns(4)
            col1.metric("Risk (Exposed)", f"{round(r_exposed*100,1)}%"); col2.metric("Risk (Unexposed)", f"{round(r_unexposed*100,1)}%")
            col3.metric("AR", f"{round(ar*100,1)}%"); col4.metric("AR%", f"{round(ar_pct,1)}%")
            st.success(f"AR = {round(ar*100,1)}%: absolute excess risk per 100 exposed people.")
            st.success(f"AR% = {round(ar_pct,1)}%: fraction of disease in the exposed group attributable to the exposure.")

    elif measure == "Number Needed to Harm / Treat (NNH/NNT)":
        st.subheader("NNH / NNT")
        data_mode = st.radio("Data entry", ["Use preset scenario","Enter my own data"], horizontal=True, key="nnt_mode")
        if data_mode == "Use preset scenario":
            scenario = st.selectbox("Scenario", ["Statins & Cardiac Events (NNT)","Aspirin & GI Bleeding (NNH)","Smoking Cessation (NNT)"], key="nnt_scenario")
            if scenario == "Statins & Cardiac Events (NNT)": r_treatment,r_control,label_treatment,label_control = 0.04,0.06,"Statin","Placebo"
            elif scenario == "Aspirin & GI Bleeding (NNH)": r_treatment,r_control,label_treatment,label_control = 0.025,0.010,"Daily aspirin","No aspirin"
            else: r_treatment,r_control,label_treatment,label_control = 0.22,0.08,"Cessation program","No program"
        else:
            label_treatment = st.text_input("Treatment group", "Treatment")
            label_control = st.text_input("Control group", "Control")
            r_treatment = st.number_input(f"Risk ({label_treatment})", min_value=0.001, max_value=1.0, value=0.04, step=0.01)
            r_control = st.number_input(f"Risk ({label_control})", min_value=0.001, max_value=1.0, value=0.06, step=0.01)
        if st.button("Calculate NNT/NNH"):
            risk_diff = abs(r_treatment - r_control)
            if risk_diff > 0:
                nnt = round(1/risk_diff, 1)
                col1,col2,col3 = st.columns(3)
                col1.metric(f"Risk ({label_treatment})", f"{round(r_treatment*100,1)}%")
                col2.metric(f"Risk ({label_control})", f"{round(r_control*100,1)}%")
                col3.metric("Risk Difference", f"{round(risk_diff*100,1)}%")
                if r_treatment < r_control: st.success(f"NNT = {nnt}: treat {nnt} people to prevent one additional outcome.")
                else: st.error(f"NNH = {nnt}: {nnt} people exposed before one additional harm expected.")

    elif measure == "Hazard Ratio (HR)":
        st.subheader("Hazard Ratio (HR)")
        st.info("HR compares the instantaneous event rate over time. Output of Cox proportional hazards regression.")
        data_mode = st.radio("Data entry", ["Use preset scenario","Enter my own data"], horizontal=True, key="hr_mode")
        if data_mode == "Use preset scenario":
            scenario = st.selectbox("Scenario", ["Statins & Time to MI","HIV & Time to AIDS","Physical Activity & Dementia"], key="hr_scenario")
            if scenario == "Statins & Time to MI": hr,ci_low_hr,ci_high_hr,exposed_label,outcome_label = 0.68,0.54,0.85,"Statin therapy","first MI"
            elif scenario == "HIV & Time to AIDS": hr,ci_low_hr,ci_high_hr,exposed_label,outcome_label = 2.31,1.74,3.07,"CD4 < 200","AIDS-defining illness"
            else: hr,ci_low_hr,ci_high_hr,exposed_label,outcome_label = 0.72,0.58,0.89,"High physical activity","dementia"
            col1,col2,col3 = st.columns(3)
            col1.metric("HR", round(hr,2)); col2.metric("CI Lower", round(ci_low_hr,2)); col3.metric("CI Upper", round(ci_high_hr,2))
            if ci_low_hr <= 1 <= ci_high_hr: st.warning(f"HR = {round(hr,2)} — CI includes 1. Not significant.")
            elif hr < 1: st.success(f"HR = {round(hr,2)}: {exposed_label} had {round((1-hr)*100,1)}% lower hazard of {outcome_label}. Significant.")
            else: st.error(f"HR = {round(hr,2)}: {exposed_label} had {round((hr-1)*100,1)}% higher hazard of {outcome_label}. Significant.")
            draw_ci("HR", hr, ci_low_hr, ci_high_hr)
        else:
            hr = st.number_input("HR", min_value=0.01, value=0.68, step=0.01)
            ci_low_hr = st.number_input("CI Lower", min_value=0.001, value=0.54, step=0.01)
            ci_high_hr = st.number_input("CI Upper", min_value=0.001, value=0.85, step=0.01)
            exposed_label = st.text_input("Exposed group", "Exposed")
            outcome_label = st.text_input("Outcome", "the outcome")
            if st.button("Interpret HR"):
                if ci_low_hr <= 1 <= ci_high_hr: st.warning(f"HR = {round(hr,2)} — CI includes 1. Not significant.")
                elif hr < 1: st.success(f"HR = {round(hr,2)}: {round((1-hr)*100,1)}% lower hazard. Significant.")
                else: st.error(f"HR = {round(hr,2)}: {round((hr-1)*100,1)}% higher hazard. Significant.")
                draw_ci("HR", hr, ci_low_hr, ci_high_hr)

    st.markdown("---")
    st.markdown("*Strong epidemiologists think structurally before computing.*")


# ==================================================
# MODULE 3: STANDARDIZATION (existing tab3)
# ==================================================
elif current_page == "standardization":
    st.title("📏 Standardization")
    col_t3, col_r3 = st.columns([5,1])
    with col_r3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reset", key="reset_std"):
            for k in ["std_preset_choice"]:
                if k in st.session_state: del st.session_state[k]
            st.rerun()
    st.markdown("**Standardization** allows fair comparison of rates between populations with different age structures.")
    STD_PRESETS = {
        "None — I'll enter my own data": None,
        "Urban vs. Rural CVD Mortality": {
            "description":"**Scenario:** Compare CVD mortality between an urban (younger) and rural (older) county. *Adapted from CDC WONDER.*",
            "age_groups":["0–44","45–54","55–64","65–74","75+"],"std_pop":[150000,40000,35000,25000,15000],
            "pop_a":[80000,15000,12000,8000,4000],"deaths_a":[12,45,120,280,310],
            "pop_b":[30000,18000,22000,20000,14000],"deaths_b":[5,55,145,430,580],
            "label_a":"Urban County","label_b":"Rural County","outcome":"CVD deaths","ref_label":"State population"
        },
        "Coal Miners vs. Office Workers (Lung Disease)": {
            "description":"**Scenario:** Compare lung disease mortality between coal miners and office workers. *Adapted from NIOSH.*",
            "age_groups":["20–34","35–44","45–54","55–64","65–74"],"std_pop":[5000,6000,5500,4000,2000],
            "pop_a":[800,1800,2100,1600,900],"deaths_a":[1,6,18,38,32],
            "pop_b":[2000,2200,1800,1200,600],"deaths_b":[0,2,5,10,8],
            "label_a":"Coal Miners","label_b":"Office Workers","outcome":"lung disease deaths","ref_label":"Workforce population"
        },
        "State A vs. State B: Diabetes Mortality": {
            "description":"**Scenario:** Two states with different age distributions compared on diabetes mortality rates.",
            "age_groups":["0–44","45–54","55–64","65–74","75+"],"std_pop":[200000,55000,48000,35000,22000],
            "pop_a":[420000,80000,65000,42000,18000],"deaths_a":[8,62,180,390,420],
            "pop_b":[180000,55000,62000,58000,45000],"deaths_b":[4,48,198,620,890],
            "label_a":"State A","label_b":"State B","outcome":"diabetes deaths","ref_label":"National population"
        },
    }
    std_preset_choice = st.selectbox("Load a preset:", list(STD_PRESETS.keys()), key="std_preset_choice")
    std_preset = STD_PRESETS[std_preset_choice]
    if std_preset: st.info(std_preset["description"])
    st.divider()

    if std_preset:
        age_groups=std_preset["age_groups"]; std_pop=std_preset["std_pop"]
        pop_a=std_preset["pop_a"]; deaths_a=std_preset["deaths_a"]
        pop_b=std_preset["pop_b"]; deaths_b=std_preset["deaths_b"]
        label_a=std_preset["label_a"]; label_b=std_preset["label_b"]
        outcome_lbl=std_preset["outcome"]; ref_label=std_preset["ref_label"]; n_groups=len(age_groups)
    else:
        col1,col2 = st.columns(2)
        with col1: label_a=st.text_input("Population A","Population A"); label_b=st.text_input("Population B","Population B")
        with col2: ref_label=st.text_input("Reference population","Standard Population"); outcome_lbl=st.text_input("Outcome","deaths")
        n_groups = st.number_input("Number of age groups", min_value=2, max_value=10, value=5)
        age_groups,std_pop,pop_a,deaths_a,pop_b,deaths_b = [],[],[],[],[],[]
        for i in range(int(n_groups)):
            cols = st.columns([2,2,2,2,2,2])
            age_groups.append(cols[0].text_input("",f"Group {i+1}",key=f"ag_{i}",label_visibility="collapsed"))
            std_pop.append(cols[1].number_input("",min_value=1,value=10000,key=f"sp_{i}",label_visibility="collapsed"))
            pop_a.append(cols[2].number_input("",min_value=1,value=1000,key=f"pa_{i}",label_visibility="collapsed"))
            deaths_a.append(cols[3].number_input("",min_value=0,value=0,key=f"da_{i}",label_visibility="collapsed"))
            pop_b.append(cols[4].number_input("",min_value=1,value=1000,key=f"pb_{i}",label_visibility="collapsed"))
            deaths_b.append(cols[5].number_input("",min_value=0,value=0,key=f"db_{i}",label_visibility="collapsed"))

    if st.button("Run Standardization Analysis"):
        if sum(pop_a) == 0 or sum(pop_b) == 0:
            st.warning("Population sizes cannot be zero.")
        else:
            rate_a = [deaths_a[i]/max(pop_a[i],1)*100000 for i in range(n_groups)]
            rate_b = [deaths_b[i]/max(pop_b[i],1)*100000 for i in range(n_groups)]
            ref_rate = [(deaths_a[i]+deaths_b[i])/max(pop_a[i]+pop_b[i],1)*100000 for i in range(n_groups)]
            expected_a_direct = [rate_a[i]/100000*std_pop[i] for i in range(n_groups)]
            expected_b_direct = [rate_b[i]/100000*std_pop[i] for i in range(n_groups)]
            age_adj_rate_a = sum(expected_a_direct)/sum(std_pop)*100000
            age_adj_rate_b = sum(expected_b_direct)/sum(std_pop)*100000
            expected_a_indirect = [ref_rate[i]/100000*pop_a[i] for i in range(n_groups)]
            expected_b_indirect = [ref_rate[i]/100000*pop_b[i] for i in range(n_groups)]
            total_obs_a = sum(deaths_a); total_obs_b = sum(deaths_b)
            total_exp_a = sum(expected_a_indirect); total_exp_b = sum(expected_b_indirect)
            smr_a = round(total_obs_a/total_exp_a, 3) if total_exp_a > 0 else None
            smr_b = round(total_obs_b/total_exp_b, 3) if total_exp_b > 0 else None
            crude_rate_a = sum(deaths_a)/sum(pop_a)*100000
            crude_rate_b = sum(deaths_b)/sum(pop_b)*100000
            crude_higher = label_a if crude_rate_a > crude_rate_b else label_b
            adj_higher = label_a if age_adj_rate_a > age_adj_rate_b else label_b

            st.subheader("📊 Results")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### {label_a}")
                st.metric("Crude Rate (per 100,000)", round(crude_rate_a, 1))
                st.metric("Age-Adjusted Rate — Direct", round(age_adj_rate_a, 1),
                          delta=f"{round(age_adj_rate_a - crude_rate_a, 1)} vs. crude", delta_color="off")
                if smr_a: st.metric("SMR — Indirect", smr_a)
            with col2:
                st.markdown(f"### {label_b}")
                st.metric("Crude Rate (per 100,000)", round(crude_rate_b, 1))
                st.metric("Age-Adjusted Rate — Direct", round(age_adj_rate_b, 1),
                          delta=f"{round(age_adj_rate_b - crude_rate_b, 1)} vs. crude", delta_color="off")
                if smr_b: st.metric("SMR — Indirect", smr_b)

            st.divider()
            if crude_higher != adj_higher:
                st.error(f"⚠️ **Confounding by age detected!** Crude rates suggest {crude_higher} has higher mortality, but after adjustment {adj_higher} has the higher rate.")
            else:
                crude_diff = abs(crude_rate_a - crude_rate_b)
                adj_diff = abs(age_adj_rate_a - age_adj_rate_b)
                if adj_diff < crude_diff * 0.7:
                    st.warning(f"⚠️ **Age partially explains the difference.** Crude gap: {round(crude_diff,1)}, adjusted gap: {round(adj_diff,1)} per 100,000.")
                else:
                    st.success("✅ Age structure had minimal impact. Crude and age-adjusted rates tell a similar story.")

            st.divider()
            st.subheader("🔍 Interpretation")
            st.markdown(f"""
**Crude rates** compare raw death counts relative to total population. Can mislead when populations have different age distributions.

**Direct age-adjusted rates** answer: *what would the rate look like if both populations had the same age structure?* The standard population ({ref_label}) is the common reference.

**SMR** applies the reference population's rates to each group's age structure to calculate expected deaths. SMR = Observed ÷ Expected.
- SMR = 1.0: mortality matches reference
- SMR > 1.0: excess mortality
- SMR < 1.0: lower mortality (possibly healthy worker effect)

**{label_a}:** Observed {int(total_obs_a)} deaths, Expected {round(total_exp_a, 1)} → SMR = {smr_a}
**{label_b}:** Observed {int(total_obs_b)} deaths, Expected {round(total_exp_b, 1)} → SMR = {smr_b}
            """)

            with st.expander("🔢 Show me the math — Direct Standardization"):
                st.markdown("**Step 1: Age-specific rates per 100,000**")
                rate_df = pd.DataFrame({"Age Group": age_groups, f"{label_a} Rate": [round(r, 1) for r in rate_a], f"{label_b} Rate": [round(r, 1) for r in rate_b], "Reference Rate": [round(r, 1) for r in ref_rate]})
                st.table(rate_df)
                st.markdown("**Step 2: Apply rates to standard population**")
                exp_df = pd.DataFrame({"Age Group": age_groups, "Std Pop": std_pop, f"Expected ({label_a})": [round(e, 1) for e in expected_a_direct], f"Expected ({label_b})": [round(e, 1) for e in expected_b_direct]})
                st.table(exp_df)
                st.markdown(f"""
**Step 3:** Sum expected ÷ total std pop × 100,000

{label_a}: {round(sum(expected_a_direct),1)} ÷ {sum(std_pop):,} × 100,000 = **{round(age_adj_rate_a,1)} per 100,000**
{label_b}: {round(sum(expected_b_direct),1)} ÷ {sum(std_pop):,} × 100,000 = **{round(age_adj_rate_b,1)} per 100,000**
                """)

    st.markdown("---")
    st.markdown("*Strong epidemiologists think structurally before computing.*")


# ==================================================
# MODULE 3: HYPOTHESIS TESTING + POWER (existing tab6 + new power section)
# ==================================================
elif current_page == "hypothesis_testing":
    st.title("🧪 Hypothesis Testing & Power")
    col_t6, col_r6 = st.columns([5,1])
    with col_r6:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reset", key="reset_ht"):
            for k in list(st.session_state.keys()):
                if any(k.startswith(p) for p in ["h0_","h1_","tails_","ht_section","chi2_slider","dof_select","tail_radio"]):
                    del st.session_state[k]
            st.rerun()
    st.markdown("Build your understanding of hypothesis testing, interpreting p-values, and statistical power.")

    ht_section = st.radio("Choose a section:", [
        "1️⃣ One vs. Two Tailed Tests",
        "2️⃣ What Does Rejecting the Null Actually Mean?",
        "3️⃣ Hypothesis Builder",
        "4️⃣ Statistical Power & Sample Size"
    ], horizontal=True, key="ht_section")
    st.divider()

    if ht_section == "1️⃣ One vs. Two Tailed Tests":
        st.subheader("One-Tailed vs. Two-Tailed Tests")
        st.markdown("""
**Two-tailed (default):** You're testing whether any difference exists, regardless of direction. Your 5% tolerance for Type I error is split — 2.5% in each tail.

**One-tailed:** You've predicted a specific direction before collecting data. All 5% goes in one tail, making the test more sensitive in that direction but unable to detect effects in the other.

**In practice:** Use two-tailed as the default. One-tailed requires a strong, pre-specified directional hypothesis. Using one-tailed post-hoc (after seeing the data) inflates Type I error.

**Chi-square tests are always two-tailed** — the chi-square statistic is always positive (it measures squared deviations), so there's no "direction" concept. The p-value from chi-square is always two-tailed.
        """)

    elif ht_section == "2️⃣ What Does Rejecting the Null Actually Mean?":
        st.subheader("What Does Rejecting the Null Actually Mean?")
        with st.expander("🔵 What the p-value IS", expanded=True):
            st.markdown("""
**The p-value is the probability of observing a result as extreme as yours (or more extreme) if the null hypothesis were true.**

Small p (e.g., 0.003): data this extreme would occur only 0.3% of the time under H₀ — very surprising.
Large p (e.g., 0.42): data this extreme would occur 42% of the time under H₀ — not surprising.

The 0.05 threshold means we accept a 5% chance of rejecting a true H₀ (Type I error / false positive).
            """)
        with st.expander("🔴 What the p-value is NOT"):
            st.markdown("""
| ❌ Common Misconception | ✅ What's Actually True |
|---|---|
| "p = 0.03 means 3% chance H₀ is true" | p-value says nothing about the probability H₀ is true |
| "p = 0.06 means no association" | Failing to reject H₀ does not prove no effect |
| "p < 0.05 means the result is important" | Statistical significance ≠ practical significance |
| "We accept the null hypothesis" | You never *accept* H₀ — you fail to reject it |
| "p = 0.049 is meaningful, p = 0.051 is not" | The 0.05 cutoff is arbitrary |
| "Smaller p = stronger association" | p reflects both sample size AND effect size |
            """)
        with st.expander("🟡 Type I and Type II Errors", expanded=True):

            # Visual 2x2 grid
            visual_html = """
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:8px 0 20px 0;">

  <!-- Column headers -->
  <div style="display:grid;grid-template-columns:180px 1fr 1fr;gap:4px;margin-bottom:4px;">
    <div></div>
    <div style="text-align:center;background:#1e40af;color:white;border-radius:8px 8px 0 0;
         padding:10px;font-weight:700;font-size:13px;">
      🌍 Reality: H₀ is TRUE<br>
      <span style="font-weight:400;font-size:11px;opacity:0.85;">(no real effect exists)</span>
    </div>
    <div style="text-align:center;background:#166534;color:white;border-radius:8px 8px 0 0;
         padding:10px;font-weight:700;font-size:13px;">
      🌍 Reality: H₀ is FALSE<br>
      <span style="font-weight:400;font-size:11px;opacity:0.85;">(a real effect exists)</span>
    </div>
  </div>

  <!-- Row 1: Reject H₀ -->
  <div style="display:grid;grid-template-columns:180px 1fr 1fr;gap:4px;margin-bottom:4px;">
    <div style="background:#374151;color:white;border-radius:8px 0 0 0;
         padding:12px;font-weight:700;font-size:13px;display:flex;align-items:center;">
      🔬 Your decision:<br>Reject H₀<br>
      <span style="font-weight:400;font-size:10px;opacity:0.8;">("significant result")</span>
    </div>
    <div style="background:#fef2f2;border:3px solid #dc2626;border-radius:0 8px 0 0;padding:16px;text-align:center;">
      <div style="font-size:28px;margin-bottom:6px;">🚨</div>
      <div style="font-weight:800;font-size:15px;color:#dc2626;">TYPE I ERROR</div>
      <div style="font-weight:600;font-size:12px;color:#dc2626;margin:4px 0;">False Positive (α)</div>
      <div style="font-size:12px;color:#7f1d1d;margin-top:8px;line-height:1.5;">
        You declared an effect that <b>doesn't exist</b>.<br>
        You were fooled by chance.
      </div>
      <div style="background:#dc2626;color:white;border-radius:6px;padding:6px 10px;
           font-size:11px;margin-top:10px;font-weight:600;">
        Probability = α (usually 0.05)
      </div>
      <div style="font-size:11px;color:#991b1b;margin-top:8px;font-style:italic;">
        "A fire alarm goes off — but there's no fire."
      </div>
    </div>
    <div style="background:#f0fdf4;border:3px solid #16a34a;border-radius:0;padding:16px;text-align:center;">
      <div style="font-size:28px;margin-bottom:6px;">✅</div>
      <div style="font-weight:800;font-size:15px;color:#166534;">CORRECT DECISION</div>
      <div style="font-weight:600;font-size:12px;color:#166534;margin:4px 0;">True Positive (Power = 1−β)</div>
      <div style="font-size:12px;color:#14532d;margin-top:8px;line-height:1.5;">
        You correctly detected a real effect.<br>
        This is what studies aim for.
      </div>
      <div style="background:#16a34a;color:white;border-radius:6px;padding:6px 10px;
           font-size:11px;margin-top:10px;font-weight:600;">
        Probability = Power (1−β)
      </div>
      <div style="font-size:11px;color:#166534;margin-top:8px;font-style:italic;">
        "Fire alarm goes off — and there IS a fire."
      </div>
    </div>
  </div>

  <!-- Row 2: Fail to reject H₀ -->
  <div style="display:grid;grid-template-columns:180px 1fr 1fr;gap:4px;">
    <div style="background:#374151;color:white;border-radius:0 0 0 8px;
         padding:12px;font-weight:700;font-size:13px;display:flex;align-items:center;">
      🔬 Your decision:<br>Fail to reject H₀<br>
      <span style="font-weight:400;font-size:10px;opacity:0.8;">("not significant")</span>
    </div>
    <div style="background:#f0fdf4;border:3px solid #16a34a;border-radius:0;padding:16px;text-align:center;">
      <div style="font-size:28px;margin-bottom:6px;">✅</div>
      <div style="font-weight:800;font-size:15px;color:#166534;">CORRECT DECISION</div>
      <div style="font-weight:600;font-size:12px;color:#166534;margin:4px 0;">True Negative</div>
      <div style="font-size:12px;color:#14532d;margin-top:8px;line-height:1.5;">
        No effect exists, and you didn't find one.<br>
        The null was correct — and you kept it.
      </div>
      <div style="background:#16a34a;color:white;border-radius:6px;padding:6px 10px;
           font-size:11px;margin-top:10px;font-weight:600;">
        Probability = 1−α (specificity)
      </div>
      <div style="font-size:11px;color:#166534;margin-top:8px;font-style:italic;">
        "No alarm — and there's no fire."
      </div>
    </div>
    <div style="background:#fffbeb;border:3px solid #d97706;border-radius:0 0 8px 0;padding:16px;text-align:center;">
      <div style="font-size:28px;margin-bottom:6px;">😴</div>
      <div style="font-weight:800;font-size:15px;color:#92400e;">TYPE II ERROR</div>
      <div style="font-weight:600;font-size:12px;color:#92400e;margin:4px 0;">False Negative (β)</div>
      <div style="font-size:12px;color:#78350f;margin-top:8px;line-height:1.5;">
        A real effect exists, but you <b>missed it</b>.<br>
        Study was underpowered or effect too small.
      </div>
      <div style="background:#d97706;color:white;border-radius:6px;padding:6px 10px;
           font-size:11px;margin-top:10px;font-weight:600;">
        Probability = β (usually target ≤ 0.20)
      </div>
      <div style="font-size:11px;color:#92400e;margin-top:8px;font-style:italic;">
        "No alarm — but there IS a fire."
      </div>
    </div>
  </div>

</div>

<!-- Key takeaways row -->
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px;">
  <div style="background:#fef2f2;border-left:4px solid #dc2626;border-radius:6px;padding:12px 14px;">
    <div style="font-weight:700;color:#dc2626;margin-bottom:6px;">🚨 Type I Error (α = false positive)</div>
    <div style="font-size:12px;color:#7f1d1d;line-height:1.6;">
      • Claiming an effect that doesn't exist<br>
      • Probability controlled by setting α = 0.05<br>
      • Reduced by: lower α threshold, replication<br>
      • <b>Analogy:</b> Convicting an innocent person<br>
      • Common in: underpowered studies fishing for p &lt; 0.05, multiple comparisons
    </div>
  </div>
  <div style="background:#fffbeb;border-left:4px solid #d97706;border-radius:6px;padding:12px 14px;">
    <div style="font-weight:700;color:#92400e;margin-bottom:6px;">😴 Type II Error (β = false negative)</div>
    <div style="font-size:12px;color:#78350f;line-height:1.6;">
      • Missing a real effect (insufficient power)<br>
      • Probability = β; Power = 1 − β<br>
      • Reduced by: larger sample size, larger effect size<br>
      • <b>Analogy:</b> Acquitting a guilty person<br>
      • Common in: small studies, weak exposures, noisy measurements
    </div>
  </div>
</div>

<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:6px;padding:12px 14px;margin-top:12px;font-size:12px;color:#1e40af;line-height:1.6;">
  <b>⚖️ The tradeoff:</b> You cannot simultaneously minimize both errors without increasing sample size.
  Reducing α (stricter threshold) → fewer Type I errors but <i>more</i> Type II errors (easier to miss real effects).
  The only way to reduce both is to collect more data or improve measurement precision.
</div>
"""
            st.markdown(visual_html, unsafe_allow_html=True)
        with st.expander("🟢 CI Connection"):
            st.markdown("""
**95% CI and p-value always agree:**
- CI includes 1 → p ≥ 0.05 → fail to reject H₀
- CI excludes 1 → p < 0.05 → reject H₀

The CI gives more information — it shows the range of plausible effect sizes, not just whether to reject H₀.
            """)

    elif ht_section == "3️⃣ Hypothesis Builder":
        st.subheader("Hypothesis Builder")
        st.markdown("For each scenario, select the correct null hypothesis, alternative hypothesis, and whether the test should be one- or two-tailed. Feedback appears immediately after each selection.")
        with st.expander("📖 Quick Reference: Null vs. Alternative Hypothesis"):
            st.markdown("""
**H₀ (Null):** No association, no difference. Always written as an equality (RR = 1, μ₁ = μ₂). What you're trying to find evidence against.

**H₁ (Alternative):** States an association or effect exists.
- **Two-tailed (≠):** You're not predicting direction — just that a difference exists. 5% split across both tails. Default.
- **One-tailed (< or >):** Predicting a specific direction before data collection. All 5% in one tail. Only use when strong prior evidence justifies direction.

**Key principle:** You never *prove* H₀ true. You either reject it (p < 0.05) or fail to reject it (p ≥ 0.05).

**Chi-square reminder:** Always two-tailed regardless of the research question.
            """)

        HYP_SCENARIOS = [
            {
                "id": "h1", "title": "Scenario A: Aerobic Exercise & Blood Pressure",
                "description": "A researcher tests whether a 12-week aerobic program reduces systolic BP in hypertensive adults. She expects it to **decrease** BP based on extensive prior research.",
                "null_options": ["The program has no effect on systolic BP (μ_before = μ_after)", "The program reduces BP (μ_before > μ_after)", "BP changes in either direction (μ_before ≠ μ_after)"],
                "alt_options": ["The program reduces systolic BP (μ_before > μ_after)", "The program has no effect", "BP changes in either direction (μ_before ≠ μ_after)"],
                "correct_null_idx": 0, "correct_alt_idx": 0, "correct_tails": "one-tailed",
                "null_feedback": "✅ Correct. H₀ states no effect — BP before = BP after.",
                "null_wrong_feedback": "❌ H₀ must state no effect — an equality. BP before = BP after.",
                "alt_feedback": "✅ Correct. Specific directional prediction (reduction) → one-tailed H₁.",
                "alt_wrong_feedback": "❌ The researcher has strong prior evidence for a decrease. That directional prediction makes H₁ one-tailed (μ_before > μ_after).",
                "tails_connection": "🎯 **One-tailed** — H₁ specifies a direction (reduction). All 5% of error tolerance is focused on detecting a decrease. More sensitive in that direction, but blind to BP increases.",
            },
            {
                "id": "h2", "title": "Scenario B: New Cholesterol Drug & Liver Enzymes",
                "description": "A pharmaceutical company tests whether a new cholesterol-lowering drug changes liver enzyme elevation rates vs. placebo. **No prior evidence** about whether it raises or lowers enzymes.",
                "null_options": ["Drug and placebo have the same enzyme elevation rate (p_drug = p_placebo)", "Drug increases enzyme elevation (p_drug > p_placebo)", "Drug changes enzyme levels in either direction (p_drug ≠ p_placebo)"],
                "alt_options": ["Drug changes enzyme elevation rate in either direction (p_drug ≠ p_placebo)", "Drug increases enzyme elevation (p_drug > p_placebo)", "Drug has no effect on enzymes"],
                "correct_null_idx": 0, "correct_alt_idx": 0, "correct_tails": "two-tailed",
                "null_feedback": "✅ Correct. H₀ states no difference — equal rates in both groups.",
                "null_wrong_feedback": "❌ H₀ must state no difference. Equal rates = equality.",
                "alt_feedback": "✅ Correct. No prior directional basis → two-tailed H₁ (≠).",
                "alt_wrong_feedback": "❌ No prior evidence about direction. Without a directional prediction established before data collection, H₁ must be two-tailed (≠).",
                "tails_connection": "🎯 **Two-tailed** — H₁ uses ≠. 5% split: 2.5% in each tail. Can detect an increase or a decrease in enzyme rates.",
            },
            {
                "id": "h3", "title": "Scenario C: Screen Time & Obesity (Chi-Square)",
                "description": "A cross-sectional survey tests whether obesity prevalence differs between high and low screen time groups using a chi-square test. No prior directional hypothesis.",
                "null_options": ["No association between screen time and obesity (PR = 1 / independent)", "High screen time is associated with higher obesity (PR > 1)", "Screen time and obesity are associated (PR ≠ 1)"],
                "alt_options": ["There is an association between screen time and obesity (PR ≠ 1)", "High screen time causes higher obesity (PR > 1)", "No association between screen time and obesity"],
                "correct_null_idx": 0, "correct_alt_idx": 0, "correct_tails": "two-tailed",
                "null_feedback": "✅ Correct. H₀: no association — independence.",
                "null_wrong_feedback": "❌ H₀ must state no association — independence between screen time and obesity.",
                "alt_feedback": "✅ Correct. No directional prediction, and chi-square is always two-tailed.",
                "alt_wrong_feedback": "❌ No directional prediction was made. Also, chi-square tests are **always two-tailed** — they measure total discrepancy without regard to direction.",
                "tails_connection": "🎯 **Two-tailed (chi-square)** — Chi-square tests are always two-tailed regardless of how the hypotheses are framed. The statistic measures squared deviations, so direction is irrelevant.",
            },
            {
                "id": "h4", "title": "Scenario D: Vaccine Efficacy Against Influenza",
                "description": "Epidemiologists test a new influenza vaccine in a randomized trial. Based on the known immunological mechanism and prior flu vaccine data, they **expect protection** (reduction in incidence).",
                "null_options": ["Vaccine and placebo groups have equal influenza incidence (IRR = 1)", "Vaccine reduces influenza incidence (IRR < 1)", "Vaccine changes incidence in either direction (IRR ≠ 1)"],
                "alt_options": ["Vaccine reduces influenza incidence (IRR < 1)", "Vaccine increases influenza incidence (IRR > 1)", "Vaccine changes incidence in either direction (IRR ≠ 1)"],
                "correct_null_idx": 0, "correct_alt_idx": 0, "correct_tails": "one-tailed",
                "null_feedback": "✅ Correct. H₀: no effect — equal incidence rates (IRR = 1).",
                "null_wrong_feedback": "❌ H₀ must be an equality — IRR = 1 (no difference in incidence between groups).",
                "alt_feedback": "✅ Correct. Known protective mechanism → directional prediction → one-tailed (IRR < 1).",
                "alt_wrong_feedback": "❌ The known mechanism and prior data support a directional prediction (protection). H₁ should specify IRR < 1 — one-tailed.",
                "tails_connection": "🎯 **One-tailed** — strong biological prior supports protection (IRR < 1). All 5% error tolerance goes toward detecting a reduction in incidence. If the vaccine somehow increased incidence, this test would not detect it.",
            },
            {
                "id": "h5", "title": "Scenario E: Sugar-Sweetened Beverages & Type 2 Diabetes",
                "description": "A prospective cohort study examines whether daily consumption of sugar-sweetened beverages (SSBs) is associated with incident Type 2 Diabetes over 10 years. The research question is open — the team wants to detect any association.",
                "null_options": ["No association between SSB consumption and T2D incidence (RR = 1)", "SSB consumption increases T2D incidence (RR > 1)", "SSB consumption is associated with T2D in either direction (RR ≠ 1)"],
                "alt_options": ["SSB consumption is associated with T2D incidence in either direction (RR ≠ 1)", "SSB consumption increases T2D incidence (RR > 1)", "SSB consumption has no association with T2D"],
                "correct_null_idx": 0, "correct_alt_idx": 0, "correct_tails": "two-tailed",
                "null_feedback": "✅ Correct. H₀: no association — RR = 1.",
                "null_wrong_feedback": "❌ H₀ must state no association. RR = 1 is the null value for a risk ratio.",
                "alt_feedback": "✅ Correct. Open research question with no pre-specified direction → two-tailed.",
                "alt_wrong_feedback": "❌ Although biologically we might suspect SSBs increase T2D risk, the research question is framed openly ('any association'). Without a pre-specified directional hypothesis, use two-tailed (RR ≠ 1).",
                "tails_connection": "🎯 **Two-tailed** — the team wants to detect any association. Even when one direction seems likely, if it wasn't formally pre-specified, the default is two-tailed to avoid inflating Type I error.",
            },
            {
                "id": "h6", "title": "Scenario F: Night Shift Work & Breast Cancer",
                "description": "A case-control study uses an odds ratio to test whether night shift work is associated with breast cancer in female nurses. The investigators have no strong prior evidence about direction.",
                "null_options": ["Night shift work and breast cancer are not associated (OR = 1)", "Night shift work increases breast cancer odds (OR > 1)", "Night shift work is associated with breast cancer in either direction (OR ≠ 1)"],
                "alt_options": ["Night shift work is associated with breast cancer in either direction (OR ≠ 1)", "Night shift work increases breast cancer odds (OR > 1)", "Night shift work decreases breast cancer odds (OR < 1)"],
                "correct_null_idx": 0, "correct_alt_idx": 0, "correct_tails": "two-tailed",
                "null_feedback": "✅ Correct. H₀: no association — OR = 1.",
                "null_wrong_feedback": "❌ H₀ must state no association. The null value for an OR is 1.",
                "alt_feedback": "✅ Correct. No strong prior directional evidence → two-tailed (OR ≠ 1).",
                "alt_wrong_feedback": "❌ Without strong pre-existing directional evidence, H₁ should be two-tailed (OR ≠ 1). The association could theoretically go in either direction.",
                "tails_connection": "🎯 **Two-tailed** — no prior directional hypothesis. OR ≠ 1 detects any association regardless of direction.",
            },
            {
                "id": "h7", "title": "Scenario G: Smoking Cessation Program & Quit Rates",
                "description": "A public health team evaluates a new smoking cessation app vs. standard counseling. They believe the app will **improve** quit rates based on a pilot study. They pre-register their directional hypothesis before the trial begins.",
                "null_options": ["App and counseling have equal 6-month quit rates (p_app = p_counseling)", "App has higher quit rates than counseling (p_app > p_counseling)", "Quit rates differ between groups in either direction (p_app ≠ p_counseling)"],
                "alt_options": ["App has higher quit rates than standard counseling (p_app > p_counseling)", "App has lower quit rates than standard counseling (p_app < p_counseling)", "Quit rates differ in either direction (p_app ≠ p_counseling)"],
                "correct_null_idx": 0, "correct_alt_idx": 0, "correct_tails": "one-tailed",
                "null_feedback": "✅ Correct. H₀: equal quit rates — no difference.",
                "null_wrong_feedback": "❌ H₀ must state equality — equal quit rates in both groups.",
                "alt_feedback": "✅ Correct. Pre-registered directional hypothesis (app improves rates) → one-tailed.",
                "alt_wrong_feedback": "❌ The team pre-registered a directional hypothesis (app > counseling) based on pilot data. Pre-registration of direction is what justifies a one-tailed test.",
                "tails_connection": "🎯 **One-tailed** — the directional hypothesis was pre-registered before data collection, which is what legitimizes a one-tailed test. Using one-tailed after seeing the data would be p-hacking.",
            },
            {
                "id": "h8", "title": "Scenario H: Lead Exposure & IQ in Children",
                "description": "A cohort study examines whether prenatal lead exposure affects IQ scores at age 7. The team expects lead to **reduce** IQ based on established neurotoxicology literature.",
                "null_options": ["Prenatal lead exposure has no effect on IQ at age 7 (μ_exposed = μ_unexposed)", "Lead exposure reduces IQ (μ_exposed < μ_unexposed)", "Lead exposure affects IQ in either direction (μ_exposed ≠ μ_unexposed)"],
                "alt_options": ["Lead exposure reduces IQ at age 7 (μ_exposed < μ_unexposed)", "Lead exposure has no effect on IQ", "Lead exposure affects IQ in either direction (μ_exposed ≠ μ_unexposed)"],
                "correct_null_idx": 0, "correct_alt_idx": 0, "correct_tails": "one-tailed",
                "null_feedback": "✅ Correct. H₀: no difference in mean IQ between exposed and unexposed.",
                "null_wrong_feedback": "❌ H₀ must be an equality — no difference in mean IQ between groups.",
                "alt_feedback": "✅ Correct. Established neurotoxicology literature strongly supports directional prediction (reduction) → one-tailed.",
                "alt_wrong_feedback": "❌ The established literature on lead neurotoxicity strongly supports a directional prediction (lower IQ). This justifies a one-tailed H₁: μ_exposed < μ_unexposed.",
                "tails_connection": "🎯 **One-tailed** — decades of neurotoxicology research establish that lead reduces cognitive function. Strong prior evidence in one direction justifies one-tailed testing when specified before data collection.",
            },
        ]

        for sc in HYP_SCENARIOS:
            st.divider()
            st.markdown(f"**{sc['title']}**")
            st.markdown(sc["description"])
            sid = sc["id"]
            st.markdown("**Step 1: Select the correct null hypothesis (H₀):**")
            null_choice = st.radio("H₀:", sc["null_options"], key=f"h0_{sid}", index=None, label_visibility="collapsed")
            if null_choice is not None:
                if sc["null_options"].index(null_choice) == sc["correct_null_idx"]:
                    st.success(sc["null_feedback"])
                else:
                    st.error(sc["null_wrong_feedback"])
            st.markdown("**Step 2: Select the correct alternative hypothesis (H₁):**")
            alt_choice = st.radio("H₁:", sc["alt_options"], key=f"h1_{sid}", index=None, label_visibility="collapsed")
            if alt_choice is not None:
                if sc["alt_options"].index(alt_choice) == sc["correct_alt_idx"]:
                    st.success(sc["alt_feedback"])
                else:
                    st.error(sc["alt_wrong_feedback"])
            st.markdown("**Step 3: Should this be a one-tailed or two-tailed test?**")
            tails_choice = st.radio("Tails:", ["one-tailed", "two-tailed"], key=f"tails_{sid}", index=None, label_visibility="collapsed")
            if tails_choice is not None:
                if tails_choice == sc["correct_tails"]:
                    st.success(f"✅ Correct — **{sc['correct_tails']}**.")
                else:
                    st.error(f"❌ This should be **{sc['correct_tails']}**.")
            if null_choice is not None and alt_choice is not None and tails_choice is not None:
                all_correct = (
                    sc["null_options"].index(null_choice) == sc["correct_null_idx"] and
                    sc["alt_options"].index(alt_choice) == sc["correct_alt_idx"] and
                    tails_choice == sc["correct_tails"]
                )
                if all_correct:
                    st.info(sc["tails_connection"])

    elif ht_section == "4️⃣ Statistical Power & Sample Size":
        st.subheader("Statistical Power & Sample Size")
        st.markdown("""
**Statistical power** is the probability of correctly rejecting a false null hypothesis — detecting a real effect when it exists.

**Power = 1 − β** where β = probability of Type II error (false negative)

Convention: aim for 80% or 90% power (β = 0.20 or 0.10).
        """)

        st.info("""
**Four factors determine power. Change any one and power changes:**
1. **Effect size** — larger true effect → easier to detect → higher power
2. **Sample size** — more participants → higher power
3. **α level** — larger α (e.g., 0.10 vs 0.05) → higher power, but more Type I error
4. **Variability** — less noise in the data → higher power
        """)

        st.subheader("Interactive Power Explorer")
        st.markdown("Adjust the parameters below and see how power changes.")

        col1, col2 = st.columns(2)
        with col1:
            true_rr = st.slider("True Risk Ratio (effect size)", 1.1, 4.0, 2.0, 0.1)
            r0 = st.slider("Baseline risk in unexposed (%)", 1, 40, 10) / 100
            alpha = st.select_slider("Alpha (α)", options=[0.01, 0.05, 0.10], value=0.05)
        with col2:
            n_per_group = st.slider("Sample size per group", 20, 2000, 200, 10)

        # Approximate power calculation for two-proportion z-test
        r1 = r0 * true_rr
        if r1 > 1.0: r1 = 1.0
        p_pool = (r0 + r1) / 2
        se_null = math.sqrt(2 * p_pool * (1 - p_pool) / n_per_group)
        se_alt = math.sqrt((r0*(1-r0) + r1*(1-r1)) / n_per_group)
        z_alpha = {0.01: 2.576, 0.05: 1.96, 0.10: 1.645}[alpha]
        if se_alt > 0:
            ncp = abs(r1 - r0) / se_alt
            power = max(0, min(1, 1 - 0.5 * (1 + math.erf((z_alpha - ncp) / math.sqrt(2)))))
        else:
            power = 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Statistical Power", f"{round(power*100,1)}%")
        col2.metric("Total Sample Size", f"{n_per_group*2:,}")
        col3.metric("Risk in exposed (r₁)", f"{round(r1*100,1)}%")

        if power >= 0.80:
            st.success(f"✅ Power = {round(power*100,1)}% ≥ 80% — adequately powered to detect RR = {true_rr} at α = {alpha}.")
        elif power >= 0.60:
            st.warning(f"⚠️ Power = {round(power*100,1)}% — marginal. Consider increasing sample size.")
        else:
            st.error(f"❌ Power = {round(power*100,1)}% — underpowered. High risk of false negative.")

        with st.expander("🔢 What sample size do I need for 80% power?"):
            # Solve for n at 80% power
            z_beta_80 = 0.842  # z for 80% power
            if abs(r1 - r0) > 0:
                n_needed = (z_alpha + z_beta_80)**2 * (r0*(1-r0) + r1*(1-r1)) / (r1 - r0)**2
                n_needed = math.ceil(n_needed)
                st.metric(f"Required n per group (80% power, α={alpha})", f"{n_needed:,}")
                st.metric("Total required sample", f"{n_needed*2:,}")
                st.markdown(f"""
**Formula:** n = (z_α + z_β)² × (p₁(1-p₁) + p₀(1-p₀)) ÷ (p₁ - p₀)²

Where z_α = {z_alpha} (for α = {alpha}, two-tailed) and z_β = {z_beta_80} (for 80% power)

**In plain terms:** You need {n_needed:,} people per group ({n_needed*2:,} total) to have an 80% chance of detecting an RR of {true_rr} (baseline risk {round(r0*100)}%) at α = {alpha}.
                """)

        with st.expander("📊 Power Across Sample Sizes"):
            ns = list(range(20, 2001, 40))
            powers = []
            for n in ns:
                se_a = math.sqrt((r0*(1-r0) + r1*(1-r1)) / n)
                if se_a > 0:
                    ncp_n = abs(r1 - r0) / se_a
                    pw = max(0, min(1, 1 - 0.5*(1 + math.erf((z_alpha - ncp_n)/math.sqrt(2)))))
                else:
                    pw = 0
                powers.append(round(pw*100,1))
            power_df = pd.DataFrame({"N per group": ns, "Power (%)": powers})
            st.line_chart(power_df.set_index("N per group"))
            st.caption(f"Power curve for RR = {true_rr}, baseline risk = {round(r0*100)}%, α = {alpha}. The dotted line at 80% is the conventional minimum threshold.")

    st.markdown("---")
    st.markdown("*Strong epidemiologists think structurally before computing.*")


# ==================================================
# MODULE 4: PRACTICE — STUDY DESIGN (existing tab4)
# ==================================================
elif current_page == "practice_design":
    st.title("🎯 Practice: Study Design & Classification")
    st.markdown("Read each scenario, make **all three decisions**, then click **Submit My Answers**. Feedback is hidden until you commit.")

    PRACTICE_SCENARIOS = [
        {"id":"s1","title":"Scenario 1: Lead Exposure & Cognitive Development",
         "description":"400 children near a lead smelting plant and 400 from unexposed neighborhoods followed 3 years. New learning disability diagnoses recorded.",
         "correct_design":"Cohort","correct_outcome":"Binary","correct_exposure":"Binary (2 groups)",
         "design_hint":"Classified by exposure → followed to outcome = cohort.",
         "outcome_hint":"Learning disability: present or absent — binary.",
         "exposure_hint":"Exposed vs. unexposed neighborhoods — two groups = binary.",
         "design_wrong":{"Case-Control":"❌ Case-control starts with cases. Here researchers started with exposure status.","Cross-sectional":"❌ Cross-sectional is a snapshot. Here children followed 3 years."},
         "outcome_wrong":{"Categorical (Nominal >2 levels)":"❌ Categorical requires 3+. Diagnosis is yes/no = binary.","Ordinal":"❌ A diagnosis is yes or no = binary.","Rate (person-time)":"❌ All followed same 3-year period — binary outcome works."},
         "exposure_wrong":{"Categorical (>2 groups)":"❌ Categorical requires 3+ groups. Two groups here = binary."},
         "data":{"type":"contingency","context":"3-year follow-up data. Calculate RR, OR, and p-value.",
                 "row_names":["Lead-exposed","Unexposed"],"col_names":["Learning Disability","No Learning Disability"],"cells":[[52,348],[21,379]]}},
        {"id":"s2","title":"Scenario 2: Fast Food & Obesity",
         "description":"One-time survey of 2,500 adults. Weekly fast food frequency (never/1–2x/3–4x/5+x) and obesity (BMI ≥30) measured simultaneously.",
         "correct_design":"Cross-sectional","correct_outcome":"Binary","correct_exposure":"Categorical (>2 groups)",
         "design_hint":"One-time survey — both measured simultaneously = cross-sectional.",
         "outcome_hint":"Obesity: BMI ≥30 vs. <30 — two categories = binary.",
         "exposure_hint":"Four frequency categories — more than 2 = categorical.",
         "design_wrong":{"Cohort":"❌ Cohort follows people over time. One-time survey = no follow-up.","Case-Control":"❌ Case-control recruits by disease status and looks back. Survey measured everything at once."},
         "outcome_wrong":{"Categorical (Nominal >2 levels)":"❌ Obesity is yes or no — two categories = binary.","Ordinal":"❌ A diagnosis is binary.","Rate (person-time)":"❌ One-time survey — no follow-up time."},
         "exposure_wrong":{"Binary (2 groups)":"❌ Four frequency categories = categorical."},
         "data":{"type":"contingency_wide","context":"Survey data by fast food frequency and obesity.",
                 "row_names":["Never","1–2x/week","3–4x/week","5+x/week"],"col_names":["Obese","Not Obese"],"cells":[[62,538],[118,682],[189,561],[141,209]]}},
        {"id":"s3","title":"Scenario 3: HPV Vaccine & Cervical Cancer",
         "description":"250 women with cervical cancer and 500 without recruited. Vaccination history assessed from medical records.",
         "correct_design":"Case-Control","correct_outcome":"Binary","correct_exposure":"Binary (2 groups)",
         "design_hint":"Started with disease status (cases vs. controls) then looked backward at vaccination = case-control.",
         "outcome_hint":"Cervical cancer: present or absent — binary.",
         "exposure_hint":"Vaccinated vs. unvaccinated — two groups = binary.",
         "design_wrong":{"Cohort":"❌ Cohort classifies by vaccination then tracks who gets cancer. Here recruited by cancer status.","Cross-sectional":"❌ Cross-sectional measures simultaneously. Here recruited by disease status and looked back."},
         "outcome_wrong":{"Categorical (Nominal >2 levels)":"❌ Binary.","Ordinal":"❌ Binary.","Rate (person-time)":"❌ In case-control, outcome is determined before study begins."},
         "exposure_wrong":{"Categorical (>2 groups)":"❌ Vaccinated vs. unvaccinated = two groups = binary."},
         "data":{"type":"contingency","context":"Case-control data. Odds Ratio is appropriate.",
                 "row_names":["Unvaccinated","Vaccinated"],"col_names":["Cervical Cancer (Case)","No Cancer (Control)"],"cells":[[178,182],[72,318]]}},
        {"id":"s4","title":"Scenario 4: Shift Work & Metabolic Syndrome",
         "description":"1,200 hospital employees classified by shift: day only, rotating, or night. Followed 5 years. Metabolic syndrome (yes/no) assessed at end.",
         "correct_design":"Cohort","correct_outcome":"Binary","correct_exposure":"Categorical (>2 groups)",
         "design_hint":"Classified by exposure (shift type) → followed to outcome = cohort.",
         "outcome_hint":"Metabolic syndrome: present or absent — binary.",
         "exposure_hint":"Three shift types — more than 2 = categorical.",
         "design_wrong":{"Case-Control":"❌ Case-control starts with people who already have metabolic syndrome. Here employees classified by shift type first.","Cross-sectional":"❌ Employees followed 5 years — not a snapshot."},
         "outcome_wrong":{"Categorical (Nominal >2 levels)":"❌ Metabolic syndrome = yes/no = binary.","Ordinal":"❌ Binary.","Rate (person-time)":"❌ All followed same 5-year period."},
         "exposure_wrong":{"Binary (2 groups)":"❌ Three categories = categorical."},
         "data":{"type":"contingency_wide","context":"5-year follow-up data by shift type.",
                 "row_names":["Day shift","Rotating shift","Night shift"],"col_names":["Metabolic Syndrome","No Metabolic Syndrome"],"cells":[[62,338],[98,302],[121,279]]}},
        {"id":"s5","title":"Scenario 5: Air Pollution & ED Visits",
         "description":"3,000 adults monitored for PM2.5 over 2 years. Participants vary in outdoor time — each contributes different observation time. Outcome: new ED visits for respiratory illness.",
         "correct_design":"Cohort","correct_outcome":"Rate (person-time)","correct_exposure":"Binary (2 groups)",
         "design_hint":"Classified by PM2.5 level → tracked for new events = cohort.",
         "outcome_hint":"Varying follow-up time — must use person-time. Rate outcome.",
         "exposure_hint":"High vs. low PM2.5 — two groups = binary.",
         "design_wrong":{"Case-Control":"❌ Case-control would start with people who already had ED visits. Here classified by exposure first.","Cross-sectional":"❌ Followed over 2 years — not a snapshot."},
         "outcome_wrong":{"Binary":"❌ Follow-up time varies. Need person-time denominator.","Categorical (Nominal >2 levels)":"❌ Rate, not unordered categories.","Ordinal":"❌ Rate per person-time."},
         "exposure_wrong":{"Categorical (>2 groups)":"❌ High vs. low = two groups = binary."},
         "data":{"type":"rate","context":"Person-time data. Calculate IRR.",
                 "row_names":["High PM2.5","Low PM2.5"],"cases":[187,64],"person_time":[4200,5100]}},
        {"id":"s7","title":"Scenario 6: Air Pollution Spikes & MI",
         "description":"2,100 MI patients. PM2.5 in hour before symptom onset (hazard period) compared to PM2.5 at same time one week earlier for same patient (control period). No separate control group.",
         "correct_design":"Case-Crossover","correct_outcome":"Binary","correct_exposure":"Binary (2 groups)",
         "design_hint":"Each patient compared to themselves at a different time — no separate control group = case-crossover.",
         "outcome_hint":"MI: occurred or did not occur — binary.",
         "exposure_hint":"High vs. low PM2.5 — two groups = binary.",
         "design_wrong":{"Cohort":"❌ Cohort groups by exposure and follows forward. Here everyone already had MI.","Case-Control":"❌ Standard case-control recruits a separate control group. Here each case is their own control.","Cross-sectional":"❌ Cross-sectional is one time point. Here two time windows per person."},
         "outcome_wrong":{"Categorical (Nominal >2 levels)":"❌ MI: yes or no = binary.","Ordinal":"❌ Binary.","Rate (person-time)":"❌ Comparison between two windows per person, not varying follow-up."},
         "exposure_wrong":{"Categorical (>2 groups)":"❌ High vs. low = two groups = binary."},
         "data":{"type":"contingency","context":"Matched data. OR appropriate — each person is their own control.",
                 "row_names":["High PM2.5 (hazard)","Low PM2.5 (hazard)"],"col_names":["High PM2.5 (control)","Low PM2.5 (control)"],"cells":[[210,480],[95,1315]]}},
        {"id":"s8","title":"Scenario 7: Sodium Intake & Hypertension (Retrospective Cohort)",
         "description":"Researchers identify 3,500 adults from a health system's electronic medical records. They review dietary assessments recorded at enrollment 8 years ago, classify each person as high or low sodium intake, then look at hypertension diagnoses recorded in the years since. The researchers are conducting the study today using historical records.",
         "correct_design":"Cohort",
         "correct_outcome":"Binary",
         "correct_exposure":"Binary (2 groups)",
         "design_hint":"Participants are classified by **exposure** (sodium intake recorded in the past) and the outcome (hypertension) is ascertained afterward — this is the defining logic of a cohort study. Using historical records makes it retrospective, but the design is still cohort.",
         "outcome_hint":"Hypertension: diagnosed or not diagnosed — two categories = binary.",
         "exposure_hint":"High vs. low sodium — two groups = binary.",
         "design_wrong":{
             "Case-Control":"❌ Case-control starts by recruiting people who already have hypertension (cases) and looks back at sodium intake. Here participants were classified by sodium intake first, then outcomes were assessed — that's cohort logic, even with historical data.",
             "Cross-sectional":"❌ Cross-sectional measures exposure and outcome at the same point in time. Here sodium was measured first (8 years ago) and hypertension was assessed afterward — temporal separation = cohort.",
         },
         "outcome_wrong":{
             "Categorical (Nominal >2 levels)":"❌ Hypertension is diagnosed or not — two categories = binary.",
             "Ordinal":"❌ A diagnosis is yes/no = binary.",
             "Rate (person-time)":"❌ All participants have the same 8-year follow-up period — binary outcome is appropriate here.",
         },
         "exposure_wrong":{"Categorical (>2 groups)":"❌ High vs. low sodium = two groups = binary."},
         "data":{"type":"contingency","context":"Retrospective cohort data. Calculate RR, OR, and chi-square.",
                 "row_names":["High sodium","Low sodium"],"col_names":["Hypertension","No Hypertension"],"cells":[[312,1188],[198,1802]]}},
        {"id":"s9","title":"Scenario 8: Country-Level Alcohol Consumption & Liver Cirrhosis",
         "description":"A researcher compiles data from 42 countries. For each country, she records the national average alcohol consumption (liters per capita per year) and the national age-standardized liver cirrhosis mortality rate (per 100,000). She finds a strong positive correlation (r = 0.74) between the two country-level measures.",
         "correct_design":"Ecological",
         "correct_outcome":"Rate (person-time)",
         "correct_exposure":"Categorical (>2 groups)",
         "design_hint":"The unit of analysis is **countries**, not individuals. Exposure and outcome are both measured at the aggregate (population) level — this is an ecological study.",
         "outcome_hint":"Liver cirrhosis mortality rate per 100,000 is a **rate with a person-time denominator** — countries contribute population-years of observation.",
         "exposure_hint":"Average alcohol consumption is a continuous measure recorded across 42 countries — more than 2 levels = categorical.",
         "design_wrong":{
             "Cohort":"❌ A cohort study would follow individual people classified by their own alcohol consumption. Here the data are country averages — no individual-level data exist.",
             "Cross-sectional":"❌ Cross-sectional studies measure exposure and outcome for individuals at one point in time. Here both are aggregated to the country level — that's ecological.",
             "Case-Control":"❌ Case-control recruits individuals with and without disease. Here the units are entire countries, not individuals.",
         },
         "outcome_wrong":{
             "Binary":"❌ The outcome is a mortality rate per 100,000 — a continuous rate variable, not a yes/no for each person.",
             "Categorical (Nominal >2 levels)":"❌ A continuous rate is not an unordered categorical variable.",
             "Ordinal":"❌ A mortality rate is a continuous measure, not ordered categories.",
         },
         "exposure_wrong":{"Binary (2 groups)":"❌ Alcohol consumption in liters per capita is continuous across 42 countries — more than 2 levels = categorical."},
         "data":None},
    ]

    design_options   = ["— Select —","Cohort","Case-Control","Cross-sectional","Ecological","Case-Crossover"]
    outcome_options  = ["— Select —","Binary","Categorical (Nominal >2 levels)","Ordinal","Rate (person-time)"]
    exposure_options = ["— Select —","Binary (2 groups)","Categorical (>2 groups)"]

    if "prac_scenario_order" not in st.session_state:
        order = list(range(len(PRACTICE_SCENARIOS))); random.shuffle(order)
        st.session_state["prac_scenario_order"] = order
    SHUFFLED_PRACTICE = [PRACTICE_SCENARIOS[i] for i in st.session_state["prac_scenario_order"]]

    if "prac_reset_count" not in st.session_state:
        st.session_state["prac_reset_count"] = 0

    col_hdr, col_rst = st.columns([5,1])
    with col_hdr: st.caption(f"**{len(PRACTICE_SCENARIOS)} scenarios** — randomized order. Reset to shuffle.")
    with col_rst:
        if st.button("🔄 Reset", key="reset_prac4"):
            st.session_state["prac_reset_count"] += 1
            keys_to_delete = [k for k in st.session_state.keys() if k.startswith("prac_") and k not in ["prac_scenario_order","prac_reset_count"]]
            for k in keys_to_delete: del st.session_state[k]
            if "prac_scenario_order" in st.session_state: del st.session_state["prac_scenario_order"]
            st.rerun()

    rc4 = st.session_state["prac_reset_count"]

    for sc in SHUFFLED_PRACTICE:
        st.divider(); st.subheader(sc["title"]); st.markdown(sc["description"])
        sid = sc["id"]
        submitted_key = f"prac_{sid}_submitted_{rc4}"
        already_submitted = st.session_state.get(submitted_key, False)

        design_choice = st.selectbox("What is the study design?", design_options, key=f"prac_{sid}_design_{rc4}", disabled=already_submitted)
        outcome_choice = st.selectbox("What is the outcome variable type?", outcome_options, key=f"prac_{sid}_outcome_{rc4}", disabled=already_submitted)
        exposure_choice = st.selectbox("What is the exposure variable type?", exposure_options, key=f"prac_{sid}_exposure_{rc4}", disabled=already_submitted)

        all_selected = all(st.session_state.get(f"prac_{sid}_{f}_{rc4}") not in [None,"— Select —"] for f in ["design","outcome","exposure"])

        if not already_submitted:
            if all_selected:
                if st.button("Submit My Answers", key=f"submit_{sid}_{rc4}", type="primary"):
                    st.session_state[submitted_key] = True; st.rerun()
            else:
                st.caption("⬆️ Make all three selections before submitting.")

        if already_submitted:
            dv = st.session_state.get(f"prac_{sid}_design_{rc4}")
            ov = st.session_state.get(f"prac_{sid}_outcome_{rc4}")
            ev = st.session_state.get(f"prac_{sid}_exposure_{rc4}")
            dc = dv == sc["correct_design"]; oc = ov == sc["correct_outcome"]; ec = ev == sc["correct_exposure"]
            all_correct = dc and oc and ec
            correct_count = sum([dc, oc, ec])
            if not all_correct:
                st.error(f"📋 **{correct_count}/3 correct — here's what you missed:**")
                if not dc:
                    wrong_hint = sc.get("design_wrong",{}).get(dv,"")
                    if wrong_hint: st.markdown(f"**Study Design** — You selected: *{dv}*\n\n{wrong_hint}")
                    st.markdown(f"✅ **Correct:** {sc['correct_design']} — {sc['design_hint']}")
                if not oc:
                    wrong_hint = sc.get("outcome_wrong",{}).get(ov,"")
                    if wrong_hint: st.markdown(f"**Outcome Type** — You selected: *{ov}*\n\n{wrong_hint}")
                    st.markdown(f"✅ **Correct:** {sc['correct_outcome']} — {sc['outcome_hint']}")
                if not ec:
                    wrong_hint = sc.get("exposure_wrong",{}).get(ev,"")
                    if wrong_hint: st.markdown(f"**Exposure Type** — You selected: *{ev}*\n\n{wrong_hint}")
                    st.markdown(f"✅ **Correct:** {sc['correct_exposure']} — {sc['exposure_hint']}")
                if st.button("🔄 Try Again", key=f"retry_{sid}_{rc4}"):
                    for f in ["design","outcome","exposure","submitted"]:
                        k = f"prac_{sid}_{f}_{rc4}"
                        if k in st.session_state: del st.session_state[k]
                    st.rerun()
            else:
                st.success("🎯 Perfect — all three correct!")
                st.markdown(f"**Design:** {sc['correct_design']} — {sc['design_hint']}")
                st.markdown(f"**Outcome:** {sc['correct_outcome']} — {sc['outcome_hint']}")
                st.markdown(f"**Exposure:** {sc['correct_exposure']} — {sc['exposure_hint']}")
                if sc.get("data") is None and all_correct:
                    st.warning("""
⚠️ **Ecological fallacy reminder:** The correlation of r = 0.74 between country-level alcohol consumption and cirrhosis mortality is compelling — but it cannot tell us whether *individuals* who drink more have higher cirrhosis risk. Countries with high alcohol consumption may differ from low-consumption countries in many other ways (diet, healthcare access, genetic factors, reporting quality). The ecological association could be entirely driven by these confounders. Always be cautious about inferring individual-level risk from group-level data.
                    """)

            if all_correct and "data" in sc and sc["data"] is not None:
                st.markdown("---"); st.markdown("### 📋 Now run the analysis")
                st.markdown(sc["data"]["context"]); d = sc["data"]
                if d["type"] in ["contingency","contingency_wide"]:
                    df_d = pd.DataFrame(d["cells"], columns=d["col_names"], index=d["row_names"])
                    df_d["Row Total"] = df_d.sum(axis=1)
                    tr = df_d.sum(); tr.name = "Column Total"
                    df_d = pd.concat([df_d, tr.to_frame().T]); st.table(df_d)
                    if st.button("Run Statistical Analysis", key=f"run_{sid}_{rc4}"):
                        table = np.array(d["cells"]); chi2_val, p_val, dof, _ = chi2_contingency(table)
                        st.write(f"χ²({dof}) = {round(chi2_val,3)}, p = {round(p_val,4) if p_val >= 0.0001 else '< 0.0001'}")
                        if p_val < 0.05: st.success("Statistically significant. Reject H₀.")
                        else: st.warning("Insufficient evidence. Fail to reject H₀.")
                        chi2_explanation_expander(chi2_val, p_val, dof, table, d["col_names"], d["row_names"])
                        if d["type"] == "contingency":
                            a,b = table[0]; c,dd = table[1]
                            if all(v > 0 for v in [a,b,c,dd]):
                                rr=(a/(a+b))/(c/(c+dd)); se_log_rr=math.sqrt((1/a)-(1/(a+b))+(1/c)-(1/(c+dd)))
                                ci_low_rr=math.exp(math.log(rr)-1.96*se_log_rr); ci_high_rr=math.exp(math.log(rr)+1.96*se_log_rr)
                                or_val=(a*dd)/(b*c); se_log_or=math.sqrt(1/a+1/b+1/c+1/dd)
                                ci_low_or=math.exp(math.log(or_val)-1.96*se_log_or); ci_high_or=math.exp(math.log(or_val)+1.96*se_log_or)
                                st.subheader("Risk Ratio (RR)")
                                if ci_low_rr <= 1 <= ci_high_rr: st.warning(f"RR = {round(rr,2)} — Not significant.")
                                else:
                                    direction = "higher" if rr > 1 else "lower"
                                    st.success(f"RR = {round(rr,2)} (95% CI: {round(ci_low_rr,2)}–{round(ci_high_rr,2)}). {round(rr,2)}× {direction}.")
                                draw_ci("RR", rr, ci_low_rr, ci_high_rr)
                                st.subheader("Odds Ratio (OR)")
                                if ci_low_or <= 1 <= ci_high_or: st.warning(f"OR = {round(or_val,2)} — Not significant.")
                                else:
                                    direction = "higher" if or_val > 1 else "lower"
                                    st.success(f"OR = {round(or_val,2)} (95% CI: {round(ci_low_or,2)}–{round(ci_high_or,2)}). {round(or_val,2)}× {direction}.")
                                draw_ci("OR", or_val, ci_low_or, ci_high_or)
                                rr_or_explanation_expander(a,b,c,dd,d["row_names"],d["col_names"],rr,or_val,ci_low_rr,ci_high_rr,ci_low_or,ci_high_or)
                        else:
                            st.info("With 3+ categories, chi-square is the appropriate test. RR/OR require a 2×2 table.")
                elif d["type"] == "rate":
                    df_d = pd.DataFrame({"Group":d["row_names"],"Cases":d["cases"],"Person-Time":d["person_time"],
                        "Rate per 100k":[round(d["cases"][i]/d["person_time"][i]*100000,1) for i in range(len(d["cases"]))]})
                    st.table(df_d)
                    if st.button("Run Statistical Analysis", key=f"run_{sid}_{rc4}"):
                        c1,c2=d["cases"]; pt1,pt2=d["person_time"]
                        irr=(c1/pt1)/(c2/pt2); se_log_irr=math.sqrt((1/c1)+(1/c2))
                        ci_low_irr=math.exp(math.log(irr)-1.96*se_log_irr); ci_high_irr=math.exp(math.log(irr)+1.96*se_log_irr)
                        st.write(f"IRR = {round(irr,3)}, 95% CI: ({round(ci_low_irr,3)}, {round(ci_high_irr,3)})")
                        if ci_low_irr <= 1 <= ci_high_irr: st.warning("CI includes 1. Not significant.")
                        else:
                            direction = "higher" if irr > 1 else "lower"
                            st.success(f"IRR = {round(irr,2)} — Rate in {d['row_names'][0]} is {round(irr,2)}× {direction}.")
                        draw_ci("IRR", irr, ci_low_irr, ci_high_irr)

    st.divider()
    total_correct=0; answered=0
    for sc in PRACTICE_SCENARIOS:
        sid=sc["id"]
        if st.session_state.get(f"prac_{sid}_submitted_{rc4}"):
            answered+=3
            total_correct+=sum([st.session_state.get(f"prac_{sid}_design_{rc4}")==sc["correct_design"],
                                 st.session_state.get(f"prac_{sid}_outcome_{rc4}")==sc["correct_outcome"],
                                 st.session_state.get(f"prac_{sid}_exposure_{rc4}")==sc["correct_exposure"]])
    if answered > 0:
        pct = round(total_correct/answered*100)
        st.subheader(f"📊 Score: {total_correct} / {answered}")
        st.progress(pct/100)
        if pct==100: st.success("🏆 Perfect on all submitted scenarios!")
        elif pct>=75: st.info("Good work — review any missed scenarios.")
        else: st.warning("Review the feedback above and try again.")

    st.markdown("---")
    st.markdown("*Strong epidemiologists think structurally before computing.*")


# ==================================================
# MODULE 4: PRACTICE — ADVANCED MEASURES (existing tab5, condensed)
# ==================================================
elif current_page == "practice_advanced":
    st.title("🎯 Practice: Advanced Epi Measures")
    st.markdown("Select the most appropriate advanced measure for each scenario, then click **Submit My Answer**.")

    ADV_SCENARIOS = [
        {"id":"adv_1","title":"Scenario 1: Obesity & Coronary Heart Disease",
         "description":"A public health analyst wants to estimate how much CHD burden could be eliminated if obesity were eradicated. 42% of US adults have obesity; they have 1.8× the risk of CHD.",
         "correct_measure":"Population Attributable Risk (PAR)",
         "measure_hint":"Population-level preventable burden — PAR combines exposure prevalence (Pe) with RR.",
         "measure_wrong":{"Standardized Mortality Ratio (SMR)":"❌ SMR compares observed to expected deaths. This asks about population-level preventable fraction — PAR.","Attributable Risk & AR%":"❌ AR% estimates fraction within the **exposed group**. PAR estimates across the **total population**.","Number Needed to Harm / Treat (NNH/NNT)":"❌ NNT/NNH express per-person benefit/harm. This asks about a population fraction — PAR.","Hazard Ratio (HR)":"❌ HR compares event rates over time. Population fraction = PAR."},
         "data":{"type":"par","context":"Calculate PAR%.","Pe":0.42,"RR":1.8}},
        {"id":"adv_2","title":"Scenario 2: Rubber Workers & Bladder Cancer",
         "description":"4,200 rubber workers followed 15 years. 38 developed bladder cancer. Applying general population rates to this cohort's age structure predicts only 18.4 expected cases.",
         "correct_measure":"Standardized Mortality Ratio (SMR)",
         "measure_hint":"Observed cases (38) and expected cases (18.4) from reference rates. Observed ÷ Expected = SMR.",
         "measure_wrong":{"Population Attributable Risk (PAR)":"❌ PAR requires exposure prevalence and RR. You have observed vs. expected = SMR.","Attributable Risk & AR%":"❌ AR% compares two groups within same study. Here comparing to reference population = SMR.","Number Needed to Harm / Treat (NNH/NNT)":"❌ NNH requires a risk difference. Observed vs. expected = SMR.","Hazard Ratio (HR)":"❌ HR requires Cox regression. Total observed vs. expected = SMR."},
         "data":{"type":"smr","context":"Calculate SMR.","observed":38,"expected":18.4,"group_label":"Rubber workers","outcome_label":"bladder cancer"}},
        {"id":"adv_3","title":"Scenario 3: Hypertension & Stroke",
         "description":"14% of uncontrolled hypertension patients had a stroke over 10 years vs. 4% of controlled patients. Of all strokes in the uncontrolled group, what fraction is due to uncontrolled BP?",
         "correct_measure":"Attributable Risk & AR%",
         "measure_hint":"Fraction of disease within the exposed group (uncontrolled hypertension) = AR%.",
         "measure_wrong":{"Population Attributable Risk (PAR)":"❌ PAR estimates fraction across the **entire population**. This asks about fraction within the **exposed group** = AR%.","Standardized Mortality Ratio (SMR)":"❌ SMR compares to reference population. This compares two groups = AR%.","Number Needed to Harm / Treat (NNH/NNT)":"❌ NNT tells how many need treatment to prevent one event. Fraction attributable = AR%.","Hazard Ratio (HR)":"❌ HR uses Cox regression. 10-year cumulative risks = AR%."},
         "data":{"type":"ar","context":"Calculate AR and AR%.","r_exposed":0.14,"r_unexposed":0.04,"exposed_label":"Uncontrolled hypertension","unexposed_label":"Controlled hypertension"}},
        {"id":"adv_4","title":"Scenario 4: Naloxone Programs & Overdose Deaths",
         "description":"3% of communities with naloxone programs had overdose deaths vs. 7% without. How many communities need the program to prevent one additional overdose death?",
         "correct_measure":"Number Needed to Harm / Treat (NNH/NNT)",
         "measure_hint":"How many need the intervention to prevent one event = NNT = 1 / Risk Difference.",
         "measure_wrong":{"Population Attributable Risk (PAR)":"❌ PAR estimates preventable fraction. This asks how many need treatment = NNT.","Standardized Mortality Ratio (SMR)":"❌ SMR compares to reference population. Per-community benefit = NNT.","Attributable Risk & AR%":"❌ AR% estimates fraction attributable. Per-community figure = NNT.","Hazard Ratio (HR)":"❌ HR uses time-to-event Cox regression. Simple proportions + per-treatment benefit = NNT."},
         "data":{"type":"nnt","context":"Calculate NNT.","r_treatment":0.03,"r_control":0.07,"treatment_label":"Naloxone program","control_label":"No program","outcome_label":"overdose death"}},
        {"id":"adv_5","title":"Scenario 5: Physical Activity & Hip Fracture",
         "description":"2,800 adults 65+ followed up to 10 years. Follow-up varies due to deaths and losses. A Cox proportional hazards model was fitted.",
         "correct_measure":"Hazard Ratio (HR)",
         "measure_hint":"Varying follow-up, censoring, and Cox model = HR.",
         "measure_wrong":{"Population Attributable Risk (PAR)":"❌ PAR requires exposure prevalence and RR. Cox regression = HR.","Standardized Mortality Ratio (SMR)":"❌ SMR requires observed vs. expected. Cox regression = HR.","Attributable Risk & AR%":"❌ AR% requires complete fixed follow-up. Censored + Cox = HR.","Number Needed to Harm / Treat (NNH/NNT)":"❌ NNT requires fixed time point. Censored data + Cox = HR.","Incidence Rate Ratio (IRR)":"❌ IRR uses person-time rates. Cox regression produces a Hazard Ratio, not an IRR."},
         "data":{"type":"hr","context":"Interpret the HR from the Cox model.","hr":0.61,"ci_low":0.48,"ci_high":0.78,"exposed_label":"Physically active","outcome_label":"hip fracture"}},
        {"id":"adv_6","title":"Scenario 6: Long-Term PPI Use & Chronic Kidney Disease",
         "description":"A large cohort study finds that 3.2% of daily proton pump inhibitor (PPI) users developed chronic kidney disease (CKD) over 5 years, compared to 1.1% of non-users. A nephrologist asks: for every how many patients prescribed long-term PPIs will one additional case of CKD occur that would not have occurred otherwise?",
         "correct_measure":"Number Needed to Harm / Treat (NNH/NNT)",
         "measure_hint":"'For every how many patients exposed will one additional harm occur' = **NNH**. This is the harm-direction version of NNT. NNH = 1 ÷ |Risk Difference|.",
         "measure_wrong":{
             "Population Attributable Risk (PAR)":"❌ PAR estimates the population-level preventable fraction. This asks for a per-patient harm count — NNH.",
             "Standardized Mortality Ratio (SMR)":"❌ SMR compares observed to expected deaths vs. a reference population. This is a per-patient clinical harm question — NNH.",
             "Attributable Risk & AR%":"❌ AR% tells what fraction of disease in the exposed group is attributable to the exposure. The nephrologist wants an intuitive per-patient figure — NNH.",
             "Hazard Ratio (HR)":"❌ HR compares event rates over time using Cox regression. Here we have simple 5-year cumulative risks — NNH.",
             "Incidence Rate Ratio (IRR)":"❌ IRR uses person-time denominators. Here we have fixed 5-year cumulative proportions — NNH.",
         },
         "data":{"type":"nnt","context":"Calculate NNH — how many patients need long-term PPI exposure before one additional CKD case occurs.",
                 "r_treatment":0.032,"r_control":0.011,
                 "treatment_label":"Long-term PPI use","control_label":"No PPI use","outcome_label":"chronic kidney disease"}},
        {"id":"adv_7","title":"Scenario 7: Air Pollution & Asthma Hospitalizations",
         "description":"A cohort of 6,200 children in an urban area is monitored for asthma-related hospitalizations. Children are classified as living in high vs. low PM2.5 exposure zones. Because families move in and out of the city during the study, each child contributes a different amount of observation time. The research team calculates rates using person-years at risk.",
         "correct_measure":"Incidence Rate Ratio (IRR)",
         "measure_hint":"**Varying follow-up time** + rates expressed per person-years = IRR. When participants contribute different amounts of observation time, you must use person-time denominators and compare incidence rates — not cumulative risks.",
         "measure_wrong":{
             "Population Attributable Risk (PAR)":"❌ PAR requires exposure prevalence in the population and an RR or IRR — it's calculated from IRR, not instead of it.",
             "Standardized Mortality Ratio (SMR)":"❌ SMR compares one group's observed events to expected events from a reference population's rates. Here we're comparing two internal groups with varying follow-up — IRR.",
             "Attributable Risk & AR%":"❌ AR% uses cumulative risks over a fixed period. When follow-up varies, rates per person-time are needed — IRR.",
             "Number Needed to Harm / Treat (NNH/NNT)":"❌ NNH/NNT require a fixed-time risk difference. Varying follow-up requires person-time rates — IRR.",
             "Hazard Ratio (HR)":"❌ HR comes from Cox proportional hazards regression. A simple comparison of person-time rates between two exposure groups produces an IRR, not an HR.",
         },
         "data":{"type":"rate","context":"Calculate the IRR comparing asthma hospitalization rates between high and low PM2.5 exposure zones.",
                 "row_names":["High PM2.5 zone","Low PM2.5 zone"],"cases":[148,62],"person_time":[9800,11200]}},
        {"id":"adv_8","title":"Scenario 8: Sedentary Behavior & Type 2 Diabetes — How Much Is Preventable?",
         "description":"A national health survey finds that 38% of adults are highly sedentary (>8 hours sitting per day). A meta-analysis estimates that highly sedentary adults have 1.4× the risk of Type 2 Diabetes compared to less sedentary adults. A public health team wants to know: if the entire population reduced their sedentary time, what fraction of all T2D cases could theoretically be prevented? Compare this to smoking & lung cancer (Pe = 0.14, RR = 15.0) to understand how exposure prevalence and effect size interact.",
         "correct_measure":"Population Attributable Risk (PAR)",
         "measure_hint":"'What fraction of all disease in the population is attributable to this exposure?' = **PAR%**. PAR combines how common the exposure is (Pe) with how strongly it causes disease (RR) to estimate population-level preventable burden.",
         "measure_wrong":{
             "Standardized Mortality Ratio (SMR)":"❌ SMR compares observed to expected deaths vs. a reference population. Population-level preventable fraction = PAR.",
             "Attributable Risk & AR%":"❌ AR% estimates the fraction within the **exposed group** only. PAR% estimates across the **entire population** — accounting for how common sedentary behavior is.",
             "Number Needed to Harm / Treat (NNH/NNT)":"❌ NNH/NNT express per-person benefit or harm. Population-level preventable fraction = PAR.",
             "Hazard Ratio (HR)":"❌ HR compares instantaneous event rates. Population-level preventable fraction = PAR.",
             "Incidence Rate Ratio (IRR)":"❌ IRR compares incidence rates between groups. To estimate what fraction of population disease is attributable to an exposure, use PAR.",
         },
         "data":{"type":"par_compare",
                 "context":"Calculate PAR% for sedentary behavior, then compare to smoking & lung cancer to see how Pe and RR interact.",
                 "scenarios":[
                     {"label":"Sedentary behavior & T2D","Pe":0.38,"RR":1.4},
                     {"label":"Smoking & Lung Cancer","Pe":0.14,"RR":15.0},
                 ]}},
    ]

    measure_options = ["— Select —","Population Attributable Risk (PAR)","Standardized Mortality Ratio (SMR)","Attributable Risk & AR%","Number Needed to Harm / Treat (NNH/NNT)","Hazard Ratio (HR)","Incidence Rate Ratio (IRR)"]

    if "adv_reset_count" not in st.session_state: st.session_state["adv_reset_count"] = 0
    if "adv_scenario_order" not in st.session_state:
        order = list(range(len(ADV_SCENARIOS))); random.shuffle(order)
        st.session_state["adv_scenario_order"] = order
    SHUFFLED_ADV = [ADV_SCENARIOS[i] for i in st.session_state["adv_scenario_order"]]

    col_hdr5, col_rst5 = st.columns([5,1])
    with col_hdr5: st.caption(f"**{len(ADV_SCENARIOS)} scenarios** — randomized.")
    with col_rst5:
        if st.button("🔄 Reset", key="reset_adv_prac"):
            st.session_state["adv_reset_count"] += 1
            keys_to_delete = [k for k in st.session_state.keys() if k.startswith("adv_") and k not in ["adv_scenario_order","adv_reset_count"]]
            for k in keys_to_delete: del st.session_state[k]
            if "adv_scenario_order" in st.session_state: del st.session_state["adv_scenario_order"]
            st.rerun()

    rc5 = st.session_state["adv_reset_count"]

    for sc in SHUFFLED_ADV:
        st.divider(); st.subheader(sc["title"]); st.markdown(sc["description"])
        sid = sc["id"]
        submitted_key = f"adv_submitted_{sid}_{rc5}"
        already_submitted = st.session_state.get(submitted_key, False)
        measure_choice = st.selectbox("Which advanced measure is most appropriate?", measure_options, key=f"adv_measure_{sid}_{rc5}", disabled=already_submitted)
        selected = st.session_state.get(f"adv_measure_{sid}_{rc5}") not in [None, "— Select —"]
        if not already_submitted:
            if selected:
                if st.button("Submit My Answer", key=f"adv_submit_{sid}_{rc5}", type="primary"):
                    st.session_state[submitted_key] = True; st.rerun()
            else: st.caption("⬆️ Select a measure before submitting.")

        if already_submitted:
            measure_val = st.session_state.get(f"adv_measure_{sid}_{rc5}")
            correct = measure_val == sc["correct_measure"]
            if not correct:
                wrong_hint = sc.get("measure_wrong",{}).get(measure_val,"")
                st.error("📋 **Incorrect — here's what you missed:**")
                if wrong_hint: st.markdown(f"**You selected:** *{measure_val}*\n\n{wrong_hint}")
                st.markdown(f"✅ **Correct:** {sc['correct_measure']} — {sc['measure_hint']}")
                if st.button("🔄 Try Again", key=f"adv_retry_{sid}_{rc5}"):
                    for k_suffix in ["measure","submitted"]:
                        k = f"adv_{k_suffix}_{sid}_{rc5}"
                        if k in st.session_state: del st.session_state[k]
                    st.rerun()
            else:
                st.success(f"✅ Correct! **{sc['correct_measure']}** — {sc['measure_hint']}")

            if correct:
                st.markdown("---"); st.markdown("### 📋 Now run the analysis")
                st.markdown(sc["data"]["context"]); d = sc["data"]
                if d["type"] == "par":
                    col1,col2 = st.columns(2)
                    col1.metric("Exposure Prevalence (Pe)", f"{round(d['Pe']*100,1)}%")
                    col2.metric("Risk Ratio (RR)", d["RR"])
                    if st.button("Calculate PAR%", key=f"run_{sid}_{rc5}"):
                        Pe=d["Pe"]; RR=d["RR"]
                        PAR_pct = (Pe*(RR-1))/(1+Pe*(RR-1))*100
                        st.metric("PAR%", f"{round(PAR_pct,1)}%")
                        st.success(f"{round(PAR_pct,1)}% of all cases attributable to this exposure.")

                elif d["type"] == "par_compare":
                    st.markdown("Calculate PAR% for each scenario, then compare them.")
                    if st.button("Calculate Both PAR%s", key=f"run_{sid}_{rc5}"):
                        results = []
                        for sc2 in d["scenarios"]:
                            Pe = sc2["Pe"]; RR = sc2["RR"]
                            par = (Pe*(RR-1))/(1+Pe*(RR-1))*100
                            results.append((sc2["label"], Pe, RR, par))
                        cols = st.columns(len(results))
                        for i, (label, Pe, RR, par) in enumerate(results):
                            cols[i].metric(f"PAR% — {label}", f"{round(par,1)}%")
                            cols[i].caption(f"Pe = {round(Pe*100,0):.0f}%, RR = {RR}")
                        st.divider()
                        par1, par2 = results[0][3], results[1][3]
                        st.info(f"""
**Key insight — Pe × RR interaction:**

**{results[0][0]}:** Pe = {round(results[0][1]*100,0):.0f}%, RR = {results[0][2]} → PAR% = {round(par1,1)}%
**{results[1][0]}:** Pe = {round(results[1][1]*100,0):.0f}%, RR = {results[1][2]} → PAR% = {round(par2,1)}%

Smoking has a far stronger individual effect (RR = {results[1][2]}) but sedentary behavior is much more common (Pe = {round(results[0][1]*100,0):.0f}% vs {round(results[1][1]*100,0):.0f}%). Yet PAR% for smoking is still higher because the RR of {results[1][2]} dominates.

**The lesson:** PAR% depends on BOTH how common the exposure is AND how strongly it causes disease. A very common exposure with a modest RR can have a surprisingly large PAR% — and a rare exposure with a huge RR can have a smaller PAR% than you'd expect.
                        """)
                        with st.expander("🔢 Show me the math"):
                            for label, Pe, RR, par in results:
                                st.markdown(f"""
**{label}**
PAR% = [Pe × (RR − 1)] ÷ [1 + Pe × (RR − 1)] × 100
= [{Pe} × ({RR} − 1)] ÷ [1 + {Pe} × ({RR} − 1)] × 100
= **{round(par,1)}%**
                                """)

                elif d["type"] == "rate":
                    df_d = pd.DataFrame({
                        "Group": d["row_names"],
                        "Cases": d["cases"],
                        "Person-Time (years)": d["person_time"],
                        "Rate per 100,000": [round(d["cases"][i]/d["person_time"][i]*100000, 1) for i in range(len(d["cases"]))]
                    })
                    st.table(df_d)
                    if st.button("Calculate IRR", key=f"run_{sid}_{rc5}"):
                        c1, c2 = d["cases"]; pt1, pt2 = d["person_time"]
                        irr = (c1/pt1)/(c2/pt2)
                        se_log_irr = math.sqrt((1/c1)+(1/c2))
                        ci_low_irr = math.exp(math.log(irr)-1.96*se_log_irr)
                        ci_high_irr = math.exp(math.log(irr)+1.96*se_log_irr)
                        col1, col2, col3 = st.columns(3)
                        col1.metric(f"Rate ({d['row_names'][0]})", f"{round(c1/pt1*100000,1)}/100k py")
                        col2.metric(f"Rate ({d['row_names'][1]})", f"{round(c2/pt2*100000,1)}/100k py")
                        col3.metric("IRR", round(irr, 3))
                        st.write(f"95% CI: ({round(ci_low_irr,3)}, {round(ci_high_irr,3)})")
                        if ci_low_irr <= 1 <= ci_high_irr:
                            st.warning("CI includes 1.0 — not statistically significant.")
                        else:
                            direction = "higher" if irr > 1 else "lower"
                            st.success(f"IRR = {round(irr,2)} — Rate in {d['row_names'][0]} is {round(irr,2)}× {direction} than in {d['row_names'][1]}.")
                        draw_ci("IRR", irr, ci_low_irr, ci_high_irr)
                        with st.expander("🔢 Show me the math — IRR"):
                            st.markdown(f"""
**IRR = Rate₁ ÷ Rate₂**

Rate ({d['row_names'][0]}) = {c1} cases ÷ {pt1} person-years = {round(c1/pt1*100000,1)} per 100,000 person-years

Rate ({d['row_names'][1]}) = {c2} cases ÷ {pt2} person-years = {round(c2/pt2*100000,1)} per 100,000 person-years

IRR = {round(c1/pt1,6)} ÷ {round(c2/pt2,6)} = **{round(irr,3)}**

95% CI: exp(ln(IRR) ± 1.96 × √(1/cases₁ + 1/cases₂)) = **({round(ci_low_irr,3)}, {round(ci_high_irr,3)})**
                            """)
                elif d["type"] == "smr":
                    col1,col2 = st.columns(2)
                    col1.metric("Observed Deaths", d["observed"]); col2.metric("Expected Deaths", d["expected"])
                    if st.button("Calculate SMR", key=f"run_{sid}_{rc5}"):
                        smr = d["observed"]/d["expected"]
                        ci_low_s = max(0, smr-1.96*(smr/math.sqrt(d["observed"])))
                        ci_high_s = smr+1.96*(smr/math.sqrt(d["observed"]))
                        st.metric("SMR", round(smr,3))
                        st.write(f"95% CI: ({round(ci_low_s,3)}, {round(ci_high_s,3)})")
                        draw_ci("SMR", smr, ci_low_s, ci_high_s)
                        if ci_low_s <= 1 <= ci_high_s: st.warning("Not significantly different from reference.")
                        elif smr > 1: st.error(f"SMR = {round(smr,2)} — Excess mortality.")
                        else: st.success(f"SMR = {round(smr,2)} — Lower than expected. Possible healthy worker effect.")
                elif d["type"] == "ar":
                    col1,col2 = st.columns(2)
                    col1.metric(f"Risk ({d['exposed_label']})", f"{round(d['r_exposed']*100,1)}%")
                    col2.metric(f"Risk ({d['unexposed_label']})", f"{round(d['r_unexposed']*100,1)}%")
                    if st.button("Calculate AR & AR%", key=f"run_{sid}_{rc5}"):
                        ar = d["r_exposed"] - d["r_unexposed"]
                        ar_pct = (ar / d["r_exposed"]) * 100
                        col1,col2 = st.columns(2)
                        col1.metric("AR", f"{round(ar*100,1)}%"); col2.metric("AR%", f"{round(ar_pct,1)}%")
                        st.success(f"AR% = {round(ar_pct,1)}%: fraction of strokes in the exposed group attributable to the exposure.")
                elif d["type"] == "nnt":
                    col1,col2 = st.columns(2)
                    col1.metric(f"Risk ({d['treatment_label']})", f"{round(d['r_treatment']*100,1)}%")
                    col2.metric(f"Risk ({d['control_label']})", f"{round(d['r_control']*100,1)}%")
                    if st.button("Calculate NNT/NNH", key=f"run_{sid}_{rc5}"):
                        risk_diff = abs(d["r_treatment"] - d["r_control"])
                        nnt = round(1/risk_diff, 1)
                        is_benefit = d["r_treatment"] < d["r_control"]
                        label = "NNT" if is_benefit else "NNH"
                        st.metric(label, nnt)
                        if is_benefit: st.success(f"NNT = {nnt}: treat {nnt} people to prevent one additional {d['outcome_label']}.")
                        else: st.error(f"NNH = {nnt}: {nnt} people exposed before one additional {d['outcome_label']}.")
                elif d["type"] == "hr":
                    col1,col2,col3 = st.columns(3)
                    col1.metric("HR", d["hr"]); col2.metric("CI Lower", d["ci_low"]); col3.metric("CI Upper", d["ci_high"])
                    if st.button("Interpret HR", key=f"run_{sid}_{rc5}"):
                        draw_ci("HR", d["hr"], d["ci_low"], d["ci_high"])
                        if d["ci_low"] <= 1 <= d["ci_high"]: st.warning(f"HR = {d['hr']} — Not significant.")
                        elif d["hr"] < 1: st.success(f"HR = {d['hr']}: {d['exposed_label']} had {round((1-d['hr'])*100,1)}% lower hazard of {d['outcome_label']}.")
                        else: st.error(f"HR = {d['hr']}: {d['exposed_label']} had {round((d['hr']-1)*100,1)}% higher hazard of {d['outcome_label']}.")

    st.divider()
    adv_answered = sum(1 for sc in ADV_SCENARIOS if st.session_state.get(f"adv_submitted_{sc['id']}_{rc5}"))
    adv_correct = sum(1 for sc in ADV_SCENARIOS if st.session_state.get(f"adv_submitted_{sc['id']}_{rc5}") and st.session_state.get(f"adv_measure_{sc['id']}_{rc5}") == sc["correct_measure"])
    if adv_answered > 0:
        st.subheader(f"📊 Score: {adv_correct} / {adv_answered} submitted")
        st.progress(adv_correct/adv_answered)

    st.markdown("---")
    st.markdown("*Strong epidemiologists think structurally before computing.*")


# ==================================================
# MODULE 4: PRACTICE — CONFOUNDING & BIAS
# ==================================================
elif current_page == "practice_confounding":
    st.title("🎯 Practice: Confounding & Bias")
    st.markdown("For each scenario, identify the issue (confounding or a specific type of bias), its direction, and how to address it.")

    CB_SCENARIOS = [
        {"id":"cb1",
         "title":"Scenario 1: Coffee & Pancreatic Cancer",
         "description":"A case-control study finds that coffee drinkers have 2.1× the odds of pancreatic cancer (OR = 2.1, 95% CI: 1.6–2.8). However, coffee consumption is strongly associated with cigarette smoking in this population, and smoking is a known risk factor for pancreatic cancer.",
         "question":"What is the primary methodological concern?",
         "options":["Confounding by smoking","Recall bias","Berkson's bias","Non-differential misclassification","Effect modification by smoking"],
         "correct":"Confounding by smoking",
         "explanation":"Smoking meets all three criteria for confounding: associated with coffee (exposure), associated with pancreatic cancer (outcome), and not on the causal pathway between coffee and cancer. The OR of 2.1 likely overestimates any true coffee-cancer relationship. After stratifying or adjusting for smoking, the association may attenuate substantially.",
         "follow_up":"How would you control for this confounding?",
         "follow_up_options":["Stratify by smoking status or adjust in multivariable regression","Use only hospital-based controls","Increase sample size","Use a one-tailed test"],
         "correct_follow_up":"Stratify by smoking status or adjust in multivariable regression",
         "follow_up_explanation":"Stratification by smoking status would give stratum-specific ORs for coffee among smokers and non-smokers separately. If the ORs in each stratum are close to 1.0 but the crude OR was 2.1, this confirms confounding by smoking. Multivariable logistic regression with smoking as a covariate is the standard analytic approach."},
        {"id":"cb2",
         "title":"Scenario 2: Processed Meat & Colorectal Cancer",
         "description":"A case-control study finds OR = 1.8 for processed meat and colorectal cancer. Cases (cancer patients) were asked to recall their diet over the past 5 years. Controls were healthy volunteers recruited from the community. Cases spent considerable time reflecting on their diet after diagnosis; controls had no such motivation.",
         "question":"What is the primary methodological concern?",
         "options":["Recall bias (differential misclassification)","Confounding by fiber intake","Berkson's bias","Non-differential misclassification","Healthy worker effect"],
         "correct":"Recall bias (differential misclassification)",
         "explanation":"Cases have a strong incentive to remember dietary exposures after receiving a cancer diagnosis — they may search their memory more carefully than controls. This differential recall inflates the apparent OR. The true association may be weaker than 1.8.",
         "follow_up":"Which direction does recall bias push the OR in this scenario?",
         "follow_up_options":["Away from null (OR is overestimated)","Toward null (OR is underestimated)","Cannot determine direction","No effect on OR"],
         "correct_follow_up":"Away from null (OR is overestimated)",
         "follow_up_explanation":"Cases over-report past exposure (processed meat) relative to controls. This increases the numerator (a — cases who report exposure) relative to controls who report exposure (b). OR = (a×d)/(b×c). Inflating 'a' relative to 'b' increases the OR — pushing it away from null."},
        {"id":"cb3",
         "title":"Scenario 3: Physical Activity & Depression",
         "description":"A cross-sectional survey finds that physically active people have 40% lower prevalence of depression (PR = 0.60). The study population is a random sample of employed adults aged 25–65.",
         "question":"What is the primary limitation of this study for causal inference?",
         "options":["Cannot establish temporality — direction of causation is unclear","Recall bias","Berkson's bias","Confounding by age","Non-differential misclassification of physical activity"],
         "correct":"Cannot establish temporality — direction of causation is unclear",
         "explanation":"Cross-sectional studies measure exposure and outcome simultaneously. It's impossible to determine whether physical inactivity leads to depression, or whether depression causes physical inactivity (reverse causation). Both directions are biologically plausible. The PR of 0.60 cannot be interpreted causally from this design alone.",
         "follow_up":"What study design would best address this limitation?",
         "follow_up_options":["Prospective cohort — measure physical activity first, follow for new depression","Larger cross-sectional study","Case-control — recruit depressed and non-depressed patients","Ecological study comparing countries"],
         "correct_follow_up":"Prospective cohort — measure physical activity first, follow for new depression",
         "follow_up_explanation":"A prospective cohort study classifies participants by physical activity at baseline, then follows them forward to identify new (incident) depression cases. This establishes temporal order — exposure before outcome — which is the only Bradford Hill criterion that is mandatory for causal inference."},
        {"id":"cb4",
         "title":"Scenario 4: Hormone Therapy & Cardiovascular Disease (The WHI Story)",
         "description":"Multiple observational cohort studies throughout the 1980s–90s found that post-menopausal women taking hormone therapy (HT) had substantially lower rates of cardiovascular disease (RR ≈ 0.5). The Women's Health Initiative RCT was then conducted — and found HT *increased* cardiovascular risk (HR = 1.29). The observational studies were wrong. Women who chose HT were more educated, higher socioeconomic status, more health-conscious, and had better access to care — all of which independently reduced cardiovascular risk.",
         "question":"What type of bias most explains the discrepancy between the observational studies and the RCT?",
         "options":["Confounding by socioeconomic status and health behaviors (healthy user bias)","Recall bias","Berkson's bias","Non-differential misclassification of HT use","Loss to follow-up bias"],
         "correct":"Confounding by socioeconomic status and health behaviors (healthy user bias)",
         "explanation":"Women who chose hormone therapy were not comparable to those who did not — they were systematically healthier on many unmeasured dimensions. This is 'healthy user bias,' a form of confounding by indication where the type of person who receives a treatment differs from those who don't. No statistical adjustment fully controlled this confounding in observational studies. The RCT randomized women, eliminating this confounding — and reversed the apparent finding."},
    ]

    if "cb_rc" not in st.session_state: st.session_state["cb_rc"] = 0
    rc = st.session_state["cb_rc"]
    col_hdr, col_rst = st.columns([5,1])
    with col_hdr: st.caption(f"**{len(CB_SCENARIOS)} scenarios**")
    with col_rst:
        if st.button("🔄 Reset", key="reset_cb"):
            st.session_state["cb_rc"] += 1; st.rerun()

    for sc in CB_SCENARIOS:
        st.divider(); st.subheader(sc["title"]); st.markdown(sc["description"])
        sid = sc["id"]
        sub_key = f"cb_submitted_{sid}_{rc}"
        already = st.session_state.get(sub_key, False)

        choice = st.radio(sc["question"], ["— Select —"] + sc["options"], key=f"cb_choice_{sid}_{rc}", index=0, disabled=already)
        if not already and choice != "— Select —":
            if st.button("Submit", key=f"cb_submit_{sid}_{rc}", type="primary"):
                st.session_state[sub_key] = True; st.rerun()

        if already:
            val = st.session_state.get(f"cb_choice_{sid}_{rc}")
            if val == sc["correct"]:
                st.success(f"✅ Correct! **{sc['correct']}**")
            else:
                st.error(f"❌ You selected: *{val}*")
                st.markdown(f"✅ **Correct:** {sc['correct']}")
            st.info(f"**Explanation:** {sc['explanation']}")

            if "follow_up" in sc:
                st.markdown(f"**Follow-up: {sc['follow_up']}**")
                sub_key2 = f"cb_fu_submitted_{sid}_{rc}"
                already2 = st.session_state.get(sub_key2, False)
                fu_choice = st.radio("", ["— Select —"] + sc["follow_up_options"], key=f"cb_fu_{sid}_{rc}", index=0, disabled=already2, label_visibility="collapsed")
                if not already2 and fu_choice != "— Select —":
                    if st.button("Submit Follow-up", key=f"cb_fu_submit_{sid}_{rc}"):
                        st.session_state[sub_key2] = True; st.rerun()
                if already2:
                    fu_val = st.session_state.get(f"cb_fu_{sid}_{rc}")
                    if fu_val == sc["correct_follow_up"]:
                        st.success(f"✅ Correct!")
                    else:
                        st.error(f"❌ You selected: *{fu_val}*")
                        st.markdown(f"✅ **Correct:** {sc['correct_follow_up']}")
                    st.info(f"**Explanation:** {sc['follow_up_explanation']}")

            if st.button("🔄 Try Again", key=f"cb_retry_{sid}_{rc}"):
                for k in [sub_key, f"cb_choice_{sid}_{rc}", f"cb_fu_submitted_{sid}_{rc}", f"cb_fu_{sid}_{rc}"]:
                    if k in st.session_state: del st.session_state[k]
                st.rerun()

    st.markdown("---")
    st.markdown("*Strong epidemiologists think structurally before computing.*")


# ==================================================
# MODULE 4: PRACTICE — SCREENING & FREQUENCY
# ==================================================
elif current_page == "practice_screening":
    st.title("🎯 Practice: Screening & Disease Frequency")
    st.markdown("Apply your knowledge of screening test performance and disease frequency measures to real scenarios.")

    SCREEN_SCENARIOS = [
        {
            "id": "ss1",
            "title": "Scenario 1: HIV Screening in Two Clinics",
            "description": """
An HIV rapid antibody test has **sensitivity = 99%** and **specificity = 99%**.

**Clinic A** serves a high-risk population: HIV prevalence = 10%
**Clinic B** serves a general primary care population: HIV prevalence = 0.1%

Both clinics use the same test.
            """,
            "question": "In which clinic will a positive test result be more likely to represent true HIV infection?",
            "options": ["Clinic A (high-prevalence)", "Clinic B (low-prevalence)", "The same in both — sensitivity and specificity don't depend on prevalence"],
            "correct": "Clinic A (high-prevalence)",
            "explanation": """
**Clinic A (prevalence = 10%):** In 10,000 patients → 1,000 truly infected, 9,000 uninfected
- True positives: 1,000 × 0.99 = 990
- False positives: 9,000 × 0.01 = 90
- PPV = 990 ÷ (990+90) = **91.7%**

**Clinic B (prevalence = 0.1%):** In 10,000 patients → 10 truly infected, 9,990 uninfected
- True positives: 10 × 0.99 = 9.9 ≈ 10
- False positives: 9,990 × 0.01 = 99.9 ≈ 100
- PPV = 10 ÷ (10+100) = **9.1%**

Same test, completely different clinical meaning. In Clinic B, 9 out of every 10 positive tests are false alarms. This is why confirmatory testing (e.g., Western blot) is required — it raises the effective specificity and PPV.
            """,
            "follow_up": "Which metric stays the same regardless of which clinic uses the test?",
            "follow_up_options": ["Sensitivity and specificity", "PPV", "NPV", "All four metrics change with prevalence"],
            "correct_follow_up": "Sensitivity and specificity",
            "follow_up_explanation": "Sensitivity and specificity are intrinsic properties of the test — they describe how well the test performs on people who truly have or don't have the disease, regardless of how many such people are in the room. PPV and NPV change with prevalence because they depend on the mix of true cases and non-cases being tested."
        },
        {
            "id": "ss2",
            "title": "Scenario 2: Outbreak Attack Rate",
            "description": """
At a catered dinner of 180 attendees, 63 people develop gastrointestinal illness within 12 hours.

Of the 63 ill attendees, 58 had eaten the chicken salad. Of the 117 who were not ill, 42 had also eaten the chicken salad.
            """,
            "question": "What is the attack rate among those who ate the chicken salad?",
            "options": ["32.2% (63 ÷ 180 × 100)", "58.0% (58 ÷ 100 × 100)", "43.8% (7/16 calculation)", "35.0%"],
            "correct": "58.0% (58 ÷ 100 × 100)",
            "explanation": """
**Setting up the 2×2 table:**

|  | Ill | Not ill | Total |
|---|---|---|---|
| Ate chicken salad | 58 | 42 | 100 |
| Did not eat chicken salad | 5 | 75 | 80 |
| Total | 63 | 117 | 180 |

**Attack rate (exposed):** 58 ÷ 100 = **58.0%**
**Attack rate (unexposed):** 5 ÷ 80 = **6.3%**

**Risk Ratio:** 58.0% ÷ 6.3% = **9.2** — people who ate the chicken salad were 9.2× more likely to become ill.

The overall attack rate (35.0%) is less informative — it mixes exposed and unexposed. The food-specific attack rate comparison is what identifies the vehicle.
            """,
        },
        {
            "id": "ss3",
            "title": "Scenario 3: Prevalence vs. Incidence",
            "description": """
A regional health department reports:
- **2021:** 4,200 people living with Type 2 diabetes in a county of 85,000 adults
- **2022:** 340 new diabetes diagnoses among the 80,800 adults who did not have diabetes at the start of 2022

The department also knows the average duration of diabetes in this population is approximately 18 years.
            """,
            "question": "What is the 2022 incidence rate (per 1,000)?",
            "options": [
                "4.21 per 1,000 (340 ÷ 80,800 × 1,000)",
                "49.4 per 1,000 (4,200 ÷ 85,000 × 1,000)",
                "3.99 per 1,000 (340 ÷ 85,000 × 1,000)",
                "0.42 per 1,000"
            ],
            "correct": "4.21 per 1,000 (340 ÷ 80,800 × 1,000)",
            "explanation": """
**2021 Prevalence:** 4,200 ÷ 85,000 × 1,000 = **49.4 per 1,000** (4.94%)

**2022 Cumulative Incidence:** 340 ÷ 80,800 × 1,000 = **4.21 per 1,000**

The denominator for incidence must be the **disease-free population at the start** (80,800) — not the full county population. Including people who already have diabetes in the denominator would underestimate incidence.

**P ≈ I × D check:** 4.21/1,000 × 18 years = 75.8/1,000 ≈ 7.6% — this overestimates our observed 4.94% prevalence, which may reflect that the 18-year average duration is an overestimate, or that mortality from diabetes reduces the prevalent pool.
            """,
            "follow_up": "Why is the denominator for incidence (80,800) different from the denominator for prevalence (85,000)?",
            "follow_up_options": [
                "Incidence requires a disease-free denominator — people who already have diabetes cannot develop it again",
                "The two measures use different years, so the population changed",
                "Prevalence always uses a larger denominator than incidence",
                "It's a data entry error — both should use 85,000"
            ],
            "correct_follow_up": "Incidence requires a disease-free denominator — people who already have diabetes cannot develop it again",
            "follow_up_explanation": "Incidence measures new cases. To be a new case, you must have been at risk of developing the disease — meaning you didn't already have it. The 4,200 prevalent cases in 2021 are excluded from the 2022 incidence denominator because they were not 'at risk' of newly developing diabetes. Using the full population as denominator would dilute the incidence estimate."
        },
        {
            "id": "ss4",
            "title": "Scenario 4: Choosing the Right Test Characteristic",
            "description": """
A hospital is deciding which screening test to use for two different purposes:

**Purpose A:** Initial screening of all patients admitted to the ICU for a rare but rapidly fatal fungal infection. Treatment must begin immediately if positive — waiting for confirmation risks death.

**Purpose B:** Confirmatory testing after a positive initial screen for a common but stigmatizing chronic condition. A false positive would lead to unnecessary treatment with serious side effects.
            """,
            "question": "For Purpose A (initial ICU screening), which property should be maximized?",
            "options": [
                "Sensitivity — to minimize false negatives (missing true cases)",
                "Specificity — to minimize false positives",
                "PPV — to ensure positives are real",
                "NPV — to ensure negatives are real"
            ],
            "correct": "Sensitivity — to minimize false negatives (missing true cases)",
            "explanation": """
**Purpose A — maximize sensitivity:**
- The disease is rapidly fatal and treatment can start immediately
- A false negative (missed case) = patient dies untreated
- A false positive = patient receives treatment unnecessarily, but that's an acceptable tradeoff
- **SnNout:** Sensitive test, Negative result rules Out. If a highly sensitive test is negative, you can be confident the patient doesn't have the infection.

**Purpose B — maximize specificity:**
- A positive screen has already occurred; now you need to *rule in*
- False positive = unnecessary treatment with serious side effects + stigma
- High specificity means few false positives → positive result is more reliable
- **SpPin:** Specific test, Positive result rules In. If a highly specific test is positive, the diagnosis is likely correct.

The general principle: use high-sensitivity tests to screen, high-specificity tests to confirm.
            """,
        },
        {
            "id": "ss5",
            "title": "Scenario 5: Interpreting an Epidemic Curve",
            "description": """
An epidemiologist plots an epidemic curve for a foodborne illness outbreak at a company picnic. The curve shows:
- Cases begin 2–6 hours after the picnic
- The curve rises sharply and falls sharply
- Nearly all cases occur within a single 8-hour window
- No secondary cases are reported in households of ill attendees
            """,
            "question": "What transmission pattern does this epidemic curve suggest?",
            "options": [
                "Point-source — all cases exposed to the same source at approximately the same time",
                "Propagated — person-to-person spread over multiple incubation periods",
                "Mixed — initial point source followed by person-to-person transmission",
                "Endemic — background level of disease in the community"
            ],
            "correct": "Point-source — all cases exposed to the same source at approximately the same time",
            "explanation": """
**Classic point-source features present:**
1. ✅ Sharp rise and fall — compressed time course
2. ✅ All cases within approximately one incubation period (2–6 hours suggests bacterial toxin or chemical)
3. ✅ No secondary cases — no person-to-person spread after the event
4. ✅ Linked to a single event (the picnic)

**What this tells you epidemiologically:**
- The incubation period (2–6 hours) suggests a preformed toxin (e.g., *Staphylococcus aureus*, *Bacillus cereus*) rather than an infection requiring bacterial replication
- Next step: identify the food vehicle by calculating food-specific attack rates and RRs for each food served
- The width of the epidemic curve approximates the incubation period range

**Contrast with propagated:** A propagated epidemic curve would show multiple waves, each approximately one incubation period apart, with cases spreading over days to weeks.
            """
        },
    ]

    if "ss_rc" not in st.session_state:
        st.session_state["ss_rc"] = 0
    rc = st.session_state["ss_rc"]

    col_hdr, col_rst = st.columns([5,1])
    with col_hdr: st.caption(f"**{len(SCREEN_SCENARIOS)} scenarios** covering screening performance and disease frequency measures.")
    with col_rst:
        if st.button("🔄 Reset", key="reset_ss"):
            st.session_state["ss_rc"] += 1; st.rerun()

    for sc in SCREEN_SCENARIOS:
        st.divider()
        st.subheader(sc["title"])
        st.markdown(sc["description"])
        sid = sc["id"]
        sub_key = f"ss_submitted_{sid}_{rc}"
        already = st.session_state.get(sub_key, False)

        choice = st.radio(
            sc["question"],
            ["— Select —"] + sc["options"],
            key=f"ss_choice_{sid}_{rc}",
            index=0,
            disabled=already
        )

        if not already and choice != "— Select —":
            if st.button("Submit", key=f"ss_submit_{sid}_{rc}", type="primary"):
                st.session_state[sub_key] = True
                st.rerun()
        elif not already:
            st.caption("⬆️ Select an answer before submitting.")

        if already:
            val = st.session_state.get(f"ss_choice_{sid}_{rc}")
            if val == sc["correct"]:
                st.success(f"✅ Correct!")
            else:
                st.error(f"❌ You selected: *{val}*")
                st.markdown(f"✅ **Correct answer:** {sc['correct']}")
            st.info(sc["explanation"])

            if "follow_up" in sc:
                st.markdown(f"---")
                st.markdown(f"**Follow-up question: {sc['follow_up']}**")
                sub_key2 = f"ss_fu_submitted_{sid}_{rc}"
                already2 = st.session_state.get(sub_key2, False)
                fu_choice = st.radio(
                    "",
                    ["— Select —"] + sc["follow_up_options"],
                    key=f"ss_fu_{sid}_{rc}",
                    index=0,
                    disabled=already2,
                    label_visibility="collapsed"
                )
                if not already2 and fu_choice != "— Select —":
                    if st.button("Submit Follow-up", key=f"ss_fu_submit_{sid}_{rc}"):
                        st.session_state[sub_key2] = True; st.rerun()
                if already2:
                    fu_val = st.session_state.get(f"ss_fu_{sid}_{rc}")
                    if fu_val == sc["correct_follow_up"]:
                        st.success("✅ Correct!")
                    else:
                        st.error(f"❌ You selected: *{fu_val}*")
                        st.markdown(f"✅ **Correct:** {sc['correct_follow_up']}")
                    st.info(sc["follow_up_explanation"])

            if st.button("🔄 Try Again", key=f"ss_retry_{sid}_{rc}"):
                for k in [sub_key, f"ss_choice_{sid}_{rc}",
                          f"ss_fu_submitted_{sid}_{rc}", f"ss_fu_{sid}_{rc}"]:
                    if k in st.session_state: del st.session_state[k]
                st.rerun()

    # Score tracker
    st.divider()
    ss_answered = sum(1 for sc in SCREEN_SCENARIOS if st.session_state.get(f"ss_submitted_{sc['id']}_{rc}"))
    ss_correct = sum(1 for sc in SCREEN_SCENARIOS
                     if st.session_state.get(f"ss_submitted_{sc['id']}_{rc}")
                     and st.session_state.get(f"ss_choice_{sc['id']}_{rc}") == sc["correct"])
    if ss_answered > 0:
        pct = round(ss_correct / ss_answered * 100)
        st.subheader(f"📊 Score: {ss_correct} / {ss_answered}")
        st.progress(pct / 100)
        if pct == 100: st.success("🏆 Perfect!")
        elif pct >= 75: st.info("Good work — review any missed scenarios.")
        else: st.warning("Review the explanations above and try again.")

    st.markdown("---")
    st.markdown("*Strong epidemiologists think structurally before computing.*")


# ==================================================
# REFERENCE: GLOSSARY (existing tab7, expanded)
# ==================================================
elif current_page == "glossary":
    st.title("📖 Glossary of Key Terms")
    st.markdown("A complete reference for all major concepts in the lab. Use this while working through practice scenarios.")

    with st.expander("📐 Study Designs", expanded=False):
        st.markdown("""
**Cohort Study**
Participants classified by **exposure status** and followed to outcome. Can be prospective or retrospective. The defining feature is grouping by exposure before outcome. Produces RR or IRR.

**Case-Control Study**
Participants recruited by **outcome status** — cases (have disease) and controls (don't). Researchers look **backward** at past exposure. Produces OR. Efficient for rare diseases.

**Matched Case-Control:** Each case paired with controls matched on potential confounders (age, sex). Controls confounding by design; requires matched analysis (conditional logistic regression).

**Cross-Sectional Study**
Exposure and outcome measured **at the same point in time** — a snapshot. Produces PR. Cannot establish temporal order.

**Case-Crossover Study**
Each case serves as **their own control** — exposure during a hazard period (just before event) vs. a control period (same person, no event). Eliminates between-person confounding. Best for transient exposures with acute effects. Produces OR.

**RCT (Randomized Controlled Trial)**
Participants **randomly assigned** to treatment or control. Gold standard for causation. Randomization distributes known and unknown confounders.
        """)

    with st.expander("⚠️ Bias"):
        st.markdown("""
**Bias**
A systematic error that distorts the true association between exposure and outcome. Unlike random error, does not average out with larger samples.

**Selection Bias**
Who is included in the study is related to both exposure and outcome.

**Berkson's Bias**
Hospital patients are not representative of the general population — both the exposure and disease independently increase hospitalization probability.

**Healthy Worker Effect**
Employed workers are systematically healthier than the general population. Causes SMR < 1 in occupational studies even without protective effects.

**Loss to Follow-Up Bias**
Dropout related to both exposure and outcome distorts results.

**Information Bias (Misclassification)**
Exposure or outcome is measured incorrectly.

**Non-Differential Misclassification**
Measurement error equal across groups. Biases toward null (attenuates the association).

**Differential Misclassification**
Measurement error differs between groups. Can bias in either direction.

**Recall Bias**
Cases remember past exposures more carefully than controls. Common in case-control studies. Biases OR away from null.
        """)

    with st.expander("🔀 Confounding & Effect Modification"):
        st.markdown("""
**Confounding**
A variable that distorts the apparent association between exposure and outcome. Must be: (1) associated with exposure, (2) independently associated with outcome, (3) not on the causal pathway.

**Confounder Control — Design:** Randomization, restriction, matching.

**Confounder Control — Analysis:** Stratification (Mantel-Haenszel), multivariable regression, propensity scores.

**Residual Confounding**
Confounding that remains after adjustment, due to imperfect measurement of confounders.

**Effect Modification (Interaction)**
The association between exposure and outcome differs across levels of a third variable. A real phenomenon to be reported, not a bias to be removed.

**10% Rule**
If adjusting for a variable changes RR/OR by >10%, it is a meaningful confounder.
        """)

    with st.expander("🔗 Causal Inference"):
        st.markdown("""
**Bradford Hill Criteria (1965)**
Nine criteria for evaluating causal evidence: strength, consistency, specificity, temporality, biological gradient, plausibility, coherence, experiment, analogy. Temporality is the only mandatory criterion.

**Counterfactual Framework**
Causation requires asking: what would have happened to the same person if their exposure status had been different? The fundamental problem of causal inference is that we can never observe both potential outcomes for the same person.
        """)

    with st.expander("📊 Disease Frequency Measures"):
        st.markdown("""
**Prevalence**
Proportion of population with condition at a point in time. Numerator: existing cases. No time unit.

**Cumulative Incidence (Attack Rate)**
Proportion of disease-free population that develops disease during a specified period. Numerator: new cases. Denominator: disease-free at start.

**Incidence Rate (Incidence Density)**
New cases per unit person-time at risk. Used when follow-up varies. Units: per 1,000 person-years.

**Case Fatality Rate (CFR)**
Deaths from disease ÷ total cases. Measures lethality, not frequency. A proportion, not a true rate.

**Prevalence-Incidence Relationship**
P ≈ I × D (Prevalence ≈ Incidence × Duration). Holds at steady state with low prevalence.

**Point-Source Epidemic**
All cases exposed to same source at same time. Sharp epidemic curve; width ≈ one incubation period.

**Propagated Epidemic**
Person-to-person spread. Multiple waves in epidemic curve; each wave ≈ one incubation period apart.
        """)

    with st.expander("🔬 Screening & Diagnostic Tests"):
        st.markdown("""
**Sensitivity**
True positive rate: proportion of true cases that test positive. High sensitivity → few false negatives → rules OUT disease when negative. **SnNout.**

**Specificity**
True negative rate: proportion of true non-cases that test negative. High specificity → few false positives → rules IN disease when positive. **SpPin.**

**PPV (Positive Predictive Value)**
Probability that a positive test reflects true disease. Depends on prevalence — low in low-prevalence populations.

**NPV (Negative Predictive Value)**
Probability that a negative test reflects true absence of disease. Increases as prevalence decreases.

**Sensitivity-Specificity Tradeoff**
Lowering the test cutpoint increases sensitivity but decreases specificity. ROC curve plots this tradeoff.

**Prevalence Effect on PPV**
Same test, different population prevalence → different PPV. Even an excellent test has poor PPV when disease prevalence is very low.
        """)

    with st.expander("📐 Study Design — Measures of Association"):
        st.markdown("""
**Risk Ratio (RR)**
Risk in exposed ÷ risk in unexposed. Cohort studies. RR = 1: no difference; RR > 1: higher risk; RR < 1: protective.

**Prevalence Ratio (PR)**
Same formula as RR but used in cross-sectional studies where the outcome is prevalent (existing), not incident (new).

**Odds Ratio (OR)**
Odds of outcome in exposed ÷ odds in unexposed. Used in case-control studies. OR always farther from 1 than RR when outcome is common. When outcome rare (<10%), OR ≈ RR.

**Incidence Rate Ratio (IRR)**
Rate in exposed ÷ rate in unexposed using person-time denominators. Used when follow-up time varies.

**Hazard Ratio (HR)**
Ratio of instantaneous event rates at any moment in time. Output of Cox proportional hazards regression. Used when follow-up varies and participants may be censored.

**Risk Difference (Attributable Risk)**
Risk in exposed − risk in unexposed. Absolute excess risk.
        """)

    with st.expander("📉 Advanced Epi Measures"):
        st.markdown("""
**Attributable Risk (AR) / Risk Difference**
Risk in exposed − risk in unexposed. Absolute excess risk per 100 exposed.

**Attributable Risk Percent (AR%)**
AR ÷ risk in exposed × 100. Fraction of disease in the exposed group attributable to exposure.

**Population Attributable Risk Percent (PAR%)**
Fraction of all disease in the total population attributable to exposure. Formula: Pe(RR−1) / [1+Pe(RR−1)] × 100. Accounts for both exposure prevalence and strength of association.

**Standardized Mortality Ratio (SMR)**
Observed deaths ÷ Expected deaths (expected = reference rates × study group age structure). SMR > 1: excess mortality. SMR < 1: lower mortality (possibly healthy worker effect).

**Healthy Worker Effect**
Workers healthier than general population → SMR < 1 in occupational cohorts even without true protection.

**Number Needed to Treat (NNT)**
How many need treatment for one additional person to benefit. NNT = 1 / Risk Difference.

**Number Needed to Harm (NNH)**
How many need exposure for one additional person to be harmed. NNH = 1 / Risk Difference.
        """)

    with st.expander("🧪 Hypothesis Testing & Power"):
        st.markdown("""
**Null Hypothesis (H₀)**
Default: no association, no difference. Always an equality (RR = 1, μ₁ = μ₂).

**Alternative Hypothesis (H₁)**
States an association exists. Two-tailed (≠) or one-tailed (< or >).

**p-value**
Probability of observing a result as extreme as yours if H₀ were true. NOT the probability H₀ is true.

**Type I Error (α)**
Rejecting true H₀. False positive. Probability = 0.05.

**Type II Error (β)**
Failing to reject false H₀. False negative. Power = 1 − β.

**Statistical Power**
Probability of correctly detecting a real effect. Power = 1 − β. Conventional minimum: 80%.

**Confidence Interval (CI)**
Range of plausible values for the true effect. 95% CI excluding 1 → p < 0.05. More informative than p-value alone.

**Chi-Square (χ²)**
Tests whether observed counts differ from expected under independence. Always two-tailed. Larger χ² → smaller p.

**One-Tailed Test**
Tests effect in one specific direction. All 5% in one tail. Only appropriate with strong prior directional hypothesis.

**Two-Tailed Test**
Tests any difference regardless of direction. Default in epidemiology. Chi-square always two-tailed.
        """)

    with st.expander("📏 Standardization"):
        st.markdown("""
**Crude Rate**
Overall rate without adjusting for confounders. Can mislead when populations have different age structures.

**Direct Standardization**
Applies each population's age-specific rates to a single standard population. Produces age-adjusted rate. Best for comparing two populations.

**Indirect Standardization**
Applies reference population rates to study group's age structure. Produces SMR. Best when study group has small numbers making age-specific rates unstable.

**Confounding by Age**
Apparent rate difference due to different age structures, not true disease burden. Standardization removes this.
        """)

    st.divider()
    st.markdown("*Return to any module to apply these concepts in context.*")

