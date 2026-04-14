import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import math
import random
import os
import json

st.set_page_config(page_title="Epidemiology Decision Simulator", layout="wide")

# ==================================================
# LOGIN GATE
# ==================================================

def load_users():
    """
    Load users from three sources in priority order:
    1. Railway environment variable EPILAB_USERS (JSON string) — for buyer deployments
    2. Streamlit Cloud secrets [users] section — for course deployment
    3. Hardcoded fallback — for local development only

    Railway env var format (set in Railway dashboard):
    EPILAB_USERS = {"buyer001": "pass123", "buyer002": "pass456"}
    """
    # 1. Railway / any platform environment variable
    env_users_raw = os.environ.get("EPILAB_USERS", "")
    if env_users_raw:
        try:
            return json.loads(env_users_raw)
        except Exception:
            pass

    # 2. Streamlit Cloud secrets
    try:
        cloud_users = st.secrets.get("users", {})
        if cloud_users:
            return dict(cloud_users)
    except Exception:
        pass

    # 3. Local development fallback — NOT used in either production deployment
    return {
        "marymathis": "epilab2024",
        "student1":   "epilab2024",
        "student2":   "epilab2024",
        "guest":      "epilab2024",
    }

def check_credentials(username, password):
    users = load_users()
    return username in users and users[username] == password

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
        st.caption("Access issues? Contact support.")


if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login_screen()
    st.stop()

# ==================================================
# HELPER FUNCTIONS
# ==================================================

def interpret_nnt(nnt, is_benefit, treatment_label, control_label, outcome_label):
    """Render a full contextual interpretation block for NNT or NNH."""
    label = "NNT" if is_benefit else "NNH"
    direction_word = "prevent" if is_benefit else "cause"
    comparison = f"compared to {control_label}" if control_label else ""

    # Contextual scale
    if nnt <= 5:
        scale_label = "Very high impact"
        scale_color = "#1a7a1a" if is_benefit else "#b71c1c"
        scale_note  = "Exceptionally few people need treatment/exposure for one event. Seen in highly effective treatments or very toxic exposures."
    elif nnt <= 15:
        scale_label = "High impact"
        scale_color = "#2e7d32" if is_benefit else "#c62828"
        scale_note  = "Strong real-world effect. Most clinically meaningful interventions fall here."
    elif nnt <= 50:
        scale_label = "Moderate impact"
        scale_color = "#f57f17"
        scale_note  = "Meaningful but modest. Common in preventive interventions applied to lower-risk populations."
    elif nnt <= 200:
        scale_label = "Small impact"
        scale_color = "#e65100"
        scale_note  = "Many people must be treated/exposed for one event. Typical of mass-population screening or low-prevalence exposures."
    else:
        scale_label = "Marginal impact"
        scale_color = "#6a1a1a" if not is_benefit else "#4a4a4a"
        scale_note  = f"Very large {label} — the absolute risk difference is tiny. Even if statistically significant, clinical/public health significance is limited."

    main_interp = (
        f"**{label} = {nnt}** — you must treat/expose **{nnt} people** with **{treatment_label}** "
        f"to {direction_word} **one additional {outcome_label}** {comparison}."
    )

    scale_html = f"""
<div style="margin:10px 0 6px 0; background:#f8f8f8; border-left:4px solid {scale_color};
     padding:10px 16px; border-radius:4px; font-size:13px;">
  <span style="font-weight:700; color:{scale_color};">{scale_label}</span> &nbsp;|&nbsp;
  <span style="color:#555;">{scale_note}</span>
</div>
"""

    # Reference scale table
    scale_table = """
<div style="font-size:12px; margin-top:10px; color:#555;">
<strong>Contextual scale for {lbl}:</strong>
<table style="border-collapse:collapse; width:100%; margin-top:6px;">
  <tr style="background:#f0f0f0;">
    <th style="padding:5px 10px; text-align:left; border:1px solid #ddd;">{lbl} value</th>
    <th style="padding:5px 10px; text-align:left; border:1px solid #ddd;">Interpretation</th>
    <th style="padding:5px 10px; text-align:left; border:1px solid #ddd;">Example</th>
  </tr>
  <tr><td style="padding:5px 10px; border:1px solid #ddd;">≤ 5</td>
      <td style="padding:5px 10px; border:1px solid #ddd;">Very high impact</td>
      <td style="padding:5px 10px; border:1px solid #ddd; color:#888;">IV antibiotics for bacterial meningitis (NNT ≈ 2–3)</td></tr>
  <tr style="background:#f9f9f9;"><td style="padding:5px 10px; border:1px solid #ddd;">6–15</td>
      <td style="padding:5px 10px; border:1px solid #ddd;">High impact</td>
      <td style="padding:5px 10px; border:1px solid #ddd; color:#888;">Statins in high-risk patients (NNT ≈ 10–20)</td></tr>
  <tr><td style="padding:5px 10px; border:1px solid #ddd;">16–50</td>
      <td style="padding:5px 10px; border:1px solid #ddd;">Moderate impact</td>
      <td style="padding:5px 10px; border:1px solid #ddd; color:#888;">Statins in average-risk patients (NNT ≈ 50)</td></tr>
  <tr style="background:#f9f9f9;"><td style="padding:5px 10px; border:1px solid #ddd;">51–200</td>
      <td style="padding:5px 10px; border:1px solid #ddd;">Small impact</td>
      <td style="padding:5px 10px; border:1px solid #ddd; color:#888;">Mammography screening (NNT ≈ 100–200)</td></tr>
  <tr><td style="padding:5px 10px; border:1px solid #ddd;">&gt; 200</td>
      <td style="padding:5px 10px; border:1px solid #ddd;">Marginal impact</td>
      <td style="padding:5px 10px; border:1px solid #ddd; color:#888;">Population salt reduction (NNT ≈ 1000+)</td></tr>
</table>
<p style="margin-top:8px; font-size:11px; color:#888;">
⚠️ {lbl} is always relative to a baseline risk — the same intervention can have different {lbl} values in high-risk vs. low-risk populations.
Always interpret {lbl} alongside the absolute risk difference and the severity of the outcome.
</p>
</div>
""".format(lbl=label)

    if is_benefit:
        st.success(main_interp)
    else:
        st.error(main_interp)

    import streamlit.components.v1 as _nnt_comp
    _nnt_comp.html(f"<div style='font-family:sans-serif;'>{scale_html}</div>", height=70, scrolling=False)

    with st.expander(f"📏 {label} Contextual Scale & Interpretation Guide"):
        st.markdown(f"""
**What does {label} = {nnt} mean in plain language?**

{"NNT measures treatment benefit: out of every " + str(int(nnt)) + " people treated with " + treatment_label + ", exactly one person avoids " + outcome_label + " who would otherwise have experienced it. The other " + str(int(nnt)-1) + " people either were never going to get " + outcome_label + " anyway, or got it despite treatment." if is_benefit else
 "NNH measures treatment/exposure harm: out of every " + str(int(nnt)) + " people exposed to " + treatment_label + ", exactly one additional person experiences " + outcome_label + " who would not have without the exposure. The other " + str(int(nnt)-1) + " experience no additional harm from this exposure."}

**Why does baseline risk matter?**
The same RR can produce very different NNT values depending on who is being treated:
- High-risk population (10% baseline): RR 0.5 → ARD 5% → NNT = **20**
- Low-risk population (1% baseline): RR 0.5 → ARD 0.5% → NNT = **200**
Same relative benefit, 10× different absolute benefit.
        """)
        st.markdown(scale_table, unsafe_allow_html=True)


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

The resulting p-value of {p_str}{tail_note} means: **conditional on the null hypothesis being true** (no association), a chi-square statistic this large or larger would arise by chance {p_str} of the time. This is not the probability that the null is true — it is the probability of the data given the null. {'That is rare enough to reject H₀.' if p_val < 0.05 else 'That is not rare enough to reject H₀ at α = 0.05.'}
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

**When do {pabbr} and OR agree?** As a rough teaching heuristic, when outcome prevalence is low (often cited as <10%), OR approximates {pabbr} reasonably well. As outcomes become more common, OR diverges increasingly from 1 — there is no sharp cutoff, and the approximation degrades gradually.
        """)


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

NAV_STRUCTURE = [
    ("MODULE 1 — Study Design & Causation", [
        ("foundations",         "🏛️", "Foundations",                "Prevention, natural history, PICO"),
        ("study_designs",       "🔬", "Study Designs",             "Cohort, case-control, RCT, crossover"),
        ("bias",                "⚠️", "Bias",                       "Selection, recall, misclassification"),
        ("confounding",         "🔀", "Confounding & Effect Mod.",  "Control methods, stratification, DAGs"),
        ("causal_inference",    "🔗", "Causal Inference",           "Bradford Hill, Rothman's pies"),
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
        ("outbreak_lab",        "🔍", "Outbreak Lab",               "Investigate 3 real-style outbreaks"),
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
    ("🏛️ Foundations",                   "foundations"),
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
    ("🔍 Outbreak Lab",                  "outbreak_lab"),
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
# MODULE 1: FOUNDATIONS
# ==================================================
if current_page == "foundations":
    st.title("🏛️ Foundations of Epidemiology")
    st.markdown("Core frameworks that underpin all of epidemiology — how disease develops, how we prevent it, how we study it, and what counts as a cause.")

    found_section = st.radio("Section:", [
        "1️⃣ Epidemiology Triangle",
        "2️⃣ Natural History & Levels of Prevention",
        "3️⃣ Chain of Infection & Infectious Disease",
        "4️⃣ Herd Immunity & R₀",
        "5️⃣ Outbreak Investigation — The 10 Steps",
        "6️⃣ PICO Framework",
    ], horizontal=True)
    st.divider()

    # ── SECTION 1: EPIDEMIOLOGY TRIANGLE ──
    if found_section == "1️⃣ Epidemiology Triangle":
        st.subheader("The Epidemiology Triangle (Epidemiologic Triad)")
        st.markdown("""
The **epidemiology triangle** is one of the oldest and most widely used frameworks for understanding why disease occurs in populations. It holds that disease results from the interaction of three elements: **agent**, **host**, and **environment** — with **time** at the center, recognizing that the relationship among all three unfolds over time.

Remove or modify any one element, and the disease dynamic changes.
        """)

        import streamlit.components.v1 as _tri_comp
        _tri_comp.html("""
<div style="font-family:sans-serif;text-align:center;padding:10px 0;">
<div style="font-weight:700;font-size:13px;margin-bottom:10px;color:#1a202c;">The Epidemiology Triangle</div>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 340" width="500" height="340">
  <!-- Triangle fill -->
  <polygon points="250,28 40,300 460,300" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
  <!-- Triangle sides colored -->
  <line x1="250" y1="28" x2="40" y2="300" stroke="#1565c0" stroke-width="2.5" stroke-dasharray="7,4"/>
  <line x1="250" y1="28" x2="460" y2="300" stroke="#2e7d32" stroke-width="2.5" stroke-dasharray="7,4"/>
  <line x1="40" y1="300" x2="460" y2="300" stroke="#c62828" stroke-width="2.5" stroke-dasharray="7,4"/>

  <!-- AGENT top -->
  <rect x="185" y="8" width="130" height="52" rx="9" fill="#fce4ec" stroke="#c62828" stroke-width="2.5"/>
  <text x="250" y="31" font-size="14" font-weight="700" fill="#c62828" text-anchor="middle">AGENT</text>
  <text x="250" y="49" font-size="10" fill="#888" text-anchor="middle">What causes disease</text>

  <!-- HOST bottom-left -->
  <rect x="8" y="274" width="120" height="52" rx="9" fill="#e3f2fd" stroke="#1565c0" stroke-width="2.5"/>
  <text x="68" y="297" font-size="14" font-weight="700" fill="#1565c0" text-anchor="middle">HOST</text>
  <text x="68" y="314" font-size="10" fill="#888" text-anchor="middle">Who gets disease</text>

  <!-- ENVIRONMENT bottom-right -->
  <rect x="372" y="274" width="120" height="52" rx="9" fill="#e8f5e9" stroke="#2e7d32" stroke-width="2.5"/>
  <text x="432" y="297" font-size="14" font-weight="700" fill="#2e7d32" text-anchor="middle">ENVIRONMENT</text>
  <text x="432" y="314" font-size="10" fill="#888" text-anchor="middle">Where it occurs</text>

  <!-- TIME center ellipse -->
  <ellipse cx="250" cy="190" rx="62" ry="36" fill="#fff8e1" stroke="#f9a825" stroke-width="2.5"/>
  <text x="250" y="185" font-size="13" font-weight="700" fill="#f57f17" text-anchor="middle">TIME</text>
  <text x="250" y="202" font-size="10" fill="#999" text-anchor="middle">When it unfolds</text>

  <!-- Arrows: TIME → vertices -->
  <defs>
    <marker id="arr" markerWidth="9" markerHeight="9" refX="5" refY="4.5" orient="auto">
      <path d="M0,0 L9,4.5 L0,9 Z" fill="#f9a825"/>
    </marker>
  </defs>
  <line x1="250" y1="154" x2="250" y2="66" stroke="#f9a825" stroke-width="1.8" marker-end="url(#arr)"/>
  <line x1="192" y1="202" x2="118" y2="284" stroke="#f9a825" stroke-width="1.8" marker-end="url(#arr)"/>
  <line x1="308" y1="202" x2="382" y2="284" stroke="#f9a825" stroke-width="1.8" marker-end="url(#arr)"/>

  <!-- Caption -->
  <text x="250" y="335" font-size="9" fill="#94a3b8" text-anchor="middle" font-style="italic">Disease occurs at the intersection of agent, host, environment — unfolding over time</text>
</svg>
</div>
        """, height=420, scrolling=False)

        st.markdown("""
| Element | Definition | Key factors |
|---|---|---|
| **Agent** | The cause of the disease — biological, chemical, physical, or nutritional | Infectivity, pathogenicity, virulence, dose |
| **Host** | The person or animal that harbors the disease | Age, sex, genetics, immunity, behavior, nutritional status |
| **Environment** | External conditions that affect the agent and host interaction | Climate, sanitation, housing, social determinants, season |
| **Time** | The temporal dimension — incubation, duration, trends, seasonality | Incubation period, duration of infectiousness, secular trends |
        """)

        st.info("""
🔑 **Why the triangle matters for intervention:**
Each vertex represents a point of intervention:
- **Agent** → destroy, deactivate, or reduce infectivity (disinfection, antibiotics, food safety)
- **Host** → increase resistance (vaccination, nutrition, behavior change)
- **Environment** → modify conditions that allow transmission (sanitation, housing, vector control)
- **Time** → act early in the natural history; prevent chronic exposure; use time-limited interventions

Public health interventions rarely target just one vertex. A comprehensive response addresses all simultaneously.
        """)

        st.markdown("#### ⚠️ Limitations of the triangle")
        st.warning("""
The epidemiology triangle was designed for **infectious disease with a single identifiable agent**. For chronic disease, mental health, and injury:
- There is rarely a single agent — causes are **multifactorial**
- Social and structural determinants don't fit neatly into "environment"
- The model doesn't capture dose-response relationships or causal pathways

This is why more complex models — the **Web of Causation**, **Rothman's causal pies**, and **DAGs** — were developed as extensions and alternatives.
        """)

        st.divider()
        st.markdown("#### 🧠 Apply the triangle")

        tri_scenarios = [
            {
                "q": "**Scenario 1:** Childhood lead poisoning from old paint in urban housing. Which element does lead-safe housing remediation primarily target?",
                "opts": ["— Select —",
                         "Agent — removes the lead from the environment",
                         "Host — increases resistance in children",
                         "Environment — modifies the condition supporting exposure",
                         "Both agent and environment — they overlap here"],
                "correct": "Both agent and environment — they overlap here",
                "fb_correct": "✅ Correct. Housing remediation removes the lead (agent) by modifying the environment. These vertices blur here — a known limitation of the model. This is also a time intervention: acting before long-term exposure accumulates.",
                "fb_wrong": "❌ Lead remediation modifies both the agent (removes lead) and the environment (changes housing condition). The model's limitation shows here — agent and environment aren't always separable.",
                "key": "tri_q1"
            },
            {
                "q": "**Scenario 2:** During the 1918 influenza pandemic, soldiers living in crowded barracks had dramatically higher attack rates than civilians in rural areas. Using the triangle, what primarily explains the higher transmission in barracks?",
                "opts": ["— Select —",
                         "Agent — the influenza virus was more virulent in military settings",
                         "Host — soldiers had weaker immune systems than civilians",
                         "Environment — crowded indoor living conditions facilitated airborne transmission",
                         "Time — soldiers were exposed at a different point in the epidemic curve"],
                "correct": "Environment — crowded indoor living conditions facilitated airborne transmission",
                "fb_correct": "✅ Correct. The agent (influenza virus) was the same. The hosts (healthy young adults) were actually immunologically comparable. The environment — crowded indoor barracks with poor ventilation — was the primary differentiating factor. Environmental modification (spacing bunks, improving ventilation) was one of the few available interventions. Time also matters: sustained exposure in shared quarters increased cumulative dose.",
                "fb_wrong": "❌ Same virus, comparable hosts. The environmental difference (crowded indoor barracks vs. dispersed rural settings) is what drove higher transmission. Environmental factors are a powerful intervention target for airborne disease.",
                "key": "tri_q2"
            },
            {
                "q": "**Scenario 3:** HIV transmission rates are far higher in sub-Saharan Africa than in Western Europe despite similar viral biology. A public health team wants to prioritize intervention using the triangle. Which element(s) most explain the geographic disparity?",
                "opts": ["— Select —",
                         "Agent — HIV is more virulent in Africa",
                         "Host — genetic susceptibility differs by population",
                         "Environment — structural factors including poverty, gender inequality, limited healthcare access, and concurrent infections that increase susceptibility",
                         "Time — Africa is earlier in the epidemic"],
                "correct": "Environment — structural factors including poverty, gender inequality, limited healthcare access, and concurrent infections that increase susceptibility",
                "fb_correct": "✅ Correct. HIV-1 biology is essentially the same globally. Genetic host factors play a minor role. The environmental vertex captures structural determinants: poverty limiting condom access, gender inequality reducing women's negotiating power, co-infections (HSV-2, STIs) increasing mucosal susceptibility, and limited antiretroviral treatment availability. This illustrates both the triangle's strength (identifies modifiable targets) and its limitation (the 'environment' box is doing a lot of work for what are really complex social structures).",
                "fb_wrong": "❌ HIV viral biology is similar globally. The geographic disparity is driven by structural environmental factors — poverty, gender inequality, healthcare infrastructure, and co-infection rates — all of which are modifiable intervention targets.",
                "key": "tri_q3"
            },
        ]

        for scen in tri_scenarios:
            ans = st.radio(scen["q"], scen["opts"], key=scen["key"])
            if ans == scen["correct"]:
                st.success(scen["fb_correct"])
            elif ans != "— Select —":
                st.error(scen["fb_wrong"])
            st.markdown("")



    elif found_section == "2️⃣ Natural History & Levels of Prevention":
        st.subheader("Natural History of Disease")
        st.markdown("""
The **natural history of disease** describes the progression of a disease process in an individual over time without medical intervention. Understanding it tells us *when* and *how* to intervene most effectively.
        """)

        nh_html = """
<div style="overflow-x:auto;margin:16px 0;">
<div style="display:flex;align-items:stretch;gap:0;min-width:600px;">
  <div style="flex:1;background:#f0fdf4;border:2px solid #16a34a;border-radius:10px 0 0 10px;padding:14px 12px;text-align:center;">
    <div style="font-size:22px;">🌱</div>
    <div style="font-weight:700;font-size:12px;color:#166534;margin:4px 0;">STAGE 1</div>
    <div style="font-weight:700;font-size:13px;color:#166534;">Susceptibility</div>
    <div style="font-size:11px;color:#14532d;margin-top:6px;line-height:1.5;">No disease present. Risk factors accumulate. Host, agent, and environment interact.<br><br><b>Intervention point:</b><br>Primary prevention</div>
  </div>
  <div style="width:28px;display:flex;align-items:center;justify-content:center;background:#e2e8f0;font-size:18px;color:#64748b;">→</div>
  <div style="flex:1;background:#fffbeb;border:2px solid #d97706;padding:14px 12px;text-align:center;">
    <div style="font-size:22px;">🔬</div>
    <div style="font-weight:700;font-size:12px;color:#92400e;margin:4px 0;">STAGE 2</div>
    <div style="font-weight:700;font-size:13px;color:#92400e;">Subclinical Disease</div>
    <div style="font-size:11px;color:#78350f;margin-top:6px;line-height:1.5;">Pathological changes underway. No symptoms yet. Disease detectable by screening tests.<br><br><b>Intervention point:</b><br>Secondary prevention</div>
  </div>
  <div style="width:28px;display:flex;align-items:center;justify-content:center;background:#e2e8f0;font-size:18px;color:#64748b;">→</div>
  <div style="flex:1;background:#fef2f2;border:2px solid #dc2626;padding:14px 12px;text-align:center;">
    <div style="font-size:22px;">🤒</div>
    <div style="font-weight:700;font-size:12px;color:#991b1b;margin:4px 0;">STAGE 3</div>
    <div style="font-weight:700;font-size:13px;color:#991b1b;">Clinical Disease</div>
    <div style="font-size:11px;color:#7f1d1d;margin-top:6px;line-height:1.5;">Signs and symptoms appear. Patient seeks care. Diagnosis made.<br><br><b>Intervention point:</b><br>Secondary & tertiary prevention</div>
  </div>
  <div style="width:28px;display:flex;align-items:center;justify-content:center;background:#e2e8f0;font-size:18px;color:#64748b;">→</div>
  <div style="flex:1;background:#f5f3ff;border:2px solid #7c3aed;border-radius:0 10px 10px 0;padding:14px 12px;text-align:center;">
    <div style="font-size:22px;">♿</div>
    <div style="font-weight:700;font-size:12px;color:#5b21b6;margin:4px 0;">STAGE 4</div>
    <div style="font-weight:700;font-size:13px;color:#5b21b6;">Resolution</div>
    <div style="font-size:11px;color:#4c1d95;margin-top:6px;line-height:1.5;">Recovery, disability, or death. Chronic disease = prolonged stage 3/4.<br><br><b>Intervention point:</b><br>Tertiary prevention</div>
  </div>
</div>
<div style="margin-top:8px;text-align:center;font-size:11px;color:#718096;">
  ← The <b>incubation period</b> begins after infection and ends at symptom onset (within stage 2). The <b>latency period</b> in chronic disease (exposure to detectable pathology) does not map cleanly onto a single stage boundary.
</div>
</div>"""
        st.markdown(nh_html, unsafe_allow_html=True)

        st.divider()
        st.subheader("Levels of Prevention")
        st.markdown("Prevention is categorized by *when* in the natural history it acts — before disease, during subclinical stages, or after diagnosis.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 🟢 Primary Prevention")
            st.markdown("""
**Goal:** Prevent disease from occurring at all.

**Acts during:** Susceptibility stage — before any pathological change.

**Mechanisms:**
- Remove or reduce risk factors
- Strengthen host resistance
- Alter the environment

**Examples:**
- Vaccination (before exposure)
- Seat belts, helmets
- Tobacco cessation programs
- Water fluoridation
- Healthy diet and exercise promotion
- Occupational safety standards

**Epidemiologic measure:** Incidence — did we prevent new cases?
            """)

        with col2:
            st.markdown("#### 🟡 Secondary Prevention")
            st.markdown("""
**Goal:** Detect and treat disease early, before symptoms appear or before it progresses.

**Acts during:** Subclinical stage — disease is present but not yet symptomatic.

**Mechanisms:**
- Screening programs
- Early diagnosis
- Prompt treatment to halt progression

**Examples:**
- Mammography for breast cancer
- Pap smear for cervical cancer
- Blood pressure screening
- Newborn metabolic screening
- HIV testing in high-risk populations

**Key concept:** Effectiveness requires that early treatment changes the outcome — screening is only valuable if catching disease earlier improves prognosis.

**Epidemiologic measure:** Prevalence detection, lead-time bias
            """)

        with col3:
            st.markdown("#### 🔴 Tertiary Prevention")
            st.markdown("""
**Goal:** Reduce disability and improve function in those who already have established disease.

**Acts during:** Clinical disease and resolution stages.

**Mechanisms:**
- Treatment to prevent complications
- Rehabilitation
- Disease management programs
- Palliative care

**Examples:**
- Cardiac rehab after heart attack
- Insulin management for diabetics
- Physical therapy after stroke
- Cancer pain management
- Support groups for chronic illness

**Epidemiologic measure:** Case fatality rate, disability-adjusted life years (DALYs), quality of life

**Quaternary prevention:** Protect patients from unnecessary or harmful interventions (overdiagnosis, overtreatment) — increasingly recognized as a fourth level.
            """)

        st.divider()
        with st.expander("⚠️ Lead-Time Bias — The Hidden Trap of Screening"):
            st.markdown("""
**Lead-time bias** occurs when screening appears to extend survival, but only because disease is detected earlier — not because treatment is more effective.

**Example:** Without screening, a cancer is detected at symptoms and patient lives 2 more years (total disease duration: 10 years). With screening, cancer detected 3 years earlier — patient lives 5 more years from diagnosis. But total lifespan is unchanged. It *looks* like survival improved (5 > 2 years), but no extra time was gained — we just moved the diagnosis earlier.

**The fix:** Use mortality rates (not survival time) to evaluate screening effectiveness, or compare age-standardized disease-specific mortality in screened vs. unscreened populations.
            """)

        with st.expander("📋 Levels of Prevention — Quick Reference"):
            st.markdown("""
| Level | Stage | Goal | Example | Key Measure |
|---|---|---|---|---|
| **Primary** | Susceptibility | Prevent disease occurrence | Vaccination, seat belts | Incidence |
| **Secondary** | Subclinical | Early detection & treatment | Mammography, Pap smear | Prevalence detected, CFR |
| **Tertiary** | Clinical/Resolution | Reduce disability & complications | Cardiac rehab, diabetes management | DALYs, QoL, CFR |
| **Quaternary** | Any | Prevent over-medicalization | Avoiding unnecessary surgery | Iatrogenic harm rates |
            """)

    # ── SECTION 2: CHAIN OF INFECTION ──
    elif found_section == "3️⃣ Chain of Infection & Infectious Disease":
        st.subheader("The Chain of Infection")
        st.markdown("""
Infectious disease transmission requires an unbroken **chain of infection** — six linked components. Breaking any single link prevents transmission. This framework guides outbreak investigation and infection control.
        """)

        chain_html = """
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:16px 0;overflow-x:auto;">
<div style="display:flex;align-items:center;gap:0;min-width:700px;justify-content:center;">

  <div style="text-align:center;width:110px;">
    <div style="background:#fef2f2;border:2px solid #dc2626;border-radius:50%;width:80px;height:80px;margin:0 auto;display:flex;align-items:center;justify-content:center;font-size:28px;">🦠</div>
    <div style="font-weight:700;font-size:11px;color:#dc2626;margin-top:6px;">1. AGENT</div>
    <div style="font-size:10px;color:#555;margin-top:2px;">Pathogen causing disease<br>(bacteria, virus, prion)</div>
  </div>
  <div style="font-size:20px;color:#94a3b8;padding:0 4px;margin-bottom:20px;">→</div>

  <div style="text-align:center;width:110px;">
    <div style="background:#fff7ed;border:2px solid #ea580c;border-radius:50%;width:80px;height:80px;margin:0 auto;display:flex;align-items:center;justify-content:center;font-size:28px;">🏠</div>
    <div style="font-weight:700;font-size:11px;color:#ea580c;margin-top:6px;">2. RESERVOIR</div>
    <div style="font-size:10px;color:#555;margin-top:2px;">Where agent lives & multiplies<br>(human, animal, soil, water)</div>
  </div>
  <div style="font-size:20px;color:#94a3b8;padding:0 4px;margin-bottom:20px;">→</div>

  <div style="text-align:center;width:110px;">
    <div style="background:#fefce8;border:2px solid #ca8a04;border-radius:50%;width:80px;height:80px;margin:0 auto;display:flex;align-items:center;justify-content:center;font-size:28px;">🚪</div>
    <div style="font-weight:700;font-size:11px;color:#ca8a04;margin-top:6px;">3. PORTAL OF EXIT</div>
    <div style="font-size:10px;color:#555;margin-top:2px;">How agent leaves reservoir<br>(respiratory, fecal, blood)</div>
  </div>
  <div style="font-size:20px;color:#94a3b8;padding:0 4px;margin-bottom:20px;">→</div>

  <div style="text-align:center;width:110px;">
    <div style="background:#f0fdf4;border:2px solid #16a34a;border-radius:50%;width:80px;height:80px;margin:0 auto;display:flex;align-items:center;justify-content:center;font-size:28px;">✈️</div>
    <div style="font-weight:700;font-size:11px;color:#16a34a;margin-top:6px;">4. MODE OF TRANSMISSION</div>
    <div style="font-size:10px;color:#555;margin-top:2px;">How agent travels<br>(droplet, contact, vector)</div>
  </div>
  <div style="font-size:20px;color:#94a3b8;padding:0 4px;margin-bottom:20px;">→</div>

  <div style="text-align:center;width:110px;">
    <div style="background:#eff6ff;border:2px solid #2563eb;border-radius:50%;width:80px;height:80px;margin:0 auto;display:flex;align-items:center;justify-content:center;font-size:28px;">🚪</div>
    <div style="font-weight:700;font-size:11px;color:#2563eb;margin-top:6px;">5. PORTAL OF ENTRY</div>
    <div style="font-size:10px;color:#555;margin-top:2px;">How agent enters host<br>(respiratory, GI, skin break)</div>
  </div>
  <div style="font-size:20px;color:#94a3b8;padding:0 4px;margin-bottom:20px;">→</div>

  <div style="text-align:center;width:110px;">
    <div style="background:#fdf4ff;border:2px solid #9333ea;border-radius:50%;width:80px;height:80px;margin:0 auto;display:flex;align-items:center;justify-content:center;font-size:28px;">🧑</div>
    <div style="font-weight:700;font-size:11px;color:#9333ea;margin-top:6px;">6. SUSCEPTIBLE HOST</div>
    <div style="font-size:10px;color:#555;margin-top:2px;">Individual who can be infected<br>(lacking immunity)</div>
  </div>

</div>
<div style="text-align:center;margin-top:10px;font-size:11px;color:#718096;">Breaking any single link prevents transmission ✂️</div>
</div>"""
        st.markdown(chain_html, unsafe_allow_html=True)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Modes of Transmission")
            st.markdown("""
**Direct transmission:**
- *Direct contact:* Physical touching, sexual contact, biting
- *Droplet spread:* Large droplets (>5μm) travel short distances (<1m) — classically influenza
- *Short-range aerosol/respiratory particle transmission:* A spectrum of particle sizes; inhalation at close range and in poorly ventilated spaces is now recognized as the primary route for COVID-19, which does not fit neatly into the historic droplet/airborne binary
- *Direct inoculation:* Needlestick, animal bite

**Indirect transmission:**
- *Airborne:* Droplet nuclei (<5μm) suspended in air, travel >1m — TB, measles, chickenpox
- *Vehicle-borne:* Contaminated food, water, blood products — Salmonella, hepatitis A
- *Vector-borne:* Biological (agent replicates in vector — malaria, dengue) or mechanical (agent carried without replication — housefly + feces)
- *Fomite:* Contaminated inanimate objects — C. diff on hospital surfaces

**Vertical transmission:** Mother to child — in utero, during delivery, breastfeeding (HIV, CMV, syphilis)
            """)

        with col2:
            st.markdown("#### Breaking the Chain — Intervention Points")
            st.markdown("""
| Link | Intervention examples |
|---|---|
| **Agent** | Antivirals, antibiotics, pasteurization |
| **Reservoir** | Animal control, treating infected individuals |
| **Portal of exit** | Respiratory precautions, wound coverage |
| **Transmission** | Hand hygiene, masks, condoms, vector control |
| **Portal of entry** | PPE, food safety, safe injection practices |
| **Host** | Vaccination, chemoprophylaxis, nutrition |

**Most effective interventions** target multiple links simultaneously. Vaccination addresses the host *and* reduces the reservoir when coverage is high enough to achieve herd immunity.
            """)

        st.divider()
        with st.expander("🦟 Vector-Borne Disease — Biological vs. Mechanical"):
            st.markdown("""
**Biological vector:** The pathogen undergoes development or multiplication inside the vector before transmission. The vector is essential to the disease cycle.
- Mosquito → malaria (Plasmodium matures in mosquito)
- Tick → Lyme disease (Borrelia multiplies in tick)
- Sandfly → leishmaniasis

**Mechanical vector:** The vector physically carries the pathogen without it replicating or developing. The vector is just transport.
- Housefly carrying Salmonella from feces to food
- Cockroach contaminating food surfaces

**Why it matters for control:** Biological vectors require control of the vector itself (insecticides, bed nets, habitat modification). Mechanical transmission is controlled more by environmental hygiene and food safety.
            """)

    # ── SECTION 3: HERD IMMUNITY & R₀ ──
    elif found_section == "4️⃣ Herd Immunity & R₀":
        st.subheader("Herd Immunity & the Basic Reproduction Number (R₀)")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### R₀ — Basic Reproduction Number")
            st.markdown("""
**Definition:** The average number of secondary cases generated by one infectious individual in a completely susceptible population, in the absence of intervention.

**R₀ = β × κ × D**
Where:
- **β** = probability of transmission per contact
- **κ** = contact rate (contacts per unit time)
- **D** = duration of infectiousness

**Interpretation:**
- R₀ < 1 → epidemic dies out
- R₀ = 1 → endemic equilibrium
- R₀ > 1 → epidemic grows

**Examples:**
| Disease | R₀ estimate (approximate, context-dependent) |
|---|---|
| Measles | 12–18 |
| Chickenpox | 8–12 |
| COVID-19 (original Wuhan strain) | 2–3 |
| Influenza (seasonal) | 1.2–1.4 |
| Ebola | 1.5–2.5 |
| Smallpox | 5–7 |

**Important:** These are approximate ranges from specific studies and populations — treat them as illustrative order-of-magnitude estimates, not fixed biological constants. R₀ values shift with population density, contact patterns, immunity levels, and pathogen evolution (e.g., Omicron R₀ estimates ranged from 8–15, far above the original strain).

**Important distinction:** R₀ is a property of the pathogen-population interaction, not of the pathogen alone. The same virus can have different R₀ in different settings.
            """)

        with col2:
            st.markdown("### Herd Immunity")
            st.markdown("""
**Definition:** Indirect protection of susceptible individuals when a sufficient proportion of the population is immune — immune individuals act as a barrier to transmission, breaking chains of infection.

**Herd immunity threshold (HIT):**

**HIT = 1 − (1/R₀)**

Examples:
| Disease | R₀ | HIT |
|---|---|---|
| Measles | 15 | 93% |
| Polio | 6 | 83% |
| COVID-19 | 3 | 67% |
| Influenza | 1.3 | 23% |

**Sources of herd immunity:**
- Vaccination (the preferred and safest route — confers immunity without disease burden)
- Natural infection (contributes to population immunity but at the cost of disease, death, and long-term complications)
- Combination of both

**Key nuance:** HIT assumes random mixing. In real populations with clustered susceptibles (vaccine refusers in communities), local outbreaks can occur even with overall population immunity above the HIT.
            """)

        st.divider()

        # Interactive R₀ calculator
        st.markdown("### 🔢 Interactive R₀ & Herd Immunity Calculator")
        col_a, col_b = st.columns(2)
        with col_a:
            r0_val = st.slider("R₀ value", 1.0, 20.0, 3.0, 0.1, key="r0_slider")
            hit = (1 - 1/r0_val) * 100
            st.metric("Herd Immunity Threshold", f"{round(hit, 1)}%")
            st.caption(f"Need ≥{round(hit,1)}% immune to stop epidemic spread")
        with col_b:
            current_immunity = st.slider("Current population immunity (%)", 0, 100, 60, 1, key="immunity_slider")
            effective_r = r0_val * (1 - current_immunity/100)
            st.metric("Effective R (Rₑ)", round(effective_r, 2),
                      delta="epidemic growing" if effective_r > 1 else "epidemic declining")
            if effective_r > 1:
                st.error(f"Rₑ = {round(effective_r,2)} > 1 — epidemic will grow. Need {round(hit-current_immunity,1)}% more immune.")
            elif effective_r < 1:
                st.success(f"Rₑ = {round(effective_r,2)} < 1 — epidemic will decline. Herd immunity achieved.")
            else:
                st.warning("Rₑ ≈ 1 — endemic equilibrium.")

        with st.expander("🔢 The math behind effective R"):
            st.markdown(f"""
**Effective reproduction number (Rₑ):** R₀ × (proportion susceptible)
= {r0_val} × (1 − {current_immunity/100}) = **{round(effective_r, 3)}**

When {current_immunity}% of the population is immune, only {100-current_immunity}% are susceptible.
Each infectious person can only transmit to susceptible contacts, so the effective number of secondary cases is reduced from R₀ = {r0_val} to Rₑ = {round(effective_r, 2)}.

**Herd immunity threshold:** 1 − 1/R₀ = 1 − 1/{r0_val} = **{round(hit,1)}%**
            """)

        st.divider()
        with st.expander("⚠️ Limitations and Misconceptions"):
            st.markdown("""
**R₀ is not fixed:** It changes with behavior, interventions, population density, and viral evolution. Delta and Omicron variants of COVID-19 had much higher R₀ than the original strain.

**Herd immunity is not binary:** It's a threshold, and population immunity is heterogeneous. Even above the HIT, local clusters of susceptibles can sustain outbreaks.

**Vaccine-induced vs. infection-induced immunity:** Both contribute to population immunity and both count toward the herd immunity threshold. Vaccination is the preferred and safest route — it confers protection without the disease burden, mortality, and long-term complications of natural infection. Policy discussions sometimes consider the contribution of prior infection to population immunity, particularly when vaccination coverage is incomplete.

**Waning immunity:** As immunity wanes (through time or new variants), effective R rises. Booster programs exist to maintain Rₑ < 1.
            """)

    # ── SECTION 4: OUTBREAK INVESTIGATION 10 STEPS ──
    elif found_section == "5️⃣ Outbreak Investigation — The 10 Steps":
        st.subheader("The 10-Step Outbreak Investigation")
        st.markdown("""
When a cluster of cases is reported, epidemiologists follow a systematic process. The steps are not strictly sequential — several happen simultaneously — but the framework ensures nothing is missed.
        """)

        STEPS = [
            ("1", "Prepare for Field Work", "#1e40af", "🧳",
             "Before going anywhere: review the literature on the suspected disease, consult with experts, assemble supplies and lab materials, arrange logistics, ensure legal and ethical authority to investigate.",
             "Key questions: What is already known about this disease? What lab tests are needed? Who has authority to implement control measures?"),
            ("2", "Establish the Existence of an Outbreak", "#1d4ed8", "📊",
             "Determine whether the number of cases exceeds the expected (baseline) level. Compare reported cases to historical rates for the same time, place, and population.",
             "An epidemic threshold is one way to flag unusual case counts — one common approach uses mean + 2 standard deviations of historical baseline data, though different surveillance systems use different methods (percentile ranks, moving averages, CUSUM). Not all clusters are true outbreaks — some are artifacts of improved surveillance or reporting."),
            ("3", "Verify the Diagnosis", "#2563eb", "🔬",
             "Confirm that cases represent the disease suspected. Review clinical findings, lab results, and case histories. Prevent false alarms from lab error, reporting artifacts, or misdiagnosis.",
             "Contact the lab, review pathology, interview clinicians. Ensure the diagnostic criteria are applied consistently across cases."),
            ("4", "Construct a Working Case Definition", "#3b82f6", "📋",
             "Define who counts as a case. A case definition has four components: person (who), place (where), time (when), and clinical criteria (what symptoms/lab findings).",
             """**Case definition levels:**
- **Confirmed:** Laboratory-confirmed disease
- **Probable:** Clinical criteria met + epidemiological link
- **Suspected:** Some clinical criteria, no lab or epi link

Case definitions should be sensitive early in an outbreak (to find cases) and refined later as the picture clarifies. A definition that is too narrow misses cases; too broad includes non-cases."""),
            ("5", "Find Cases Systematically — Case Finding", "#60a5fa", "🔍",
             "Actively search for additional cases beyond those already reported. Passive surveillance misses cases. Active case finding uses lab records, hospital records, school absenteeism data, and direct community outreach.",
             "Ask every case: 'Do you know anyone else who is ill?' Review emergency department logs, lab submissions, and pharmacy records for syndromic patterns."),
            ("6", "Describe the Outbreak — Descriptive Epidemiology", "#7c3aed", "📈",
             "Characterize cases by person, place, and time. Draw an epidemic curve. Map cases geographically. Describe demographics of cases (age, sex, occupation, behaviors).",
             """**The epidemic curve** tells you:
- *Shape:* Point source vs. propagated vs. mixed
- *Timing:* When did the outbreak peak?
- *Incubation period:* Width of the curve approximates the incubation range
- *Are new cases still occurring?* Is the outbreak ongoing?"""),
            ("7", "Develop Hypotheses", "#9333ea", "💡",
             "Based on descriptive data, generate hypotheses about the source, mode of transmission, and risk factors. What do cases have in common? Where were they? What did they eat or do?",
             "Hypotheses should be specific and testable. 'The contaminated chicken salad served at Table 3 at the Saturday event caused illness' is a testable hypothesis. 'Something at the event caused illness' is not."),
            ("8", "Evaluate Hypotheses — Analytic Epidemiology", "#a855f7", "🧮",
             "Test hypotheses using analytic study designs. In a defined cohort (e.g., event attendees), calculate attack rates and RRs for each potential vehicle. In a community outbreak, conduct a case-control study.",
             """**Cohort approach:** All attendees form the cohort. Calculate food-specific attack rates (AR exposed vs. AR unexposed) and RR for each food item. The food with the highest RR and lowest p-value is the likely vehicle.

**Case-control approach:** Cases (ill people) vs. controls (well people). Calculate OR for each exposure. Used when the at-risk population cannot be enumerated."""),
            ("9", "Implement Control Measures", "#dc2626", "🛑",
             "Control measures should begin as soon as possible — do not wait for a complete investigation. Implement measures targeted at the most likely source while investigation continues.",
             """**Control by chain of infection link:**
- Source: recall contaminated product, close restaurant, treat reservoir
- Transmission: handwashing advisories, isolation, quarantine
- Host: prophylaxis, vaccination, advisories to at-risk populations

Document all control measures and their timing for the outbreak report."""),
            ("10", "Communicate Findings", "#16a34a", "📣",
             "Write an outbreak investigation report. Communicate findings to public health authorities, the media (if appropriate), affected communities, and the scientific literature.",
             "The report should include: background, methods, results (epidemic curve, attack rates, OR/RR tables), conclusions, and recommendations. Timely communication prevents additional cases and builds public trust."),
        ]

        for step in STEPS:
            num, title, color, icon, desc, detail = step
            with st.expander(f"{icon} **Step {num}: {title}**"):
                st.markdown(f"**{desc}**")
                st.markdown(detail)

        st.divider()
        with st.expander("📋 Quick Reference — The 10 Steps"):
            steps_table = "| Step | Name | Key Action |\n|---|---|---|\n"
            for step in STEPS:
                num, title, color, icon, desc, detail = step
                first_sentence = desc.split(".")[0] + "."
                steps_table += f"| **{num}** | {title} | {first_sentence} |\n"
            st.markdown(steps_table)

    # ── SECTION 5: PICO ──
    elif found_section == "6️⃣ PICO Framework":
        st.subheader("The PICO Framework — Structuring Research Questions")
        st.markdown("""
A well-formed research question is the foundation of a good study. The **PICO framework** breaks any clinical or epidemiologic question into four components that map directly onto study design decisions.
        """)

        pico_html = """
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0;">
  <div style="background:#eff6ff;border-left:5px solid #2563eb;border-radius:6px;padding:16px;">
    <div style="font-size:32px;font-weight:900;color:#1d4ed8;">P</div>
    <div style="font-weight:700;font-size:16px;color:#1e40af;margin-bottom:6px;">Population / Patient</div>
    <div style="font-size:13px;color:#1e3a8a;line-height:1.6;">
      Who are you studying? Define the target population by relevant characteristics.<br><br>
      <b>Questions to ask:</b> What age group? What disease or condition? What setting (community, hospital)? What exposure history?<br><br>
      <b>Example:</b> "Adults aged 40–65 with newly diagnosed Type 2 diabetes in primary care settings"
    </div>
  </div>
  <div style="background:#f0fdf4;border-left:5px solid #16a34a;border-radius:6px;padding:16px;">
    <div style="font-size:32px;font-weight:900;color:#16a34a;">I</div>
    <div style="font-weight:700;font-size:16px;color:#166534;margin-bottom:6px;">Intervention / Exposure</div>
    <div style="font-size:13px;color:#14532d;line-height:1.6;">
      What is the exposure, intervention, or risk factor of interest?<br><br>
      <b>Questions to ask:</b> What is the treatment or exposure? What is the dose/intensity? What is the timing?<br><br>
      <b>Example:</b> "Structured dietary counseling program (12 sessions over 6 months)"
    </div>
  </div>
  <div style="background:#fffbeb;border-left:5px solid #d97706;border-radius:6px;padding:16px;">
    <div style="font-size:32px;font-weight:900;color:#d97706;">C</div>
    <div style="font-weight:700;font-size:16px;color:#92400e;margin-bottom:6px;">Comparison / Control</div>
    <div style="font-size:13px;color:#78350f;line-height:1.6;">
      What is the alternative? What are you comparing the intervention/exposure to?<br><br>
      <b>Questions to ask:</b> Usual care? No treatment? A different exposure level? A different drug?<br><br>
      <b>Example:</b> "Standard written dietary advice (one session at diagnosis)"
    </div>
  </div>
  <div style="background:#fdf4ff;border-left:5px solid #9333ea;border-radius:6px;padding:16px;">
    <div style="font-size:32px;font-weight:900;color:#9333ea;">O</div>
    <div style="font-weight:700;font-size:16px;color:#6b21a8;margin-bottom:6px;">Outcome</div>
    <div style="font-size:13px;color:#581c87;line-height:1.6;">
      What are you measuring? What change are you hoping to detect?<br><br>
      <b>Questions to ask:</b> Primary outcome? Secondary outcomes? How measured? Over what time period?<br><br>
      <b>Example:</b> "HbA1c reduction at 12 months; secondary: body weight, medication adherence"
    </div>
  </div>
</div>"""
        st.markdown(pico_html, unsafe_allow_html=True)

        st.info("""
**Full PICO question from the example:**
"In adults aged 40–65 with newly diagnosed Type 2 diabetes in primary care (P), does structured dietary counseling over 6 months (I), compared to standard written advice (C), reduce HbA1c at 12 months (O)?"

This question immediately suggests: **RCT** (if assigning intervention) or **cohort study** (if observational). The outcome (HbA1c) suggests continuous measurement. The comparison group is defined. The population is specific enough to guide sampling.
        """)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### PICO → Study Design")
            st.markdown("""
| PICO component | Study design implication |
|---|---|
| **I = assigned intervention** | RCT |
| **I = observed exposure** | Cohort or case-control |
| **O = rare disease** | Case-control |
| **O = incidence over time** | Cohort |
| **O = prevalence** | Cross-sectional |
| **P = defined event attendees** | Cohort (outbreak) |
| **C = no comparison** | Descriptive only |
            """)

        with col2:
            st.markdown("#### PICO Extensions")
            st.markdown("""
**PICOT** — adds **T (Time):** Over what follow-up period?

**PICOS** — adds **S (Study design):** What design is appropriate?

**PECO** — used in environmental/observational epi:
- P = Population
- E = Exposure (not intervention)
- C = Comparison
- O = Outcome

**Systematic reviews use PICO** to define eligibility criteria — studies are included only if their P, I, C, and O match the review's question.
            """)

        with st.expander("✏️ PICO Builder — Write Your Own"):
            st.markdown("Use the fields below to structure a research question:")
            p = st.text_input("P — Population/Patient:", placeholder="e.g., HIV-positive adults on ART in sub-Saharan Africa", key="pico_p")
            i = st.text_input("I — Intervention/Exposure:", placeholder="e.g., daily high-dose vitamin D supplementation", key="pico_i")
            c = st.text_input("C — Comparison:", placeholder="e.g., placebo", key="pico_c")
            o = st.text_input("O — Outcome:", placeholder="e.g., CD4 count at 6 months; tuberculosis incidence", key="pico_o")
            if p and i and c and o:
                st.success(f"**Your PICO question:**\n\nIn {p} ({p and 'P'}), does {i} ({i and 'I'}), compared to {c} ({c and 'C'}), affect {o} ({o and 'O'})?")
                st.markdown(f"> *\"In {p}, does {i}, compared to {c}, affect {o}?\"*")

    st.markdown("---")
    st.markdown("*Strong epidemiologists ask the right question before choosing the right method.*")


# ==================================================
# MODULE 1: STUDY DESIGNS
# ==================================================
elif current_page == "study_designs":
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


        st.divider()
        st.markdown("#### ⚠️ The Ecologic Fallacy")
        st.error("""
**The ecologic fallacy** occurs when you draw conclusions about *individuals* from data measured at the *group (ecologic) level*.

**Classic example:** Countries with higher fat consumption have higher rates of breast cancer. But this does not mean that *individuals* who eat more fat get more breast cancer — other individual-level factors (age, genetics, reproductive history) may explain the pattern, and the group-level correlation may be confounded in ways that can't be detected from group data alone.

**Why it happens:** Ecologic data measure *averages* or *rates* for groups. Individual variation within those groups is invisible. An association that exists at the group level may disappear, reverse, or be spurious when studied at the individual level.

**When ecologic studies are appropriate:**
- Generating hypotheses for individual-level studies
- Studying exposures that are truly group-level (e.g., air quality laws, water fluoridation policy)
- Surveillance when individual data are unavailable

**The rule:** Ecologic correlations cannot substitute for individual-level associations. Always note the unit of analysis and be explicit about what level the conclusions apply to.
        """)

        epi_scenarios = [
            {
                "q": "**Scenario 1:** A study finds that counties with more fast food restaurants per capita have higher obesity rates. A researcher concludes that people who eat at fast food restaurants are more likely to be obese. This is:",
                "opts": ["— Select —",
                         "A valid causal inference from ecologic data",
                         "The ecologic fallacy — individual-level conclusions drawn from group-level data",
                         "Confounding by income",
                         "Selection bias"],
                "correct": "The ecologic fallacy — individual-level conclusions drawn from group-level data",
                "fb_correct": "✅ Correct. County-level restaurant density and county-level obesity rate are group measures. Whether individuals who eat at fast food chains are more obese than individuals in those same counties who don't — that's an individual-level question the ecologic data cannot answer.",
                "fb_wrong": "❌ The primary error is drawing individual-level conclusions from group-level data. The unit of analysis is counties; the inference is about individuals. Income confounding may exist too, but the structural error is the ecologic fallacy.",
                "key": "eco_q1"
            },
            {
                "q": "**Scenario 2:** Countries with higher chocolate consumption per capita win more Nobel Prizes per capita. A journalist concludes that eating chocolate makes people smarter. What is the most likely explanation?",
                "opts": ["— Select —",
                         "Chocolate consumption causes cognitive improvement — a valid biological mechanism",
                         "The ecologic fallacy — wealth (a country-level variable) drives both chocolate consumption and Nobel Prize rates",
                         "Reverse causation — winning Nobel Prizes causes people to eat more chocolate",
                         "This is a valid randomized finding"],
                "correct": "The ecologic fallacy — wealth (a country-level variable) drives both chocolate consumption and Nobel Prize rates",
                "fb_correct": "✅ Correct. This is a famous real example (Messerli, 2012 — published somewhat tongue-in-cheek). Wealthy countries have both higher per-capita chocolate consumption AND higher per-capita Nobel Prizes, because wealth drives investment in education and research. The correlation is ecologic confounding. No individual-level data support a chocolate-intelligence link. Classic ecologic fallacy + confounding at the group level.",
                "fb_wrong": "❌ Wealthy nations have both higher chocolate consumption AND greater investment in education and research. A country-level correlation driven by a common cause (wealth) cannot support individual-level causal claims. This is the ecologic fallacy compounded by ecologic confounding.",
                "key": "eco_q2"
            },
            {
                "q": "**Scenario 3:** A researcher uses county-level data to study the relationship between opioid prescribing rates and opioid overdose death rates. She finds a strong positive correlation (r = 0.74). She wants to use this to estimate the individual-level risk of overdose among people who receive opioid prescriptions. Is this appropriate?",
                "opts": ["— Select —",
                         "Yes — the correlation is strong, so the inference is valid",
                         "No — this is an ecologic correlation; it cannot directly estimate individual-level risk",
                         "Yes — county-level data are more reliable than individual data",
                         "It depends on whether the data are from the same year"],
                "correct": "No — this is an ecologic correlation; it cannot directly estimate individual-level risk",
                "fb_correct": "✅ Correct. The county-level correlation is informative for policy (target high-prescribing counties for intervention) but cannot estimate the individual-level probability of overdose among people who receive a prescription. That requires individual-level data with person-level exposure and outcome. The ecologic correlation is real and useful — but it answers a different question than individual risk.",
                "fb_wrong": "❌ Ecologic correlations — even strong ones — cannot substitute for individual-level associations. The same county-level pattern could exist even if the highest-prescribing doctors serve the lowest-risk patients. Individual data are needed for individual-level inference.",
                "key": "eco_q3"
            },
        ]

        for scen in epi_scenarios:
            ans = st.radio(scen["q"], scen["opts"], key=scen["key"])
            if ans == scen["correct"]:
                st.success(scen["fb_correct"])
            elif ans != "— Select —":
                st.error(scen["fb_wrong"])
            st.markdown("")


    elif section == "2️⃣ Design Selector":
        st.subheader("Design Selector")
        st.markdown("""
Work through the decision tree to identify the correct study design, characterize your data types, and determine the appropriate statistical test.
        """)

        # ── Step 1: Experimental vs Observational ──
        q1 = st.radio("**Step 1.** Is this an experimental study where the researcher assigns exposure?",
                       ["— Select —", "Yes — researcher assigns", "No — observational"], key="ds_q1")

        design_identified = None

        if q1 == "Yes — researcher assigns":
            st.success("**Design: Randomized Controlled Trial (RCT)** — researcher controls exposure assignment. Randomization distributes known and unknown confounders.")
            design_identified = "RCT"

        elif q1 == "No — observational":
            q2 = st.radio("**Step 2.** What is the unit of analysis?",
                           ["— Select —", "Individuals", "Groups or populations (countries, cities, time periods)"], key="ds_q2")
            if q2 == "Groups or populations (countries, cities, time periods)":
                st.success("**Design: Ecological Study** — exposure and outcome measured at the group level. Cannot establish individual-level causation. Beware the ecological fallacy.")
                design_identified = "Ecological"
            elif q2 == "Individuals":
                q3 = st.radio("**Step 3.** How were participants sampled?",
                               ["— Select —",
                                "By exposure status (then followed to outcome)",
                                "By outcome / disease status (then looked back at exposure)",
                                "Neither — random sample or whole population at one time",
                                "Cases only — each compared to themselves at a different time"],
                               key="ds_q3")
                if q3 == "By exposure status (then followed to outcome)":
                    st.success("**Design: Cohort Study** — grouped by exposure, then followed to outcome. Prospective (going forward) or retrospective (historical records), but always exposure → outcome in logic.")
                    design_identified = "Cohort"
                elif q3 == "By outcome / disease status (then looked back at exposure)":
                    st.success("**Design: Case-Control Study** — cases (have disease) and controls (don't) identified, then past exposure assessed. Always retrospective in logic.")
                    design_identified = "Case-Control"
                elif q3 == "Neither — random sample or whole population at one time":
                    st.success("**Design: Cross-Sectional Study** — exposure and outcome measured simultaneously. No temporal ordering possible.")
                    design_identified = "Cross-Sectional"
                elif q3 == "Cases only — each compared to themselves at a different time":
                    st.success("**Design: Case-Crossover Study** — each case serves as their own control. Exposure during a hazard period vs. a control period for the same person.")
                    design_identified = "Case-Crossover"

        # ── Data Type & Statistical Test Section ──
        if design_identified:
            st.divider()
            st.markdown("### 📊 Step 2 — Characterize Your Data")
            st.markdown("The statistical test you use depends on the **measurement level** of your outcome and exposure. Work through the questions below.")

            # Data type explainer
            with st.expander("📖 Data Type Reference — Click to review before answering"):
                st.markdown("""
**Nominal (Categorical, Unordered)**
Categories with no inherent order. Cannot do arithmetic on them.
- Binary: two categories (disease yes/no, exposed yes/no, male/female)
- Multinomial: three or more unordered categories (blood type A/B/AB/O, race/ethnicity)
- Examples: diagnosis present/absent, vaccination status, cause of death

**Ordinal**
Categories with a meaningful order but unequal or unknown intervals between levels.
- You know the rank, not the distance between ranks
- Examples: pain scale (mild/moderate/severe), education level (some HS / HS / college / graduate), cancer stage (I/II/III/IV), Likert scales

**Discrete (Count)**
Whole numbers representing counts of things. Cannot be fractional.
- Examples: number of hospitalizations, number of sexual partners, parity (number of births), number of cigarettes per day
- Often modeled using Poisson or negative binomial regression (modeling assumption, not an inherent property of the data — actual count distributions vary)

**Continuous**
Can take any value within a range, including decimals. Measured, not counted.
- Interval: equal intervals, but no true zero (temperature in °C, calendar year)
- Ratio: equal intervals AND a true zero (blood pressure, BMI, age, income, height, weight)
- Examples: systolic BP (mmHg), BMI, HbA1c (%), serum cholesterol (mg/dL)

**Key rule:** Continuous data can always be collapsed into ordinal or nominal categories (e.g., BMI → obese/not obese), but you lose information. Nominal data cannot be made continuous.
                """)

            col_a, col_b = st.columns(2)
            with col_a:
                outcome_type = st.selectbox(
                    "**Outcome (dependent variable) data type:**",
                    ["— Select —",
                     "Binary (yes/no, present/absent)",
                     "Nominal (3+ unordered categories)",
                     "Ordinal (ordered categories)",
                     "Discrete / Count",
                     "Continuous (interval or ratio)"],
                    key=f"ds_outcome_{design_identified}"
                )
            with col_b:
                exposure_type = st.selectbox(
                    "**Exposure (independent variable) data type:**",
                    ["— Select —",
                     "Binary (2 groups)",
                     "Categorical (3+ groups, unordered)",
                     "Ordinal (ordered categories or dose levels)",
                     "Continuous"],
                    key=f"ds_exposure_{design_identified}"
                )

            if outcome_type != "— Select —" and exposure_type != "— Select —":
                st.divider()
                st.markdown("### 🧮 Recommended Measure of Association & Statistical Test")

                # ── Lookup table: design + outcome + exposure → measure + test ──
                def get_recommendation(design, outcome, exposure):
                    # Normalize
                    o = outcome.split(" ")[0].lower()   # binary / nominal / ordinal / discrete / continuous
                    e = exposure.split(" ")[0].lower()  # binary / categorical / ordinal / continuous

                    recs = []

                    if design in ["Cohort", "RCT"]:
                        if o == "binary":
                            if e == "binary":
                                recs = [
                                    ("Risk Ratio (RR)", "Ratio of cumulative incidence (or IR if varying follow-up → IRR)", "#1d4ed8"),
                                    ("Chi-square test", "Tests independence of exposure and outcome in a 2×2 table", "#166534"),
                                    ("Log-binomial regression", "Multivariable RR with covariate adjustment", "#4b5563"),
                                    ("Poisson regression", "Alternative to log-binomial; more stable convergence", "#4b5563"),
                                ]
                            elif e in ["categorical", "ordinal"]:
                                recs = [
                                    ("RR per category", "Calculate RR for each exposure level vs. reference", "#1d4ed8"),
                                    ("Chi-square test (trend)", "Cochran-Armitage for ordered exposure categories", "#166534"),
                                    ("Logistic/log-binomial regression", "Handles multiple categories + adjustment", "#4b5563"),
                                ]
                            elif e == "continuous":
                                recs = [
                                    ("RR per unit increase", "From Poisson or log-binomial regression", "#1d4ed8"),
                                    ("Dose-response analysis", "Test biological gradient (Bradford Hill)", "#166534"),
                                    ("Logistic regression (if OR acceptable)", "More common in practice despite OR vs. RR issue", "#4b5563"),
                                ]
                        elif o == "continuous":
                            if e == "binary":
                                recs = [
                                    ("Mean difference", "Difference in means between exposed and unexposed", "#1d4ed8"),
                                    ("Independent samples t-test", "Compares two group means; assumes normality", "#166534"),
                                    ("Mann-Whitney U test", "Non-parametric alternative if distribution is skewed", "#4b5563"),
                                    ("Linear regression", "Adjusts for confounders; gives beta coefficient", "#4b5563"),
                                ]
                            elif e in ["categorical", "ordinal"]:
                                recs = [
                                    ("ANOVA (Analysis of Variance)", "Compares means across 3+ groups", "#1d4ed8"),
                                    ("Kruskal-Wallis test", "Non-parametric alternative to ANOVA", "#4b5563"),
                                    ("Linear regression with dummies", "Multivariable adjustment", "#4b5563"),
                                ]
                            elif e == "continuous":
                                recs = [
                                    ("Pearson correlation (r)", "Strength and direction of linear association", "#1d4ed8"),
                                    ("Spearman correlation (ρ)", "Non-parametric; for non-normal or ordinal data", "#4b5563"),
                                    ("Linear regression", "Predicts outcome from exposure; β = change per unit E", "#166534"),
                                ]
                        elif o == "discrete":
                            recs = [
                                ("Incidence Rate Ratio (IRR)", "Ratio of event rates using person-time", "#1d4ed8"),
                                ("Poisson regression", "Models count outcomes; assumes variance = mean", "#166534"),
                                ("Negative binomial regression", "If overdispersion present (variance > mean)", "#4b5563"),
                            ]
                        elif o == "ordinal":
                            recs = [
                                ("Proportional odds ratio", "From ordinal logistic regression", "#1d4ed8"),
                                ("Mann-Whitney U / Kruskal-Wallis", "Non-parametric comparison of ordered outcomes", "#4b5563"),
                                ("Ordinal logistic regression", "Multivariable adjustment; proportional odds model", "#166534"),
                            ]

                    elif design == "Case-Control":
                        if o == "binary":
                            if e == "binary":
                                recs = [
                                    ("Odds Ratio (OR)", "Cross-product ratio from 2×2 table. OR is the only valid measure from case-control data.", "#1d4ed8"),
                                    ("Chi-square test / Fisher's exact", "Fisher's when any cell < 5", "#166534"),
                                    ("Unconditional logistic regression", "Unmatched case-control; multivariable OR", "#4b5563"),
                                    ("Conditional logistic regression", "Matched case-control design", "#4b5563"),
                                ]
                            elif e in ["categorical", "ordinal"]:
                                recs = [
                                    ("OR per category", "Reference category vs. each exposure level", "#1d4ed8"),
                                    ("Chi-square for trend", "Cochran-Armitage for dose-response in case-control", "#166634"),
                                    ("Logistic regression", "Handles multiple categories + confounders", "#4b5563"),
                                ]
                            elif e == "continuous":
                                recs = [
                                    ("OR per unit increase", "From logistic regression with continuous exposure", "#1d4ed8"),
                                    ("Logistic regression", "Treat exposure as continuous predictor", "#166534"),
                                    ("Quartile/quintile analysis", "Categorize continuous exposure for dose-response", "#4b5563"),
                                ]

                    elif design == "Cross-Sectional":
                        if o == "binary":
                            if e == "binary":
                                recs = [
                                    ("Prevalence Ratio (PR)", "Ratio of prevalences. Preferred over OR for cross-sectional data with common outcomes.", "#1d4ed8"),
                                    ("Chi-square test", "Tests independence", "#166534"),
                                    ("Modified Poisson regression", "Gives PR directly; more appropriate than logistic regression for prevalent outcomes", "#4b5563"),
                                    ("Logistic regression (gives OR)", "Often used but gives OR not PR — important distinction when outcome is common", "#9a3412"),
                                ]
                            elif e in ["categorical", "ordinal"]:
                                recs = [
                                    ("PR per category", "Modified Poisson regression for each exposure level", "#1d4ed8"),
                                    ("Chi-square test", "Overall test of independence across categories", "#166534"),
                                ]
                            elif e == "continuous":
                                recs = [
                                    ("PR per unit increase", "From modified Poisson regression", "#1d4ed8"),
                                    ("Logistic regression (OR)", "More common in practice", "#4b5563"),
                                ]
                        elif o == "continuous":
                            recs = [
                                ("Mean difference or Pearson r", "Depending on exposure type", "#1d4ed8"),
                                ("Linear regression", "Adjusts for confounders", "#166534"),
                            ]

                    elif design == "Case-Crossover":
                        recs = [
                            ("Odds Ratio (OR)", "Conditional logistic regression; each case is own control", "#1d4ed8"),
                            ("Conditional logistic regression", "Matched analysis required — each case matched to their own control periods", "#166534"),
                        ]

                    elif design == "Ecological":
                        recs = [
                            ("Pearson correlation (r)", "Between group-level exposure and outcome rates", "#1d4ed8"),
                            ("Linear regression", "Predicts group outcome rate from group exposure level", "#4b5563"),
                            ("⚠️ Cannot infer individual risk", "Ecological fallacy — group-level associations may not hold at individual level", "#991b1b"),
                        ]

                    return recs

                recs = get_recommendation(design_identified, outcome_type, exposure_type)

                if recs:
                    for measure, explanation, color in recs:
                        st.markdown(f"""
<div style="display:flex;gap:12px;align-items:flex-start;padding:10px 14px;
     background:#f8fafc;border-left:4px solid #{color[1:] if color.startswith('#') else color};
     border-radius:0 6px 6px 0;margin:6px 0;">
  <div style="min-width:0;flex:1;">
    <div style="font-weight:700;font-size:14px;color:{color};">{measure}</div>
    <div style="font-size:12px;color:#4b5563;margin-top:2px;">{explanation}</div>
  </div>
</div>""", unsafe_allow_html=True)

                    # Special notes
                    st.divider()
                    st.markdown("#### 📌 Important Notes for Your Design")

                    if design_identified == "Case-Control":
                        st.warning("**Case-control studies can only produce OR** — never RR or PR. This is because the proportion of cases to controls is set by the researcher (sampling fraction), so absolute risks cannot be calculated from these data.")
                    if design_identified == "Cross-Sectional" and "Binary" in outcome_type:
                        st.info("**PR vs. OR in cross-sectional studies:** When outcome prevalence exceeds 10%, OR diverges substantially from PR. Modified Poisson regression directly estimates PR and is the preferred approach. Logistic regression (which gives OR) is commonly used but technically incorrect for prevalent outcomes.")
                    if design_identified == "Ecological":
                        st.error("**Ecological fallacy reminder:** Statistical associations at the population level cannot be used to draw conclusions about individuals. Always state this limitation explicitly in ecological study reports.")
                    if design_identified in ["Cohort", "RCT"] and "Continuous" in outcome_type and "Binary" in exposure_type:
                        st.info("**Normality check:** t-tests assume approximately normal distribution of the outcome (or large sample). For skewed outcomes or small samples, use Mann-Whitney U. For very large samples (n > 100/group), t-tests are robust to non-normality due to the Central Limit Theorem.")

                    # Multivariable note
                    st.markdown("""
> **Multivariable adjustment:** All measures above have regression equivalents that allow adjustment for confounders. The regression model for each outcome type:
> - Binary outcome → Logistic regression (OR) or log-binomial/Poisson (RR/PR)
> - Continuous outcome → Linear regression (β coefficient = mean difference per unit exposure)
> - Count/rate outcome → Poisson or negative binomial regression (IRR)
> - Time-to-event with censoring → Cox proportional hazards regression (HR)
                    """)

                    # ── Confounding Adjustment Section ──────────────────────
                    st.divider()
                    st.markdown("### 🔀 Adjusting for Confounding in This Design")
                    st.markdown("Choose the stage at which you want to control confounding:")

                    conf_stage = st.radio(
                        "Control at:",
                        ["Design stage (before data collection)", "Analysis stage (after data collection)"],
                        key=f"conf_stage_{design_identified}",
                        horizontal=True
                    )

                    if conf_stage == "Design stage (before data collection)":
                        st.markdown("#### 🏗️ Design-Stage Confounding Control")

                        design_conf_methods = {
                            "RCT": [
                                ("🎲 Randomization", "BEST METHOD for RCTs",
                                 "Randomly assign participants to treatment or control. Randomization distributes both measured AND unmeasured confounders equally across groups — the only method that controls for unknown confounders. With sufficient sample size, all confounders balance by chance.",
                                 "#166534", "#f0fdf4"),
                                ("🔲 Stratified randomization", "When known confounders must be balanced",
                                 "Randomize within strata of an important confounder (e.g., randomize men and women separately). Ensures equal distribution of that confounder even in small samples. Also called 'block randomization by stratum.'",
                                 "#1d4ed8", "#eff6ff"),
                                ("🎭 Blinding", "Reduces performance and detection bias",
                                 "Single-blind: participants unaware of assignment. Double-blind: both participants and assessors unaware. Prevents differential behavior and outcome measurement based on exposure status.",
                                 "#7c3aed", "#fdf4ff"),
                            ],
                            "Cohort": [
                                ("🚧 Restriction", "Limit study to one level of the confounder",
                                 "If age is a confounder, restrict to a narrow age range (e.g., 40–50 year olds only). Simple and effective — but reduces sample size and limits generalizability. Cannot control for confounders you haven't thought of.",
                                 "#166534", "#f0fdf4"),
                                ("🤝 Matching", "Match exposed and unexposed on confounder values",
                                 "For each exposed person, find an unexposed person with the same value of the confounder (e.g., same age, sex). Controls confounding by design. Requires matched analysis (conditional logistic regression or McNemar's test). Cannot match on too many variables (matching failure).",
                                 "#1d4ed8", "#eff6ff"),
                                ("🔲 Stratified sampling", "Sample equal proportions from confounder strata",
                                 "Ensure equal representation of confounder levels in exposed and unexposed groups at recruitment. Less common than restriction or matching.",
                                 "#7c3aed", "#fdf4ff"),
                            ],
                            "Case-Control": [
                                ("🤝 Matching", "Most common design-stage method in case-control studies",
                                 "Match each case to one or more controls on the confounder (e.g., same age ±2 years, same sex). Eliminates the matched variable as a confounder — it cannot differ between cases and controls by design. CRITICAL: matched design requires matched analysis (conditional logistic regression). Unmatched analysis of matched data gives biased results.",
                                 "#166534", "#f0fdf4"),
                                ("🚧 Restriction", "Restrict cases and controls to one level of confounder",
                                 "Recruit only women, or only non-smokers. Controls that confounder completely but limits generalizability and may reduce the number of eligible cases.",
                                 "#1d4ed8", "#eff6ff"),
                            ],
                            "Cross-Sectional": [
                                ("🚧 Restriction", "Limit enrollment to one level of the confounder",
                                 "Enroll only one age group, one sex, one occupation. Controls that confounder but limits external validity.",
                                 "#166534", "#f0fdf4"),
                                ("🔲 Stratified sampling", "Ensure balanced confounder distribution",
                                 "Sample equal proportions from each stratum of the confounder. Less common in cross-sectional surveys.",
                                 "#1d4ed8", "#eff6ff"),
                            ],
                            "Case-Crossover": [
                                ("🧍 Self-matching (inherent to design)", "Each case is their own control — time-invariant confounders are automatically controlled",
                                 "The case-crossover design inherently controls for all stable between-person confounders (sex, genetics, baseline health, socioeconomic status, smoking history) because each person is compared only to themselves. This is the primary advantage of the design.",
                                 "#166534", "#f0fdf4"),
                            ],
                            "Ecological": [
                                ("⚠️ Limited design-stage options", "Ecological studies have very limited ability to control confounding",
                                 "Since data are aggregate (group-level), individual-level confounding cannot be controlled by design. You can restrict to groups that are similar on key confounders (e.g., only high-income countries), but this limits generalizability and still cannot address unmeasured ecological confounders.",
                                 "#991b1b", "#fef2f2"),
                            ],
                        }

                        methods = design_conf_methods.get(design_identified, [])
                        for icon_title, subtitle, explanation, color, bg in methods:
                            st.markdown(f"""
<div style="background:{bg};border-left:4px solid {color};border-radius:0 8px 8px 0;
     padding:14px 16px;margin:8px 0;">
  <div style="font-weight:700;font-size:14px;color:{color};">{icon_title}</div>
  <div style="font-size:12px;font-weight:600;color:{color};opacity:0.8;margin:2px 0 6px 0;">{subtitle}</div>
  <div style="font-size:13px;color:#374151;line-height:1.6;">{explanation}</div>
</div>""", unsafe_allow_html=True)

                    else:  # Analysis stage
                        st.markdown("#### 📊 Analysis-Stage Confounding Control")
                        st.markdown("These methods are applied after data collection and work for all observational designs.")

                        analysis_methods = [
                            ("📊 Stratified Analysis (Mantel-Haenszel)",
                             "Stratify data by the confounder and calculate a pooled, confounder-adjusted estimate",
                             """**How it works:** Divide your data into strata of the confounder (e.g., smokers and non-smokers separately). Calculate the measure of association within each stratum. Pool the stratum-specific estimates using Mantel-Haenszel weighting.

**How to detect confounding:** Compare crude estimate to MH-adjusted estimate. If they differ by >10%, the variable is a meaningful confounder.

**How to detect effect modification:** Compare stratum-specific estimates to each other. If they differ substantially, there is effect modification — report separately, don't pool.

**Best for:** Binary confounders or confounders with few levels. Becomes unwieldy with many confounders (stratification table explosion).

**In EpiLab:** The Stratified Analysis tool in Measures of Association → Section 3 demonstrates this with three preset scenarios.""",
                             "#1d4ed8"),
                            ("📈 Multivariable Regression",
                             "Include confounders as covariates in a regression model alongside the exposure",
                             """**How it works:** Add confounders as additional predictors in the regression model. The coefficient for your exposure is then adjusted for all included confounders simultaneously.

**Which model:**
- Binary outcome → Logistic regression (OR) or log-binomial regression (RR/PR)
- Continuous outcome → Linear regression (β coefficient)
- Count/rate outcome → Poisson regression (IRR)
- Time-to-event → Cox proportional hazards regression (HR)

**Advantages:** Can adjust for many confounders simultaneously. Can include continuous confounders without categorizing. Standard in modern epidemiology.

**Limitation:** Only controls for measured confounders. Residual confounding from unmeasured variables remains. Model must be correctly specified.""",
                             "#166534"),
                            ("⚖️ Propensity Score Methods",
                             "Model the probability of exposure, then use scores to create comparable groups",
                             """**How it works:** Build a model predicting the probability of being exposed given all measured covariates. This propensity score summarizes all confounders into one number.

**Uses:**
- *Matching:* Match exposed and unexposed with similar propensity scores
- *Weighting (IPTW):* Weight observations by inverse of propensity — creates a pseudo-population where exposure is independent of confounders
- *Stratification:* Stratify by propensity score quintiles

**Best for:** Situations with many confounders relative to sample size, or when you want to mimic randomization in observational data.

**Limitation:** Only balances measured confounders — unmeasured confounders still cause bias.""",
                             "#7c3aed"),
                            ("🔢 Standardization",
                             "Apply a standard population's weights to remove the effect of a confounder",
                             """**How it works (direct):** Apply your population's age-specific rates to a standard population's age structure. The resulting standardized rate is what your population's rate would be if it had the standard age structure.

**How it works (indirect):** Apply a reference population's age-specific rates to your population's age structure to calculate expected events. SMR = Observed/Expected.

**Best for:** Controlling confounding by age (or other categorical variables) in rate comparisons across populations.

**Limitation:** Can only standardize for variables with known stratum-specific rates. Does not adjust for individual-level confounders in the same way regression does.""",
                             "#b45309"),
                            ("🔬 Instrumental Variable Analysis",
                             "Advanced: use a variable that affects exposure but not outcome directly",
                             """**How it works:** Find an instrument — a variable that (1) is associated with the exposure, (2) only affects the outcome through the exposure, and (3) is not associated with confounders. Use it to estimate a causal effect.

**Classic examples:** Mendelian randomization uses genetic variants as instruments. Distance to a specialist as instrument for receiving specialist care.

**Best for:** Unmeasured confounding — when you cannot measure the confounders. The instrument must meet strict assumptions.

**Limitation:** Valid instruments are rare and hard to find. Weak instruments introduce bias. Assumptions are often untestable.""",
                             "#4b5563"),
                        ]

                        for title, subtitle, explanation, color in analysis_methods:
                            with st.expander(f"**{title}** — {subtitle}"):
                                st.markdown(explanation)

                        st.info("""
**Choosing between methods:**
- **1–2 categorical confounders:** Stratification (Mantel-Haenszel) is transparent and easy to explain
- **3+ confounders or continuous confounders:** Multivariable regression
- **Many confounders, want to mimic RCT:** Propensity score matching or IPTW
- **Age confounding in rate comparisons:** Standardization (direct or indirect)
- **Unmeasured confounding is a concern:** Instrumental variable analysis (if instrument available)

**What no method can do:** Control for confounders that were never measured. This is why *a priori* identification of confounders using DAGs before data collection is essential.
                        """)

                else:
                    st.info("This combination is unusual — check your data type selections.")





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



        st.divider()
        st.subheader("Randomization — Why It Works")
        st.markdown("""
**Randomization** is the defining feature of an RCT, and its power is frequently misunderstood. It does not simply create similar groups — it does something more fundamental: it **distributes all confounders — measured and unmeasured — equally between groups by chance**.

In observational studies, the investigator must identify and control for every confounder. In a randomized trial, randomization handles all confounders *automatically* — including ones the investigator doesn't know about and couldn't measure.
        """)

        st.markdown("""
| | Observational study | Randomized trial |
|---|---|---|
| **How groups are formed** | Participants choose exposure (or it's assigned by circumstance) | Exposure assigned by random chance |
| **Confounding** | Present — must be identified and controlled statistically | Distributed equally by chance (on average) |
| **Unmeasured confounders** | Remain a threat; cannot be adjusted | Balanced by randomization |
| **Causal inference** | Limited — association, not proven causation | Stronger basis for causal inference |
| **Feasibility** | Can study exposures that can't be randomized (smoking, poverty) | Cannot randomize harmful or impossible exposures |
        """)

        st.warning("""
⚠️ **Randomization works in expectation, not perfectly in every trial.** In small trials, chance imbalance in key variables can still occur. This is why researchers check baseline characteristics (Table 1 in a paper) and why small trials are less convincing than large ones. It's also why randomization is done in blocks and may be stratified by important variables (e.g., sex, site) to improve balance.
        """)

        st.divider()
        st.subheader("Blinding")
        st.markdown("""
**Blinding** (also called *masking*) prevents knowledge of treatment assignment from influencing outcomes, behavior, or assessment. Unblinded trials are susceptible to multiple biases.
        """)

        st.markdown("""
| Type | Who is blinded | What bias it prevents |
|---|---|---|
| **Single-blind** | Participant only | Placebo effect; behavioral change based on knowing assignment |
| **Double-blind** | Participant + investigator / assessor | Above + differential assessment and recording bias |
| **Triple-blind** | Participant + investigator + data analyst | Above + analysis bias (selective reporting, outcome switching) |
| **Open-label** | No blinding | Appropriate when blinding is impossible (e.g., surgical vs. no surgery) |
        """)

        st.info("""
🔑 **The placebo effect** is the measurable, real improvement in health outcomes that occurs when a participant *believes* they are receiving an effective treatment. It is not imaginary — it produces documented physiological changes. Double-blinding controls for it by ensuring both arms receive an indistinguishable treatment (active drug vs. identical-appearing placebo).

**Performance bias:** Even without outcome reporting issues, knowing your treatment assignment can change behavior. Participants who know they received the "real" treatment may comply more, exercise more, or seek less supplementary care — all of which confound the result.
        """)

        st.divider()
        st.subheader("Intent-to-Treat Analysis")
        st.markdown("""
**Intent-to-treat (ITT) analysis** analyzes participants in the group to which they were *randomized*, regardless of whether they actually received the treatment, complied with it, or dropped out.

**Why it matters:** If you only analyze participants who completed treatment as assigned (*per-protocol* analysis), you reintroduce selection bias — the kind of people who comply with or tolerate treatment may differ systematically from those who don't.

| Analysis approach | Who is analyzed | Preserves randomization? | Best answers |
|---|---|---|---|
| **Intent-to-treat** | Everyone as randomized | ✅ Yes | "Does offering this treatment improve outcomes in the real world?" |
| **Per-protocol** | Only those who completed treatment as assigned | ❌ No — selection bias re-enters | "Does the treatment work in ideal conditions?" |
| **As-treated** | Grouped by what they actually received | ❌ No — observational | Similar to per-protocol; susceptible to confounding |
        """)

        st.success("""
✅ **The rule:** ITT is the primary analysis in most RCTs because it mirrors real-world effectiveness. Per-protocol is often reported as a secondary/sensitivity analysis. A trial that only reports per-protocol results, especially when dropout rates differ between arms, should be read with caution.
        """)

        rct_q = st.radio(
            "**Concept check:** In a trial of a new antidepressant, 30% of participants in the treatment group stop taking the medication due to side effects. An ITT analysis would:",
            ["— Select —",
             "Exclude the dropouts — they didn't actually receive the full treatment",
             "Include all randomized participants in their original groups, including dropouts",
             "Move dropouts to the control group since they didn't receive treatment",
             "Run a separate analysis for completers only and report that as the primary result"],
            key="rct_q_itt"
        )
        if rct_q == "Include all randomized participants in their original groups, including dropouts":
            st.success("✅ Correct. ITT includes everyone as randomized. The 30% who stopped taking medication are still analyzed in the treatment group. This preserves the randomization and gives an estimate of real-world effectiveness — including the reality that 30% won't tolerate the drug. Excluding them would introduce selection bias (those who drop out due to side effects are likely different from those who don't).")
        elif rct_q != "— Select —":
            st.error("❌ ITT means 'analyze as randomized.' Everyone stays in their assigned group regardless of compliance or dropout. Excluding non-compliers reintroduces selection bias and answers a different question (efficacy in ideal conditions) rather than effectiveness in the real world.")


    st.markdown("---")
    st.markdown("*Strong epidemiologists think structurally before computing.*")


# ==================================================
# MODULE 1: BIAS
# ==================================================
elif current_page == "bias":
    st.title("⚠️ Bias")
    st.markdown("Bias is a **systematic error** that leads to an incorrect estimate of the association between exposure and outcome. Unlike random error, bias does not average out with larger sample sizes.")
    st.info("**Key principle:** Bias operates in one direction — it either inflates or deflates the true association. Recognizing the type of bias helps you predict whether your result is likely an over- or under-estimate.")

    bias_section = st.radio("Section:", [
        "1️⃣ Selection Bias",
        "2️⃣ Information Bias",
        "3️⃣ Bias Direction Exercise",
        "4️⃣ Reliability & Validity"
    ], horizontal=True)
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

    if bias_section == "4️⃣ Reliability & Validity":
        st.subheader("Reliability & Validity in Epidemiologic Measurement")
        st.markdown("""
Measurement quality underpins the entire enterprise of epidemiology. Every exposure, outcome, and covariate in a study is a measurement — and every measurement can be wrong in two distinct ways: it can be **unreliable** (inconsistent) or **invalid** (systematically off target). Understanding the difference is essential for evaluating study quality and interpreting results.
        """)

        bulls_html = """
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:16px 0;">
  <div style="font-weight:700;font-size:14px;color:#1a202c;margin-bottom:12px;text-align:center;">The Bulls-Eye Diagram — Reliability vs. Validity</div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:16px;text-align:center;">
    <div>
      <svg viewBox="0 0 120 120" width="120" height="120" xmlns="http://www.w3.org/2000/svg">
        <circle cx="60" cy="60" r="56" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
        <circle cx="60" cy="60" r="40" fill="#fecaca" stroke="#dc2626" stroke-width="2"/>
        <circle cx="60" cy="60" r="24" fill="#ef4444" stroke="#dc2626" stroke-width="2"/>
        <circle cx="60" cy="60" r="8" fill="#7f1d1d"/>
        <circle cx="60" cy="60" r="4" fill="#fbbf24"/>
        <circle cx="62" cy="58" r="5" fill="#1d4ed8"/><circle cx="58" cy="62" r="5" fill="#1d4ed8"/>
        <circle cx="61" cy="61" r="5" fill="#1d4ed8"/><circle cx="59" cy="59" r="5" fill="#1d4ed8"/>
      </svg>
      <div style="font-weight:700;color:#166534;font-size:12px;margin-top:4px;">✅ Reliable &amp; Valid</div>
      <div style="font-size:11px;color:#555;margin-top:2px;">Tight cluster on target<br>Low random + systematic error</div>
    </div>
    <div>
      <svg viewBox="0 0 120 120" width="120" height="120" xmlns="http://www.w3.org/2000/svg">
        <circle cx="60" cy="60" r="56" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
        <circle cx="60" cy="60" r="40" fill="#fecaca" stroke="#dc2626" stroke-width="2"/>
        <circle cx="60" cy="60" r="24" fill="#ef4444" stroke="#dc2626" stroke-width="2"/>
        <circle cx="60" cy="60" r="8" fill="#7f1d1d"/>
        <circle cx="60" cy="60" r="4" fill="#fbbf24"/>
        <circle cx="90" cy="28" r="5" fill="#1d4ed8"/><circle cx="92" cy="30" r="5" fill="#1d4ed8"/>
        <circle cx="88" cy="32" r="5" fill="#1d4ed8"/><circle cx="91" cy="29" r="5" fill="#1d4ed8"/>
      </svg>
      <div style="font-weight:700;color:#b45309;font-size:12px;margin-top:4px;">⚠️ Reliable, Not Valid</div>
      <div style="font-size:11px;color:#555;margin-top:2px;">Tight cluster, off target<br>Low random, high systematic error</div>
    </div>
    <div>
      <svg viewBox="0 0 120 120" width="120" height="120" xmlns="http://www.w3.org/2000/svg">
        <circle cx="60" cy="60" r="56" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
        <circle cx="60" cy="60" r="40" fill="#fecaca" stroke="#dc2626" stroke-width="2"/>
        <circle cx="60" cy="60" r="24" fill="#ef4444" stroke="#dc2626" stroke-width="2"/>
        <circle cx="60" cy="60" r="8" fill="#7f1d1d"/>
        <circle cx="60" cy="60" r="4" fill="#fbbf24"/>
        <circle cx="40" cy="50" r="5" fill="#1d4ed8"/><circle cx="75" cy="40" r="5" fill="#1d4ed8"/>
        <circle cx="55" cy="80" r="5" fill="#1d4ed8"/><circle cx="80" cy="72" r="5" fill="#1d4ed8"/>
      </svg>
      <div style="font-weight:700;color:#0369a1;font-size:12px;margin-top:4px;">🔵 Valid, Not Reliable</div>
      <div style="font-size:11px;color:#555;margin-top:2px;">Scattered around target<br>High random, low systematic error</div>
    </div>
    <div>
      <svg viewBox="0 0 120 120" width="120" height="120" xmlns="http://www.w3.org/2000/svg">
        <circle cx="60" cy="60" r="56" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
        <circle cx="60" cy="60" r="40" fill="#fecaca" stroke="#dc2626" stroke-width="2"/>
        <circle cx="60" cy="60" r="24" fill="#ef4444" stroke="#dc2626" stroke-width="2"/>
        <circle cx="60" cy="60" r="8" fill="#7f1d1d"/>
        <circle cx="60" cy="60" r="4" fill="#fbbf24"/>
        <circle cx="88" cy="32" r="5" fill="#1d4ed8"/><circle cx="96" cy="70" r="5" fill="#1d4ed8"/>
        <circle cx="78" cy="96" r="5" fill="#1d4ed8"/><circle cx="92" cy="48" r="5" fill="#1d4ed8"/>
      </svg>
      <div style="font-weight:700;color:#dc2626;font-size:12px;margin-top:4px;">❌ Neither</div>
      <div style="font-size:11px;color:#555;margin-top:2px;">Scattered and off target<br>High random + systematic error</div>
    </div>
  </div>
  <div style="text-align:center;font-size:11px;color:#718096;margin-top:8px;">🟡 = True value (bulls-eye center) &nbsp;|&nbsp; 🔵 = Measurements</div>
</div>"""
        st.markdown(bulls_html, unsafe_allow_html=True)

        st.info("**The fundamental rule:** A measure can be reliable without being valid — it consistently misses the target. But a measure cannot be valid without being reliable — if measurements are scattered randomly, they cannot consistently hit the truth.")

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🎯 Validity")
            st.markdown("""
**Definition:** Does the measure capture what it intends to measure? A valid measure has minimal systematic error (bias).

**Face validity** — Does it *look* like it measures the concept? Weakest form. Example: asking "do you exercise?" has face validity for physical activity.

**Content validity** — Does it cover all relevant dimensions? Example: a diet questionnaire asking only about fat intake has poor content validity for "overall diet quality."

**Criterion validity** — How well does it correlate with a gold standard?
- *Concurrent:* Measured at the same time as the gold standard
- *Predictive:* Measured earlier predicts a future outcome

**Construct validity** — Does the measure behave as theoretically expected? Example: a stress scale should correlate positively with cortisol and negatively with wellbeing.

**Internal validity** — Study results are unbiased for the study population. Threatened by bias and confounding.

**External validity (generalizability)** — Findings apply to other populations and settings. Requires internal validity first.
            """)

        with col2:
            st.markdown("### 🔁 Reliability")
            st.markdown("""
**Definition:** Does the measure produce consistent results under the same conditions? A reliable measure has minimal random error.

**Test-retest reliability** — Same measure, same subjects, two time points. Assumes true value hasn't changed. Example: blood pressure one week apart in stable patients.

**Inter-rater reliability** — Agreement between different raters. Critical for subjective assessments. Measured with **Kappa (κ)**.

**Intra-rater reliability** — Same rater, different occasions. Even skilled pathologists may classify the same slide differently weeks later.

**Internal consistency** — For multi-item scales: do all items measure the same construct? Measured with **Cronbach's alpha (α)**. Acceptable: α ≥ 0.70; excellent: α ≥ 0.90.

**Parallel forms** — Two versions of a test produce similar scores. Relevant for standardized tests with multiple forms.
            """)

        st.divider()
        st.markdown("### 📊 Measuring Agreement — Kappa Statistic (κ)")
        st.markdown("""
Percent agreement is inflated by **chance agreement**. Two raters randomly classifying with 50/50 probability agree ~50% of the time by chance. Kappa corrects for this.

**κ = (P_observed − P_expected) ÷ (1 − P_expected)**
        """)

        kappa_html = """
<div style="overflow-x:auto;margin:12px 0;">
<table style="border-collapse:collapse;width:100%;font-size:13px;">
  <tr style="background:#1e40af;color:white;">
    <th style="padding:10px 14px;text-align:left;">Kappa (κ)</th>
    <th style="padding:10px 14px;text-align:left;">Interpretation</th>
    <th style="padding:10px 14px;text-align:left;">Context</th>
  </tr>
  <tr style="background:#f8fafc;"><td style="padding:9px 14px;border-bottom:1px solid #e2e8f0;">&lt; 0.00</td><td style="padding:9px 14px;border-bottom:1px solid #e2e8f0;color:#dc2626;font-weight:600;">Less than chance</td><td style="padding:9px 14px;border-bottom:1px solid #e2e8f0;">Systematic disagreement</td></tr>
  <tr><td style="padding:9px 14px;border-bottom:1px solid #e2e8f0;">0.00–0.20</td><td style="padding:9px 14px;border-bottom:1px solid #e2e8f0;color:#dc2626;">Slight</td><td style="padding:9px 14px;border-bottom:1px solid #e2e8f0;">Unacceptable for research</td></tr>
  <tr style="background:#f8fafc;"><td style="padding:9px 14px;border-bottom:1px solid #e2e8f0;">0.21–0.40</td><td style="padding:9px 14px;border-bottom:1px solid #e2e8f0;color:#d97706;">Fair</td><td style="padding:9px 14px;border-bottom:1px solid #e2e8f0;">Marginal — needs improvement</td></tr>
  <tr><td style="padding:9px 14px;border-bottom:1px solid #e2e8f0;">0.41–0.60</td><td style="padding:9px 14px;border-bottom:1px solid #e2e8f0;color:#d97706;font-weight:600;">Moderate</td><td style="padding:9px 14px;border-bottom:1px solid #e2e8f0;">Acceptable for exploratory work</td></tr>
  <tr style="background:#f8fafc;"><td style="padding:9px 14px;border-bottom:1px solid #e2e8f0;">0.61–0.80</td><td style="padding:9px 14px;border-bottom:1px solid #e2e8f0;color:#16a34a;font-weight:600;">Substantial</td><td style="padding:9px 14px;border-bottom:1px solid #e2e8f0;">Good — acceptable for most research</td></tr>
  <tr><td style="padding:9px 14px;">0.81–1.00</td><td style="padding:9px 14px;color:#166534;font-weight:700;">Almost perfect</td><td style="padding:9px 14px;">Excellent — gold standard quality</td></tr>
</table>
</div>"""
        st.markdown(kappa_html, unsafe_allow_html=True)

        st.markdown("#### 🔢 Interactive Kappa Calculator")
        st.markdown("Enter the 2×2 agreement table between two raters:")
        k1, k2 = st.columns(2)
        with k1:
            a = st.number_input("Both rate POSITIVE (a)", min_value=0, value=45, key="kap_a")
            b = st.number_input("Rater 1 positive, Rater 2 negative (b)", min_value=0, value=10, key="kap_b")
        with k2:
            c = st.number_input("Rater 1 negative, Rater 2 positive (c)", min_value=0, value=5, key="kap_c")
            d = st.number_input("Both rate NEGATIVE (d)", min_value=0, value=40, key="kap_d")

        N = a + b + c + d
        if N > 0:
            p_obs = (a + d) / N
            p_exp = ((a + b) / N * (a + c) / N) + ((c + d) / N * (b + d) / N)
            if p_exp < 1:
                kappa = (p_obs - p_exp) / (1 - p_exp)
                pct_agree = round(p_obs * 100, 1)
                kappa_r = round(kappa, 3)
                if kappa < 0: interp = "Less than chance ❌"
                elif kappa < 0.21: interp = "Slight ❌"
                elif kappa < 0.41: interp = "Fair ⚠️"
                elif kappa < 0.61: interp = "Moderate ✅"
                elif kappa < 0.81: interp = "Substantial ✅"
                else: interp = "Almost perfect ✅"
                c1, c2, c3 = st.columns(3)
                c1.metric("% Agreement", f"{pct_agree}%")
                c2.metric("κ (Kappa)", kappa_r)
                c3.metric("Interpretation", interp)
                with st.expander("🔢 Show the math"):
                    st.markdown(f"""
**N = {N}**
P_observed = ({a} + {d}) / {N} = **{round(p_obs,4)}**
P_expected = [{a+b}/{N} × {a+c}/{N}] + [{c+d}/{N} × {b+d}/{N}] = **{round(p_exp,4)}**
κ = ({round(p_obs,4)} − {round(p_exp,4)}) / (1 − {round(p_exp,4)}) = **{kappa_r}** → {interp}

Simple % agreement was {pct_agree}%. Kappa adjusts for {round(p_exp*100,1)}% expected by chance.
                    """)

        st.divider()
        st.markdown("### 🔗 Connection to Misclassification Bias")
        conn_html = """
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:12px 0;">
  <div style="background:#fef2f2;border-left:4px solid #dc2626;border-radius:6px;padding:14px;">
    <div style="font-weight:700;color:#dc2626;margin-bottom:8px;">Low Reliability → Non-Differential Misclassification</div>
    <div style="font-size:12px;color:#7f1d1d;line-height:1.7;">
      Random measurement error equal across groups → same error rate in cases and controls.<br><br>
      <b>Effect:</b> Always biases RR/OR <b>toward null</b> (attenuates the true association).<br><br>
      <b>Example:</b> Self-report physical activity questionnaire with poor test-retest reliability misclassifies active/inactive equally regardless of disease status. True protective effect is underestimated.
    </div>
  </div>
  <div style="background:#fffbeb;border-left:4px solid #d97706;border-radius:6px;padding:14px;">
    <div style="font-weight:700;color:#92400e;margin-bottom:8px;">Invalid Measure → Differential Misclassification</div>
    <div style="font-size:12px;color:#78350f;line-height:1.7;">
      Systematic error differs by group — different error rates in cases vs. controls.<br><br>
      <b>Effect:</b> Can bias in <b>either direction</b> — toward or away from null.<br><br>
      <b>Example:</b> Cases recall past exposures more carefully after diagnosis (recall bias). Error rate is higher in controls → OR inflated away from null.
    </div>
  </div>
</div>
<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:6px;padding:12px 14px;margin-top:4px;font-size:12px;color:#1e40af;line-height:1.7;">
  <b>Practical implication:</b> A null result with poor measurement reliability may not mean no effect — non-differential misclassification biases toward null. Always evaluate measurement quality before interpreting results.
</div>"""
        st.markdown(conn_html, unsafe_allow_html=True)

        with st.expander("📋 Quick Reference Table"):
            st.markdown("""
| Concept | Definition | Measured by | Failure effect |
|---|---|---|---|
| **Reliability** | Consistency | Kappa (κ), Cronbach's α, ICC | Non-differential misclassification → toward null |
| **Validity** | Accuracy | Gold standard correlation | Differential misclassification → any direction |
| **Face validity** | Looks right | Expert judgment | Weakest form |
| **Content validity** | Covers all dimensions | Expert review | Incomplete construct coverage |
| **Criterion validity** | Agrees with gold standard | Concurrent/predictive r | Most objective |
| **Construct validity** | Behaves as expected | Convergent/discriminant | Tests theory |
| **Internal validity** | Unbiased for study population | Bias/confounding absence | Cannot conclude causation |
| **External validity** | Generalizes to other settings | Sample representativeness | Limited applicability |
| **κ < 0.40** | Poor agreement | — | Revise tool or training |
| **κ 0.61–0.80** | Substantial agreement | — | Acceptable for epi research |
| **α < 0.70** | Poor internal consistency | — | Items may not measure same construct |
            """)

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
In practice, researchers don't let the data decide for them. They start by drawing out what they believe the causal structure looks like — based on prior literature, biological or social plausibility, and logic. This is often done with a **DAG (Directed Acyclic Graph)**: a map of arrows showing what causes what.

The DAG tells you which variables to control for and which to leave alone, **before you run a single model.** This is not a statistical decision. It's a conceptual one. You have to commit to a causal story about your variables and then let that story guide your modeling choices.
        """)

        st.error("""
⚠️ **Controlling for the wrong variable can make things worse, not better:**

- **Control for a confounder →** removes bias ✅
- **Control for a mediator →** blocks the causal pathway you're trying to study ❌
- **Control for a collider →** opens up a new bias that didn't exist before ❌

The same variable can play any of these roles depending on the causal structure — which is why you need the DAG first.
        """)

        st.info("""
💬 **The uncomfortable truth:** Reasonable researchers can look at the same data and disagree on the causal structure. This is why published papers include **sensitivity analyses** that test different modeling choices — reporting what happens when you control for different sets of variables. It's less like following a formula and more like making an argument you have to defend. The DAG is how you make that argument explicit.
        """)

        st.divider()
        st.markdown("#### Select a DAG structure to explore")
        st.markdown("For each structure, study the diagram, understand *why* adjustment helps or hurts, and note what the correct analytical response is.")


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
  <div style="font-weight:700;font-size:13px;margin-bottom:16px;color:#1a202c;">Mediator DAG: Physical Activity → Blood Pressure → CVD</div>
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 200" width="520" height="200" style="font-family:sans-serif;">
    <defs>
      <marker id="ma" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#2e7d32"/></marker>
      <marker id="mr" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#c62828"/></marker>
      <marker id="mb" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#1565c0"/></marker>
    </defs>
    <!-- Nodes -->
    <rect x="10" y="55" width="140" height="50" rx="8" fill="#e3f2fd" stroke="#1565c0" stroke-width="2"/>
    <text x="80" y="77" font-size="12" font-weight="700" fill="#1565c0" text-anchor="middle">Physical Activity</text>
    <text x="80" y="94" font-size="10" fill="#888" text-anchor="middle">Exposure (E)</text>

    <rect x="190" y="55" width="140" height="50" rx="8" fill="#e8f5e9" stroke="#2e7d32" stroke-width="2"/>
    <text x="260" y="77" font-size="12" font-weight="700" fill="#2e7d32" text-anchor="middle">↓ Blood Pressure</text>
    <text x="260" y="94" font-size="10" fill="#888" text-anchor="middle">Mediator (M)</text>

    <rect x="370" y="55" width="140" height="50" rx="8" fill="#fce4ec" stroke="#c62828" stroke-width="2"/>
    <text x="440" y="77" font-size="12" font-weight="700" fill="#c62828" text-anchor="middle">CVD Risk</text>
    <text x="440" y="94" font-size="10" fill="#888" text-anchor="middle">Outcome (Y)</text>

    <!-- E → M -->
    <line x1="152" y1="80" x2="188" y2="80" stroke="#2e7d32" stroke-width="2" marker-end="url(#ma)"/>
    <text x="170" y="72" font-size="9" fill="#2e7d32" text-anchor="middle">causes</text>

    <!-- M → Y -->
    <line x1="332" y1="80" x2="368" y2="80" stroke="#c62828" stroke-width="2" marker-end="url(#mr)"/>
    <text x="350" y="72" font-size="9" fill="#c62828" text-anchor="middle">causes</text>

    <!-- E → Y direct (dashed, above) -->
    <path d="M80,55 Q260,10 440,55" stroke="#1565c0" stroke-width="1.8" stroke-dasharray="6,3" fill="none" marker-end="url(#mb)"/>
    <text x="260" y="22" font-size="9" fill="#1565c0" text-anchor="middle">direct effect (also exists)</text>

    <!-- Labels -->
    <text x="260" y="148" font-size="9" fill="#718096" text-anchor="middle" font-style="italic">Total effect = direct + indirect (via blood pressure). Do NOT adjust for M to estimate total effect.</text>
  </svg>
</div>"""
            import streamlit.components.v1 as _dag_comp; _dag_comp.html(dag_html, height=250, scrolling=False)

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
  <div style="font-weight:700;font-size:13px;margin-bottom:16px;color:#1a202c;">Collider DAG: Talent and Hard Work → Success (collider)</div>
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 230" width="560" height="230" style="font-family:sans-serif;">
    <defs>
      <marker id="cp" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#7b1fa2"/></marker>
      <marker id="cg" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#2e7d32"/></marker>
      <marker id="cb" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#1565c0"/></marker>
    </defs>
    <!-- Talent (top-left) -->
    <rect x="20" y="20" width="140" height="52" rx="8" fill="#e3f2fd" stroke="#1565c0" stroke-width="2"/>
    <text x="90" y="43" font-size="13" font-weight="700" fill="#1565c0" text-anchor="middle">Talent</text>
    <text x="90" y="61" font-size="10" fill="#888" text-anchor="middle">Cause 1</text>
    <!-- Hard Work (bottom-left) -->
    <rect x="20" y="158" width="140" height="52" rx="8" fill="#e8f5e9" stroke="#2e7d32" stroke-width="2"/>
    <text x="90" y="181" font-size="13" font-weight="700" fill="#2e7d32" text-anchor="middle">Hard Work</text>
    <text x="90" y="199" font-size="10" fill="#888" text-anchor="middle">Cause 2</text>
    <!-- Success (right, collider) -->
    <rect x="360" y="89" width="160" height="52" rx="8" fill="#f3e5f5" stroke="#7b1fa2" stroke-width="2.5"/>
    <text x="440" y="112" font-size="13" font-weight="700" fill="#7b1fa2" text-anchor="middle">Success</text>
    <text x="440" y="130" font-size="10" fill="#888" text-anchor="middle">COLLIDER ← ←</text>
    <!-- Arrows into collider -->
    <line x1="162" y1="46" x2="358" y2="106" stroke="#1565c0" stroke-width="2" marker-end="url(#cb)"/>
    <line x1="162" y1="184" x2="358" y2="124" stroke="#2e7d32" stroke-width="2" marker-end="url(#cg)"/>
    <!-- Do not adjust label (left-center) -->
    <rect x="180" y="95" width="160" height="28" rx="5" fill="#e8f5e9" stroke="#2e7d32" stroke-width="1"/>
    <text x="260" y="113" font-size="10" fill="#2e7d32" text-anchor="middle">✅ Path blocked naturally</text>
    <!-- Warning label (bottom-right) -->
    <rect x="300" y="160" width="240" height="28" rx="5" fill="#fce4ec" stroke="#c62828" stroke-width="1"/>
    <text x="420" y="178" font-size="10" fill="#c62828" text-anchor="middle">❌ Adjust for Success → opens spurious path</text>
    <!-- Caption -->
    <text x="280" y="220" font-size="9" fill="#718096" text-anchor="middle" font-style="italic">Conditioning on Success makes Talent and Hard Work appear negatively correlated</text>
  </svg>
</div>"""
            import streamlit.components.v1 as _dag_comp; _dag_comp.html(dag_html, height=360, scrolling=False)

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
            import streamlit.components.v1 as _dag_comp; _dag_comp.html(dag_html, height=240, scrolling=False)

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
  <div style="font-weight:700;font-size:13px;margin-bottom:16px;color:#1a202c;">M-Bias DAG — The M shape</div>
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 210" width="560" height="210" style="font-family:sans-serif;">
    <defs>
      <marker id="mg" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#9e9e9e"/></marker>
      <marker id="mr2" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#c62828"/></marker>
    </defs>
    <!-- U1 top-left -->
    <rect x="10" y="10" width="110" height="50" rx="8" fill="#f5f5f5" stroke="#9e9e9e" stroke-width="2" stroke-dasharray="4,2"/>
    <text x="65" y="32" font-size="11" font-weight="700" fill="#757575" text-anchor="middle">U₁</text>
    <text x="65" y="50" font-size="9" fill="#aaa" text-anchor="middle">Unmeasured → E</text>
    <!-- U2 top-right -->
    <rect x="440" y="10" width="110" height="50" rx="8" fill="#f5f5f5" stroke="#9e9e9e" stroke-width="2" stroke-dasharray="4,2"/>
    <text x="495" y="32" font-size="11" font-weight="700" fill="#757575" text-anchor="middle">U₂</text>
    <text x="495" y="50" font-size="9" fill="#aaa" text-anchor="middle">Unmeasured → Y</text>
    <!-- E bottom-left -->
    <rect x="10" y="150" width="120" height="50" rx="8" fill="#e3f2fd" stroke="#1565c0" stroke-width="2"/>
    <text x="70" y="172" font-size="12" font-weight="700" fill="#1565c0" text-anchor="middle">Exposure (E)</text>
    <text x="70" y="189" font-size="10" fill="#888" text-anchor="middle">e.g., alcohol use</text>
    <!-- M center -->
    <rect x="220" y="80" width="120" height="50" rx="8" fill="#fff3e0" stroke="#e65100" stroke-width="2.5"/>
    <text x="280" y="102" font-size="12" font-weight="700" fill="#e65100" text-anchor="middle">M</text>
    <text x="280" y="119" font-size="10" fill="#888" text-anchor="middle">Pre-treatment var.</text>
    <!-- Y bottom-right -->
    <rect x="430" y="150" width="120" height="50" rx="8" fill="#fce4ec" stroke="#c62828" stroke-width="2"/>
    <text x="490" y="172" font-size="12" font-weight="700" fill="#c62828" text-anchor="middle">Outcome (Y)</text>
    <text x="490" y="189" font-size="10" fill="#888" text-anchor="middle">e.g., depression</text>
    <!-- U1 → E -->
    <line x1="65" y1="62" x2="65" y2="148" stroke="#9e9e9e" stroke-width="1.8" stroke-dasharray="4,2" marker-end="url(#mg)"/>
    <!-- U1 → M -->
    <line x1="122" y1="35" x2="218" y2="95" stroke="#9e9e9e" stroke-width="1.8" stroke-dasharray="4,2" marker-end="url(#mg)"/>
    <!-- U2 → Y -->
    <line x1="495" y1="62" x2="495" y2="148" stroke="#9e9e9e" stroke-width="1.8" stroke-dasharray="4,2" marker-end="url(#mg)"/>
    <!-- U2 → M -->
    <line x1="438" y1="35" x2="342" y2="95" stroke="#9e9e9e" stroke-width="1.8" stroke-dasharray="4,2" marker-end="url(#mg)"/>
    <!-- E → Y (true effect) -->
    <path d="M130,175 Q280,205 428,175" stroke="#c62828" stroke-width="2" fill="none" marker-end="url(#mr2)"/>
    <text x="280" y="205" font-size="9" fill="#c62828" text-anchor="middle">E → Y (true causal effect)</text>
    <!-- M shape label -->
    <text x="280" y="17" font-size="10" fill="#e65100" text-anchor="middle" font-weight="700">The "M" shape: U₁ → M ← U₂, with E ← U₁ and U₂ → Y</text>
  </svg>
  <div style="margin-top:10px;display:flex;gap:12px;justify-content:center;font-size:11px;">
    <div style="background:#e8f5e9;border-radius:6px;padding:8px 12px;color:#2e7d32;">✅ <b>Without M:</b> E→Y estimate is unbiased (U₁-U₂ path blocked at M)</div>
    <div style="background:#fce4ec;border-radius:6px;padding:8px 12px;color:#c62828;">❌ <b>Adjust for M:</b> Opens E ← U₁ → M ← U₂ → Y — introduces bias</div>
  </div>
</div>"""
            import streamlit.components.v1 as _dag_comp; _dag_comp.html(dag_html, height=340, scrolling=False)

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
  <div style="font-weight:700;font-size:13px;margin-bottom:16px;color:#1a202c;">Proxy DAG: Education proxies for SES</div>
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 230" width="560" height="230" style="font-family:sans-serif;">
    <defs>
      <marker id="pg" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#9e9e9e"/></marker>
      <marker id="po" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#e65100"/></marker>
    </defs>
    <!-- SES (U) top-center -->
    <rect x="195" y="10" width="170" height="52" rx="8" fill="#f5f5f5" stroke="#9e9e9e" stroke-width="2" stroke-dasharray="5,3"/>
    <text x="280" y="33" font-size="12" font-weight="700" fill="#757575" text-anchor="middle">SES (U)</text>
    <text x="280" y="51" font-size="10" fill="#aaa" text-anchor="middle">True variable — unmeasured</text>
    <!-- Education bottom-left -->
    <rect x="40" y="155" width="160" height="52" rx="8" fill="#fff3e0" stroke="#e65100" stroke-width="2.5"/>
    <text x="120" y="178" font-size="12" font-weight="700" fill="#e65100" text-anchor="middle">Education</text>
    <text x="120" y="196" font-size="10" fill="#888" text-anchor="middle">Proxy (measured)</text>
    <!-- Health Outcome bottom-right -->
    <rect x="360" y="155" width="160" height="52" rx="8" fill="#fce4ec" stroke="#c62828" stroke-width="2"/>
    <text x="440" y="178" font-size="12" font-weight="700" fill="#c62828" text-anchor="middle">Health Outcome</text>
    <text x="440" y="196" font-size="10" fill="#888" text-anchor="middle">Outcome (Y)</text>
    <!-- SES → Education -->
    <line x1="230" y1="64" x2="168" y2="153" stroke="#9e9e9e" stroke-width="2" stroke-dasharray="4,2" marker-end="url(#pg)"/>
    <text x="188" y="105" font-size="9" fill="#9e9e9e" text-anchor="middle">causes</text>
    <!-- SES → Outcome -->
    <line x1="330" y1="64" x2="392" y2="153" stroke="#9e9e9e" stroke-width="2" stroke-dasharray="4,2" marker-end="url(#pg)"/>
    <text x="372" y="105" font-size="9" fill="#9e9e9e" text-anchor="middle">causes</text>
    <!-- Education → Outcome partial path (below nodes) -->
    <line x1="202" y1="181" x2="358" y2="181" stroke="#e65100" stroke-width="2" stroke-dasharray="5,3" marker-end="url(#po)"/>
    <text x="280" y="146" font-size="9" fill="#e65100" text-anchor="middle">adjusting for proxy = partial confounding control</text>
    <!-- Caption -->
    <text x="280" y="220" font-size="9" fill="#718096" text-anchor="middle" font-style="italic">Residual confounding remains: Education ≠ SES. Weaker proxy = more residual confounding.</text>
  </svg>
</div>"""
            import streamlit.components.v1 as _dag_comp; _dag_comp.html(dag_html, height=280, scrolling=False)

            st.warning("⚠️ **Proxy limitations:** Adjusting for a proxy only partially controls for the underlying variable. The weaker the proxy-variable relationship, the more residual confounding remains. This is why residual confounding is almost always present in observational studies.")

            with st.expander("📊 Implications for interpretation"):
                st.markdown("""
**When a proxy is used as a confounder:**
- You get partial adjustment, not full adjustment
- Incomplete adjustment may leave the estimate biased — often, but not always, somewhere between the crude and fully adjusted estimate. The direction and magnitude depend on the strength and direction of the remaining confounding.
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

        st.divider()
        st.markdown("#### 🧭 Putting it together: DAGs as arguments, not algorithms")
        st.markdown("""
The six structures above aren't just taxonomy — they're the building blocks of every confounding decision you'll ever make. Here's what working epidemiologists actually do:

**Step 1 — Draw before you model.**
Before opening a dataset, sketch your causal assumptions. What causes the exposure? What causes the outcome? What variables are on the causal pathway vs. common causes? Your DAG is a public commitment to a theoretical position.

**Step 2 — Identify the adjustment set.**
From your DAG, identify which variables to include in your model to block all backdoor paths — without conditioning on mediators or colliders. This is your *a priori* adjustment set. It comes from the DAG, not from stepwise variable selection.

**Step 3 — Acknowledge uncertainty.**
If reasonable people could draw the DAG differently, run sensitivity analyses. Control for different sets of variables and report what changes. A finding that holds across plausible causal structures is more credible than one that depends on one specific DAG.

**Step 4 — Be honest about unmeasured confounding.**
Every DAG has variables you couldn't measure. Name them explicitly. Discuss the likely direction of residual confounding. This is not a weakness — it's scientific integrity.
        """)

        st.success("""
✅ **The key insight:** Two researchers can use identical data and reach different estimates — not because one made a math error, but because they committed to different causal structures. Statistical tests cannot resolve this disagreement. Only transparent causal reasoning can. This is what separates descriptive statistics from epidemiologic thinking.
        """)

    st.markdown("---")
    st.markdown("*Strong epidemiologists think structurally before computing.*")


# ==================================================
# MODULE 1: CAUSAL INFERENCE
# ==================================================
elif current_page == "causal_inference":
    st.title("🔗 Causal Inference")
    st.markdown("Association does not equal causation. Causal inference is the process of evaluating whether an observed statistical association reflects a true cause-and-effect relationship.")

    ci_section = st.radio("Section:", [
        "1️⃣ Bradford Hill Criteria",
        "2️⃣ Criteria Application Exercise",
        "3️⃣ Rothman's Sufficient-Component Cause Model",
        "4️⃣ Web of Causation"
    ], horizontal=True)
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

    elif ci_section == "3️⃣ Rothman's Sufficient-Component Cause Model":
        st.subheader("Rothman's Sufficient-Component Cause Model (The Causal Pies)")
        st.markdown("""
Kenneth Rothman's **sufficient-component cause model** — often called the "causal pies" — provides a rigorous conceptual framework for thinking about causation in epidemiology. It moves beyond the simple "A causes B" framing to capture how disease actually arises from the joint action of multiple factors.
        """)

        st.markdown("### Core Concepts")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**Component cause**
Any factor that plays a role in producing disease. A component cause alone is not sufficient to cause disease — it must act with other components. Smoking is a component cause of lung cancer, not a sufficient cause by itself (most smokers never get lung cancer).

**Sufficient cause**
A complete causal mechanism — a minimal set of component causes that, together, inevitably produces disease. "Sufficient" means that given all components, disease is certain. "Minimal" means removing any one component would make it insufficient.

**Necessary cause**
A component that appears in *every* sufficient cause of a disease — without it, disease cannot occur. HIV is a necessary cause of AIDS. HPV is a necessary cause of cervical cancer.

**Note:** Most exposures in epidemiology are component causes, not necessary causes.
            """)
        with col2:
            st.markdown("""
**The pie analogy**
Each "pie" represents one sufficient cause — a complete causal mechanism. Each "slice" of the pie is a component cause. The pie is complete (sufficient) only when all slices are present.

**Key implications:**
- Disease can have multiple sufficient causes (multiple pies)
- The same component can appear in more than one sufficient cause
- Removing any one component from a sufficient cause prevents that pathway
- We never observe sufficient causes directly — we only observe component causes

**Why this matters for epi:**
- A factor with modest RR may be the "last piece" in many pies → high PAR%
- A factor with large RR may complete few pies → low PAR% if rare
- Interaction = two factors appear together in the same pie
- This model explains why effect modification (interaction) is the rule, not the exception
            """)

        st.divider()
        st.markdown("### 🥧 The Pies — Visual Model")

        pies_html = """
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:16px 0;background:#fff;">

  <div style="text-align:center;margin-bottom:16px;font-size:13px;color:#4b5563;">
    Three <em>example</em> sufficient causes for <b>Lung Cancer</b> — illustrative pathways only; in reality there are many more
  </div>

  <!-- LETTER KEY -->
  <div style="display:flex;justify-content:center;gap:8px;flex-wrap:wrap;margin-bottom:20px;">
    <div style="display:flex;align-items:center;gap:5px;background:#fef2f2;border:1px solid #fca5a5;border-radius:20px;padding:4px 12px;">
      <span style="background:#ef4444;color:white;font-weight:900;font-size:12px;border-radius:50%;width:20px;height:20px;display:inline-flex;align-items:center;justify-content:center;">A</span>
      <span style="font-size:12px;color:#1a202c;">Smoking <em style="font-size:10px;color:#dc2626;">(component, not necessary — but responsible for the largest PAR%)</em></span>
    </div>
    <div style="display:flex;align-items:center;gap:5px;background:#fff7ed;border:1px solid #fdba74;border-radius:20px;padding:4px 12px;">
      <span style="background:#f97316;color:white;font-weight:900;font-size:12px;border-radius:50%;width:20px;height:20px;display:inline-flex;align-items:center;justify-content:center;">B</span>
      <span style="font-size:12px;color:#1a202c;">Asbestos exposure</span>
    </div>
    <div style="display:flex;align-items:center;gap:5px;background:#fefce8;border:1px solid #fde047;border-radius:20px;padding:4px 12px;">
      <span style="background:#eab308;color:white;font-weight:900;font-size:12px;border-radius:50%;width:20px;height:20px;display:inline-flex;align-items:center;justify-content:center;">C</span>
      <span style="font-size:12px;color:#1a202c;">Radon exposure</span>
    </div>
    <div style="display:flex;align-items:center;gap:5px;background:#f5f3ff;border:1px solid #c4b5fd;border-radius:20px;padding:4px 12px;">
      <span style="background:#8b5cf6;color:white;font-weight:900;font-size:12px;border-radius:50%;width:20px;height:20px;display:inline-flex;align-items:center;justify-content:center;">D</span>
      <span style="font-size:12px;color:#1a202c;">Air pollution</span>
    </div>
    <div style="display:flex;align-items:center;gap:5px;background:#ecfdf5;border:1px solid #6ee7b7;border-radius:20px;padding:4px 12px;">
      <span style="background:#10b981;color:white;font-weight:900;font-size:12px;border-radius:50%;width:20px;height:20px;display:inline-flex;align-items:center;justify-content:center;">E</span>
      <span style="font-size:12px;color:#1a202c;">Ionizing radiation</span>
    </div>
    <div style="display:flex;align-items:center;gap:5px;background:#ecfeff;border:1px solid #67e8f9;border-radius:20px;padding:4px 12px;">
      <span style="background:#06b6d4;color:white;font-weight:900;font-size:12px;border-radius:50%;width:20px;height:20px;display:inline-flex;align-items:center;justify-content:center;">F</span>
      <span style="font-size:12px;color:#1a202c;">Genetic susceptibility</span>
    </div>
    <div style="display:flex;align-items:center;gap:5px;background:#f8fafc;border:1px solid #cbd5e1;border-radius:20px;padding:4px 12px;">
      <span style="background:#94a3b8;color:white;font-weight:900;font-size:12px;border-radius:50%;width:20px;height:20px;display:inline-flex;align-items:center;justify-content:center;">U</span>
      <span style="font-size:12px;color:#1a202c;">Unknown component(s)</span>
    </div>
  </div>

  <!-- THREE PIES -->
  <div style="display:flex;gap:24px;justify-content:center;flex-wrap:wrap;align-items:flex-start;">

    <!-- PIE I: A=120°, B=80°, C=80°, U=80° — 2.5° gap each side -->
    <div style="text-align:center;width:220px;">
      <div style="font-weight:700;color:#1d4ed8;margin-bottom:8px;font-size:14px;">Sufficient Cause I</div>
      <svg viewBox="0 0 200 200" width="200" height="200" xmlns="http://www.w3.org/2000/svg">
        <path d="M100,100 L103.84,12.08 A88,88 0 0,1 178.06,140.63 Z" fill="#ef4444"/>
        <path d="M100,100 L174.22,147.28 A88,88 0 0,1 73.54,183.93 Z" fill="#f97316"/>
        <path d="M100,100 L66.32,181.3 A88,88 0 0,1 12.75,88.51 Z" fill="#eab308"/>
        <path d="M100,100 L14.09,80.95 A88,88 0 0,1 96.16,12.08 Z" fill="#94a3b8"/>
        <!-- white gap lines -->
        <line x1="100" y1="100" x2="103.84" y2="12.08" stroke="white" stroke-width="3"/>
        <line x1="100" y1="100" x2="178.06" y2="140.63" stroke="white" stroke-width="3"/>
        <line x1="100" y1="100" x2="73.54" y2="183.93" stroke="white" stroke-width="3"/>
        <line x1="100" y1="100" x2="14.09" y2="80.95" stroke="white" stroke-width="3"/>
        <!-- Labels -->
        <text x="147.3" y="72.7"  font-size="22" fill="white" font-weight="900" text-anchor="middle" dominant-baseline="middle">A</text>
        <text x="118.7" y="151.3" font-size="22" fill="white" font-weight="900" text-anchor="middle" dominant-baseline="middle">B</text>
        <text x="52.7"  y="127.3" font-size="22" fill="white" font-weight="900" text-anchor="middle" dominant-baseline="middle">C</text>
        <text x="64.9"  y="58.2"  font-size="22" fill="white" font-weight="900" text-anchor="middle" dominant-baseline="middle">U</text>
      </svg>
      <div style="font-size:11px;color:#555;margin-top:4px;">A + B + C + U</div>
    </div>

    <!-- PIE II: A=200°, D+U=160° -->
    <div style="text-align:center;width:220px;">
      <div style="font-weight:700;color:#1d4ed8;margin-bottom:8px;font-size:14px;">Sufficient Cause II</div>
      <svg viewBox="0 0 200 200" width="200" height="200" xmlns="http://www.w3.org/2000/svg">
        <path d="M100,100 L103.84,12.08 A88,88 0 1,1 73.54,183.93 Z" fill="#ef4444"/>
        <path d="M100,100 L66.32,181.3 A88,88 0 0,1 96.16,12.08 Z" fill="#8b5cf6"/>
        <line x1="100" y1="100" x2="103.84" y2="12.08" stroke="white" stroke-width="3"/>
        <line x1="100" y1="100" x2="73.54"  y2="183.93" stroke="white" stroke-width="3"/>
        <text x="153.7" y="109.5" font-size="22" fill="white" font-weight="900" text-anchor="middle" dominant-baseline="middle">A</text>
        <text x="46.3"  y="90.5"  font-size="16" fill="white" font-weight="900" text-anchor="middle" dominant-baseline="middle">D+U</text>
      </svg>
      <div style="font-size:11px;color:#555;margin-top:4px;">A + D + U</div>
    </div>

    <!-- PIE III: F=90°, B=90°, E=90°, U=90° -->
    <div style="text-align:center;width:220px;">
      <div style="font-weight:700;color:#1d4ed8;margin-bottom:8px;font-size:14px;">Sufficient Cause III</div>
      <svg viewBox="0 0 200 200" width="200" height="200" xmlns="http://www.w3.org/2000/svg">
        <path d="M100,100 L103.84,12.08 A88,88 0 0,1 187.92,96.16 Z" fill="#06b6d4"/>
        <path d="M100,100 L187.92,103.84 A88,88 0 0,1 103.84,187.92 Z" fill="#f97316"/>
        <path d="M100,100 L96.16,187.92 A88,88 0 0,1 12.08,103.84 Z" fill="#10b981"/>
        <path d="M100,100 L12.08,96.16 A88,88 0 0,1 96.16,12.08 Z" fill="#94a3b8"/>
        <line x1="100" y1="100" x2="103.84" y2="12.08"  stroke="white" stroke-width="3"/>
        <line x1="100" y1="100" x2="187.92" y2="96.16"  stroke="white" stroke-width="3"/>
        <line x1="100" y1="100" x2="103.84" y2="187.92" stroke="white" stroke-width="3"/>
        <line x1="100" y1="100" x2="12.08"  y2="103.84" stroke="white" stroke-width="3"/>
        <text x="138.6" y="61.4"  font-size="22" fill="white" font-weight="900" text-anchor="middle" dominant-baseline="middle">F</text>
        <text x="138.6" y="138.6" font-size="22" fill="white" font-weight="900" text-anchor="middle" dominant-baseline="middle">B</text>
        <text x="61.4"  y="138.6" font-size="22" fill="white" font-weight="900" text-anchor="middle" dominant-baseline="middle">E</text>
        <text x="61.4"  y="61.4"  font-size="22" fill="white" font-weight="900" text-anchor="middle" dominant-baseline="middle">U</text>
      </svg>
      <div style="font-size:11px;color:#555;margin-top:4px;">F + B + E + U <em>(no smoking)</em></div>
    </div>

  </div>

  <!-- Callout -->
  <!-- Callout: core teaching points -->
  <div style="margin-top:20px;background:#eff6ff;border-radius:8px;padding:14px 18px;font-size:12px;color:#1e40af;line-height:1.8;">
    <div style="font-weight:700;font-size:13px;margin-bottom:6px;">📌 Reading these diagrams</div>
    <div>&#x2714; <b>A (Smoking) is in pathways I and II, but not III</b> — it is a <em>component</em> cause, not a <em>necessary</em> cause. Lung cancer can occur without it.</div>
    <div>&#x2714; <b>"Not necessary" does not mean "not important."</b> Smoking carries the highest population attributable fraction of any single lung cancer component — it just doesn't appear in every sufficient cause.</div>
    <div>&#x2714; <b>Every pathway contains U (unknown components).</b> This is not a gap in these three examples — it is a fundamental feature of the model. We never fully enumerate all component causes. This is why PAR%s for known causes rarely sum to 100%.</div>
    <div>&#x2714; <b>These are three examples, not an exhaustive list.</b> There are likely many sufficient causes for lung cancer. This diagram illustrates the logic of the model, not a complete causal inventory.</div>
    <div>&#x26A0; <b>Slice size does not represent causal strength.</b> Equal-sized slices are a visual convenience — component causes do not necessarily contribute equally, and some (like smoking) are far stronger contributors than others.</div>
  </div>

</div>"""
        import streamlit.components.v1 as _rothman_comp
        _rothman_comp.html(
            f"<!DOCTYPE html><html><body style='margin:0;padding:0;background:#fff;'>{pies_html}</body></html>",
            height=560,
            scrolling=False
        )

        st.divider()
        st.markdown("### Epidemiologic Implications")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
#### Strength of Association ≠ Importance

A component that completes many sufficient causes has a high PAR% even if its RR is modest — because removing it prevents disease through many pathways.

A component with a very high RR (e.g., a rare genetic mutation) may complete few sufficient causes and have a low PAR% despite looking "strong."

**PAR% reflects both how often a component participates in causal mechanisms and how common it is in the population.** A component appearing in many sufficient causes but rare in the population can still have a modest PAR%. Conversely, a common exposure that participates in even a few pathways can have a very high PAR%.

#### Why Interactions Are Expected

Two factors "interact" (show effect modification) when they appear in the same sufficient cause. Rothman's model predicts that biologic interactions should be the *norm* rather than the exception — because disease rarely arises from single causes acting in isolation. A practical caveat: detecting statistical interaction in data depends heavily on the scale of measurement (additive vs. multiplicative) and on sample size. Most studies are underpowered to detect interaction even when it exists biologically.

**Biologic interaction** = two factors in the same causal pie.
**Statistical interaction** = departure from additive or multiplicative effects.
            """)

        with col2:
            st.markdown("""
#### The Unknown Slices (U)

Every pie in the diagram contains "U" — unknown components. This is Rothman's acknowledgment that:
- We never identify all component causes
- PAR%s for known factors rarely sum to 100%
- The "missing" variance reflects unmeasured or undiscovered components

This contributes to incomplete adjustment and uncertainty in observational estimates. Note: unknown component causes are not automatically confounders — confounding requires a common cause of both exposure and outcome. Unknown components may be unmeasured causal factors, parts of mechanisms unrelated to the study exposure, or sources of variability we have not yet characterized.

#### Prevention Implications

**Removing any one component from a sufficient cause prevents disease through that pathway** — but with an important population-level framing.

This means:
- Even modest exposures are worth targeting if they appear in many pathways
- You don't need to address all causes simultaneously
- Removing a common component (like smoking) disrupts many complete mechanisms at once

**Two caveats for students:** First, we rarely know which sufficient cause any individual is "on" — prevention works probabilistically across populations, not deterministically for individuals. Second, removing a component prevents only the pathways that include it; other sufficient causes remain.

**This is the theoretical basis for PAR% as a public health metric.**
            """)

        st.divider()
        with st.expander("🔢 Quantifying Interaction — Synergy and Antagonism"):
            st.markdown("""
**Additive interaction (the epidemiologic standard):**
When two factors act in the same sufficient cause (same pie), their joint effect exceeds the sum of their individual effects on the additive scale.

**Synergy (positive interaction):**
RR(AB) > RR(A alone) + RR(B alone) − 1

Example: Asbestos alone → RR = 5; Smoking alone → RR = 10; Asbestos + Smoking → RR = 50
Expected additive: 5 + 10 − 1 = 14. Observed: 50. Strong synergy → both appear in same sufficient cause.

**Antagonism (negative interaction):**
The joint effect is less than the sum of individual effects. Less common — may reflect competing mechanisms.

**Why additive, not multiplicative?**
Rothman argues that the biologically meaningful scale for interaction is additive (excess cases), not multiplicative (ratio). Two factors that simply multiply each other's effects are acting independently on separate sufficient causes.
            """)

        with st.expander("📋 Key Terms — Rothman's Model"):
            st.markdown("""
| Term | Definition |
|---|---|
| **Component cause** | A factor that contributes to one or more sufficient causes but alone cannot produce disease |
| **Sufficient cause** | A minimal set of component causes that inevitably produces disease |
| **Necessary cause** | A component that appears in every sufficient cause — disease cannot occur without it |
| **Causal pie** | Visual metaphor: each pie = one sufficient cause; each slice = one component |
| **Induction period** | Time between the action of a component cause and the initiation of disease |
| **Latency period** | Time between disease initiation and disease detection |
| **Biologic synergy** | Two components appear in the same pie — joint effect exceeds additive expectation |
| **U (unknown)** | Unknown component causes that complete sufficient causes — always present |
            """)


    elif ci_section == "4️⃣ Web of Causation":
        st.subheader("Web of Causation (MacMahon & Pugh, 1960)")
        st.markdown("""
The **Web of Causation**, proposed by Brian MacMahon and Thomas Pugh, was a direct response to the limitations of single-agent causal models. It holds that disease is rarely caused by a single factor — instead, it arises from a complex network of interacting causes operating across biological, behavioral, social, and environmental levels simultaneously.

The metaphor of a *web* captures the idea that causes are interconnected: pulling on one strand affects others. There is no single root cause, and no single intervention is sufficient.
        """)

        st.markdown("""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:24px;margin:16px 0;">
<div style="font-weight:700;font-size:13px;margin-bottom:16px;color:#1a202c;">Web of Causation — Coronary Heart Disease (simplified)</div>
<div style="font-size:12px;color:#334155;line-height:2;">
<div style="display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-bottom:16px;">
  <span style="background:#fff3e0;border:1px solid #ef6c00;border-radius:6px;padding:4px 10px;">Diet high in saturated fat</span>
  <span style="background:#fff3e0;border:1px solid #ef6c00;border-radius:6px;padding:4px 10px;">Physical inactivity</span>
  <span style="background:#fff3e0;border:1px solid #ef6c00;border-radius:6px;padding:4px 10px;">Cigarette smoking</span>
  <span style="background:#e3f2fd;border:1px solid #1565c0;border-radius:6px;padding:4px 10px;">Hypertension</span>
  <span style="background:#e3f2fd;border:1px solid #1565c0;border-radius:6px;padding:4px 10px;">Hyperlipidemia</span>
  <span style="background:#e3f2fd;border:1px solid #1565c0;border-radius:6px;padding:4px 10px;">Diabetes</span>
  <span style="background:#e8f5e9;border:1px solid #2e7d32;border-radius:6px;padding:4px 10px;">Poverty</span>
  <span style="background:#e8f5e9;border:1px solid #2e7d32;border-radius:6px;padding:4px 10px;">Stress</span>
  <span style="background:#e8f5e9;border:1px solid #2e7d32;border-radius:6px;padding:4px 10px;">Limited healthcare access</span>
  <span style="background:#fce4ec;border:1px solid #c62828;border-radius:6px;padding:4px 10px;font-weight:700;">↓ CORONARY HEART DISEASE ↓</span>
</div>
<div style="font-size:11px;color:#718096;text-align:center;font-style:italic;">
Each factor connects to others — poverty → poor diet + stress + limited access; smoking → hypertension; inactivity → diabetes + hyperlipidemia.
The web has no single origin and no single solution.
</div>
</div>
</div>
        """, unsafe_allow_html=True)

        st.markdown("#### Comparing causal models")
        st.markdown("""
| Model | Key idea | Best suited for | Key limitation |
|---|---|---|---|
| **Epidemiology Triangle** | Disease = agent × host × environment | Infectious disease with one identifiable agent | Too simple for chronic disease; no causal direction |
| **Chain of infection** | Sequential steps from reservoir to host | Communicable disease control | Linear; doesn't capture feedback loops |
| **Bradford Hill criteria** | Weight-of-evidence for one exposure-outcome pair | Evaluating a specific causal hypothesis | One pair at a time; no network structure |
| **Rothman's causal pies** | Sufficient-component causes; multiple pies | Understanding why some exposed people don't get sick | Abstract; hard to operationalize for intervention |
| **Web of Causation** | Interconnected network of causes at multiple levels | Chronic disease; social determinants; policy | Complexity makes it hard to test statistically |
| **DAG** | Formal directed graph of causal assumptions | Deciding what to control for in a regression model | Requires strong prior knowledge; results depend on assumed structure |
        """)

        st.info("""
🔑 **The web's contribution to public health practice:**
Because causes are interconnected, intervening at *any* point in the web can reduce disease. You don't need to find the "root cause" — you need to find the most *modifiable*, *high-impact* nodes.

This is why seat belts reduce traffic deaths even though the "cause" of a crash is driver behavior. And why improving neighborhood walkability reduces diabetes even without targeting diet directly.
        """)

        web_q = st.radio(
            "**Apply it:** A researcher wants to reduce hypertension rates in a low-income urban neighborhood. The Web of Causation perspective would suggest:",
            ["— Select —",
             "Prescribe antihypertensive medications to all residents — target the biological pathway",
             "Address only stress reduction programs — the most direct cause",
             "Pursue multiple simultaneous interventions: access to healthy food, stress reduction, healthcare access, and built environment changes",
             "Wait for genetic research to identify the root cause"],
            key="web_q1"
        )
        if web_q == "Pursue multiple simultaneous interventions: access to healthy food, stress reduction, healthcare access, and built environment changes":
            st.success("✅ Correct. The web has no single root cause — multiple interconnected factors contribute. A web-of-causation approach recommends targeting multiple nodes simultaneously, especially upstream social and environmental determinants, rather than any single factor.")
        elif web_q != "— Select —":
            st.error("❌ The Web of Causation rejects single-cause thinking. No single intervention addresses all the interconnected factors. The model specifically motivates multi-level, simultaneous interventions — including upstream social and structural causes.")


    st.markdown("---")
    st.markdown("*Strong epidemiologists think structurally before computing.*")


# ==================================================
# MODULE 2: DISEASE FREQUENCY
# ==================================================
elif current_page == "disease_frequency":
    st.title("📊 Disease Frequency")
    st.markdown("Before comparing rates across groups, you need to be able to measure disease frequency accurately in a single population.")

    df_section = st.radio("Section:", ["1️⃣ Core Measures", "2️⃣ Interactive Calculator", "3️⃣ Prevalence-Incidence Relationship", "4️⃣ Epidemic Curves", "5️⃣ Person, Place & Time", "6️⃣ Public Health Surveillance", "7️⃣ Mortality Measures & YPLL"], horizontal=True)
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

        with st.expander("📌 Secondary Attack Rate (SAR)"):
            st.markdown("""
**Definition:** The proportion of susceptible contacts of a case who develop disease within one incubation period of exposure.

**Formula:** SAR = Secondary cases ÷ Susceptible contacts × 100

**Critical rule:** The **index case is excluded** from both numerator and denominator. The index case was infected outside the setting — they cannot be a secondary case. The denominator is only the susceptible contacts who could have been infected by the index case.

**Example:**
- Index case (child) returns home from school with norovirus
- Household has 4 other members (all susceptible)
- 3 of the 4 develop illness within the incubation period
- **SAR = 3 ÷ 4 × 100 = 75%**

**How SAR differs from attack rate:**
| Measure | Numerator | Denominator | Answers |
|---|---|---|---|
| Attack Rate (AR) | All cases | All at risk (incl. index) | How likely is exposure to cause disease? |
| Secondary Attack Rate (SAR) | Secondary cases only | Susceptible contacts (excl. index) | How likely does the index case transmit to contacts? |

**Uses of SAR:**
- Estimates household or close-contact transmissibility
- Compares effectiveness of isolation and prophylaxis
- Contributes to estimating R₀ (basic reproduction number)
- Higher SAR = more transmissible pathogen or more intimate setting

**Example values:** Norovirus household SAR ≈ 30–80%; Measles SAR in unvaccinated households ≈ 75–90%; Seasonal influenza SAR ≈ 10–30%
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
                "description": "**Scenario:** Health department investigates 38 cases of bloody diarrhea linked to a county fair petting zoo. Onset times cluster 2–6 days after the fair date (typical E. coli O157:H7 incubation: 1–10 days, average 3–4 days).",
                "peak": 3.5, "spread": 1.2, "total": 38,
                "x_label": "Days after fair visit",
                "color": "#e53935",
                "key_features": [
                    "Single peak centered ~day 3–4 post-exposure — consistent with E. coli O157:H7 average incubation of 3–4 days",
                    "Incubation range of 1–10 days produces a wider, flatter peak than toxin-mediated outbreaks (2–6 hours)",
                    "Cases begin day 1, peak day 3–4, tail off through day 8 — all within one incubation period range",
                    "No secondary person-to-person wave — point source ends when fair exposure ends",
                ],
                "next_step": "Identify the specific exposure — petting zoo animal contact, contaminated water, food handling. Conduct **case-control study** with fair attendees as controls. Collect stool cultures from cases.",
                "contrast": "Compare to Staph toxin (all cases within 2–6 hours). E. coli O157 has the same point-source shape but stretched over days, not hours — reflecting live bacterial replication vs. preformed toxin.",
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
                # Extend to peak*3 or at minimum 12 steps to show full tail
                n_steps = max(12, int(round(peak * 3)) + 1)
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

            # ── Axis scaling ────────────────────────────────────────────────
            # Y-axis: nice tick interval, y_max slightly above max bar
            def nice_tick_interval(data_max):
                """Return a round tick interval giving 4-7 ticks."""
                if data_max <= 0: return 1
                if data_max <= 5: return 1   # small counts: tick every 1
                raw = data_max / 5          # aim for ~5 ticks
                magnitude = 10 ** _math.floor(_math.log10(raw))
                candidates = [1, 2, 5, 10]
                for c in candidates:
                    val = c * magnitude
                    if val >= raw:
                        return max(1, int(val))
                return max(1, int(10 * magnitude))

            y_tick_interval = nice_tick_interval(max_count)
            y_max = y_tick_interval * (_math.ceil(max_count / y_tick_interval) + 1)
            y_ticks_vals = list(range(0, y_max + 1, y_tick_interval))

            # X-axis: evenly spaced labels — target ~8-12 labels
            n_bars = len(counts)
            target_x_labels = min(12, n_bars)
            x_step = max(1, round(n_bars / target_x_labels))
            # Round x_step to a nice number
            for nice in [1, 2, 5, 10, 20, 25, 50, 100]:
                if nice >= x_step:
                    x_step = nice
                    break

            # ---- Build SVG ------------------------------------------------
            chart_w   = 700
            chart_h   = 240
            pad_l     = 52
            pad_b     = 46
            pad_t     = 32
            pad_r     = 20
            plot_w    = chart_w - pad_l - pad_r
            plot_h    = chart_h - pad_b - pad_t
            bar_gap   = 1
            bar_w     = max(2.0, (plot_w - bar_gap * (n_bars - 1)) / n_bars)

            def xp(i):
                return pad_l + i * (bar_w + bar_gap)
            def yp(c):
                return pad_t + plot_h - (c / y_max) * plot_h

            bars = ""
            for i, c in enumerate(counts):
                bh = (c / y_max) * plot_h if y_max > 0 else 0
                bars += f'<rect x="{round(xp(i),1)}" y="{round(yp(c),1)}" width="{round(bar_w,1)}" height="{round(bh,1)}" fill="{color}" rx="1" opacity="0.82"/>'

            # Y ticks — evenly spaced, always start at 0
            y_ticks = ""
            for val in y_ticks_vals:
                ty = round(yp(val), 1)
                if pad_t - 4 <= ty <= pad_t + plot_h + 4:  # only draw if in range
                    y_ticks += f'<line x1="{pad_l}" y1="{ty}" x2="{pad_l + plot_w}" y2="{ty}" stroke="#ececec" stroke-width="1"/>'
                    y_ticks += f'<text x="{pad_l - 6}" y="{ty + 4}" text-anchor="end" font-size="11" fill="#999">{val}</text>'

            # X labels — evenly spaced
            x_lbls = ""
            for i in range(0, n_bars, x_step):
                lx = round(xp(i) + bar_w / 2, 1)
                x_lbls += f'<text x="{lx}" y="{chart_h - 4}" text-anchor="middle" font-size="10" fill="#999">{x_vals[i]}</text>'
            # Always label the last value if not already labeled
            last_i = n_bars - 1
            if last_i % x_step != 0:
                lx = round(xp(last_i) + bar_w / 2, 1)
                x_lbls += f'<text x="{lx}" y="{chart_h - 4}" text-anchor="middle" font-size="10" fill="#999">{x_vals[last_i]}</text>'

            # axis lines
            axes = (
                f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+plot_h}" stroke="#bbb" stroke-width="1.5"/>'
                f'<line x1="{pad_l}" y1="{pad_t+plot_h}" x2="{pad_l+plot_w}" y2="{pad_t+plot_h}" stroke="#bbb" stroke-width="1.5"/>'
                f'<text x="{pad_l - 36}" y="{pad_t + plot_h//2}" text-anchor="middle" font-size="11" fill="#666" '
                f'transform="rotate(-90,{pad_l-36},{pad_t + plot_h//2})">Number of Cases</text>'
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

        # ── Four-Pattern Comparison ──────────────────────────────
        st.divider()
        st.subheader("📊 All Four Epidemic Patterns — Side by Side")
        st.markdown("""
Each pattern has a distinct shape. The **dotted baseline** shows the expected background rate — the number of cases you would see in any given time period without an outbreak. Cases above this line represent **excess cases** attributable to the event or transmission chain.
        """)

        import streamlit.components.v1 as _comp4

        four_panel_html = """
<!DOCTYPE html>
<html>
<head>
<style>
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f8fafc; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px; }
  .panel { background: white; border-radius: 10px; padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
  .panel-title { font-size: 13px; font-weight: 700; margin-bottom: 2px; }
  .panel-subtitle { font-size: 11px; color: #6b7280; margin-bottom: 10px; }
  svg { width: 100%; overflow: visible; }
  .legend { font-size: 10px; color: #6b7280; margin-top: 8px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
  .legend-item { display: flex; align-items: center; gap: 4px; }
  .key-box { background: #f0f9ff; border-left: 3px solid #0284c7; padding: 8px 10px; margin-top: 10px; border-radius: 0 6px 6px 0; font-size: 11px; color: #0c4a6e; line-height: 1.5; }
</style>
</head>
<body>
<div class="grid">

  <!-- PANEL 1: POINT SOURCE -->
  <div class="panel">
    <div class="panel-title" style="color:#dc2626;">☢️ Point Source</div>
    <div class="panel-subtitle">All cases exposed to same source at same time</div>
    <svg viewBox="0 0 280 140" xmlns="http://www.w3.org/2000/svg">
      <!-- Axes -->
      <line x1="35" y1="110" x2="270" y2="110" stroke="#d1d5db" stroke-width="1.5"/>
      <line x1="35" y1="10" x2="35" y2="110" stroke="#d1d5db" stroke-width="1.5"/>
      <!-- Baseline dotted -->
      <line x1="35" y1="100" x2="270" y2="100" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5,3"/>
      <text x="272" y="103" font-size="8" fill="#94a3b8">baseline</text>
      <!-- Bars: sharp peak, bell shaped, all within short window -->
      <!-- Time points 1-12, bars only at 3-7 -->
      <rect x="68" y="102" width="14" height="8" fill="#fca5a5" rx="1"/>
      <rect x="88" y="88" width="14" height="22" fill="#f87171" rx="1"/>
      <rect x="108" y="68" width="14" height="42" fill="#ef4444" rx="1"/>
      <rect x="128" y="48" width="14" height="62" fill="#dc2626" rx="1"/>
      <rect x="148" y="62" width="14" height="48" fill="#ef4444" rx="1"/>
      <rect x="168" y="82" width="14" height="28" fill="#f87171" rx="1"/>
      <rect x="188" y="98" width="14" height="12" fill="#fca5a5" rx="1"/>
      <!-- Peak annotation -->
      <line x1="135" y1="44" x2="135" y2="20" stroke="#dc2626" stroke-width="1" stroke-dasharray="3,2"/>
      <text x="138" y="18" font-size="9" fill="#dc2626" font-weight="bold">Peak</text>
      <!-- X axis labels -->
      <text x="33" y="122" font-size="8" fill="#9ca3af" text-anchor="middle">0</text>
      <text x="75" y="122" font-size="8" fill="#9ca3af" text-anchor="middle">2</text>
      <text x="115" y="122" font-size="8" fill="#9ca3af" text-anchor="middle">4</text>
      <text x="155" y="122" font-size="8" fill="#9ca3af" text-anchor="middle">6</text>
      <text x="195" y="122" font-size="8" fill="#9ca3af" text-anchor="middle">8</text>
      <text x="235" y="122" font-size="8" fill="#9ca3af" text-anchor="middle">10</text>
      <!-- Bracket showing incubation width -->
      <line x1="70" y1="130" x2="200" y2="130" stroke="#dc2626" stroke-width="1.2"/>
      <line x1="70" y1="127" x2="70" y2="133" stroke="#dc2626" stroke-width="1.2"/>
      <line x1="200" y1="127" x2="200" y2="133" stroke="#dc2626" stroke-width="1.2"/>
      <text x="135" y="139" font-size="8" fill="#dc2626" text-anchor="middle">≈ incubation period range</text>
      <!-- Y label -->
      <text x="12" y="65" font-size="8" fill="#9ca3af" transform="rotate(-90,12,65)">Cases</text>
      <text x="150" y="115" font-size="8" fill="#9ca3af" text-anchor="middle">Time</text>
    </svg>
    <div class="key-box">
      <b>Shape:</b> Single sharp peak, rapid rise and fall<br>
      <b>Duration:</b> Width ≈ one incubation period<br>
      <b>Baseline:</b> Cases drop back to baseline quickly<br>
      <b>No secondary waves</b> — exposure ended
    </div>
  </div>

  <!-- PANEL 2: PROPAGATED -->
  <div class="panel">
    <div class="panel-title" style="color:#1d4ed8;">🔗 Propagated (Person-to-Person)</div>
    <div class="panel-subtitle">Each generation infects the next — successive waves</div>
    <svg viewBox="0 0 280 140" xmlns="http://www.w3.org/2000/svg">
      <!-- Axes -->
      <line x1="35" y1="110" x2="270" y2="110" stroke="#d1d5db" stroke-width="1.5"/>
      <line x1="35" y1="10" x2="35" y2="110" stroke="#d1d5db" stroke-width="1.5"/>
      <!-- Baseline dotted -->
      <line x1="35" y1="100" x2="270" y2="100" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5,3"/>
      <text x="272" y="103" font-size="8" fill="#94a3b8">baseline</text>
      <!-- Wave 1: small (index cases) -->
      <rect x="42" y="96" width="10" height="14" fill="#93c5fd" rx="1"/>
      <rect x="54" y="90" width="10" height="20" fill="#60a5fa" rx="1"/>
      <rect x="66" y="96" width="10" height="14" fill="#93c5fd" rx="1"/>
      <!-- Wave 2: medium -->
      <rect x="88" y="88" width="10" height="22" fill="#60a5fa" rx="1"/>
      <rect x="100" y="76" width="10" height="34" fill="#3b82f6" rx="1"/>
      <rect x="112" y="82" width="10" height="28" fill="#3b82f6" rx="1"/>
      <rect x="124" y="90" width="10" height="20" fill="#60a5fa" rx="1"/>
      <!-- Wave 3: large -->
      <rect x="146" y="72" width="10" height="38" fill="#2563eb" rx="1"/>
      <rect x="158" y="50" width="10" height="60" fill="#1d4ed8" rx="1"/>
      <rect x="170" y="58" width="10" height="52" fill="#1d4ed8" rx="1"/>
      <rect x="182" y="72" width="10" height="38" fill="#2563eb" rx="1"/>
      <rect x="194" y="84" width="10" height="26" fill="#60a5fa" rx="1"/>
      <!-- Wave 4: declining -->
      <rect x="216" y="88" width="10" height="22" fill="#93c5fd" rx="1"/>
      <rect x="228" y="94" width="10" height="16" fill="#bfdbfe" rx="1"/>
      <rect x="240" y="98" width="10" height="12" fill="#bfdbfe" rx="1"/>
      <!-- Wave labels -->
      <text x="58" y="85" font-size="8" fill="#1d4ed8" text-anchor="middle">W1</text>
      <text x="108" y="71" font-size="8" fill="#1d4ed8" text-anchor="middle">W2</text>
      <text x="168" y="45" font-size="8" fill="#1d4ed8" text-anchor="middle">W3</text>
      <text x="228" y="83" font-size="8" fill="#1d4ed8" text-anchor="middle">W4</text>
      <!-- Incubation period arrows between waves -->
      <line x1="65" y1="75" x2="98" y2="75" stroke="#1d4ed8" stroke-width="1" marker-end="url(#arrow)" stroke-dasharray="3,2"/>
      <line x1="133" y1="75" x2="146" y2="75" stroke="#1d4ed8" stroke-width="1" stroke-dasharray="3,2"/>
      <text x="82" y="72" font-size="7" fill="#1d4ed8" text-anchor="middle">≈1 incub.</text>
      <!-- Y label -->
      <text x="12" y="65" font-size="8" fill="#9ca3af" transform="rotate(-90,12,65)">Cases</text>
      <text x="150" y="115" font-size="8" fill="#9ca3af" text-anchor="middle">Time</text>
    </svg>
    <div class="key-box">
      <b>Shape:</b> Multiple waves, each growing then shrinking<br>
      <b>Wave spacing:</b> ≈ one incubation period apart<br>
      <b>Duration:</b> Weeks to months<br>
      <b>Rises above baseline</b> repeatedly until immunity exhausted
    </div>
  </div>

  <!-- PANEL 3: MIXED -->
  <div class="panel">
    <div class="panel-title" style="color:#7c3aed;">🔀 Mixed</div>
    <div class="panel-subtitle">Point source followed by person-to-person spread</div>
    <svg viewBox="0 0 280 140" xmlns="http://www.w3.org/2000/svg">
      <!-- Axes -->
      <line x1="35" y1="110" x2="270" y2="110" stroke="#d1d5db" stroke-width="1.5"/>
      <line x1="35" y1="10" x2="35" y2="110" stroke="#d1d5db" stroke-width="1.5"/>
      <!-- Baseline dotted -->
      <line x1="35" y1="100" x2="270" y2="100" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5,3"/>
      <text x="272" y="103" font-size="8" fill="#94a3b8">baseline</text>
      <!-- Point source initial peak (red) -->
      <rect x="48" y="72" width="11" height="38" fill="#fca5a5" rx="1"/>
      <rect x="61" y="50" width="11" height="60" fill="#f87171" rx="1"/>
      <rect x="74" y="40" width="11" height="70" fill="#ef4444" rx="1"/>
      <rect x="87" y="55" width="11" height="55" fill="#f87171" rx="1"/>
      <rect x="100" y="75" width="11" height="35" fill="#fca5a5" rx="1"/>
      <!-- Secondary wave (purple) -->
      <rect x="128" y="82" width="11" height="28" fill="#c4b5fd" rx="1"/>
      <rect x="141" y="65" width="11" height="45" fill="#a78bfa" rx="1"/>
      <rect x="154" y="58" width="11" height="52" fill="#8b5cf6" rx="1"/>
      <rect x="167" y="68" width="11" height="42" fill="#7c3aed" rx="1"/>
      <rect x="180" y="78" width="11" height="32" fill="#a78bfa" rx="1"/>
      <rect x="193" y="88" width="11" height="22" fill="#c4b5fd" rx="1"/>
      <!-- Third wave (smaller) -->
      <rect x="218" y="90" width="11" height="20" fill="#ddd6fe" rx="1"/>
      <rect x="231" y="94" width="11" height="16" fill="#ede9fe" rx="1"/>
      <rect x="244" y="98" width="11" height="12" fill="#ede9fe" rx="1"/>
      <!-- Labels -->
      <text x="74" y="35" font-size="8" fill="#dc2626" text-anchor="middle" font-weight="bold">Point source</text>
      <text x="163" y="53" font-size="8" fill="#7c3aed" text-anchor="middle" font-weight="bold">2° spread</text>
      <!-- Arrow from initial peak to secondary -->
      <path d="M 110 65 Q 120 55 128 70" stroke="#7c3aed" stroke-width="1.2" fill="none" stroke-dasharray="3,2"/>
      <!-- Y label -->
      <text x="12" y="65" font-size="8" fill="#9ca3af" transform="rotate(-90,12,65)">Cases</text>
      <text x="150" y="115" font-size="8" fill="#9ca3af" text-anchor="middle">Time</text>
    </svg>
    <div class="key-box">
      <b>Shape:</b> Initial sharp peak (point source), then waves<br>
      <b>Initial peak:</b> Red = common vehicle (food, water)<br>
      <b>Secondary waves:</b> Purple = person-to-person spread<br>
      <b>Example:</b> Contaminated water + household spread
    </div>
  </div>

  <!-- PANEL 4: ENDEMIC -->
  <div class="panel">
    <div class="panel-title" style="color:#166534;">📈 Endemic</div>
    <div class="panel-subtitle">Stable background rate — disease is always present</div>
    <svg viewBox="0 0 280 140" xmlns="http://www.w3.org/2000/svg">
      <!-- Axes -->
      <line x1="35" y1="110" x2="270" y2="110" stroke="#d1d5db" stroke-width="1.5"/>
      <line x1="35" y1="10" x2="35" y2="110" stroke="#d1d5db" stroke-width="1.5"/>
      <!-- Epidemic threshold (higher dotted line) -->
      <line x1="35" y1="72" x2="270" y2="72" stroke="#dc2626" stroke-width="1.2" stroke-dasharray="4,3"/>
      <text x="272" y="70" font-size="8" fill="#dc2626">epidemic threshold</text>
      <!-- Baseline (endemic level) dotted -->
      <line x1="35" y1="90" x2="270" y2="90" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5,3"/>
      <text x="272" y="93" font-size="8" fill="#94a3b8">endemic baseline</text>
      <!-- Endemic bars — fluctuating around baseline, never crossing epidemic threshold -->
      <rect x="42" y="88" width="13" height="22" fill="#86efac" rx="1"/>
      <rect x="58" y="83" width="13" height="27" fill="#4ade80" rx="1"/>
      <rect x="74" y="78" width="13" height="32" fill="#22c55e" rx="1"/>
      <rect x="90" y="85" width="13" height="25" fill="#4ade80" rx="1"/>
      <rect x="106" y="87" width="13" height="23" fill="#86efac" rx="1"/>
      <rect x="122" y="82" width="13" height="28" fill="#4ade80" rx="1"/>
      <rect x="138" y="76" width="13" height="34" fill="#16a34a" rx="1"/>
      <rect x="154" y="80" width="13" height="30" fill="#22c55e" rx="1"/>
      <rect x="170" y="85" width="13" height="25" fill="#4ade80" rx="1"/>
      <rect x="186" y="83" width="13" height="27" fill="#4ade80" rx="1"/>
      <rect x="202" y="79" width="13" height="31" fill="#22c55e" rx="1"/>
      <rect x="218" y="86" width="13" height="24" fill="#86efac" rx="1"/>
      <rect x="234" y="88" width="13" height="22" fill="#86efac" rx="1"/>
      <rect x="250" y="84" width="13" height="26" fill="#4ade80" rx="1"/>
      <!-- Bracket showing variation around baseline -->
      <line x1="36" y1="76" x2="36" y2="94" stroke="#166534" stroke-width="1.2"/>
      <line x1="33" y1="76" x2="39" y2="76" stroke="#166534" stroke-width="1.2"/>
      <line x1="33" y1="94" x2="39" y2="94" stroke="#166534" stroke-width="1.2"/>
      <text x="28" y="88" font-size="8" fill="#166534" text-anchor="middle" transform="rotate(-90,28,88)">variation</text>
      <!-- Y label -->
      <text x="12" y="65" font-size="8" fill="#9ca3af" transform="rotate(-90,12,65)">Cases</text>
      <text x="150" y="115" font-size="8" fill="#9ca3af" text-anchor="middle">Time (months/years)</text>
    </svg>
    <div class="key-box">
      <b>Shape:</b> Fluctuates around a stable baseline — no single peak<br>
      <b>Baseline:</b> Expected background rate for this population<br>
      <b>Epidemic threshold:</b> Red dashes — if cases exceed this, investigate<br>
      <b>Examples:</b> Malaria in endemic zones, TB, seasonal flu baseline
    </div>
  </div>

</div>

<!-- Legend row -->
<div style="padding: 4px 16px 12px 16px; display: flex; gap: 24px; font-size: 11px; color: #6b7280; flex-wrap: wrap;">
  <div style="display:flex;align-items:center;gap:6px;">
    <svg width="30" height="10"><line x1="0" y1="5" x2="30" y2="5" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5,3"/></svg>
    Endemic baseline — expected background case count
  </div>
  <div style="display:flex;align-items:center;gap:6px;">
    <svg width="30" height="10"><line x1="0" y1="5" x2="30" y2="5" stroke="#dc2626" stroke-width="1.2" stroke-dasharray="4,3"/></svg>
    Epidemic threshold — cases above this warrant investigation
  </div>
</div>
</body>
</html>
"""
        _comp4.html(four_panel_html, height=700, scrolling=False)

        st.markdown("""
**Reading the baseline:** The dotted gray line represents the **expected endemic level** — how many cases occur in any given time period without an unusual event. Cases above the baseline represent excess cases attributable to the outbreak or epidemic. An **epidemic threshold** (red dashes, shown in the endemic panel) represents one possible alert level — a common teaching heuristic uses 2 standard deviations above the historical mean, though real surveillance systems use varied approaches. When cases cross the threshold, formal outbreak investigation is triggered.
        """)


    elif df_section == "5️⃣ Person, Place & Time":
        st.subheader("Descriptive Epidemiology — Person, Place, and Time")
        st.markdown("""
**Descriptive epidemiology** characterizes the distribution of disease in a population. The classic organizing framework is **person, place, and time** — three dimensions that, taken together, generate hypotheses about causation by revealing *who* gets sick, *where*, and *when*.

Descriptive epi precedes analytic epi. You cannot test a hypothesis you haven't formed yet. Person-place-time analysis is how you form it.
        """)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
<div style="background:#e3f2fd;border:2px solid #1565c0;border-radius:10px;padding:16px;height:100%;">
<div style="font-weight:700;font-size:15px;color:#1565c0;margin-bottom:8px;">👤 Person</div>
<div style="font-size:12px;color:#334155;line-height:1.6;">
<b>Who</b> is getting sick?<br><br>
• Age<br>• Sex / gender<br>• Race / ethnicity<br>• Socioeconomic status<br>• Occupation<br>• Behavior (diet, activity, smoking)<br>• Immune status<br>• Marital / household status
<br><br><i>Generates hypotheses about host susceptibility and exposure patterns</i>
</div>
</div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
<div style="background:#e8f5e9;border:2px solid #2e7d32;border-radius:10px;padding:16px;height:100%;">
<div style="font-weight:700;font-size:15px;color:#2e7d32;margin-bottom:8px;">📍 Place</div>
<div style="font-size:12px;color:#334155;line-height:1.6;">
<b>Where</b> is disease concentrated?<br><br>
• Geographic region / country<br>• Urban vs. rural<br>• Neighborhood / census tract<br>• Workplace / school<br>• Healthcare setting<br>• Country of birth<br>• Water / food source
<br><br><i>Generates hypotheses about environmental exposures and transmission routes</i>
</div>
</div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
<div style="background:#fce4ec;border:2px solid #c62828;border-radius:10px;padding:16px;height:100%;">
<div style="font-weight:700;font-size:15px;color:#c62828;margin-bottom:8px;">📅 Time</div>
<div style="font-size:12px;color:#334155;line-height:1.6;">
<b>When</b> does disease occur?<br><br>
• Secular (long-term) trends<br>• Cyclic / seasonal patterns<br>• Point epidemic<br>• Epidemic curve shape<br>• Time since exposure<br>• Period vs. cohort effects
<br><br><i>Generates hypotheses about incubation periods, exposure timing, and trend drivers</i>
</div>
</div>
            """, unsafe_allow_html=True)

        st.markdown("&nbsp;", unsafe_allow_html=True)

        st.markdown("#### Time patterns in epidemiology")
        st.markdown("""
| Time pattern | Description | Example |
|---|---|---|
| **Secular trend** | Long-term change over years or decades | Declining smoking rates → declining lung cancer incidence |
| **Cyclic pattern** | Regular ups and downs, often seasonal | Influenza peaking each winter |
| **Point epidemic** | Sudden sharp rise traced to a single exposure | Food poisoning at a catered event |
| **Propagated epidemic** | Successive waves from person-to-person spread | Measles in an under-vaccinated school |
| **Cohort effect** | A generation's risk differs from others due to shared exposures at a formative period | Higher lung cancer in cohorts who smoked before health warnings |
| **Period effect** | A change in risk affects all age groups simultaneously at one calendar period | Increased opioid mortality after 2010 across all ages |
        """)

        st.info("""
🔑 **From description to hypothesis:** Person-place-time analysis doesn't prove causation — it generates the *question*. 

**Example:** If you observe that cases of hepatitis A cluster in restaurant workers (person), in a single neighborhood (place), after a one-week window (time), you now have a specific hypothesis — shared food source, specific location, known incubation window — that you can test analytically.
        """)

        st.divider()
        st.divider()
        st.markdown("#### 🧠 Practice: Person, Place, and Time")
        st.markdown("Work through each scenario and identify which descriptive dimension is being used.")

        ppt_scenarios = [
            {
                "q": "**Scenario 1:** Opioid overdose deaths are highest among white males aged 25–54 in rural Appalachian counties. Which dimension identifies *rural Appalachian counties* as a key descriptor?",
                "opts": ["— Select —", "Person — demographic characteristic", "Place — geographic concentration", "Time — temporal pattern", "All three simultaneously"],
                "correct": "Place — geographic concentration",
                "fb_correct": "✅ Correct. Geographic concentration of cases = place. Rural Appalachian counties is a location descriptor that generates hypotheses about environmental factors, access to care, economic conditions, and distribution of the drug supply. Person (white males 25–54) and time would complete the descriptive picture.",
                "fb_wrong": "❌ 'Rural Appalachian counties' is a geographic descriptor — that's place. Person would be white males aged 25–54. Time would be trend data.",
                "key": "ppt_q2"
            },
            {
                "q": "**Scenario 2:** Influenza hospitalizations peak every January–February and drop to near zero by May. This pattern repeats each year. This is best described as:",
                "opts": ["— Select —", "A secular trend — consistent long-term increase", "A cyclic/seasonal pattern — regular recurrence tied to the calendar", "A point epidemic — single sharp peak from one exposure", "A cohort effect — one generation is more affected"],
                "correct": "A cyclic/seasonal pattern — regular recurrence tied to the calendar",
                "fb_correct": "✅ Correct. Seasonal patterns recur predictably. Influenza's winter peak is a cyclic pattern driven by indoor crowding, lower humidity, and seasonal immune changes. It repeats annually without a long-term directional change — distinguishing it from a secular trend.",
                "fb_wrong": "❌ A cyclic pattern repeats regularly on a calendar basis. Secular trend = long-term directional change. Point epidemic = single sharp peak. Cohort effect = one birth cohort disproportionately affected.",
                "key": "ppt_q3"
            },
            {
                "q": "**Scenario 3:** Stomach cancer rates are 5× higher in Japan than in the US. Japanese immigrants to the US develop stomach cancer at rates intermediate between Japan and US-born individuals — and their children approach US rates. Which dimension does this pattern primarily use, and what hypothesis does it suggest?",
                "opts": ["— Select —",
                         "Time — the trend toward lower rates over generations suggests secular decline",
                         "Place — comparing rates across countries suggests environmental/dietary causes, not purely genetic ones",
                         "Person — age differences explain the variation",
                         "This is purely a genetic story — place doesn't matter"],
                "correct": "Place — comparing rates across countries suggests environmental/dietary causes, not purely genetic ones",
                "fb_correct": "✅ Correct. This is a classic migration study — a place-based analysis. If stomach cancer were purely genetic, Japanese migrants to the US would maintain Japan's high rates. Instead, rates shift toward the US level within one or two generations, strongly suggesting that environmental/dietary exposures (salted foods, H. pylori prevalence) rather than genetics drive the difference. Place comparisons are powerful hypothesis generators.",
                "fb_wrong": "❌ The key observation is the geographic variation AND the shift across migration — this is place-based analysis. The generational shift does involve time, but the primary inference is about place-specific environmental exposures, not a secular trend.",
                "key": "ppt_q4"
            },
            {
                "q": "**Scenario 4:** Lung cancer mortality among men rose sharply from 1930–1990, then began declining. Among women, it rose from 1960–2000, then plateaued. This is best explained by:",
                "opts": ["— Select —",
                         "A period effect — something in the environment changed everyone at the same time",
                         "A cohort effect — generations who smoked heavily at different historical periods are experiencing consequences decades later",
                         "A place effect — lung cancer varies by region",
                         "Measurement artifact — diagnostic criteria changed"],
                "correct": "A cohort effect — generations who smoked heavily at different historical periods are experiencing consequences decades later",
                "fb_correct": "✅ Correct. Men adopted cigarette smoking in large numbers in the 1930s–40s; lung cancer followed 20–30 years later (the latency period). Women adopted smoking later (post-WWII); their lung cancer peak follows accordingly. Each generation (cohort) carries the exposure history of their era. This is a textbook cohort effect — distinguishable from a period effect because the timing differs between sexes based on when each group began smoking.",
                "fb_wrong": "❌ A period effect would affect all age groups simultaneously at one calendar period. Here the timing differs between men and women based on historical smoking patterns — that's a cohort effect. Generations who smoked heavily are experiencing consequences 20–30 years after the exposure.",
                "key": "ppt_q5"
            },
        ]

        for scen in ppt_scenarios:
            ans = st.radio(scen["q"], scen["opts"], key=scen["key"])
            if ans == scen["correct"]:
                st.success(scen["fb_correct"])
            elif ans != "— Select —":
                st.error(scen["fb_wrong"])


    elif df_section == "6️⃣ Public Health Surveillance":
        st.subheader("Public Health Surveillance")
        st.markdown("""
**Public health surveillance** is the continuous, systematic collection, analysis, and interpretation of health data, used to plan, implement, and evaluate public health practice.

Surveillance is the foundation of everything else: you cannot detect outbreaks, identify trends, target interventions, or evaluate programs without it. The CDC defines it as *information for action*.
        """)

        st.markdown("#### Active vs. Passive Surveillance")
        st.markdown("""
| Feature | Passive surveillance | Active surveillance |
|---|---|---|
| **How data are collected** | Health providers report voluntarily when they diagnose a condition | Health department actively contacts providers and seeks out cases |
| **Effort required** | Low — relies on existing reporting channels | High — requires dedicated staff and contact |
| **Sensitivity** | Lower — underreporting common | Higher — more complete case ascertainment |
| **Cost** | Lower | Higher |
| **Best use** | Ongoing background monitoring of common conditions | Outbreak investigation; verification of eradication; high-priority conditions |
| **Example** | Physician reports a tuberculosis case to the health department | Health department calls all hospitals weekly to find influenza hospitalizations |
        """)

        st.markdown("#### Notifiable Diseases")
        st.info("""
**Notifiable diseases** (also called *reportable diseases*) are conditions that clinicians and laboratories are legally required to report to public health authorities when diagnosed. Reportability is determined by state law; the CDC maintains a national list of **Nationally Notifiable Diseases** that states are encouraged to report to CDC.

**What makes a disease notifiable?**
- Potential for epidemic spread
- Severity or public health impact
- Existence of an effective intervention that requires prompt action
- Need for national surveillance data

**Examples:** Cholera, measles, HIV, gonorrhea, Salmonellosis, pertussis, hepatitis A, rabies, COVID-19 (added 2020)

⚠️ **Important limitation:** Notifiable disease data reflect only *diagnosed and reported* cases. Underreporting is the norm — particularly for conditions with stigma, for diseases where providers don't test, and for asymptomatic infections.
        """)

        st.markdown("#### The Surveillance Cycle")
        st.markdown("""
| Step | Action | Example |
|---|---|---|
| **Collection** | Gather case reports, lab data, vital records | Lab reports positive *Salmonella* culture |
| **Analysis** | Calculate rates, identify trends, compare to baseline | Rate this week vs. expected baseline for the season |
| **Interpretation** | Determine public health significance | Is this a cluster? Has a threshold been crossed? |
| **Dissemination** | Share findings with decision makers and public | MMWR report, health alert network |
| **Action** | Implement response; evaluate effectiveness | Outbreak investigation, recall, targeted vaccination |
        """)

        st.markdown("#### Surveillance data sources")
        st.markdown("""
| Source | Examples | Strengths | Limitations |
|---|---|---|---|
| **Vital records** | Death certificates, birth records | Universal, continuous, legal | Coding errors, cause-of-death inaccuracies |
| **Disease registries** | Cancer registry, trauma registry | Detailed clinical data | Voluntary, incomplete, expensive |
| **Notifiable disease reports** | CDC NNDSS | Legally mandated, national | Underreporting, reporting delays |
| **Sentinel surveillance** | Selected sites report all cases of a condition | Timely, detailed | Not nationally representative |
| **Syndromic surveillance** | ER chief complaints, pharmacy sales | Real-time, early warning | Non-specific, many false alarms |
| **Serosurveys** | Population antibody testing | Captures subclinical cases | Expensive, periodic not continuous |
        """)

        surv_scenarios = [
            {
                "q": "**Scenario 1:** A health department wants to ensure they capture every case of a rare hemorrhagic fever during an outbreak. They assign staff to call every hospital daily. This is:",
                "opts": ["— Select —",
                         "Passive surveillance — providers report when they diagnose",
                         "Active surveillance — the health department is seeking out cases",
                         "Syndromic surveillance — based on symptom patterns",
                         "Sentinel surveillance — using selected reporting sites"],
                "correct": "Active surveillance — the health department is seeking out cases",
                "fb_correct": "✅ Correct. Active surveillance: the health authority initiates contact and seeks out cases. Higher sensitivity, higher cost. Standard for outbreak investigation and high-priority conditions where completeness matters.",
                "fb_wrong": "❌ When the health department *initiates contact* and seeks cases, that is active surveillance. Passive waits for providers to report voluntarily.",
                "key": "surv_q1"
            },
            {
                "q": "**Scenario 2:** Every physician in the state is required by law to report any diagnosed case of tuberculosis to the county health department. The health department does not follow up unless a case is reported. This is:",
                "opts": ["— Select —",
                         "Active surveillance",
                         "Passive surveillance",
                         "Sentinel surveillance",
                         "Syndromic surveillance"],
                "correct": "Passive surveillance",
                "fb_correct": "✅ Correct. The provider initiates the report upon diagnosis; the health department waits to receive it. This is passive surveillance. It relies on provider compliance, leading to underreporting — especially for conditions with stigma or where providers may not test.",
                "fb_wrong": "❌ Passive surveillance = the health department waits for providers to report. The report originates with the clinician, not the health authority.",
                "key": "surv_q2"
            },
            {
                "q": "**Scenario 3:** A state health department tracks emergency department chief complaints electronically in real time, looking for unusual clusters of 'fever and rash' or 'difficulty breathing' that might signal an emerging outbreak before diagnoses are confirmed. This is:",
                "opts": ["— Select —",
                         "Passive surveillance",
                         "Active surveillance",
                         "Syndromic surveillance — using symptom clusters as early warning signals",
                         "A disease registry"],
                "correct": "Syndromic surveillance — using symptom clusters as early warning signals",
                "fb_correct": "✅ Correct. Syndromic surveillance uses clinical indicators (symptoms, chief complaints, medication sales, school absenteeism) before confirmed diagnoses to provide real-time early warning. It is non-specific — many false alarms — but timely. Used extensively for bioterrorism detection and pandemic monitoring.",
                "fb_wrong": "❌ Syndromic surveillance uses symptom patterns and proxy indicators (not confirmed diagnoses) for early outbreak detection. It's real-time but non-specific.",
                "key": "surv_q3"
            },
            {
                "q": "**Scenario 4:** A researcher wants to estimate the true prevalence of COVID-19 infection in a state, including people who never had symptoms and never tested. She draws a random sample of 5,000 residents and tests their blood for COVID-19 antibodies. This is:",
                "opts": ["— Select —",
                         "Notifiable disease surveillance — required reporting to the health department",
                         "Active surveillance — the researcher is seeking cases",
                         "A serosurvey — population-based serologic testing to estimate true exposure",
                         "Sentinel surveillance — using selected high-risk sites"],
                "correct": "A serosurvey — population-based serologic testing to estimate true exposure",
                "fb_correct": "✅ Correct. A serosurvey (serologic survey) tests a population sample for antibodies to estimate cumulative exposure — including subclinical and asymptomatic infections that would never appear in case reports. Serosurveys revealed that COVID-19's true infection prevalence was many times higher than confirmed case counts. Limitation: expensive and periodic, not continuous.",
                "fb_wrong": "❌ This is a serosurvey — blood-based testing of a population sample to detect past infection regardless of symptoms or testing history. It captures the 'iceberg' of infection below the surveillance waterline.",
                "key": "surv_q4"
            },
            {
                "q": "**Scenario 5:** State cancer registries record every newly diagnosed cancer case, including clinical details, treatment, and survival data. A researcher uses registry data to study 5-year survival trends for colon cancer. This data source is best described as:",
                "opts": ["— Select —",
                         "Notifiable disease surveillance",
                         "A disease registry",
                         "Vital records",
                         "Syndromic surveillance"],
                "correct": "A disease registry",
                "fb_correct": "✅ Correct. Disease registries systematically collect detailed clinical information on all cases of a specific disease within a defined population. Cancer registries (like SEER) enable survival analysis, incidence trends, and treatment outcomes research. Limitation: voluntary reporting means they can be incomplete; they are expensive to maintain.",
                "fb_wrong": "❌ A disease registry systematically collects detailed clinical data on a specific disease category. This is distinct from notifiable disease reports (which are law-required single reports) and vital records (which capture births and deaths only).",
                "key": "surv_q5"
            },
        ]

        for scen in surv_scenarios:
            st.divider()
            ans = st.radio(scen["q"], scen["opts"], key=scen["key"])
            if ans == scen["correct"]:
                st.success(scen["fb_correct"])
            elif ans != "— Select —":
                st.error(scen["fb_wrong"])


    elif df_section == "7️⃣ Mortality Measures & YPLL":
        st.subheader("Mortality Measures & Years of Potential Life Lost (YPLL)")
        st.markdown("""
Beyond crude mortality rates, epidemiologists use several specialized mortality measures that capture *who* is dying, *from what*, and — critically — how *premature* the deaths are. These measures guide resource allocation and reflect priorities for prevention.
        """)

        st.markdown("#### Key Mortality Rate Formulas")
        st.markdown("""
| Measure | Formula | What it captures |
|---|---|---|
| **Crude mortality rate** | (Total deaths ÷ Midyear population) × 10ⁿ | Overall death burden; confounded by age structure |
| **Age-specific mortality rate** | (Deaths in age group ÷ Population in age group) × 10ⁿ | Mortality at a specific life stage |
| **Cause-specific mortality rate** | (Deaths from cause X ÷ Midyear population) × 10ⁿ | Burden of a specific disease on the population |
| **Case fatality rate (CFR)** | (Deaths from disease ÷ Cases of disease) × 100 | Severity of a disease once acquired |
| **Infant mortality rate (IMR)** | (Deaths under 1 year ÷ Live births) × 1,000 | Child survival; widely used population health indicator |
| **Maternal mortality ratio** | (Maternal deaths ÷ Live births) × 100,000 | Risk of death from pregnancy/childbirth |
| **Proportional mortality ratio (PMR)** | (Deaths from cause X ÷ Total deaths) × 100 | Fraction of all deaths attributable to a cause |
        """)

        st.info("""
⚠️ **PMR vs. cause-specific mortality rate:** The PMR tells you what *proportion* of deaths are from a cause — it says nothing about the absolute risk of dying from that cause in the population. A high PMR for heart disease could mean heart disease is common OR that everything else has been controlled.
        """)

        st.divider()
        st.markdown("#### Years of Potential Life Lost (YPLL)")
        st.markdown("""
**YPLL** measures the impact of *premature mortality* — deaths that occur before an expected age. Unlike crude mortality rates (which count all deaths equally), YPLL gives more weight to deaths among young people.

**Formula:** For each death before the reference age (usually **75 years** in the US):

> YPLL = Reference age (75) − Age at death

**Example:** A person dies at age 25 → contributes 50 YPLL. A person dies at age 70 → contributes 5 YPLL. A person dies at age 80 → contributes 0 YPLL (did not die prematurely by this definition).

**Population YPLL rate** = (Total YPLL ÷ Population under 75) × 100,000
        """)

        st.markdown("""
| Why YPLL matters | Explanation |
|---|---|
| **Captures burden of premature death** | Injuries and homicides rank higher in YPLL than in crude mortality because they kill young people |
| **Reframes prevention priorities** | Cancer and heart disease dominate crude death counts; unintentional injuries dominate YPLL |
| **Policy relevance** | A dollar spent preventing a death at age 30 prevents more YPLL than one preventing a death at age 72 |
| **Leading causes shift** | In YPLL rankings, unintentional injury, suicide, and homicide rise dramatically relative to their crude death rank |
        """)

        st.divider()
        st.markdown("#### 🧮 YPLL Calculator")
        ref_age = 75
        col_y1, col_y2 = st.columns(2)
        with col_y1:
            n_deaths = st.number_input("Number of deaths to enter:", min_value=1, max_value=20, value=3, key="ypll_n")
        ages = []
        for i in range(int(n_deaths)):
            age = st.number_input(f"Age at death #{i+1}:", min_value=0, max_value=74, value=min(30 + i*10, 74), key=f"ypll_age_{i}")
            ages.append(age)

        if st.button("Calculate YPLL", key="ypll_calc"):
            ypll_list = [(age, ref_age - age) for age in ages]
            total_ypll = sum(y for _, y in ypll_list)
            import pandas as pd
            ypll_df = pd.DataFrame(ypll_list, columns=["Age at death", "YPLL contributed"])
            ypll_df["Age at death"] = ypll_df["Age at death"].astype(int)
            st.dataframe(ypll_df, use_container_width=True, hide_index=True)
            st.metric("Total YPLL", total_ypll)
            avg_age = sum(ages) / len(ages)
            st.success(f"These {len(ages)} deaths contribute a total of **{total_ypll} years of potential life lost** (reference age = 75). Average age at death = {round(avg_age,1)} years. Contrast with a scenario where all deaths occurred at age 74 — that would yield only {len(ages)} YPLL total.")

        ypll_q = st.radio(
            "**Concept check:** Unintentional injuries rank #5 in crude US mortality but #1 in YPLL. What does this tell you?",
            ["— Select —",
             "Injuries are actually less important than heart disease for prevention",
             "Injuries disproportionately kill young people — each death removes many more years of life",
             "The YPLL method overcounts injury deaths",
             "YPLL and crude mortality always give the same ranking"],
            key="ypll_q1"
        )
        if ypll_q == "Injuries disproportionately kill young people — each death removes many more years of life":
            st.success("✅ Correct. Injuries are a leading killer of people aged 1–44. Each death in this age group contributes 30–74 YPLL. Heart disease deaths, though more numerous, tend to occur in older adults (70s–80s) and contribute few or no YPLL per death. YPLL thus ranks injuries as a top prevention priority even when raw death counts would suggest otherwise.")
        elif ypll_q != "— Select —":
            st.error("❌ YPLL weights deaths by how young the person was — more years lost per death = higher YPLL rank. Injuries kill disproportionately young people, making them rank higher in YPLL than in crude mortality counts.")


    st.markdown("---")
    st.markdown("*Strong epidemiologists think structurally before computing.*")


# ==================================================
# MODULE 2: SCREENING & DIAGNOSTIC TESTS
# ==================================================
elif current_page == "screening":
    st.title("🔬 Screening & Diagnostic Tests")
    st.markdown("Evaluating the performance of a test requires understanding how sensitivity, specificity, and the prevalence of disease in the population interact.")

    screen_section = st.radio("Section:", [
        "1️⃣ Core Concepts",
        "2️⃣ Interactive 2×2 Calculator",
        "3️⃣ Likelihood Ratios & Fagan Nomogram",
        "4️⃣ Prevalence Effect on PPV",
        "5️⃣ Wilson & Jungner Criteria",
        "6️⃣ ROC Curve"
    ], horizontal=True)
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
            "Mammography & Breast Cancer (50-year-olds)": {
                "a": 84, "b": 107, "c": 16, "d": 793,
                "desc": (
                    "Based on US Breast Cancer Surveillance Consortium (BCSC) data for women aged 50–59. "
                    "Sensitivity ~84%, Specificity ~88%, Prevalence ~10% (symptomatic/referred population). "
                    "Note: in true population screening (prevalence ~1%), PPV drops dramatically — "
                    "use the Prevalence Effect on PPV section to explore this. "
                    "Source: Kerlikowske et al., JAMA 1996; BCSC data."
                )
            },
            "Rapid Strep Test (RADT)": {
                "a": 257, "b": 32, "c": 43, "d": 668,
                "desc": (
                    "Based on Cochrane systematic review (Cohen et al., 2016) — 105 evaluations, 58,244 children. "
                    "Summary sensitivity 85.6% (95% CI 83.3–87.6%), specificity 95.4% (95% CI 94.5–96.2%). "
                    "Prevalence ~30% in symptomatic patients presenting with pharyngitis. "
                    "High specificity means positive result is reliable; moderate sensitivity means negative result "
                    "should prompt culture confirmation in children."
                )
            },
            "PSA Screening (>4 ng/mL) & Prostate Cancer": {
                "a": 31, "b": 54, "c": 119, "d": 796,
                "desc": (
                    "Based on the Prostate Cancer Prevention Trial (PCPT) placebo arm. "
                    "At the standard 4 ng/mL cutoff: sensitivity ~21%, specificity ~94%. "
                    "Prevalence ~15% in screened men ≥55 years. "
                    "Note the very low sensitivity — nearly 80% of cancers are missed at this threshold. "
                    "This is why PSA screening guidelines remain contested. "
                    "Source: Thompson et al., J Natl Cancer Inst 2005; Etzioni et al."
                )
            },
        }

        preset_choice = st.selectbox("Load a preset:", list(SCREEN_PRESETS.keys()), key="screen_preset")
        preset = SCREEN_PRESETS[preset_choice]
        if preset and preset.get("desc"):
            st.info(preset["desc"])

        # Force widget values to update when preset changes
        if preset:
            # Only overwrite if the preset selection just changed
            prev_key = "screen_preset_prev"
            if st.session_state.get(prev_key) != preset_choice:
                st.session_state["sc_a"] = preset["a"]
                st.session_state["sc_b"] = preset["b"]
                st.session_state["sc_c"] = preset["c"]
                st.session_state["sc_d"] = preset["d"]
                st.session_state[prev_key] = preset_choice

        col1, col2, col3, col4 = st.columns(4)
        defaults = preset if preset else {"a": 90, "b": 10, "c": 10, "d": 890}
        # Initialize session state keys if not yet set
        for k, v in [("sc_a", defaults["a"]), ("sc_b", defaults["b"]),
                     ("sc_c", defaults["c"]), ("sc_d", defaults["d"])]:
            if k not in st.session_state:
                st.session_state[k] = v
        a = col1.number_input("a (TP)", min_value=0, key="sc_a")
        b = col2.number_input("b (FP)", min_value=0, key="sc_b")
        c = col3.number_input("c (FN)", min_value=0, key="sc_c")
        d = col4.number_input("d (TN)", min_value=0, key="sc_d")

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
            lr_pos = sens / (1 - spec) if spec < 1 else float('inf')
            lr_neg = (1 - sens) / spec if spec > 0 else float('inf')

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Sensitivity", f"{round(sens*100,1)}%")
            col2.metric("Specificity", f"{round(spec*100,1)}%")
            col3.metric("PPV", f"{round(ppv*100,1)}%")
            col4.metric("NPV", f"{round(npv*100,1)}%")
            col5.metric("Prevalence", f"{round(prev*100,1)}%")

            # LR row
            st.markdown("##### Likelihood Ratios")
            lrc1, lrc2, lrc3 = st.columns([1, 1, 2])
            lrp_str = f"{round(lr_pos, 2)}" if lr_pos != float('inf') else "∞"
            lrn_str = f"{round(lr_neg, 3)}"

            # Interpret LR+
            if lr_pos >= 10:
                lrp_interp = "🟢 Strong rule-in (≥10)"
            elif lr_pos >= 5:
                lrp_interp = "🟡 Moderate rule-in (5–10)"
            elif lr_pos >= 2:
                lrp_interp = "🟠 Weak rule-in (2–5)"
            else:
                lrp_interp = "🔴 Uninformative (<2)"

            # Interpret LR−
            if lr_neg <= 0.1:
                lrn_interp = "🟢 Strong rule-out (≤0.1)"
            elif lr_neg <= 0.2:
                lrn_interp = "🟡 Moderate rule-out (0.1–0.2)"
            elif lr_neg <= 0.5:
                lrn_interp = "🟠 Weak rule-out (0.2–0.5)"
            else:
                lrn_interp = "🔴 Uninformative (>0.5)"

            with lrc1:
                st.metric("LR+", lrp_str)
                st.caption(f"Sens ÷ (1−Spec) = {round(sens,3)} ÷ {round(1-spec,3)}")
                st.caption(lrp_interp)
            with lrc2:
                st.metric("LR−", lrn_str)
                st.caption(f"(1−Sens) ÷ Spec = {round(1-sens,3)} ÷ {round(spec,3)}")
                st.caption(lrn_interp)
            with lrc3:
                st.info("""
**Reading likelihood ratios:**
- **LR+ ≥ 10** → positive result strongly increases disease probability
- **LR+ 2–5** → modest increase; useful but not definitive
- **LR− ≤ 0.1** → negative result strongly decreases disease probability
- **LR− 0.2–0.5** → modest decrease; useful but not definitive
- **LR = 1.0** → test result changes nothing
                """)

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
| LR+ | Sens ÷ (1−Spec) | {round(sens,3)} ÷ {round(1-spec,3)} | **{lrp_str}** |
| LR− | (1−Sens) ÷ Spec | {round(1-sens,3)} ÷ {round(spec,3)} | **{lrn_str}** |
                """)

    elif screen_section == "3️⃣ Likelihood Ratios & Fagan Nomogram":
        st.subheader("Likelihood Ratios — Updating Probability with Test Results")
        st.markdown("""
Sensitivity, specificity, PPV, and NPV all depend on the prevalence of disease in the population being tested. **Likelihood ratios (LRs)** are different — they describe the test's inherent discriminating ability and can be applied to any patient with a known pre-test probability.

**LR+ = Sensitivity ÷ (1 − Specificity)** — how much more likely a positive result is in a diseased person vs. a healthy person

**LR− = (1 − Sensitivity) ÷ Specificity** — how much less likely a negative result is in a diseased person vs. a healthy person
        """)

        st.markdown("""
| LR+ value | Interpretation | Rule-of-thumb |
|---|---|---|
| > 10 | Large, often conclusive increase in probability | Strong rule-in |
| 5–10 | Moderate increase | Moderate rule-in |
| 2–5 | Small increase | Weak rule-in |
| 1–2 | Minimal/no change | Uninformative |
| **LR− value** | | |
| < 0.1 | Large, often conclusive decrease in probability | Strong rule-out |
| 0.1–0.2 | Moderate decrease | Moderate rule-out |
| 0.2–0.5 | Small decrease | Weak rule-out |
| > 0.5 | Minimal/no change | Uninformative |
        """)

        st.info("""
🔑 **How to use an LR:** Convert pre-test probability → pre-test odds → multiply by LR → post-test odds → convert back to post-test probability.

**Pre-test odds** = Pre-test probability ÷ (1 − Pre-test probability)

**Post-test odds** = Pre-test odds × LR

**Post-test probability** = Post-test odds ÷ (1 + Post-test odds)
        """)

        st.divider()
        st.subheader("🧮 Dynamic Fagan Nomogram")
        st.markdown("""
The **Fagan nomogram** is a graphical tool that performs the pre-test → post-test probability conversion visually. Draw a line from pre-test probability (left) through LR (center) to read post-test probability (right).
        """)

        import math as _fm
        import streamlit.components.v1 as _fagan_comp

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            pre_prob = st.slider("Pre-test probability (%):", 1, 99, 20, 1, key="fagan_pre") / 100
        with col_f2:
            lr_type = st.radio("Apply:", ["LR+ (positive result)", "LR− (negative result)"], key="fagan_lr_type", horizontal=True)
        with col_f3:
            if "LR+" in lr_type:
                lr_val = st.slider("LR+ value:", 0.5, 50.0, 5.0, 0.5, key="fagan_lrp")
            else:
                lr_val = st.slider("LR− value:", 0.01, 2.0, 0.1, 0.01, key="fagan_lrn")

        # Calculate post-test probability
        pre_odds = pre_prob / (1 - pre_prob)
        post_odds = pre_odds * lr_val
        post_prob = post_odds / (1 + post_odds)

        # Show metrics
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Pre-test probability", f"{round(pre_prob*100,1)}%")
        mc2.metric("LR applied", round(lr_val, 2))
        mc3.metric("Post-test probability", f"{round(post_prob*100,1)}%",
                   delta=f"{round((post_prob - pre_prob)*100,1)} pp",
                   delta_color="normal" if post_prob > pre_prob else "inverse")

        # Build Fagan nomogram SVG
        # Three vertical log-odds scales: pre-test prob (left), LR (center), post-test prob (right)
        NW, NH = 480, 400
        pad_top, pad_bot = 40, 40
        scale_h = NH - pad_top - pad_bot

        def prob_to_y(p, h=scale_h, top=pad_top):
            """Convert probability to y position using log-odds scale"""
            p = max(0.001, min(0.999, p))
            odds = p / (1 - p)
            log_odds = math.log10(odds)
            lo_min, lo_max = math.log10(0.001/0.999), math.log10(0.999/0.001)
            frac = (log_odds - lo_min) / (lo_max - lo_min)
            return top + (1 - frac) * h

        def lr_to_y(lr, h=scale_h, top=pad_top):
            """Convert LR to y position using log scale"""
            lr = max(0.001, min(1000, lr))
            log_lr = math.log10(lr)
            lr_min, lr_max = math.log10(0.001), math.log10(1000)
            frac = (log_lr - lr_min) / (lr_max - lr_min)
            return top + (1 - frac) * h

        # Scale positions
        x_pre = 80
        x_lr  = NW // 2
        x_post = NW - 80

        # Pre-test probability ticks
        pre_ticks  = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4,
                      0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        lr_ticks   = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5,
                      1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        post_ticks = pre_ticks

        def tick_label(p):
            if p >= 0.1: return f"{round(p*100)}"
            elif p >= 0.01: return f"{round(p*100,1)}"
            else: return f"{p*100:.1f}"

        def lr_label(v):
            if v >= 1: return str(int(v)) if v == int(v) else str(round(v,1))
            else: return str(round(v,3)).rstrip('0').rstrip('.')

        pre_tick_svg = ""
        for p in pre_ticks:
            y = prob_to_y(p)
            pre_tick_svg += f'<line x1="{x_pre-6}" y1="{round(y,1)}" x2="{x_pre}" y2="{round(y,1)}" stroke="#555" stroke-width="1"/>'
            pre_tick_svg += f'<text x="{x_pre-8}" y="{round(y+3,1)}" font-size="9" text-anchor="end" fill="#444">{tick_label(p)}</text>'

        lr_tick_svg = ""
        for v in lr_ticks:
            y = lr_to_y(v)
            lr_tick_svg += f'<line x1="{x_lr-4}" y1="{round(y,1)}" x2="{x_lr+4}" y2="{round(y,1)}" stroke="#555" stroke-width="1"/>'
            lr_tick_svg += f'<text x="{x_lr+6}" y="{round(y+3,1)}" font-size="9" text-anchor="start" fill="#444">{lr_label(v)}</text>'

        post_tick_svg = ""
        for p in post_ticks:
            y = prob_to_y(p)
            post_tick_svg += f'<line x1="{x_post}" y1="{round(y,1)}" x2="{x_post+6}" y2="{round(y,1)}" stroke="#555" stroke-width="1"/>'
            post_tick_svg += f'<text x="{x_post+8}" y="{round(y+3,1)}" font-size="9" text-anchor="start" fill="#444">{tick_label(p)}</text>'

        # The pivot line: from pre-test through LR to post-test
        y_pre  = prob_to_y(pre_prob)
        y_lr   = lr_to_y(lr_val)
        y_post = prob_to_y(post_prob)

        line_color = "#2563eb" if "LR+" in lr_type else "#dc2626"

        nomo_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{NW}" height="{NH}"
     style="font-family:sans-serif;background:#fafafa;border-radius:8px;border:1px solid #e2e8f0;">
  <!-- Scale lines -->
  <line x1="{x_pre}" y1="{pad_top}" x2="{x_pre}" y2="{NH-pad_bot}" stroke="#888" stroke-width="1.5"/>
  <line x1="{x_lr}"  y1="{pad_top}" x2="{x_lr}"  y2="{NH-pad_bot}" stroke="#888" stroke-width="1.5"/>
  <line x1="{x_post}" y1="{pad_top}" x2="{x_post}" y2="{NH-pad_bot}" stroke="#888" stroke-width="1.5"/>

  <!-- Scale headers -->
  <text x="{x_pre}"  y="{pad_top-12}" font-size="11" font-weight="700" fill="#334155" text-anchor="middle">Pre-test %</text>
  <text x="{x_lr}"   y="{pad_top-12}" font-size="11" font-weight="700" fill="#334155" text-anchor="middle">Likelihood Ratio</text>
  <text x="{x_post}" y="{pad_top-12}" font-size="11" font-weight="700" fill="#334155" text-anchor="middle">Post-test %</text>

  <!-- Ticks -->
  {pre_tick_svg}
  {lr_tick_svg}
  {post_tick_svg}

  <!-- Pivot line through all three points -->
  <line x1="{x_pre}" y1="{round(y_pre,1)}" x2="{x_post}" y2="{round(y_post,1)}"
        stroke="{line_color}" stroke-width="2.5" stroke-dasharray="6,3" opacity="0.7"/>

  <!-- Dots at each scale -->
  <circle cx="{x_pre}"  cy="{round(y_pre,1)}"  r="7" fill="{line_color}" stroke="white" stroke-width="2"/>
  <circle cx="{x_lr}"   cy="{round(y_lr,1)}"   r="6" fill="#f59e0b"      stroke="white" stroke-width="2"/>
  <circle cx="{x_post}" cy="{round(y_post,1)}"  r="7" fill="{line_color}" stroke="white" stroke-width="2"/>

  <!-- Value labels at dots -->
  <text x="{x_pre-14}" y="{round(y_pre-9,1)}" font-size="10" font-weight="700" fill="{line_color}" text-anchor="middle">{round(pre_prob*100,1)}%</text>
  <text x="{x_lr}"     y="{round(y_lr-9,1)}"  font-size="10" font-weight="700" fill="#b45309"       text-anchor="middle">LR={round(lr_val,2)}</text>
  <text x="{x_post+14}" y="{round(y_post-9,1)}" font-size="10" font-weight="700" fill="{line_color}" text-anchor="middle">{round(post_prob*100,1)}%</text>

  <!-- LR=1 reference line -->
  <line x1="{x_lr-4}" y1="{round(lr_to_y(1),1)}" x2="{x_lr+4}" y2="{round(lr_to_y(1),1)}"
        stroke="#ef4444" stroke-width="2"/>
  <text x="{x_lr-40}" y="{round(lr_to_y(1)+4,1)}" font-size="9" fill="#ef4444" font-weight="600">LR=1</text>

  <!-- Bottom note -->
  <text x="{NW//2}" y="{NH-8}" font-size="9" fill="#94a3b8" text-anchor="middle" font-style="italic">
    Read: draw line from pre-test probability through LR to post-test probability
  </text>
</svg>"""

        _fagan_comp.html(f"<div style='font-family:sans-serif;'>{nomo_svg}</div>", height=NH+20, scrolling=False)

        # Plain language interpretation
        change = post_prob - pre_prob
        direction = "increases" if change > 0 else "decreases"
        st.markdown(f"""
**Interpretation:** Starting with a pre-test probability of **{round(pre_prob*100,1)}%**, applying an **{"LR+" if "LR+" in lr_type else "LR−"} of {round(lr_val,2)}**
{direction} the post-test probability to **{round(post_prob*100,1)}%** — a change of **{round(abs(change)*100,1)} percentage points**.

Pre-test odds: {round(pre_odds,3)} → Post-test odds: {round(post_odds,3)} → Post-test probability: {round(post_prob*100,1)}%
        """)

        if "LR+" in lr_type:
            if lr_val >= 10:
                st.success(f"✅ LR+ = {round(lr_val,2)} — a **large, often conclusive** increase. A positive test substantially raises disease probability.")
            elif lr_val >= 5:
                st.success(f"✅ LR+ = {round(lr_val,2)} — a **moderate** increase. A positive test is meaningful but not definitive.")
            elif lr_val >= 2:
                st.warning(f"⚠️ LR+ = {round(lr_val,2)} — a **small** increase. A positive test provides some, but limited, diagnostic value.")
            else:
                st.error(f"❌ LR+ = {round(lr_val,2)} — **near 1**. A positive test adds almost no diagnostic information.")
        else:
            if lr_val <= 0.1:
                st.success(f"✅ LR− = {round(lr_val,3)} — a **large, often conclusive** decrease. A negative test substantially lowers disease probability (strong rule-out).")
            elif lr_val <= 0.2:
                st.success(f"✅ LR− = {round(lr_val,3)} — a **moderate** decrease. A negative test is meaningful for ruling out disease.")
            elif lr_val <= 0.5:
                st.warning(f"⚠️ LR− = {round(lr_val,3)} — a **small** decrease. A negative test provides some but limited reassurance.")
            else:
                st.error(f"❌ LR− = {round(lr_val,3)} — **near 1**. A negative test adds almost no diagnostic information.")

        st.divider()
        st.markdown("#### 🧠 Practice: Applying Likelihood Ratios")
        lr_q1 = st.radio(
            "**Scenario:** A 55-year-old woman with chest pain has a pre-test probability of coronary artery disease of 40%. Her exercise stress test is positive. The LR+ for this test is 4.5. What is her approximate post-test probability?",
            ["— Select —", "~20% — the test lowered probability", "~50% — barely changed",
             "~75% — LR+ of 4.5 substantially raises a 40% pre-test probability", "~95% — almost certain"],
            key="lr_q1"
        )
        if lr_q1 == "~75% — LR+ of 4.5 substantially raises a 40% pre-test probability":
            st.success("""✅ Correct. Pre-test odds = 0.40/0.60 = 0.667. Post-test odds = 0.667 × 4.5 = 3.0. Post-test probability = 3.0/4.0 = 75%. A positive stress test in a patient with intermediate pre-test probability substantially raises the probability — but note it's still not 100%, which is why further testing may follow.""")
        elif lr_q1 != "— Select —":
            st.error("❌ Work through the math: Pre-test odds = 0.4/0.6 = 0.667. Post-test odds = 0.667 × 4.5 = 3.0. Post-test probability = 3.0/(1+3.0) = 75%.")

        lr_q2 = st.radio(
            "**Scenario:** A patient has a 30% pre-test probability of pulmonary embolism. A D-dimer test is negative. The LR− for D-dimer is 0.08. What does this mean clinically?",
            ["— Select —",
             "Post-test probability is still ~30% — the test didn't change anything",
             "Post-test probability drops to ~3% — PE is effectively ruled out",
             "Post-test probability rises — a negative test worsens the outlook",
             "LR− can't be used for ruling out disease"],
            key="lr_q2"
        )
        if lr_q2 == "Post-test probability drops to ~3% — PE is effectively ruled out":
            st.success("""✅ Correct. Pre-test odds = 0.3/0.7 = 0.429. Post-test odds = 0.429 × 0.08 = 0.034. Post-test probability = 0.034/1.034 ≈ 3.3%. A negative D-dimer in a patient with moderate pre-test probability reduces the probability to ~3% — below the threshold for anticoagulation. This illustrates the power of LR− for rule-out. D-dimer's high sensitivity (low LR−) is why it's used as a rule-out test.""")
        elif lr_q2 != "— Select —":
            st.error("❌ LR− below 0.1 is a strong rule-out. Pre-test odds = 0.3/0.7 = 0.429. Post-test odds = 0.429 × 0.08 = 0.034. Post-test probability ≈ 3.3%.")


    elif screen_section == "4️⃣ Prevalence Effect on PPV":
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

    elif screen_section == "5️⃣ Wilson & Jungner Criteria":
        st.subheader("Wilson & Jungner Criteria for Screening Programs")
        st.markdown("""
In 1968, Wilson and Jungner published a landmark WHO report establishing the criteria that a disease and its screening test must meet before a population-wide screening program is justified. These remain the gold standard framework for evaluating screening programs today.
        """)

        WJ_CRITERIA = [
            ("1", "The condition should be an important health problem",
             "Important means significant burden — high prevalence, serious consequences, or major impact on mortality/quality of life. Rare conditions with minor consequences don't justify population-wide screening programs.",
             "Breast cancer, colorectal cancer, cervical cancer, neonatal PKU, hypertension"),
            ("2", "The natural history of the condition should be adequately understood",
             "You need to know how disease progresses through its natural history stages to identify when screening is most effective and what early-stage disease looks like. Without this knowledge, you can't know whether early detection changes outcomes.",
             "We must know that there is a detectable preclinical phase that screening can identify"),
            ("3", "There should be a recognizable latent or early symptomatic stage",
             "Screening only works if there is a window of time when disease is present but asymptomatic, AND detectable. If disease progresses too rapidly from onset to symptoms, screening has no opportunity to intervene before harm occurs.",
             "Cervical intraepithelial neoplasia (CIN) before invasive cervical cancer; ductal carcinoma in situ (DCIS) before invasive breast cancer"),
            ("4", "There should be a suitable test or examination",
             "The test must be: acceptable to the screened population, sufficiently sensitive to detect disease in the preclinical stage, sufficiently specific to avoid excessive false positives, safe, and feasible at scale.",
             "Pap smear for cervical cancer; PSA for prostate cancer (controversial — specificity concerns); mammography for breast cancer"),
            ("5", "The test should be acceptable to the population",
             "If the target population won't accept the test — due to discomfort, embarrassment, cultural barriers, or perceived risk — uptake will be poor and the program will fail regardless of test performance. Acceptability includes both the test itself and the follow-up procedures required after a positive result.",
             "Colonoscopy has lower acceptability than FIT (fecal immunochemical test) for colorectal cancer screening"),
            ("6", "There should be an accepted treatment for patients with recognized disease",
             "Screening is only ethical if an effective treatment exists. Identifying disease early and then having nothing beneficial to offer — or offering treatment that causes harm without benefit — violates the principle of beneficence. Early detection without effective treatment only extends the period of anxiety, not survival.",
             "The controversy over prostate cancer screening partly reflects uncertain benefit of early treatment for low-risk disease"),
            ("7", "Facilities for diagnosis and treatment should be available",
             "Screening creates downstream demand for diagnostic workup (biopsy, imaging, specialist review) and treatment. If these facilities are unavailable or inaccessible, positive screens lead to anxiety without resolution. Healthcare system capacity must be considered before launching a program.",
             "Introducing mammography screening in a setting without surgical or radiotherapy capacity is counterproductive"),
            ("8", "There should be an agreed policy on whom to treat as patients",
             "The criteria for who gets treated after a positive screen must be clear and consistently applied. Ambiguous treatment thresholds lead to inconsistent care and either over-treatment of indolent disease or under-treatment of serious disease.",
             "PSA thresholds for prostate biopsy; colposcopy referral thresholds after abnormal Pap smear"),
            ("9", "The cost of case-finding should be economically balanced in relation to possible expenditure on medical care as a whole",
             "A cost-effectiveness analysis should show that the screening program provides value relative to other uses of healthcare resources. Costs include the test itself, false positive workup, treatment of screen-detected cases, and management of overdiagnosed cases.",
             "Cost per QALY (quality-adjusted life year) gained is the standard metric for evaluating screening cost-effectiveness"),
            ("10", "Case-finding should be a continuing process and not a 'once and for all' project",
             "Screening programs require ongoing surveillance, quality assurance, and re-screening at appropriate intervals. A one-time screen misses cases that develop after the screening date. Regular re-screening at evidence-based intervals is required.",
             "Cervical cancer screening every 3–5 years; mammography every 1–2 years depending on age and risk"),
        ]

        for num, criterion, explanation, example in WJ_CRITERIA:
            with st.expander(f"**Criterion {num}:** {criterion}"):
                st.markdown(f"**Why it matters:** {explanation}")
                st.info(f"**Example:** {example}")

        st.divider()

        # Modern application
        st.markdown("#### Applying Wilson & Jungner — Cervical Cancer Screening")
        st.markdown("""
Cervical cancer screening (Pap smear / HPV test) is often cited as the most successful screening program in history. It satisfies all 10 criteria:

| Criterion | How cervical screening meets it |
|---|---|
| 1. Important health problem | Leading cause of cancer death in women in low-resource settings |
| 2. Natural history understood | HPV → CIN → invasive cancer over 10–20 years is well characterized |
| 3. Latent stage | CIN (precancerous lesion) is detectable years before invasion |
| 4. Suitable test | Pap smear (sensitivity 55–80%) + HPV co-testing (sensitivity >90%) |
| 5. Acceptable | Accepted by most women; efforts underway to improve uptake in underserved groups |
| 6. Effective treatment | CIN treated by LEEP/cryotherapy; early invasive cancer curable by surgery |
| 7. Facilities available | Colposcopy and treatment available in most healthcare settings |
| 8. Treatment policy agreed | Clear guidelines for CIN 1 vs. CIN 2/3 management |
| 9. Cost-effective | One of the most cost-effective cancer interventions known |
| 10. Ongoing program | Re-screening every 3–5 years based on age and risk |
        """)

        st.warning("""
**Where Wilson & Jungner criteria are NOT met — PSA screening for prostate cancer:**
- Criterion 4: PSA has poor specificity (many false positives → unnecessary biopsies)
- Criterion 6: Treatment benefit for low-risk, screen-detected prostate cancer is uncertain
- Criterion 8: No consensus on treatment thresholds — leads to significant overtreatment
- Criterion 9: High cost from false positive workups; overdiagnosis of indolent disease

This is why PSA screening recommendations are inconsistent across countries and guidelines.
        """)

        with st.expander("📋 Wilson & Jungner Criteria — Quick Reference"):
            st.markdown("""
| # | Criterion | Key question |
|---|---|---|
| 1 | Important health problem | Is the disease burden sufficient to justify screening? |
| 2 | Natural history understood | Do we know how disease progresses? |
| 3 | Latent/early stage exists | Is there a detectable pre-symptomatic window? |
| 4 | Suitable test | Is the test sensitive, specific, safe, and feasible? |
| 5 | Acceptable to population | Will people accept the test? |
| 6 | Effective treatment exists | Does early detection improve outcomes? |
| 7 | Facilities available | Can we follow up positive screens? |
| 8 | Treatment policy agreed | Do we know who to treat and how? |
| 9 | Cost-effective | Is the benefit worth the cost? |
| 10 | Ongoing program | Is re-screening at intervals feasible? |
            """)


    elif screen_section == "6️⃣ ROC Curve":
        st.subheader("Receiver Operating Characteristic (ROC) Curve")
        st.markdown("""
The **ROC curve** is one of the most important concepts in diagnostic test evaluation. It visualizes the fundamental tradeoff between a test's ability to correctly identify people who have a disease (sensitivity) and its ability to correctly identify people who don't (specificity) — across every possible threshold for calling a test "positive."

Understanding ROC curves is essential for evaluating tests, comparing competing tests, and choosing diagnostic cutpoints in clinical and public health practice.
        """)

        st.info("""
**The core tradeoff:** Every diagnostic test that produces a continuous result (e.g., blood glucose, PSA level, test score) requires a *cutpoint* to divide results into "positive" and "negative." 
- Move the cutpoint lower → more people test positive → sensitivity ↑, specificity ↓ (more true positives, but more false positives too)
- Move the cutpoint higher → fewer people test positive → specificity ↑, sensitivity ↓ (fewer false positives, but more missed cases)

The ROC curve shows *every possible version* of that tradeoff simultaneously.
        """)

        st.markdown("#### How the ROC curve is constructed")
        st.markdown("""
For each possible cutpoint along the test's continuous scale:
1. Calculate **sensitivity** = TP / (TP + FN) — the true positive rate
2. Calculate **1 − specificity** = FP / (FP + TN) — the false positive rate

Plot each cutpoint as a point: **x = (1 − specificity), y = sensitivity**

Connect all points → the ROC curve.
        """)

        # ── SENS/SPEC TRADEOFF VISUAL ─────────────────────────────────────
        st.markdown("#### 👁️ See the tradeoff — what actually happens when you move the threshold")
        st.markdown("""
The fundamental reason sensitivity and specificity trade off is that the distributions of test scores in diseased and healthy people **overlap**. The threshold sits somewhere in that overlap. Move it left or right and you change how much of each distribution you capture.
        """)

        import numpy as _np2
        import streamlit.components.v1 as _dist_comp

        dist_preset = st.selectbox("Choose a test overlap scenario:", [
            "Good separation (AUC ≈ 0.85) — distributions mostly separate",
            "Poor separation (AUC ≈ 0.65) — distributions heavily overlap",
            "Excellent separation (AUC ≈ 0.95) — minimal overlap",
        ], key="dist_preset_tradeoff")

        if "Poor" in dist_preset:
            mu_d, sd_d, mu_h, sd_h = 6.5, 2.8, 4.5, 2.6
        elif "Excellent" in dist_preset:
            mu_d, sd_d, mu_h, sd_h = 8.5, 1.4, 3.5, 1.4
        else:
            mu_d, sd_d, mu_h, sd_h = 7.5, 2.0, 4.0, 2.0

        t_lo = min(mu_h - 3*sd_h, mu_d - 3*sd_d)
        t_hi = max(mu_h + 3*sd_h, mu_d + 3*sd_d)
        thresh_dist = st.slider("Move the diagnostic threshold:",
            min_value=round(t_lo,1), max_value=round(t_hi,1),
            value=round((mu_d + mu_h)/2, 1), step=0.1, key="dist_thresh")

        # Compute metrics at this threshold
        import math as _m2
        def norm_cdf(x, mu, sd):
            return 0.5 * (1 + math.erf((x - mu) / (sd * math.sqrt(2))))

        sens_d = 1 - norm_cdf(thresh_dist, mu_d, sd_d)
        spec_d = norm_cdf(thresh_dist, mu_h, sd_h)
        fpr_d  = 1 - spec_d

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Sensitivity (TPR)", f"{round(sens_d*100,1)}%", help="Fraction of diseased people correctly called positive")
        c2.metric("Specificity", f"{round(spec_d*100,1)}%", help="Fraction of healthy people correctly called negative")
        c3.metric("False Positive Rate", f"{round(fpr_d*100,1)}%", help="1 − Specificity")
        c4.metric("Youden's J", f"{round(sens_d + spec_d - 1, 3)}", help="Sensitivity + Specificity − 1. Max = 1 (perfect). Higher = better combined performance at this threshold.")

        # Build SVG distribution plot
        W2, H2 = 520, 220
        pad_l, pad_b, pad_t, pad_r = 40, 30, 20, 20
        pw2, ph2 = W2-pad_l-pad_r, H2-pad_b-pad_t

        xs = [t_lo + i*(t_hi-t_lo)/400 for i in range(401)]
        import math
        def gauss(x, mu, sd):
            return (1/(sd*math.sqrt(2*math.pi))) * math.exp(-0.5*((x-mu)/sd)**2)

        yd = [gauss(x, mu_d, sd_d) for x in xs]
        yh = [gauss(x, mu_h, sd_h) for x in xs]
        y_max = max(max(yd), max(yh)) * 1.1

        def px(v):
            return pad_l + (v-t_lo)/(t_hi-t_lo)*pw2
        def py(v):
            return (H2-pad_b) - v/y_max*ph2

        # Build paths
        path_d = " ".join(f"{'M' if i==0 else 'L'}{round(px(xs[i]),1)},{round(py(yd[i]),1)}" for i in range(len(xs)))
        path_h = " ".join(f"{'M' if i==0 else 'L'}{round(px(xs[i]),1)},{round(py(yh[i]),1)}" for i in range(len(xs)))

        # Shaded regions
        # TP: diseased above threshold (green fill)
        tp_pts = [(px(xs[i]), py(yd[i])) for i in range(len(xs)) if xs[i] >= thresh_dist]
        tp_path = f"M{round(px(thresh_dist),1)},{round(py(0),1)} " + " ".join(f"L{round(p[0],1)},{round(p[1],1)}" for p in tp_pts) + f" L{round(tp_pts[-1][0],1)},{round(py(0),1)} Z" if tp_pts else ""
        # FN: diseased below threshold (red fill)
        fn_pts = [(px(xs[i]), py(yd[i])) for i in range(len(xs)) if xs[i] < thresh_dist]
        fn_path = f"M{round(px(xs[0]),1)},{round(py(0),1)} " + " ".join(f"L{round(p[0],1)},{round(p[1],1)}" for p in fn_pts) + f" L{round(px(thresh_dist),1)},{round(py(0),1)} Z" if fn_pts else ""
        # FP: healthy above threshold (orange fill)
        fp_pts = [(px(xs[i]), py(yh[i])) for i in range(len(xs)) if xs[i] >= thresh_dist]
        fp_path = f"M{round(px(thresh_dist),1)},{round(py(0),1)} " + " ".join(f"L{round(p[0],1)},{round(p[1],1)}" for p in fp_pts) + f" L{round(fp_pts[-1][0],1)},{round(py(0),1)} Z" if fp_pts else ""
        # TN: healthy below threshold (blue fill)
        tn_pts = [(px(xs[i]), py(yh[i])) for i in range(len(xs)) if xs[i] < thresh_dist]
        tn_path = f"M{round(px(xs[0]),1)},{round(py(0),1)} " + " ".join(f"L{round(p[0],1)},{round(p[1],1)}" for p in tn_pts) + f" L{round(px(thresh_dist),1)},{round(py(0),1)} Z" if tn_pts else ""

        # Threshold line
        tx = round(px(thresh_dist),1)

        # X axis ticks
        x_range = t_hi - t_lo
        tick_step = round(x_range/5, 1)
        xtick_svg = ""
        for i in range(6):
            v = round(t_lo + i*x_range/5, 1)
            xtick_svg += f'<text x="{round(px(v),1)}" y="{H2-8}" font-size="8" text-anchor="middle" fill="#888">{v}</text>'

        svg2 = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W2}" height="{H2}" style="font-family:sans-serif;background:#fafafa;border-radius:8px 8px 0 0;">
  <!-- Shaded areas -->
  {"<path d='"+fn_path+"' fill='rgba(220,38,38,0.18)'/>" if fn_path else ""}
  {"<path d='"+tp_path+"' fill='rgba(22,163,74,0.22)'/>" if tp_path else ""}
  {"<path d='"+fp_path+"' fill='rgba(234,88,12,0.20)'/>" if fp_path else ""}
  {"<path d='"+tn_path+"' fill='rgba(37,99,235,0.15)'/>" if tn_path else ""}
  <!-- Distribution curves -->
  <path d="{path_h}" stroke="#2563eb" stroke-width="2.5" fill="none"/>
  <path d="{path_d}" stroke="#dc2626" stroke-width="2.5" fill="none"/>
  <!-- Threshold line -->
  <line x1="{tx}" y1="{pad_t}" x2="{tx}" y2="{H2-pad_b}" stroke="#374151" stroke-width="2" stroke-dasharray="5,3"/>
  <text x="{tx+4}" y="{pad_t+12}" font-size="9" fill="#374151" font-weight="600">Threshold</text>
  <!-- Axis -->
  <line x1="{pad_l}" y1="{H2-pad_b}" x2="{W2-pad_r}" y2="{H2-pad_b}" stroke="#999" stroke-width="1.5"/>
  {xtick_svg}
  <text x="{pad_l+pw2//2}" y="{H2-1}" font-size="9" text-anchor="middle" fill="#666">Test score</text>
  <!-- Curve peak labels only (no overlap) -->
  <text x="{round(px(mu_d)+4,0)}" y="{round(py(gauss(mu_d,mu_d,sd_d))-10,0)}" font-size="10" fill="#dc2626" font-weight="700">Diseased</text>
  <text x="{round(px(mu_h)-4,0)}" y="{round(py(gauss(mu_h,mu_h,sd_h))-10,0)}" font-size="10" fill="#2563eb" font-weight="700" text-anchor="end">Healthy</text>
</svg>
<div style="background:#f0f4f8;border-radius:0 0 8px 8px;padding:8px 16px;display:flex;gap:20px;flex-wrap:wrap;font-family:sans-serif;font-size:11px;">
  <span><span style="display:inline-block;width:12px;height:12px;background:rgba(22,163,74,0.5);border-radius:2px;margin-right:4px;vertical-align:middle;"></span><span style="color:#166534;font-weight:600;">TP</span> — sens = {round(sens_d*100,0):.0f}%</span>
  <span><span style="display:inline-block;width:12px;height:12px;background:rgba(220,38,38,0.4);border-radius:2px;margin-right:4px;vertical-align:middle;"></span><span style="color:#991b1b;font-weight:600;">FN</span> — missed = {round((1-sens_d)*100,0):.0f}%</span>
  <span><span style="display:inline-block;width:12px;height:12px;background:rgba(37,99,235,0.35);border-radius:2px;margin-right:4px;vertical-align:middle;"></span><span style="color:#1e40af;font-weight:600;">TN</span> — spec = {round(spec_d*100,0):.0f}%</span>
  <span><span style="display:inline-block;width:12px;height:12px;background:rgba(234,88,12,0.4);border-radius:2px;margin-right:4px;vertical-align:middle;"></span><span style="color:#9a3412;font-weight:600;">FP</span> — false+ = {round(fpr_d*100,0):.0f}%</span>
</div>"""

        _dist_comp.html(f"<div style='font-family:sans-serif;'>{svg2}</div>", height=H2+50, scrolling=False)
        st.caption("Red curve = diseased population. Blue curve = healthy population. Move the threshold slider to see how the four regions (TP/FN/FP/TN) shift — and why sensitivity and specificity cannot both be maximized simultaneously when distributions overlap.")

        st.info("""
🔑 **The fundamental insight:** Sensitivity and specificity trade off because the diseased and healthy distributions **overlap**. In that overlap zone, some healthy people have high scores (→ false positives) and some diseased people have low scores (→ false negatives). No threshold can eliminate both simultaneously — it can only shift which error you make more of.

- **Move threshold left** → capture more diseased people (↑ sensitivity) but also more healthy people (↑ false positives, ↓ specificity)
- **Move threshold right** → exclude more healthy people (↑ specificity) but also miss more diseased (↑ false negatives, ↓ sensitivity)
- **Better test** → distributions further apart → less overlap → both sensitivity and specificity can be high simultaneously
        """)


        # Interactive ROC with adjustable threshold
        import numpy as np
        import pandas as pd
        import streamlit.components.v1 as _roc_comp

        st.markdown("#### 🧮 Interactive ROC Explorer")
        st.markdown("Adjust the diagnostic threshold to see how sensitivity and specificity change — and where that threshold sits on the ROC curve.")

        roc_preset = st.selectbox("Select a test scenario:", [
            "Moderate test (AUC ≈ 0.75) — e.g., PSA for prostate cancer",
            "Good test (AUC ≈ 0.85) — e.g., HbA1c for diabetes",
            "Excellent test (AUC ≈ 0.95) — e.g., HIV ELISA",
            "Poor test (AUC ≈ 0.60) — near-random discrimination",
        ], key="roc_preset")

        np.random.seed(42)
        n = 500
        if "PSA" in roc_preset:
            diseased = np.random.normal(6.0, 2.5, n//2)
            healthy  = np.random.normal(3.5, 2.0, n//2)
        elif "HbA1c" in roc_preset:
            diseased = np.random.normal(7.5, 1.5, n//2)
            healthy  = np.random.normal(5.5, 1.2, n//2)
        elif "HIV" in roc_preset:
            diseased = np.random.normal(8.0, 1.0, n//2)
            healthy  = np.random.normal(3.0, 1.2, n//2)
        else:
            diseased = np.random.normal(5.5, 2.5, n//2)
            healthy  = np.random.normal(4.5, 2.5, n//2)

        all_scores = np.concatenate([diseased, healthy])
        labels = np.concatenate([np.ones(n//2), np.zeros(n//2)])
        t_min, t_max = float(np.min(all_scores)), float(np.max(all_scores))
        t_mid = float(np.median(diseased) * 0.6 + np.median(healthy) * 0.4)

        threshold = st.slider("Diagnostic threshold (test positive if score ≥ threshold):",
                              min_value=round(t_min,1), max_value=round(t_max,1),
                              value=round(t_mid,1), step=0.1, key="roc_threshold")

        tp = np.sum((all_scores >= threshold) & (labels == 1))
        fp = np.sum((all_scores >= threshold) & (labels == 0))
        fn = np.sum((all_scores < threshold)  & (labels == 1))
        tn = np.sum((all_scores < threshold)  & (labels == 0))
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (fp + tn) if (fp + tn) > 0 else 0
        ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv  = tn / (fn + tn) if (fn + tn) > 0 else 0

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Sensitivity", f"{round(sens*100,1)}%")
        c2.metric("Specificity", f"{round(spec*100,1)}%")
        c3.metric("PPV", f"{round(ppv*100,1)}%")
        c4.metric("NPV", f"{round(npv*100,1)}%")

        # Build ROC curve data
        thresholds = np.linspace(t_min, t_max, 300)
        roc_pts = []
        for t in thresholds:
            tp_t = np.sum((all_scores >= t) & (labels == 1))
            fp_t = np.sum((all_scores >= t) & (labels == 0))
            fn_t = np.sum((all_scores < t)  & (labels == 1))
            tn_t = np.sum((all_scores < t)  & (labels == 0))
            s = tp_t/(tp_t+fn_t) if (tp_t+fn_t)>0 else 0
            sp = tn_t/(fp_t+tn_t) if (fp_t+tn_t)>0 else 0
            roc_pts.append((1-sp, s))

        # AUC via trapezoid
        xs = [p[0] for p in roc_pts]
        ys = [p[1] for p in roc_pts]
        auc = 0.0
        for i in range(len(xs)-1):
            auc += (xs[i]-xs[i+1]) * (ys[i]+ys[i+1]) / 2
        auc = abs(auc)

        # SVG ROC curve
        W, H = 400, 320
        pad = 50
        pw, ph = W-pad-20, H-pad-20
        def to_px(fpr, tpr):
            x = pad + fpr * pw
            y = (H-pad) - tpr * ph
            return x, y

        path_d = " ".join([f"{'M' if i==0 else 'L'}{round(to_px(x,y)[0],1)},{round(to_px(x,y)[1],1)}"
                           for i,(x,y) in enumerate(roc_pts)])
        # Current threshold point
        cur_fpr = 1 - spec
        cur_tpr = sens
        cpx, cpy = to_px(cur_fpr, cur_tpr)

        # Diagonal reference line
        diag = f"M{pad},{H-pad} L{pad+pw},{H-pad-ph}"

        # Y axis ticks
        yticks = "".join([f'<text x="{pad-5}" y="{round(to_px(0,v)[1]+4,0)}" font-size="9" text-anchor="end" fill="#666">{round(v*100)}%</text>' +
                          f'<line x1="{pad-3}" y1="{round(to_px(0,v)[1],0)}" x2="{pad}" y2="{round(to_px(0,v)[1],0)}" stroke="#ccc"/>' 
                          for v in [0,.2,.4,.6,.8,1.0]])
        xticks = "".join([f'<text x="{round(to_px(v,0)[0],0)}" y="{H-pad+14}" font-size="9" text-anchor="middle" fill="#666">{round(v*100)}%</text>'
                          for v in [0,.2,.4,.6,.8,1.0]])

        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" style="font-family:sans-serif;background:#fafafa;border-radius:8px;">
  <!-- Axes -->
  <line x1="{pad}" y1="{H-pad}" x2="{pad+pw}" y2="{H-pad}" stroke="#999" stroke-width="1.5"/>
  <line x1="{pad}" y1="{H-pad}" x2="{pad}" y2="{H-pad-ph}" stroke="#999" stroke-width="1.5"/>
  {yticks}{xticks}
  <!-- Labels -->
  <text x="{pad+pw//2}" y="{H-4}" font-size="10" text-anchor="middle" fill="#555">1 − Specificity (False Positive Rate)</text>
  <text x="12" y="{H-pad-ph//2}" font-size="10" text-anchor="middle" fill="#555" transform="rotate(-90,12,{H-pad-ph//2})">Sensitivity (True Positive Rate)</text>
  <!-- Diagonal reference (random = AUC 0.50) -->
  <path d="{diag}" stroke="#bbb" stroke-width="1" stroke-dasharray="4,3" fill="none"/>
  <text x="{pad+pw-5}" y="{H-pad-ph+10}" font-size="9" fill="#aaa" text-anchor="end">Random (AUC=0.50)</text>
  <!-- ROC curve -->
  <path d="{path_d}" stroke="#2563eb" stroke-width="2.5" fill="none"/>
  <!-- Current threshold point -->
  <circle cx="{round(cpx,1)}" cy="{round(cpy,1)}" r="7" fill="#ef4444" stroke="white" stroke-width="2"/>
  <text x="{round(cpx+12,1)}" y="{round(cpy-4,1)}" font-size="9" fill="#ef4444">Current threshold</text>
  <!-- AUC label -->
  <text x="{pad+pw-8}" y="{H-pad-ph+28}" font-size="11" font-weight="bold" text-anchor="end" fill="#2563eb">AUC = {round(auc,3)}</text>
</svg>"""

        _roc_comp.html(f"<div style='font-family:sans-serif;'>{svg}</div>", height=H+10, scrolling=False)

        st.caption("Blue curve = ROC curve for this test. Red dot = current threshold. Dashed diagonal = a test that performs no better than chance (AUC = 0.50).")

        st.divider()
        st.markdown("#### The Area Under the Curve (AUC)")
        st.markdown(f"""
**AUC = {round(auc,3)}** for this scenario.

The AUC (also called the **c-statistic**) summarizes the ROC curve in a single number — the probability that the test assigns a higher score to a randomly chosen diseased person than to a randomly chosen healthy person.
        """)

        st.markdown("""
| AUC | Interpretation | Example context |
|---|---|---|
| **1.0** | Perfect discrimination — no overlap between diseased and healthy distributions | Rare in practice |
| **0.90–0.99** | Excellent | HIV ELISA, some molecular tests |
| **0.80–0.89** | Good | HbA1c for diabetes |
| **0.70–0.79** | Fair/Moderate | Many clinical risk scores |
| **0.60–0.69** | Poor | PSA for prostate cancer |
| **0.50** | No discrimination — equivalent to flipping a coin | Useless test |
| **< 0.50** | Worse than chance — test result is inverted | Test score is inverse predictor |
        """)

        st.info("""
🔑 **Choosing a cutpoint from the ROC curve:**
There is no single "correct" threshold — the right choice depends on the consequences of false positives vs. false negatives:

- **Screening for a serious, treatable disease** → set threshold low → prioritize sensitivity (catch all cases even if many false positives)
- **Confirmatory test before invasive procedure** → set threshold high → prioritize specificity (minimize unnecessary harm)
- **Youden's J statistic** (sensitivity + specificity − 1) identifies the point on the ROC curve that maximizes the combined performance

The point at the upper-left corner of the ROC curve (high sensitivity AND high specificity) is ideal — but how close you can get depends on the underlying biological overlap between diseased and healthy populations.
        """)

        st.divider()
        st.markdown("#### 🧠 Concept check")
        roc_q1 = st.radio(
            "**Q1:** You lower the diagnostic threshold for a blood glucose screening test. What happens on the ROC curve?",
            ["— Select —",
             "The AUC increases — you've improved the test",
             "You move along the curve toward higher sensitivity and higher false positive rate",
             "You move along the curve toward higher specificity and lower sensitivity",
             "The curve shape changes"],
            key="roc_q1"
        )
        if roc_q1 == "You move along the curve toward higher sensitivity and higher false positive rate":
            st.success("✅ Correct. Lowering the threshold calls more people positive — you catch more true cases (sensitivity ↑) but also flag more healthy people incorrectly (false positive rate ↑ = specificity ↓). This moves the red dot along the existing curve — it doesn't change the curve's shape or the AUC, because AUC reflects the test's inherent discriminating ability, not the chosen cutpoint.")
        elif roc_q1 != "— Select —":
            st.error("❌ Changing the threshold moves you along the existing ROC curve but doesn't change its shape or AUC. AUC is a property of the test itself, not the threshold. Lowering the threshold → more positives → sensitivity ↑, specificity ↓ → move toward the upper-right corner of the curve.")

        roc_q2 = st.radio(
            "**Q2:** Test A has AUC = 0.65. Test B has AUC = 0.88. What can you conclude?",
            ["— Select —",
             "Test B is better at discriminating diseased from non-diseased, across all possible thresholds",
             "Test B has higher sensitivity than Test A",
             "Test B has higher specificity than Test A",
             "Test A is useless and should be abandoned"],
            key="roc_q2"
        )
        if roc_q2 == "Test B is better at discriminating diseased from non-diseased, across all possible thresholds":
            st.success("✅ Correct. AUC summarizes performance across all thresholds. Test B discriminates better overall. You cannot conclude which test has higher sensitivity or specificity at any specific threshold — that depends on where you set the cutpoint. And AUC = 0.65 is poor but not necessarily useless, especially if it provides incremental value alongside other tests.")
        elif roc_q2 != "— Select —":
            st.error("❌ AUC compares overall discrimination ability across all thresholds. It doesn't tell you sensitivity or specificity at any specific cutpoint — those depend on the chosen threshold, not the AUC. Test B is the better discriminator overall.")


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

                        # Derived measures
                        risk_exp   = a/(a+b)
                        risk_unexp = c/(c+d)
                        ar_abs     = risk_exp - risk_unexp
                        ar_pct     = (ar_abs / risk_exp * 100) if risk_exp > 0 else 0
                        nnt_nnh    = abs(1/ar_abs) if ar_abs != 0 else float("inf")

                        exp_label  = row_names[0] if row_names[0] else "exposed group"
                        unexp_label = row_names[1] if row_names[1] else "unexposed group"
                        out_label  = col_names[0] if col_names[0] else "the outcome"
                        risk_word  = "prevalence" if is_cs else "risk"

                        # Magnitude helper
                        def rr_magnitude(v):
                            v = abs(v - 1)
                            if v < 0.1: return "negligible"
                            elif v < 0.5: return "weak"
                            elif v < 1.0: return "moderate"
                            elif v < 2.0: return "strong"
                            else: return "very strong"

                        # ── RR / PR ──────────────────────────────────────────
                        st.subheader(f"{'Prevalence Ratio (PR)' if is_cs else 'Risk Ratio (RR)'}")

                        sig_rr = not (ci_low_rr <= 1 <= ci_high_rr)
                        direction_rr = "higher" if rr > 1 else "lower"
                        mag_rr = rr_magnitude(rr)

                        if sig_rr:
                            st.success(f"{pabbr} = {round(rr,2)} (95% CI: {round(ci_low_rr,2)}–{round(ci_high_rr,2)})")
                        else:
                            st.warning(f"{pabbr} = {round(rr,2)} (95% CI: {round(ci_low_rr,2)}–{round(ci_high_rr,2)}) — CI includes 1")

                        # Plain-language interpretation
                        if rr > 1:
                            interp_rr = (
                                f"The {risk_word} of **{out_label}** among **{exp_label}** is "
                                f"**{round(rr,2)} times higher** than among **{unexp_label}** "
                                f"({round(risk_exp*100,1)}% vs. {round(risk_unexp*100,1)}%). "
                                f"This is a **{mag_rr}** positive association."
                            )
                        elif rr < 1:
                            interp_rr = (
                                f"The {risk_word} of **{out_label}** among **{exp_label}** is "
                                f"**{round((1-rr)*100,1)}% lower** than among **{unexp_label}** "
                                f"({round(risk_exp*100,1)}% vs. {round(risk_unexp*100,1)}%). "
                                f"This is a **{mag_rr}** negative association (possible protective effect)."
                            )
                        else:
                            interp_rr = f"{pabbr} = 1.0 — no difference in {risk_word} between groups."

                        ci_interp_rr = (
                            f"We are 95% confident the true population {pabbr} lies between "
                            f"**{round(ci_low_rr,2)} and {round(ci_high_rr,2)}**. "
                            f"{'The CI excludes 1, consistent with a statistically significant association.' if sig_rr else 'The CI includes 1, consistent with a non-significant result — the true value could be no association.'}"
                        )

                        st.markdown(interp_rr)
                        st.caption(ci_interp_rr)
                        draw_ci(pabbr, rr, ci_low_rr, ci_high_rr)

                        # ── OR ───────────────────────────────────────────────
                        if design != "Cross-sectional":
                            st.subheader("Odds Ratio (OR)")
                            sig_or = not (ci_low_or <= 1 <= ci_high_or)
                            direction_or = "higher" if or_val > 1 else "lower"
                            mag_or = rr_magnitude(or_val)

                            if sig_or:
                                st.success(f"OR = {round(or_val,2)} (95% CI: {round(ci_low_or,2)}–{round(ci_high_or,2)})")
                            else:
                                st.warning(f"OR = {round(or_val,2)} (95% CI: {round(ci_low_or,2)}–{round(ci_high_or,2)}) — CI includes 1")

                            if design == "Case-Control":
                                or_design_note = (
                                    f"In a **case-control study**, the OR estimates the odds of exposure in cases vs. controls — "
                                    f"**not** the risk or probability of disease. "
                                    f"The OR of **{round(or_val,2)}** means the odds of **{exp_label}** exposure were "
                                    f"{'**' + str(round(or_val,2)) + '× higher** in cases' if or_val > 1 else '**' + str(round(1/or_val,2)) + '× lower** in cases'} "
                                    f"than in controls. "
                                    f"{'When outcome is rare, OR ≈ RR as a rough heuristic.' if (a+c)/(a+b+c+d) < 0.10 else 'Outcome prevalence here is not low, so OR and RR diverge — OR exaggerates the magnitude of association.'}"
                                )
                            else:
                                or_design_note = (
                                    f"The odds of **{out_label}** among **{exp_label}** are "
                                    f"**{round(or_val,2)}× {'higher' if or_val > 1 else 'lower'}** than among **{unexp_label}**. "
                                    f"This is a **{mag_or}** association. "
                                    f"{'Outcome prevalence here is low, so OR ≈ RR as a rough heuristic.' if (a+c)/(a+b+c+d) < 0.10 else 'Outcome prevalence is not low — OR diverges from RR and should not be interpreted as a risk ratio.'}"
                                )

                            st.markdown(or_design_note)
                            ci_interp_or = (
                                f"95% CI: **{round(ci_low_or,2)}–{round(ci_high_or,2)}**. "
                                f"{'CI excludes 1 — statistically significant.' if sig_or else 'CI includes 1 — result is not statistically significant at α=0.05.'}"
                            )
                            st.caption(ci_interp_or)
                            draw_ci("OR", or_val, ci_low_or, ci_high_or)

                        # ── AR% & NNT ─────────────────────────────────────────
                        with st.expander("📐 Attributable Risk & NNT/NNH"):
                            col_ar1, col_ar2, col_ar3 = st.columns(3)
                            col_ar1.metric("Absolute Risk Difference", f"{round(ar_abs*100,2)} pp",
                                           help="Risk in exposed minus risk in unexposed (percentage points)")
                            col_ar2.metric("AR% (Attributable Fraction)", f"{round(ar_pct,1)}%",
                                           help="What % of disease in exposed is attributable to the exposure?")
                            col_ar3.metric("NNT/NNH", f"{round(nnt_nnh,1)}",
                                           help="Number needed to treat/harm — how many people exposed per extra case?")

                            if ar_abs > 0:
                                st.markdown(f"""
**Absolute risk difference:** Among every 100 {exp_label}, **{round(ar_abs*100,1)} more** develop {out_label} compared to {unexp_label}.

**AR% (Attributable Fraction in Exposed):** Of all {out_label} cases occurring in {exp_label}, **{round(ar_pct,1)}%** are attributable to the exposure itself — the remainder would have occurred anyway.
                                """)
                                interpret_nnt(round(nnt_nnh, 1), False, exp_label, unexp_label, out_label)
                            else:
                                st.markdown(f"""
**Absolute risk difference:** Among every 100 {exp_label}, **{round(abs(ar_abs)*100,1)} fewer** develop {out_label} compared to {unexp_label}.
                                """)
                                interpret_nnt(round(nnt_nnh, 1), True, exp_label, unexp_label, out_label)

                        # ── Design caution ────────────────────────────────────
                        if is_cs:
                            st.info("⏱️ **Cross-sectional caution:** Prevalence ratios from cross-sectional data cannot establish temporal precedence (exposure before outcome). Associations observed here may reflect disease duration as much as incidence.")
                        elif design == "Case-Control":
                            st.info("📋 **Case-control note:** Attack rates and RR cannot be directly calculated from case-control data — only OR. The OR above is the appropriate measure for this design.")

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
                if ci_low_irr <= 1 <= ci_high_irr:
                    st.warning(f"IRR = {round(irr,2)} (95% CI: {round(ci_low_irr,3)}–{round(ci_high_irr,3)}) — CI includes 1. No significant difference in rates.")
                else:
                    direction = "higher" if irr > 1 else "lower"
                    st.success(f"IRR = {round(irr,2)} (95% CI: {round(ci_low_irr,3)}–{round(ci_high_irr,3)})")
                    if irr > 1:
                        st.markdown(
                            f"The incidence rate in **{group_names[0]}** is **{round(irr,2)}× higher** "
                            f"than in **{group_names[1]}** ({round(r1*100000,1)} vs. {round(r2*100000,1)} per 100,000 person-time). "
                            f"We are 95% confident the true IRR lies between **{round(ci_low_irr,2)} and {round(ci_high_irr,2)}**, "
                            f"which excludes 1 — consistent with a statistically significant association."
                        )
                    else:
                        st.markdown(
                            f"The incidence rate in **{group_names[0]}** is **{round((1-irr)*100,1)}% lower** "
                            f"than in **{group_names[1]}** ({round(r1*100000,1)} vs. {round(r2*100000,1)} per 100,000 person-time). "
                            f"95% CI: **{round(ci_low_irr,2)}–{round(ci_high_irr,2)}**, excluding 1."
                        )
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
                is_benefit = r_treatment < r_control
                interpret_nnt(nnt, is_benefit, label_treatment, label_control,
                              "the outcome")

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

**Chi-square tests are nondirectional** — the chi-square statistic is always non-negative (it measures total squared departure from independence), so the test cannot distinguish which group has higher rates. The p-value comes from the right tail of the chi-square distribution. It is more precise to call this a nondirectional test rather than "two-tailed," since the mechanics differ from a two-sided t-test.
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
                "alt_feedback": "✅ Correct. No directional prediction, and chi-square is nondirectional — it tests total departure from independence without distinguishing which group has higher rates.",
                "alt_wrong_feedback": "❌ No directional prediction was made. Also, chi-square is a nondirectional test — it measures total departure from independence without regard to which direction the difference goes.",
                "tails_connection": "🎯 **Nondirectional (chi-square)** — Chi-square tests departure from independence without regard to direction. The statistic is always non-negative (squared deviations), so the concept of 'one-tailed vs. two-tailed' does not apply in the same way as for t-tests.",
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
         "outcome_wrong":{"Continuous":"❌ Learning disability is recorded as present or absent — a yes/no outcome, not a numeric measurement.","Categorical (Nominal >2 levels)":"❌ Categorical requires 3+. Diagnosis is yes/no = binary.","Ordinal":"❌ A diagnosis is yes or no = binary.","Rate (person-time)":"❌ All followed same 3-year period — binary outcome works."},
         "exposure_wrong":{"Categorical (>2 groups)":"❌ Categorical requires 3+ groups. Two groups here = binary.","Ordinal":"❌ Ordinal implies ordered categories (e.g., low/medium/high). Exposed vs. unexposed is simply two groups = binary.","Continuous":"❌ Exposure is group membership (exposed vs. unexposed neighborhoods), not a measured quantity — binary."},
         "data":{"type":"contingency","context":"3-year follow-up data. Calculate RR, OR, and p-value.",
                 "row_names":["Lead-exposed","Unexposed"],"col_names":["Learning Disability","No Learning Disability"],"cells":[[52,348],[21,379]]}},
        {"id":"s2","title":"Scenario 2: Fast Food & Obesity",
         "description":"One-time survey of 2,500 adults. Weekly fast food frequency (never/1–2x/3–4x/5+x) and obesity (BMI ≥30) measured simultaneously.",
         "correct_design":"Cross-sectional","correct_outcome":"Binary","correct_exposure":"Categorical (>2 groups)",
         "design_hint":"One-time survey — both measured simultaneously = cross-sectional.",
         "outcome_hint":"Obesity: BMI ≥30 vs. <30 — two categories = binary.",
         "exposure_hint":"Four frequency categories — more than 2 = categorical.",
         "design_wrong":{"Cohort":"❌ Cohort follows people over time. One-time survey = no follow-up.","Case-Control":"❌ Case-control recruits by disease status and looks back. Survey measured everything at once."},
         "outcome_wrong":{"Continuous":"❌ Obesity (BMI ≥30) is a yes/no classification — not a continuous measurement.","Categorical (Nominal >2 levels)":"❌ Obesity is yes or no — two categories = binary.","Ordinal":"❌ A diagnosis is binary.","Rate (person-time)":"❌ One-time survey — no follow-up time."},
         "exposure_wrong":{"Binary (2 groups)":"❌ Four frequency categories = categorical.","Ordinal":"❌ Close — the categories do have a natural order, but they are named discrete groups, not a measured quantity. Categorical (>2 groups) is the standard classification here.","Continuous":"❌ Fast food frequency is recorded in four discrete named categories, not as a measured quantity on a numeric scale — categorical."},
         "data":{"type":"contingency_wide","context":"Survey data by fast food frequency and obesity.",
                 "row_names":["Never","1–2x/week","3–4x/week","5+x/week"],"col_names":["Obese","Not Obese"],"cells":[[62,538],[118,682],[189,561],[141,209]]}},
        {"id":"s3","title":"Scenario 3: HPV Vaccine & Cervical Cancer",
         "description":"250 women with cervical cancer and 500 without recruited. Vaccination history assessed from medical records.",
         "correct_design":"Case-Control","correct_outcome":"Binary","correct_exposure":"Binary (2 groups)",
         "design_hint":"Started with disease status (cases vs. controls) then looked backward at vaccination = case-control.",
         "outcome_hint":"Cervical cancer: present or absent — binary.",
         "exposure_hint":"Vaccinated vs. unvaccinated — two groups = binary.",
         "design_wrong":{"Cohort":"❌ Cohort classifies by vaccination then tracks who gets cancer. Here recruited by cancer status.","Cross-sectional":"❌ Cross-sectional measures simultaneously. Here recruited by disease status and looked back."},
         "outcome_wrong":{"Continuous":"❌ Cervical cancer is present or absent — a yes/no outcome, not a numeric measurement.","Categorical (Nominal >2 levels)":"❌ Binary.","Ordinal":"❌ Binary.","Rate (person-time)":"❌ In case-control, outcome is determined before study begins."},
         "exposure_wrong":{"Categorical (>2 groups)":"❌ Vaccinated vs. unvaccinated = two groups = binary.","Ordinal":"❌ Vaccination status has no natural ordering — it is simply vaccinated or not = binary.","Continuous":"❌ Vaccination status is a group classification (vaccinated vs. unvaccinated), not a measured quantity — binary."},
         "data":{"type":"contingency","context":"Case-control data. Odds Ratio is appropriate.",
                 "row_names":["Unvaccinated","Vaccinated"],"col_names":["Cervical Cancer (Case)","No Cancer (Control)"],"cells":[[178,182],[72,318]]}},
        {"id":"s4","title":"Scenario 4: Shift Work & Metabolic Syndrome",
         "description":"1,200 hospital employees classified by shift: day only, rotating, or night. Followed 5 years. Metabolic syndrome (yes/no) assessed at end.",
         "correct_design":"Cohort","correct_outcome":"Binary","correct_exposure":"Categorical (>2 groups)",
         "design_hint":"Classified by exposure (shift type) → followed to outcome = cohort.",
         "outcome_hint":"Metabolic syndrome: present or absent — binary.",
         "exposure_hint":"Three shift types — more than 2 = categorical.",
         "design_wrong":{"Case-Control":"❌ Case-control starts with people who already have metabolic syndrome. Here employees classified by shift type first.","Cross-sectional":"❌ Employees followed 5 years — not a snapshot."},
         "outcome_wrong":{"Continuous":"❌ Metabolic syndrome is recorded as present or absent — a yes/no outcome, not a numeric measurement.","Categorical (Nominal >2 levels)":"❌ Metabolic syndrome = yes/no = binary.","Ordinal":"❌ Binary.","Rate (person-time)":"❌ All followed same 5-year period."},
         "exposure_wrong":{"Binary (2 groups)":"❌ Three categories = categorical.","Ordinal":"❌ Shift type (day/rotating/night) is an unordered nominal grouping, not a naturally ranked scale — categorical.","Continuous":"❌ Shift type is a named category (day/rotating/night), not a quantity measured on a numeric scale — categorical."},
         "data":{"type":"contingency_wide","context":"5-year follow-up data by shift type.",
                 "row_names":["Day shift","Rotating shift","Night shift"],"col_names":["Metabolic Syndrome","No Metabolic Syndrome"],"cells":[[62,338],[98,302],[121,279]]}},
        {"id":"s5","title":"Scenario 5: Air Pollution & ED Visits",
         "description":"3,000 adults monitored for PM2.5 over 2 years. Participants vary in outdoor time — each contributes different observation time. Outcome: new ED visits for respiratory illness.",
         "correct_design":"Cohort","correct_outcome":"Rate (person-time)","correct_exposure":"Binary (2 groups)",
         "design_hint":"Classified by PM2.5 level → tracked for new events = cohort.",
         "outcome_hint":"Varying follow-up time — must use person-time. Rate outcome.",
         "exposure_hint":"High vs. low PM2.5 — two groups = binary.",
         "design_wrong":{"Case-Control":"❌ Case-control would start with people who already had ED visits. Here classified by exposure first.","Cross-sectional":"❌ Followed over 2 years — not a snapshot."},
         "outcome_wrong":{"Binary":"❌ Follow-up time varies. Need person-time denominator.","Continuous":"❌ The outcome is a count of events relative to time at risk — that requires a rate with a person-time denominator, not a generic continuous measure.","Categorical (Nominal >2 levels)":"❌ Rate, not unordered categories.","Ordinal":"❌ Rate per person-time."},
         "exposure_wrong":{"Categorical (>2 groups)":"❌ High vs. low = two groups = binary.","Ordinal":"❌ High vs. low PM2.5 is two groups with no meaningful in-between ranking — binary.","Continuous":"❌ PM2.5 exposure is classified into two groups (high vs. low) here — binary, not a raw continuous measure."},
         "data":{"type":"rate","context":"Person-time data. Calculate IRR.",
                 "row_names":["High PM2.5","Low PM2.5"],"cases":[187,64],"person_time":[4200,5100]}},
        {"id":"s7","title":"Scenario 6: Air Pollution Spikes & MI",
         "description":"2,100 MI patients. PM2.5 in hour before symptom onset (hazard period) compared to PM2.5 at same time one week earlier for same patient (control period). No separate control group.",
         "correct_design":"Case-Crossover","correct_outcome":"Binary","correct_exposure":"Binary (2 groups)",
         "design_hint":"Each patient compared to themselves at a different time — no separate control group = case-crossover.",
         "outcome_hint":"MI: occurred or did not occur — binary.",
         "exposure_hint":"High vs. low PM2.5 — two groups = binary.",
         "design_wrong":{"Cohort":"❌ Cohort groups by exposure and follows forward. Here everyone already had MI.","Case-Control":"❌ Standard case-control recruits a separate control group. Here each case is their own control.","Cross-sectional":"❌ Cross-sectional is one time point. Here two time windows per person."},
         "outcome_wrong":{"Continuous":"❌ MI either occurred or did not — a yes/no outcome, not a numeric measurement.","Categorical (Nominal >2 levels)":"❌ MI: yes or no = binary.","Ordinal":"❌ Binary.","Rate (person-time)":"❌ Comparison between two windows per person, not varying follow-up."},
         "exposure_wrong":{"Categorical (>2 groups)":"❌ High vs. low = two groups = binary.","Ordinal":"❌ High vs. low PM2.5 is two groups — binary.","Continuous":"❌ PM2.5 is classified into two groups (high vs. low) here — binary, not a raw continuous measure."},
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
             "Continuous":"❌ Hypertension is diagnosed or not — a yes/no outcome, not a numeric measurement.",
             "Categorical (Nominal >2 levels)":"❌ Hypertension is diagnosed or not — two categories = binary.",
             "Ordinal":"❌ A diagnosis is yes/no = binary.",
             "Rate (person-time)":"❌ All participants have the same 8-year follow-up period — binary outcome is appropriate here.",
         },
         "exposure_wrong":{"Categorical (>2 groups)":"❌ High vs. low sodium = two groups = binary.","Ordinal":"❌ High vs. low sodium is two groups — binary.","Continuous":"❌ Sodium intake is classified into two groups (high vs. low) here — binary, not a raw continuous measure."},
         "data":{"type":"contingency","context":"Retrospective cohort data. Calculate RR, OR, and chi-square.",
                 "row_names":["High sodium","Low sodium"],"col_names":["Hypertension","No Hypertension"],"cells":[[312,1188],[198,1802]]}},
        {"id":"s9","title":"Scenario 8: Country-Level Alcohol Consumption & Liver Cirrhosis",
         "description":"A researcher compiles data from 42 countries. For each country, she records the national average alcohol consumption (liters per capita per year) and the national age-standardized liver cirrhosis mortality rate (per 100,000). She finds a strong positive correlation (r = 0.74) between the two country-level measures.",
         "correct_design":"Ecological",
         "correct_outcome":"Rate (person-time)",
         "correct_exposure":"Continuous",
         "design_hint":"The unit of analysis is **countries**, not individuals. Exposure and outcome are both measured at the aggregate (population) level — this is an ecological study.",
         "outcome_hint":"Liver cirrhosis mortality rate per 100,000 is a **rate with a person-time denominator** — countries contribute population-years of observation.",
         "exposure_hint":"Average alcohol consumption in liters per capita is a **continuous** measure — it takes any numeric value along a scale, not discrete groups.",
         "design_wrong":{
             "Cohort":"❌ A cohort study would follow individual people classified by their own alcohol consumption. Here the data are country averages — no individual-level data exist.",
             "Cross-sectional":"❌ Cross-sectional studies measure exposure and outcome for individuals at one point in time. Here both are aggregated to the country level — that's ecological.",
             "Case-Control":"❌ Case-control recruits individuals with and without disease. Here the units are entire countries, not individuals.",
         },
         "outcome_wrong":{
             "Binary":"❌ The outcome is a mortality rate per 100,000 — a continuous rate variable, not a yes/no for each person.",
             "Continuous":"❌ Close — it is numerically continuous, but because it has a person-time denominator (population-years), the precise type is Rate (person-time).",
             "Categorical (Nominal >2 levels)":"❌ A continuous rate is not an unordered categorical variable.",
             "Ordinal":"❌ A mortality rate is a continuous measure, not ordered categories.",
         },
         "exposure_wrong":{
             "Binary (2 groups)":"❌ Alcohol consumption in liters per capita spans a full numeric range across 42 countries — that is continuous, not two groups.",
             "Categorical (>2 groups)":"❌ While there are more than 2 countries, the exposure is a measured quantity on a continuous scale, not discrete named categories.",
             "Ordinal":"❌ Liters per capita is a precise numeric measurement, not ordered rank categories — continuous.",
         },
         "data":None},
    ]

    design_options   = ["— Select —","Cohort","Case-Control","Cross-sectional","Ecological","Case-Crossover"]
    outcome_options  = ["— Select —","Binary","Continuous","Categorical (Nominal >2 levels)","Ordinal","Rate (person-time)"]
    exposure_options = ["— Select —","Binary (2 groups)","Categorical (>2 groups)","Ordinal","Continuous"]

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
                        interpret_nnt(nnt, is_benefit, d["treatment_label"],
                                      d["control_label"], d["outcome_label"])
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
        {"id":"cb5",
         "title":"Scenario 5: Self-Reported Physical Activity & Cardiovascular Disease",
         "description":"A cohort study uses a self-reported physical activity questionnaire to classify 12,000 participants as active or inactive at baseline. The questionnaire has known measurement error — it misclassifies approximately 15% of truly active people as inactive, and 15% of truly inactive people as active. This error rate is the same regardless of whether participants later develop CVD.",
         "question":"What type of bias is present, and in which direction does it push the RR?",
         "options":["Non-differential misclassification — biases RR toward null (underestimates true association)","Differential misclassification — biases RR away from null (overestimates)","Confounding by age — RR direction unpredictable","Berkson's bias — biases toward null","Recall bias — biases away from null"],
         "correct":"Non-differential misclassification — biases RR toward null (underestimates true association)",
         "explanation":"Misclassification is non-differential because the error rate (15%) is the same in those who develop CVD and those who don't — it doesn't depend on the outcome. Non-differential misclassification of a binary exposure typically attenuates simple measures of association toward the null (RR toward 1.0) — this attenuation is the most common and expected pattern. Some truly active people are classified as inactive, diluting the 'active' group. The estimated RR will generally be closer to 1 than the truth — potentially masking a real protective effect. Note: this attenuation toward null is well-established for simple binary exposure misclassification in standard 2×2 analyses, though more complex settings can produce exceptions.",
         "follow_up":"If the study finds RR = 1.05 (not significant), what should you conclude?",
         "follow_up_options":["Physical activity may still be protective — non-differential misclassification attenuates the true RR toward null","Physical activity definitely has no effect on CVD","The result is a false positive due to misclassification","Misclassification causes overestimation, so the true RR is even higher than 1.05"],
         "correct_follow_up":"Physical activity may still be protective — non-differential misclassification attenuates the true RR toward null",
         "follow_up_explanation":"A null result (RR ≈ 1) in the presence of non-differential misclassification cannot be interpreted as evidence of no effect. The true RR could be meaningfully below 1.0 (protective), but misclassification has attenuated it toward 1.0. This is why measurement precision matters — imprecise exposure measurement can make real associations invisible."},
        {"id":"cb6",
         "title":"Scenario 6: HIV Treatment & Opportunistic Infections (Confounding by Indication)",
         "description":"A database study of HIV-positive patients finds that those who received antiretroviral therapy (ART) had higher rates of opportunistic infections (RR = 1.8) compared to those not on ART. A researcher concludes ART is harmful.",
         "question":"What is the most likely explanation for this finding?",
         "options":["Confounding by indication — sicker patients were more likely to receive ART","Recall bias — ART patients remember more infections","Berkson's bias — ART patients are more likely to be hospitalized","Non-differential misclassification of ART use","Loss to follow-up bias"],
         "correct":"Confounding by indication — sicker patients were more likely to receive ART",
         "explanation":"Confounding by indication occurs when the reason for receiving treatment (indication) is itself associated with the outcome. Patients with more advanced HIV disease (lower CD4 counts, more symptoms) were preferentially prescribed ART — and they were also at higher baseline risk of opportunistic infections. The disease severity confounds the ART-infection relationship. After adjusting for CD4 count and disease stage, ART is strongly protective. The crude RR of 1.8 is completely reversed by confounding.",
         "follow_up":"What analytic approach best addresses confounding by indication in this database study?",
         "follow_up_options":["Adjust for disease severity (CD4 count, viral load, clinical stage) in multivariable regression or use propensity score methods","Restrict to patients who received ART","Use a one-tailed test","Increase sample size"],
         "correct_follow_up":"Adjust for disease severity (CD4 count, viral load, clinical stage) in multivariable regression or use propensity score methods",
         "follow_up_explanation":"Confounding by indication requires measuring and adjusting for the factors that drove treatment decisions. CD4 count, viral load, and clinical stage are the primary determinants of ART initiation — and they independently predict opportunistic infections. Propensity score methods model the probability of treatment given all measured covariates, creating groups of treated and untreated patients with similar disease profiles."},
        {"id":"cb7",
         "title":"Scenario 7: Occupational Cohort — Loss to Follow-Up",
         "description":"A 10-year cohort study follows 4,000 factory workers to examine whether benzene exposure causes leukemia. By year 10, 28% of the original cohort could not be traced. Workers in the high-exposure group were significantly more likely to have left employment (often due to illness) and be lost to follow-up. Workers lost to follow-up had higher baseline benzene exposure than those who remained.",
         "question":"What is the primary concern about this loss to follow-up?",
         "options":["Differential loss to follow-up — the association between benzene and leukemia will be underestimated","Non-differential loss to follow-up — no systematic bias introduced","Recall bias — lost workers misremembered exposures","Berkson's bias — hospital admission patterns are distorted","Confounding by age — lost workers were older"],
         "correct":"Differential loss to follow-up — the association between benzene and leukemia will be underestimated",
         "explanation":"Loss to follow-up is differential because it is related to both the exposure (high benzene workers more likely to leave) and the outcome (workers who left due to illness had higher risk). The most exposed and highest-risk workers disproportionately left the study. This removes events from the exposed group, artificially lowering the apparent leukemia rate in that group and pushing the RR toward null — underestimating the true association. The study will appear to show benzene is less harmful than it truly is.",
         "follow_up":"Which direction does this differential loss bias the RR?",
         "follow_up_options":["Toward null — the benzene-leukemia association is underestimated","Away from null — the association is overestimated","Cannot determine direction","No effect — loss to follow-up only affects precision, not validity"],
         "correct_follow_up":"Toward null — the benzene-leukemia association is underestimated",
         "follow_up_explanation":"The highest-risk exposed workers are lost preferentially. Their leukemia cases never get counted. The remaining exposed workers are a healthier, lower-risk subset — making the exposed group look healthier than it truly is. This compresses the RR toward 1.0. In occupational epidemiology, this pattern is common and contributes to underestimation of occupational hazards."},
        {"id":"cb8",
         "title":"Scenario 8: Aspirin & Stroke — Confounding vs. Effect Modification",
         "description":"A cohort study finds that aspirin users have lower stroke risk overall (RR = 0.78). When stratified by sex, the results diverge sharply: in men, RR = 0.62 (38% risk reduction); in women, RR = 0.97 (essentially no effect). The crude and sex-stratified RRs differ substantially.",
         "question":"What does this pattern indicate?",
         "options":["Effect modification by sex — aspirin's protective effect differs by sex","Confounding by sex — sex distorts the crude aspirin-stroke association","Recall bias — women under-report aspirin use","Non-differential misclassification by sex","Berkson's bias — sex affects hospitalization rates"],
         "correct":"Effect modification by sex — aspirin's protective effect differs by sex",
         "explanation":"Effect modification (interaction) occurs when the magnitude of an association differs across levels of a third variable. Here, sex modifies the aspirin-stroke relationship: aspirin is substantially protective in men but not in women. This is a real biological phenomenon to be reported, not a bias to be removed. The appropriate response is to report sex-stratified RRs, not a single adjusted estimate, because pooling would give a misleading 'average' effect that applies to neither sex accurately.",
         "follow_up":"How does effect modification differ from confounding in terms of what you do about it?",
         "follow_up_options":["Confounding is removed by adjustment; effect modification is reported by presenting stratum-specific estimates","Both are removed by multivariable adjustment","Confounding is reported; effect modification is adjusted away","They are the same phenomenon described differently"],
         "correct_follow_up":"Confounding is removed by adjustment; effect modification is reported by presenting stratum-specific estimates",
         "follow_up_explanation":"The fundamental distinction: confounding is a bias — an alternative explanation for your finding that you want to eliminate. Effect modification is a finding — real variation in the association across subgroups that you want to describe and report. Adjusting for an effect modifier would hide clinically and biologically important heterogeneity. Stratified reporting gives each subgroup the accurate estimate that applies to them."},
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
        {
            "id": "ss6",
            "title": "Scenario 6: Propagated vs. Point-Source Epidemic Curve",
            "description": """
A city health department investigates an outbreak of influenza-like illness. The epidemic curve shows:
- First cases appear in week 1 (likely index cases)
- A second, larger wave of cases appears approximately 2–3 weeks later
- A third smaller wave appears 2–3 weeks after that
- Cases continue spreading for 8 weeks total before declining
- Cases are occurring in multiple households, schools, and workplaces across the city
            """,
            "question": "What transmission pattern does this epidemic curve suggest?",
            "options": [
                "Propagated — person-to-person spread over successive incubation periods",
                "Point-source — single common exposure event",
                "Mixed — point source followed by person-to-person spread",
                "Endemic — stable background rate of disease"
            ],
            "correct": "Propagated — person-to-person spread over successive incubation periods",
            "explanation": """
**Classic propagated epidemic curve features:**
1. ✅ Multiple waves of cases, each approximately one incubation period apart
2. ✅ Extended time course (weeks to months)
3. ✅ Cases spread geographically across households, schools, workplaces
4. ✅ Consistent with influenza's ~2–3 day incubation and respiratory transmission

**The pattern reflects chains of transmission:**
- Wave 1: Index cases infect close contacts
- Wave 2: Those contacts infect the next generation (~1 incubation period later)
- Wave 3: The third generation of cases — classic propagated pattern

**How to distinguish from point-source:**
- Point-source: single sharp peak, all cases within one incubation period, no geographic spread
- Propagated: successive waves, spread over many incubation periods, multiple settings affected
- Mixed: starts sharp (point source), then secondary waves emerge (person-to-person)

**Public health implication:** Propagated outbreaks require interrupting transmission chains — isolation, contact tracing, vaccination — rather than simply removing a contaminated food source.
            """,
            "follow_up": "The incubation period for influenza is 1–4 days. Approximately how far apart should the peaks of successive waves be?",
            "follow_up_options": [
                "1–4 days apart — approximately one incubation period between generations",
                "2–3 weeks apart — one generation of spread takes weeks",
                "The peaks should all occur simultaneously",
                "Peak spacing is unrelated to incubation period"
            ],
            "correct_follow_up": "1–4 days apart — approximately one incubation period between generations",
            "follow_up_explanation": "Each new wave represents a new generation of cases — people infected by the previous generation. The time between waves equals approximately one incubation period, because it takes that long for newly infected people to become symptomatic. This is why epidemic curve wave spacing is used to estimate incubation periods in novel outbreaks."
        },
        {
            "id": "ss7",
            "title": "Scenario 7: Attack Rate vs. Case Fatality Rate",
            "description": """
During an Ebola outbreak in a remote village of 400 residents:
- 60 residents develop Ebola disease over the course of the outbreak
- 42 of the 60 cases die from the disease
- The remaining 340 residents were exposed but did not develop illness

A public health team reports two key metrics to characterize the outbreak severity.
            """,
            "question": "What is the case fatality rate (CFR) for this outbreak?",
            "options": [
                "70% (42 ÷ 60 × 100) — deaths among those who developed disease",
                "10.5% (42 ÷ 400 × 100) — deaths among all village residents",
                "15% (60 ÷ 400 × 100) — cases among all village residents",
                "85% (340 ÷ 400 × 100) — survivors among all residents"
            ],
            "correct": "70% (42 ÷ 60 × 100) — deaths among those who developed disease",
            "explanation": """
**Case Fatality Rate (CFR):** Deaths ÷ Cases × 100 = 42 ÷ 60 × 100 = **70%**

CFR measures the **probability of death given disease** — it uses only people who developed the disease as the denominator. It is a measure of disease severity (virulence), not exposure risk.

**Attack Rate (AR):** Cases ÷ Population at risk × 100 = 60 ÷ 400 × 100 = **15%**

AR measures the **probability of developing disease given exposure** — it uses the full at-risk population as denominator. It reflects how transmissible or how potent the exposure is.

**The critical distinction:**
| Measure | Numerator | Denominator | Answers |
|---|---|---|---|
| Attack Rate | Cases | Population at risk | How likely is exposure to cause disease? |
| CFR | Deaths | Cases (ill people only) | How deadly is the disease once you have it? |
| Mortality Rate | Deaths | Population at risk | How likely is the population to die? |

Ebola's CFR of 70% in this outbreak reflects its extreme virulence. The attack rate of 15% reflects the transmission dynamics in this specific setting.
            """,
            "follow_up": "What is the mortality rate (deaths per population at risk) for this outbreak?",
            "follow_up_options": [
                "10.5% (42 ÷ 400 × 100) — proportion of the whole village who died",
                "70% (42 ÷ 60 × 100) — that's the CFR, not mortality rate",
                "15% (60 ÷ 400 × 100) — that's the attack rate",
                "42% (42 ÷ 100 × 100) — incorrect denominator"
            ],
            "correct_follow_up": "10.5% (42 ÷ 400 × 100) — proportion of the whole village who died",
            "follow_up_explanation": "Mortality rate uses the full population at risk as the denominator — it answers 'what fraction of people in the exposed population died?' CFR (70%) is higher than mortality rate (10.5%) because CFR is conditional on being a case. Mortality rate accounts for the fact that most exposed people (340/400) never developed disease at all."
        },
        {
            "id": "ss8",
            "title": "Scenario 8: Secondary Attack Rate",
            "description": """
A norovirus outbreak begins when one child (the index case) returns from school with gastroenteritis. Over the next week, 3 of the 4 other household members develop the same illness.

Meanwhile, in a separate household nearby, a different child is the index case. Their household has 5 members besides the index case; 2 develop illness.
            """,
            "question": "What is the secondary attack rate (SAR) in Household 1?",
            "options": [
                "75% (3 ÷ 4 × 100) — cases among susceptible household contacts",
                "80% (4 ÷ 5 × 100) — cases including the index case",
                "60% (3 ÷ 5 × 100) — cases among all household members",
                "25% (1 ÷ 4 × 100) — those who didn't get sick"
            ],
            "correct": "75% (3 ÷ 4 × 100) — cases among susceptible household contacts",
            "explanation": """
**Secondary Attack Rate (SAR):** Secondary cases ÷ Susceptible contacts × 100

**Household 1:** 3 secondary cases ÷ 4 susceptible contacts = **75%**

The index case is **excluded** from both numerator and denominator. SAR measures transmission from the index case to contacts — so the denominator is only the people who could have been infected by the index case (susceptible contacts).

**Household 2 SAR:** 2 ÷ 5 = **40%**

**Why SAR matters:**
- SAR estimates the probability that a susceptible contact of a case will become infected
- Higher SAR = more transmissible pathogen or more intimate contact setting
- Used to estimate R₀ (basic reproduction number) in household studies
- Helps evaluate the effectiveness of isolation and prophylaxis interventions

**Comparison:** Norovirus is highly transmissible in household settings (SAR often 30–80%). The SAR difference between households reflects the stochastic nature of transmission and potentially household crowding or hygiene practices.
            """,
            "follow_up": "Why is the index case excluded from the SAR denominator?",
            "follow_up_options": [
                "Because the index case was infected outside the household — they cannot be a secondary case",
                "Because the index case is always immune after recovery",
                "To make the SAR calculation simpler",
                "The index case should be included — the denominator should be all household members"
            ],
            "correct_follow_up": "Because the index case was infected outside the household — they cannot be a secondary case",
            "follow_up_explanation": "The secondary attack rate specifically measures household transmission from the index case. The index case was infected from an external source (school), not from within the household. Including them would conflate community transmission with household transmission. The SAR denominator is only those at risk of becoming secondary cases — the household contacts who were susceptible when the index case arrived home."
        },
        {
            "id": "ss9",
            "title": "Scenario 9: Prevalence–Incidence Relationship (P = I × D)",
            "description": """
A disease epidemiologist is reviewing data from a national surveillance system:

- Incidence of Type 1 diabetes: **22 new cases per 100,000 per year**
- Prevalence of Type 1 diabetes: **220 per 100,000**

A colleague notes that the prevalence seems unexpectedly high. The epidemiologist wants to use the steady-state relationship between prevalence, incidence, and duration to check the implied average disease duration.
            """,
            "question": "Using P ≈ I × D, what is the implied average duration of Type 1 diabetes in this population?",
            "options": [
                "10 years (220 ÷ 22 = 10)",
                "242 years (220 + 22)",
                "0.1 years (22 ÷ 220)",
                "4,840 years (220 × 22)"
            ],
            "correct": "10 years (220 ÷ 22 = 10)",
            "explanation": """
**The prevalence-incidence relationship:**

At steady state: **P ≈ I × D**

Rearranging: **D = P ÷ I** = 220 ÷ 22 = **10 years**

This means the average person with Type 1 diabetes in this population has had the disease for approximately 10 years — which seems low for a lifelong condition. This might suggest:
- Significant mortality reducing the prevalent pool (people dying with the disease)
- Recent changes in incidence (P lags behind recent incidence changes)
- Migration of cases out of the population

**When P = I × D is valid:**
- Disease incidence is stable over time (steady state)
- Average duration is relatively stable
- The population is closed (no major in/out migration)

**Using the formula in reverse:** If you know prevalence and average duration, you can estimate incidence even without direct case surveillance — useful in settings with incomplete reporting.

**Key insight:** Prevalence is a function of BOTH how fast new cases occur (incidence) AND how long people stay in the prevalent pool (duration). A disease can have high prevalence with low incidence if it lasts a long time (e.g., HIV on treatment). A disease can have low prevalence with high incidence if it resolves quickly or causes rapid death.
            """,
            "follow_up": "HIV incidence is 0.4/1,000/year in a population. With modern ART, average duration is now 40 years. What prevalence does P = I × D predict?",
            "follow_up_options": [
                "16 per 1,000 (0.4 × 40)",
                "40.4 per 1,000 (0.4 + 40)",
                "100 per 1,000 (0.4 × 40 / some factor)",
                "0.01 per 1,000 (0.4 ÷ 40)"
            ],
            "correct_follow_up": "16 per 1,000 (0.4 × 40)",
            "follow_up_explanation": "P ≈ I × D = 0.4/1,000/year × 40 years = 16/1,000. This illustrates a critical point about HIV in the ART era: even as incidence has declined, prevalence has risen because people are living much longer with HIV. This is why HIV prevalence continues to increase in many settings despite falling incidence — longer duration dominates. Policy implications: more people need sustained treatment, but fewer new infections are occurring."
        },
        {
            "id": "ss10",
            "title": "Scenario 10: Likelihood Ratios — Changing Pre-Test Probability",
            "description": """
A 55-year-old male smoker presents with a chronic cough. You are considering lung cancer.

**Pre-test probability of lung cancer:** 8% (based on age, smoking history, symptoms)

**Chest CT scan performance:**
- Sensitivity = 96%
- Specificity = 80%
- LR+ = sensitivity ÷ (1 − specificity) = 0.96 ÷ 0.20 = **4.8**
- LR− = (1 − sensitivity) ÷ specificity = 0.04 ÷ 0.80 = **0.05**

The CT comes back **positive**.
            """,
            "question": "What does an LR+ of 4.8 tell you about this positive CT result?",
            "options": [
                "A positive CT is 4.8× more likely in someone with lung cancer than in someone without it",
                "The patient has a 4.8% chance of having lung cancer",
                "4.8% of positive tests are true positives",
                "The test increases sensitivity by a factor of 4.8"
            ],
            "correct": "A positive CT is 4.8× more likely in someone with lung cancer than in someone without it",
            "explanation": """
**Likelihood Ratio (LR) interpretation:**

**LR+ = P(positive test | disease) ÷ P(positive test | no disease)**
= Sensitivity ÷ (1 − Specificity) = 0.96 ÷ 0.20 = **4.8**

LR+ = 4.8 means: *a positive test result is 4.8 times more likely to occur in a person who has the disease than in a person who doesn't.*

**Converting to post-test probability using Bayes:**
- Pre-test odds = 0.08 ÷ 0.92 = **0.087**
- Post-test odds = pre-test odds × LR+ = 0.087 × 4.8 = **0.416**
- Post-test probability = 0.416 ÷ (1 + 0.416) = **29%**

The positive CT raises the probability of lung cancer from **8% to 29%**.

**LR− interpretation:** If the CT had been negative, post-test odds = 0.087 × 0.05 = 0.0043, post-test probability = **0.4%**. A negative CT nearly rules out lung cancer in this patient.

**LR benchmarks:**
| LR value | Interpretation |
|---|---|
| > 10 | Large, often conclusive change |
| 5–10 | Moderate increase |
| 2–5 | Small but sometimes important |
| 1 | No change in probability |
| 0.1–0.5 | Small decrease |
| < 0.1 | Large decrease, nearly rules out |
            """,
            "follow_up": "If the pre-test probability were 50% instead of 8% (high-risk screening setting), what would the post-test probability be after a positive CT (LR+ = 4.8)?",
            "follow_up_options": [
                "82.8% — pre-test odds 1.0 × 4.8 = post-test odds 4.8 → probability 4.8/5.8",
                "50% × 4.8 = 240% — not possible",
                "96% — the sensitivity of the test",
                "54.8% — adding LR to pre-test probability"
            ],
            "correct_follow_up": "82.8% — pre-test odds 1.0 × 4.8 = post-test odds 4.8 → probability 4.8/5.8",
            "follow_up_explanation": "Pre-test probability 50% → pre-test odds = 0.5/0.5 = 1.0. Post-test odds = 1.0 × 4.8 = 4.8. Post-test probability = 4.8/(1+4.8) = 82.8%. The same LR+ of 4.8 takes a low-risk patient from 8% to 29%, but takes a high-risk patient from 50% to 83%. This demonstrates why LRs must always be applied to a specific pre-test probability — the same test result has completely different clinical meaning depending on who was tested."
        },
        {
            "id": "ss11",
            "title": "Scenario 11: When NPV Matters Most",
            "description": """
A primary care physician sees two patients in one day. Both receive a negative mammogram result.

**Patient A:** 65-year-old woman with a strong family history of breast cancer (first-degree relative diagnosed at 42), carries a BRCA1 mutation. Pre-test probability of breast cancer: **15%**.

**Patient B:** 35-year-old woman with no family history, no risk factors, routine screening. Pre-test probability of breast cancer: **0.4%**.

**Mammogram performance:**
- Sensitivity = 85%, Specificity = 90%
- NPV varies by prevalence
            """,
            "question": "For which patient is the negative mammogram result more reassuring (higher NPV)?",
            "options": [
                "Patient B (low pre-test probability) — NPV is higher when disease is rare",
                "Patient A (high pre-test probability) — the test worked harder to find cancer",
                "The same for both — NPV depends only on sensitivity and specificity",
                "Cannot determine without knowing the exact number of patients tested"
            ],
            "correct": "Patient B (low pre-test probability) — NPV is higher when disease is rare",
            "explanation": """
**Calculating NPV for each patient (per 10,000 tested):**

**Patient A (prevalence = 15%):**
- True cases: 1,500; Disease-free: 8,500
- True negatives: 8,500 × 0.90 = 7,650
- False negatives: 1,500 × 0.15 = 225 (missed cancers)
- NPV = 7,650 ÷ (7,650 + 225) = **97.1%**

**Patient B (prevalence = 0.4%):**
- True cases: 40; Disease-free: 9,960
- True negatives: 9,960 × 0.90 = 8,964
- False negatives: 40 × 0.15 = 6
- NPV = 8,964 ÷ (8,964 + 6) = **99.9%**

**The negative result is far more reassuring for Patient B.**

For Patient A, despite a negative mammogram, there is still a **2.9% chance she has breast cancer** that was missed. This is not negligible given her BRCA1 status — additional imaging (MRI) is warranted.

**Key principle:** NPV is highest when disease prevalence is low. In high-prevalence populations, a negative test cannot as confidently rule out disease because there are more true cases, and some will inevitably be missed (false negatives).
            """,
            "follow_up": "What is the correct clinical action for Patient A after a negative mammogram?",
            "follow_up_options": [
                "Additional imaging (breast MRI) — NPV of 97.1% means a 2.9% miss rate is clinically unacceptable for a BRCA1 carrier",
                "Reassure and return in 2 years — the mammogram was negative",
                "Repeat the mammogram immediately",
                "Biopsy all BRCA1 carriers regardless of imaging results"
            ],
            "correct_follow_up": "Additional imaging (breast MRI) — NPV of 97.1% means a 2.9% miss rate is clinically unacceptable for a BRCA1 carrier",
            "follow_up_explanation": "For a BRCA1 carrier with 15% pre-test probability, a 2.9% post-negative-test probability of missed cancer is clinically significant. Current guidelines recommend annual MRI in addition to mammography for BRCA1/2 carriers — MRI has higher sensitivity (>90%) for this population. This scenario illustrates why 'negative' doesn't mean 'all clear' in high-risk populations, and why understanding NPV is essential for clinical decision-making."
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
# OUTBREAK LAB
# ==================================================
elif current_page == "outbreak_lab":
    st.title("🔍 Outbreak Lab")
    st.markdown("""
You are an Epidemic Intelligence Service (EIS) officer. Three outbreaks have been reported. Work through the clues, make decisions, calculate the numbers, and solve each investigation. Every decision maps to the **10-step outbreak investigation framework**.
    """)

    import math as _omath

    OB1_STEPS = [
        "Step 1 — Verify the diagnosis & establish the outbreak",
        "Step 2 — Construct a case definition",
        "Step 3 — Epidemic curve & descriptive epidemiology",
        "Step 4 — Generate & test hypotheses (attack rates)",
        "Step 5 — Control measures & resolution",
    ]
    OB2_STEPS = [
        "Step 1 — Verify diagnosis & chain of infection",
        "Step 2 — Herd immunity & the math behind the outbreak",
        "Step 3 — Contact tracing & case finding",
        "Step 4 — Control measures",
        "Step 5 — Could this have been prevented?",
    ]
    OB3_STEPS = [
        "Step 1 — Build the case definition & line list",
        "Step 2 — Epidemic curve & incubation period estimation",
        "Step 3 — Food-specific attack rates (calculate)",
        "Step 4 — Environmental investigation",
        "Step 5 — Control, report & prevent recurrence",
    ]

    def next_step_button(current_step, all_steps, idx_key, label="Next Step"):
        """Advance step using a plain (non-widget-bound) index in session state."""
        # Initialize if not set
        if idx_key not in st.session_state:
            st.session_state[idx_key] = 0
        idx = all_steps.index(current_step) if current_step in all_steps else 0
        if idx >= 0 and idx < len(all_steps) - 1:
            next_label = all_steps[idx + 1]
            st.markdown("---")
            col_nb1, col_nb2, col_nb3 = st.columns([3, 2, 3])
            with col_nb2:
                if st.button(f"➡️ {label}", key=f"next_{idx_key}_{idx}", use_container_width=True):
                    st.session_state[idx_key] = idx + 1
                    st.rerun()
            st.caption(f"Next: **{next_label}**")
        elif idx == len(all_steps) - 1:
            st.markdown("---")
            st.success("🎉 **Scenario complete!** Select a new outbreak above, or jump back to any step to review.")


    # ── Compendium reference ──────────────────────────────────────────────────
    with st.expander("📋 Field Reference — Compendium of Acute Foodborne GI Diseases (keep open while investigating)"):
        st.markdown("Use this table to match incubation period, symptoms, and food vehicle to the most likely agent.")
        st.divider()

        st.markdown("#### Group I — Short incubation, vomiting predominant, little or no fever")
        group1 = [
            ["Bacillus cereus (emetic)", "0.5–6 hours", "Nausea, vomiting, cramps; diarrhea occasionally", "Boiled or fried rice"],
            ["Staphylococcus aureus", "2–4 h (range 0.5–8 h)", "Nausea, cramps, vomiting, diarrhea; fever may be present", "Ham, beef, poultry; cream-filled pastries; custard; high-protein leftovers"],
            ["Heavy metals (arsenic, cadmium, copper, mercury, lead, zinc)", "<1–6 hours", "Nausea, vomiting, cramps, diarrhea", "High-acid food/beverages stored in coated or metal-contaminated containers"],
        ]
        import pandas as pd
        g1_df = pd.DataFrame(group1, columns=["Agent", "Incubation", "Symptoms", "Characteristic foods"])
        st.dataframe(g1_df, use_container_width=True, hide_index=True)

        st.markdown("#### Group II — Moderate to long incubation, diarrhea predominant, often with fever")
        group2 = [
            ["Bacillus cereus (diarrheal)", "6–24 hours", "Abdominal cramps, watery diarrhea; vomiting occasionally", "Custards, cereal products, meat loaf, sauces, refried beans, dried potatoes"],
            ["Campylobacter jejuni", "2–5 days (1–10 d)", "Diarrhea (often bloody), cramps, fever, nausea, vomiting", "Raw milk, poultry, water, raw clams, beef liver"],
            ["Clostridium perfringens", "8–12 h (6–24 h)", "Abdominal cramps, watery diarrhea; vomiting and fever rare", "Inadequately heated/reheated meats, stews, gravy, refried beans"],
            ["ETEC (Enterotoxigenic E. coli)", "10–72 hours", "Abdominal cramps, watery diarrhea", "Uncooked vegetables, salads, water"],
            ["STEC / E. coli O157:H7", "3–4 days (2–10 d)", "Bloody diarrhea, cramps; fever infrequent; HUS risk", "Undercooked ground beef, raw milk, produce, soft cheese, water"],
            ["Norovirus", "24–48 h (10–50 h)", "Nausea, vomiting, cramps, watery diarrhea, low fever", "Fecally contaminated ready-to-eat foods, frostings, clams, oysters, water"],
            ["Salmonella spp. (non-typhoidal)", "12–72 h (6 h–7 d)", "Diarrhea, cramps, fever, headache; vomiting occasionally", "Poultry, eggs, meat, raw milk, produce"],
            ["Shigella spp.", "1–3 days (1–7 d)", "Cramps, fever, diarrhea (often bloody), watery diarrhea, nausea", "Fecally contaminated foods, salads, cut fruit, water"],
            ["Vibrio cholerae", "24–72 h (hours–5 d)", "Profuse watery diarrhea (rice-water stools)", "Raw fish/shellfish, crustacean, fecally contaminated water/foods"],
            ["Vibrio parahaemolyticus", "12–24 h (4–96 h)", "Cramps, watery diarrhea, nausea, vomiting, fever; bloody diarrhea occasionally", "Marine fish, shellfish, crustacean (raw or undercooked)"],
            ["Yersinia enterocolitica", "4–6 days (1–14 d)", "Fever, diarrhea, cramps, vomiting; may mimic appendicitis", "Raw milk, tofu, water, undercooked pork"],
        ]
        g2_df = pd.DataFrame(group2, columns=["Agent", "Incubation", "Symptoms", "Characteristic foods"])
        st.dataframe(g2_df, use_container_width=True, hide_index=True)

        st.markdown("#### Group III — Special presentations")
        group3 = [
            ["Clostridium botulinum", "12–48 h (6 h–8 d)", "Nausea, vomiting, diarrhea; blurred vision; descending paralysis", "Canned low-acid foods, smoked fish, cooked potatoes, marine mammals"],
            ["Cryptosporidium spp.", "7 days (2–14 d)", "Watery diarrhea, cramps, nausea, vomiting, fever", "Water, fecally contaminated foods"],
            ["Giardia intestinalis", "7–10 days (3–25 d)", "Cramps, diarrhea, watery diarrhea, fatty stools, bloating", "Water, fecally contaminated foods"],
            ["Hepatitis A virus", "28–30 days (15–50 d)", "Fever, nausea, diarrhea, anorexia, jaundice", "Raw shellfish, cold fecally contaminated foods, water"],
            ["Scombroid fish poisoning", "Minutes–1 hour", "Headache, nausea, vomiting, flushing, dizziness, burning mouth/throat", "Temperature-abused fish (tuna, mahi mahi, bluefish, mackerel, marlin, bonito)"],
        ]
        g3_df = pd.DataFrame(group3, columns=["Agent", "Incubation", "Symptoms", "Characteristic foods"])
        st.dataframe(g3_df, use_container_width=True, hide_index=True)

        st.markdown("""
<div style="font-size:11px;color:#6b7280;margin-top:8px;line-height:1.6;">
<b>Symptom key:</b> AC = cramps; D = diarrhea; BD = bloody diarrhea; WD = watery diarrhea; F = fever; H = headache; N = nausea; V = vomiting<br>
<b>Sources:</b> Heymann DL. <i>Control of Communicable Diseases Manual</i> (19th ed.), APHA 2008; CDC; Wisconsin Division of Public Health P-01257 (4/2016).<br>
<b>How to use:</b> (1) Note incubation period from exposure to symptom onset. (2) Identify predominant symptom pattern. (3) Match to characteristic food vehicle. These three together narrow the differential significantly before lab results return.
</div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("#### 🔍 Quick differential by incubation period")
        diff_col1, diff_col2, diff_col3 = st.columns(3)
        with diff_col1:
            st.markdown("""
**< 6 hours**
- Staph aureus toxin
- B. cereus (emetic)
- Heavy metals
- Scombroid (minutes)

*Vomiting dominant; preformed toxin — no replication needed*
            """)
        with diff_col2:
            st.markdown("""
**6–72 hours**
- Clostridium perfringens
- B. cereus (diarrheal)
- Norovirus
- Salmonella
- ETEC

*Mixed picture; diarrhea often prominent*
            """)
        with diff_col3:
            st.markdown("""
**> 3 days**
- Campylobacter (2–5 d)
- STEC/O157 (3–4 d)
- Yersinia (4–6 d)
- Giardia (7–10 d)
- Hepatitis A (28–30 d)

*Longer incubation; fever common; HUS/complications possible*
            """)

    # Track scenario to reset step indices when scenario changes
    if "ob_prev_scenario" not in st.session_state:
        st.session_state["ob_prev_scenario"] = ""

    ob_scenario = st.selectbox("Select an outbreak to investigate:", [
        "— Choose an outbreak —",
        "🍽️ Scenario 1: Norovirus at a University Dining Hall",
        "📚 Scenario 2: Measles in an Under-Vaccinated Elementary School",
        "🥘 Scenario 3: Salmonellosis at a Community Church Potluck",
    ], key="ob_scenario_select")

    # Reset step index when scenario changes
    if ob_scenario != st.session_state["ob_prev_scenario"]:
        st.session_state["ob1_idx"] = 0
        st.session_state["ob2_idx"] = 0
        st.session_state["ob3_idx"] = 0
        st.session_state["ob_prev_scenario"] = ob_scenario

    st.divider()

    # ════════════════════════════════════════════════════════════════
    # SCENARIO 1: NOROVIRUS
    # ════════════════════════════════════════════════════════════════
    if ob_scenario == "🍽️ Scenario 1: Norovirus at a University Dining Hall":

        col_brief, col_stats = st.columns([2,1])
        with col_brief:
            st.markdown("""
### 🎯 Your Mission
A university student health center has reported an unusual cluster of gastrointestinal illness among students who ate in the main dining hall last Tuesday evening. Students report vomiting, diarrhea, and nausea starting 18–36 hours after the meal. Your job: identify the vehicle, control the outbreak, and prevent further spread.
            """)
        with col_stats:
            st.markdown("""
<div style="background:#fef2f2;border-radius:8px;padding:14px;font-size:13px;">
<b>📋 Outbreak Brief</b><br><br>
🤒 <b>Cases reported:</b> 47<br>
🏥 <b>Hospitalizations:</b> 3<br>
💀 <b>Deaths:</b> 0<br>
📍 <b>Location:</b> University campus<br>
⏱️ <b>Exposure date:</b> Tuesday dinner<br>
🦠 <b>Suspected agent:</b> Unknown
</div>
            """, unsafe_allow_html=True)

        ob1_step = st.radio("Jump to step:", [
            "Step 1 — Verify the diagnosis & establish the outbreak",
            "Step 2 — Construct a case definition",
            "Step 3 — Epidemic curve & descriptive epidemiology",
            "Step 4 — Generate & test hypotheses (attack rates)",
            "Step 5 — Control measures & resolution",
        ], index=st.session_state.get("ob1_idx", 0), horizontal=False)
        st.divider()

        # ── STEP 1 ──
        if ob1_step == "Step 1 — Verify the diagnosis & establish the outbreak":
            st.subheader("Step 1 — Does an outbreak actually exist?")
            st.markdown("""
The student health center has seen 47 students with vomiting and diarrhea in 48 hours. The usual Tuesday volume is 2–3 GI cases per week.

**Lab results so far:** 6 stool samples submitted. Results pending. Students report symptoms began 18–36 hours after Tuesday dinner.

**Clinical picture:** Sudden onset nausea, vomiting (projectile in some), watery diarrhea (non-bloody), low-grade fever, muscle aches. Symptoms resolving in 24–48 hours.
            """)
            st.info("💡 **Step 1 of 10:** Prepare for field work + Establish the outbreak exists")

            q1 = st.radio("**Decision 1A:** Based on the information above, does an outbreak exist?", [
                "— Select —",
                "Yes — 47 cases vs. expected 2–3/week clearly exceeds baseline",
                "No — wait for lab results before declaring an outbreak",
                "Maybe — need to interview students first",
            ], key="ob1_q1")

            if q1 == "Yes — 47 cases vs. expected 2–3/week clearly exceeds baseline":
                st.success("""
✅ **Correct.** An outbreak exists when case counts significantly exceed the expected baseline. 47 cases in 48 hours vs. 2–3/week = approximately 16× the baseline rate. You don't need lab confirmation to establish that an outbreak is occurring — epidemiologic evidence is sufficient to begin the investigation.
                """)
                st.markdown("**10-step connection:** Step 2 — *Establish the existence of an outbreak*")

            elif q1 == "No — wait for lab results before declaring an outbreak":
                st.error("""
❌ **Incorrect.** Waiting for lab results before acting is a common error that allows outbreaks to grow. Epidemiologic criteria (cases exceeding expected baseline by time, place, and person) are sufficient to declare and investigate an outbreak. Lab confirmation identifies the agent — it doesn't define whether an outbreak is occurring.
                """)

            elif q1 == "Maybe — need to interview students first":
                st.warning("""
⚠️ **Partially correct.** Interviewing is essential, but you have enough information right now to establish that case counts exceed the baseline. You can declare an outbreak AND begin interviews simultaneously — these are not sequential steps.
                """)

            if q1 != "— Select —":
                st.divider()
                q1b = st.radio("**Decision 1B:** What agent does the clinical picture most suggest?", [
                    "— Select —",
                    "Staphylococcus aureus toxin (onset 2–6 hours)",
                    "Norovirus (onset 12–48 hours, rapid spread, projectile vomiting)",
                    "Salmonella (onset 6–72 hours, bloody diarrhea common)",
                    "E. coli O157 (onset 1–10 days, bloody diarrhea, HUS risk)",
                ], key="ob1_q1b")

                if q1b == "Norovirus (onset 12–48 hours, rapid spread, projectile vomiting)":
                    st.success("""
✅ **Correct.** The 18–36 hour incubation, projectile vomiting, brief duration (24–48h), and high attack rate in a congregate setting are the classic norovirus signature. Staph toxin would present in 2–6 hours. Salmonella typically produces more diarrhea than vomiting. E. coli O157 rarely causes projectile vomiting and has a longer incubation.
                    """)
                elif q1b != "— Select —":
                    st.error("""
❌ **Incorrect.** Review the incubation periods: Staph toxin = 2–6h (preformed toxin). Norovirus = 12–48h. Salmonella = 6–72h (longer, more diarrhea-predominant). E. coli O157 = 1–10 days (bloody diarrhea, HUS risk). The 18–36h onset + projectile vomiting + brief illness duration = norovirus pattern.
                    """)

        # ── STEP 2 ──

            next_step_button(ob1_step, OB1_STEPS, "ob1_idx")

        elif ob1_step == "Step 2 — Construct a case definition":
            st.subheader("Step 2 — Who counts as a case?")
            st.markdown("""
You need a **case definition** before you can count cases, calculate attack rates, or analyze the data. A case definition has four components: **person, place, time, and clinical criteria**.

You currently have:
- Person: Students (and potentially staff) at the university
- Place: Main dining hall, Tuesday dinner service
- Time: Symptoms began between Tuesday evening and Thursday morning
- Clinical: Vomiting and/or diarrhea (3+ loose stools/24h) after eating at the dining hall
            """)
            st.info("💡 **Step 4 of 10:** Construct a working case definition")

            q2a = st.radio("**Decision 2A:** How sensitive should your initial case definition be?", [
                "— Select —",
                "Narrow (confirmed lab-positive only) — precise but will miss most cases",
                "Broad (any GI symptoms after Tuesday dinner) — sensitive, captures more cases early",
                "Moderate (vomiting OR ≥3 loose stools within 72h of Tuesday dinner) — balances sensitivity and specificity",
            ], key="ob1_q2a")

            if q2a == "Moderate (vomiting OR ≥3 loose stools within 72h of Tuesday dinner) — balances sensitivity and specificity":
                st.success("""
✅ **Correct.** Early in an investigation, case definitions should be broad enough to capture cases without being so loose they include unrelated illness. Starting with a moderate definition — vomiting OR ≥3 loose stools within the plausible incubation window — is standard practice. You refine it as more information emerges.
                """)
            elif q2a == "Broad (any GI symptoms after Tuesday dinner) — sensitive, captures more cases early":
                st.warning("""
⚠️ **Acceptable but not ideal.** Being broadly sensitive early is reasonable, but "any GI symptoms" risks including students with pre-existing conditions, mild unrelated illness, or anxiety responses. A minimum symptom threshold (vomiting OR ≥3 loose stools) improves specificity without losing too many true cases.
                """)
            elif q2a == "Narrow (confirmed lab-positive only) — precise but will miss most cases":
                st.error("""
❌ **Incorrect.** Lab-confirmed cases only would capture maybe 5–10% of the true outbreak. Most norovirus cases are never lab-confirmed. Requiring confirmation before counting cases would make your attack rates meaningless and delay control measures by days to weeks.
                """)

            if q2a != "— Select —":
                st.divider()
                st.markdown("#### ✏️ Build Your Case Definition")
                st.markdown("Using the components below, construct the full working case definition:")

                cc_who = st.selectbox("Person:", ["Any person", "Student or staff member", "Student only"], key="ob1_cc1")
                cc_where = st.selectbox("Place:", ["Anywhere on campus", "Who ate in the main dining hall", "Who ate any campus meal"], key="ob1_cc2")
                cc_when = st.selectbox("Time:", ["At any point this semester", "On Tuesday evening (Nov 5)", "Between Nov 4–7"], key="ob1_cc3")
                cc_clinical = st.selectbox("Clinical:", [
                    "With any GI complaint",
                    "With vomiting OR ≥3 loose stools within 72 hours of the meal",
                    "With lab-confirmed norovirus",
                ], key="ob1_cc4")

                if cc_who and cc_where and cc_when and cc_clinical:
                    st.info(f"""
**Your case definition:**
"{cc_who} {cc_where} {cc_when} with {cc_clinical.lower().replace('with ', '')}."
                    """)
                    if "Student or staff" in cc_who and "main dining hall" in cc_where and "Tuesday" in cc_when and "72 hours" in cc_clinical:
                        st.success("✅ This is a strong working case definition — specific enough to be meaningful, sensitive enough to capture cases, time-bounded to the exposure window.")
                    elif "lab-confirmed" in cc_clinical:
                        st.error("❌ Lab confirmation requirement will miss most cases and delay your investigation.")
                    else:
                        st.info("This definition will work for now. Note your choices — they affect who gets counted as a case.")

        # ── STEP 3 ──

            next_step_button(ob1_step, OB1_STEPS, "ob1_idx")

        elif ob1_step == "Step 3 — Epidemic curve & descriptive epidemiology":
            st.subheader("Step 3 — Describe the outbreak: Person, Place, Time")
            st.markdown("""
You have now interviewed 89 students who ate Tuesday dinner. 47 meet your case definition. Below is what you know about the distribution of cases.
            """)
            st.info("💡 **Step 6 of 10:** Describe the outbreak in terms of person, place, and time")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📊 Cases by time of symptom onset")
                # Time-ordered data: Tuesday dinner 5-8pm, norovirus incubation 18-36h
                # Onset runs Wed 6am through Thu 6am — clean unimodal bell
                onset_labels_ordered = [
                    "Wed\n6am", "Wed\n9am", "Wed\n12pm", "Wed\n3pm",
                    "Wed\n6pm", "Wed\n9pm", "Thu\n12am", "Thu\n3am", "Thu\n6am"
                ]
                onset_counts_ordered = [1, 4, 9, 13, 11, 6, 2, 1, 0]
                # Render as SVG via components so order is guaranteed
                import streamlit.components.v1 as _ob1_comp
                n_bars = len(onset_labels_ordered)
                max_c = max(onset_counts_ordered)
                cw, ch = 420, 180
                pad_l, pad_b, pad_t, pad_r = 36, 48, 20, 10
                pw = cw - pad_l - pad_r
                ph = ch - pad_b - pad_t
                bw = pw / n_bars - 2
                import math as _m

                # Nice y ticks
                tick_int = 5
                y_max_tick = tick_int * (_m.ceil(max_c / tick_int) + 1)

                bars_svg = ""
                for i, (lbl, cnt) in enumerate(zip(onset_labels_ordered, onset_counts_ordered)):
                    bh = (cnt / y_max_tick) * ph if y_max_tick > 0 else 0
                    bx = pad_l + i * (pw / n_bars) + 1
                    by = pad_t + ph - bh
                    bars_svg += f'<rect x="{round(bx,1)}" y="{round(by,1)}" width="{round(bw,1)}" height="{round(bh,1)}" fill="#3b82f6" rx="2"/>'
                    # x label (split on \n)
                    parts = lbl.split("\n")
                    lx = round(bx + bw/2, 1)
                    bars_svg += f'<text x="{lx}" y="{ch-28}" font-size="8" fill="#6b7280" text-anchor="middle">{parts[0]}</text>'
                    bars_svg += f'<text x="{lx}" y="{ch-18}" font-size="8" fill="#6b7280" text-anchor="middle">{parts[1]}</text>'

                # Y ticks
                yticks_svg = ""
                for v in range(0, y_max_tick + 1, tick_int):
                    ty = round(pad_t + ph - (v / y_max_tick) * ph, 1)
                    yticks_svg += f'<line x1="{pad_l}" y1="{ty}" x2="{pad_l+pw}" y2="{ty}" stroke="#e5e7eb" stroke-width="1"/>'
                    yticks_svg += f'<text x="{pad_l-4}" y="{ty+3}" font-size="8" fill="#9ca3af" text-anchor="end">{v}</text>'

                axes_svg = (
                    f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+ph}" stroke="#d1d5db" stroke-width="1.5"/>'
                    f'<line x1="{pad_l}" y1="{pad_t+ph}" x2="{pad_l+pw}" y2="{pad_t+ph}" stroke="#d1d5db" stroke-width="1.5"/>'
                    f'<text x="14" y="{pad_t+ph//2}" font-size="9" fill="#6b7280" text-anchor="middle" transform="rotate(-90,14,{pad_t+ph//2})">Cases</text>'
                )
                title_svg = f'<text x="{pad_l + pw//2}" y="13" font-size="10" font-weight="bold" fill="#374151" text-anchor="middle">Symptom Onset — Norovirus Outbreak</text>'

                svg_html = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;">
<svg xmlns="http://www.w3.org/2000/svg" width="{cw}" height="{ch}" style="font-family:sans-serif;background:#fafafa;border-radius:6px;display:block;">
  {title_svg}{yticks_svg}{bars_svg}{axes_svg}
</svg></body></html>"""
                _ob1_comp.html(svg_html, height=ch + 10, scrolling=False)
                st.caption("Tuesday dinner served 5:00–8:00 PM. X-axis shows time of symptom onset (Wednesday–Thursday).")

            with col2:
                st.markdown("#### 👥 Person characteristics")
                st.markdown("""
| Characteristic | Cases (n=47) | Non-cases (n=42) |
|---|---|---|
| Mean age | 19.8 years | 20.1 years |
| Female | 55% | 52% |
| Freshman | 62% | 58% |
| Ate salad bar | 87% | 41% |
| Ate hot entrée | 71% | 68% |
| Ate dessert bar | 44% | 42% |
| Sat in east section | 61% | 18% |
                """)

            st.divider()
            q3a = st.radio("**Decision 3A:** Based on the epidemic curve, what transmission pattern does this represent?", [
                "— Select —",
                "Propagated (person-to-person) — multiple waves",
                "Point source — single peak, all cases within one incubation period range",
                "Endemic — stable background rate",
                "Mixed — initial point source with secondary spread",
            ], key="ob1_q3a")

            if q3a == "Point source — single peak, all cases within one incubation period range":
                st.success("""
✅ **Correct.** The curve shows a single unimodal peak — cases rise from Wednesday morning, peak Wednesday afternoon, and decline through Thursday. All cases fall within an ~30-hour window (6am Wed to 6am Thu), consistent with a single common exposure at Tuesday dinner 18–36 hours earlier. This is a classic point-source epidemic curve. No secondary wave has appeared yet, suggesting person-to-person spread has not started.
                """)

            elif q3a == "Propagated (person-to-person) — multiple waves":
                st.error("❌ A propagated curve would show multiple waves separated by one incubation period. This curve has one peak — consistent with a single common exposure.")
            elif q3a != "— Select —":
                st.error("❌ The single sharp peak rising and falling within 24 hours, with all cases tracing to a single meal, is a classic point-source pattern.")

            if q3a != "— Select —":
                st.divider()
                st.markdown("""
> **Note before you answer:** Look carefully at both the gaps *and* the type of variable. The east seating section gap (61% vs. 18%) is actually larger than the salad bar gap (87% vs. 41%). Think about *why* one makes more epidemiologic sense as a vehicle than the other.
                """)
                q3b = st.radio("**Decision 3B:** What does the descriptive data suggest as the most likely vehicle?", [
                    "— Select —",
                    "Hot entrée — 71% of cases ate it",
                    "Salad bar — 87% of cases ate it vs. only 41% of non-cases",
                    "Dessert bar — similar rates in cases and non-cases",
                    "East seating section — cases concentrated there (61% vs. 18%)",
                ], key="ob1_q3b")
                if q3b == "Salad bar — 87% of cases ate it vs. only 41% of non-cases":
                    st.success("""
✅ **Correct.** The salad bar is the right hypothesis — and it's worth understanding *why*, because the east section gap is actually larger (43 points vs. 46 points for the salad bar).

**The critical distinction is biological plausibility and causal logic:**
- **Salad bar** is a *food vehicle* — it can be directly contaminated and ingested. This is a biologically plausible route of transmission for norovirus. The difference in exposure rates between cases and non-cases is large and epidemiologically meaningful.
- **East seating section** is a *place*, not a vehicle. Eating in the east section doesn't cause illness — it's almost certainly a confounder or proxy. The most likely explanation: the salad bar was located near or in the east section, so students who sat there were also more likely to eat from it. The seating section correlates with the exposure but is not the cause.

**The lesson:** When both a food item and a place show large gaps, ask whether the place association is explained by differential access to the food. Always prefer the biologically plausible vehicle over a geographical correlate. You'll confirm this with attack rate calculations in Step 4.
                    """)
                elif q3b == "East seating section — cases concentrated there (61% vs. 18%)":
                    st.warning("""
⚠️ **Good observation, but not the vehicle.** You correctly noticed that the east section gap (61% vs. 18%) is the largest in the table — that's careful reading. But a seating section is a *place*, not a food vehicle. Place cannot directly transmit norovirus.

**The better interpretation:** The east section concentration is almost certainly a proxy for salad bar exposure — the salad bar was likely positioned near or in the east section, so students who sat there disproportionately ate from it. In epi terms, seating section is a *confounder* or *surrogate marker* of the actual exposure (salad bar), not the cause.

**The rule:** When you see a strong association with a place, ask whether the place correlates with a food exposure. Always prioritize the biologically plausible food vehicle over a geographical correlate. The attack rate analysis in Step 4 will test the food hypothesis directly.
                    """)
                elif q3b == "Hot entrée — 71% of cases ate it":
                    st.error("❌ 71% of cases AND 68% of non-cases ate the hot entrée — almost identical rates, meaning no meaningful difference in exposure. When cases and non-cases ate something at nearly the same rate, that item is unlikely to be the vehicle.")
                elif q3b == "Dessert bar — similar rates in cases and non-cases":
                    st.error("❌ The dessert bar shows 44% vs. 42% — nearly identical rates. No association with illness. The vehicle will show a large gap between cases and non-cases.")


        # ── STEP 4 ──

            next_step_button(ob1_step, OB1_STEPS, "ob1_idx")

        elif ob1_step == "Step 4 — Generate & test hypotheses (attack rates)":
            st.subheader("Step 4 — Calculate attack rates and test your hypothesis")
            st.markdown("""
You have completed interviews with all 89 students who ate Tuesday dinner. Now you'll calculate food-specific attack rates and risk ratios to identify the vehicle.
            """)
            st.info("💡 **Steps 7–8 of 10:** Develop hypotheses → Test hypotheses analytically")

            st.markdown("#### 🧮 Interactive Attack Rate Calculator")
            st.markdown("""
For each food item, calculate:
- **Attack rate (exposed)** = sick among those who ate ÷ total who ate × 100
- **Attack rate (unexposed)** = sick among those who didn't eat ÷ total who didn't eat × 100
- **Risk Ratio (RR)** = AR exposed ÷ AR unexposed
            """)

            food_data = {
                "Salad bar (mixed greens)": {"ate_sick": 41, "ate_well": 6, "notate_sick": 6, "notate_well": 36},
                "Caesar salad dressing": {"ate_sick": 38, "ate_well": 5, "notate_sick": 9, "notate_well": 37},
                "Hot entrée (pasta)": {"ate_sick": 33, "ate_well": 30, "notate_sick": 14, "notate_well": 12},
                "Rolls/bread": {"ate_sick": 28, "ate_well": 25, "notate_sick": 19, "notate_well": 17},
                "Soft-serve ice cream": {"ate_sick": 20, "ate_well": 19, "notate_sick": 27, "notate_well": 23},
            }

            results = []
            for food, d in food_data.items():
                ate_total = d["ate_sick"] + d["ate_well"]
                notate_total = d["notate_sick"] + d["notate_well"]
                ar_exp = round(d["ate_sick"] / ate_total * 100, 1) if ate_total > 0 else 0
                ar_unexp = round(d["notate_sick"] / notate_total * 100, 1) if notate_total > 0 else 0
                rr = round(ar_exp / ar_unexp, 2) if ar_unexp > 0 else float("inf")
                results.append({
                    "Food item": food,
                    "Ate (sick/total)": f"{d['ate_sick']}/{ate_total}",
                    "AR exposed (%)": ar_exp,
                    "Did not eat (sick/total)": f"{d['notate_sick']}/{notate_total}",
                    "AR unexposed (%)": ar_unexp,
                    "RR": rr
                })

            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("### 🔍 Analyze the table — work through these before drawing conclusions")
            st.markdown("*Answer each question in order. Each one builds on the last.*")

            # ── ANALYSIS Q1: Null RR ──
            aq1 = st.radio(
                "**Analysis 1:** What RR value would indicate that a food item has NO association with illness?",
                ["— Select —", "RR = 0", "RR = 1.0", "RR = 0.5", "RR > 2"],
                key="ob1_aq1"
            )
            if aq1 == "RR = 1.0":
                st.success("✅ Correct. RR = 1.0 means the attack rate is identical in those who ate vs. those who didn't — eating that food conveys no additional risk. Look at rolls/bread (RR = 1.0) and soft-serve ice cream (RR = 0.95) — near-null RRs, no association.")
            elif aq1 != "— Select —":
                st.error("❌ RR = 1.0 is the null value — it means the attack rate in exposed equals the attack rate in unexposed. RR > 1 means increased risk; RR < 1 means decreased risk; RR = 1 means no difference.")

            if aq1 == "RR = 1.0":
                st.divider()
                # ── ANALYSIS Q2: Why AR unexposed matters ──
                aq2 = st.radio(
                    "**Analysis 2:** Hot entrée (pasta) has AR exposed = 52.4% — that seems high. Why is it NOT a strong vehicle candidate?",
                    ["— Select —",
                     "Because only 63 students ate it",
                     "Because AR unexposed is 53.8% — nearly identical — so eating it made no difference",
                     "Because pasta can't carry norovirus",
                     "Because the RR should be calculated differently for hot foods"],
                    key="ob1_aq2"
                )
                if aq2 == "Because AR unexposed is 53.8% — nearly identical — so eating it made no difference":
                    st.success("""
✅ Exactly right. This is the single most important concept in foodborne outbreak analysis: **a high AR exposed means nothing without a low AR unexposed.**

Hot entrée: 52.4% of those who ate it got sick. But 53.8% of those who *didn't* eat it also got sick. RR = 0.97 — essentially 1. Whether you ate the pasta or not made no difference to your risk.

This happens when a food is just popular — many people eat it, so many sick people ate it, but many well people did too. Absolute counts mislead; the ratio is what matters.
                    """)
                elif aq2 != "— Select —":
                    st.error("❌ The issue is the AR unexposed — what happened to people who DIDN'T eat it. If the unexposed get sick at the same rate as the exposed, eating it made no difference.")

                if aq2 == "Because AR unexposed is 53.8% — nearly identical — so eating it made no difference":
                    st.divider()
                    # ── ANALYSIS Q3: Rank the two candidates ──
                    aq3 = st.radio(
                        "**Analysis 3:** Two items show a large gap between AR exposed and AR unexposed: salad bar (mixed greens) and Caesar dressing. Which has the stronger signal and why?",
                        ["— Select —",
                         "Salad bar — more people ate it (47 vs. 43), so the sample is larger",
                         "Caesar dressing — AR exposed 88.4% vs. AR unexposed 19.6%, RR 4.51 vs. salad bar RR 6.1. Wait — salad bar has the higher RR",
                         "Caesar dressing — slightly lower AR exposed but far lower AR unexposed than salad bar (19.6% vs. 14.3%), giving RR 4.51 vs. 6.1. Salad bar actually has the higher RR",
                         "They are identical — both are strong candidates and you cannot distinguish them"],
                        key="ob1_aq3"
                    )
                    if aq3 == "Caesar dressing — slightly lower AR exposed but far lower AR unexposed than salad bar (19.6% vs. 14.3%), giving RR 4.51 vs. 6.1. Salad bar actually has the higher RR":
                        st.success("""
✅ Sharp reading — the salad bar actually has the higher RR (6.1 vs. 4.51). Both are strong signals. So why do investigators ultimately point to the dressing rather than the greens?

This is where **biological plausibility and ingredient overlap** enter the analysis: nearly every student who took mixed greens also added Caesar dressing, but some students took dressing alone (on other items or as a dip). The dressing is the more *specific* item — it narrows the hypothesis. Caesar dressing made with raw egg is a well-established norovirus vehicle when handled by an ill food worker.

**The method:** Use RR to identify candidates, then use ingredient overlap and biological plausibility to narrow to the specific vehicle. You'll confirm in Step 5 when the environmental swabs come back.
                        """)
                    elif aq3 == "Caesar dressing — AR exposed 88.4% vs. AR unexposed 19.6%, RR 4.51 vs. salad bar RR 6.1. Wait — salad bar has the higher RR":
                        st.success("""
✅ You caught the correction mid-answer — that's exactly right. Salad bar RR = 6.1, Caesar dressing RR = 4.51. The greens have the stronger statistical signal.

The reason investigators focus on the dressing anyway comes down to ingredient specificity and biological plausibility — Caesar dressing made with raw egg, handled by an ill worker, is the more actionable specific vehicle. Greens and dressing were nearly always consumed together, making statistical separation difficult.

This illustrates an important limitation of attack rate analysis: when two items are almost always eaten together, it can be hard to separate their individual contributions statistically.
                        """)
                    elif aq3 != "— Select —":
                        st.error("❌ Look carefully at both RRs in the table. Compare salad bar greens (RR = 6.1) vs. Caesar dressing (RR = 4.51). Which is numerically higher? Then think about why investigators might still focus on the dressing despite that.")

                    if aq3 != "— Select —" and aq3 != "— Select —":
                        st.divider()
                        # ── ANALYSIS Q4: What makes a strong vehicle overall ──
                        aq4 = st.radio(
                            "**Analysis 4:** Using the table, which of the following best describes the pattern of a STRONG vehicle vs. a NON-vehicle?",
                            ["— Select —",
                             "Strong vehicle: high AR exposed AND high AR unexposed / Non-vehicle: low AR in both",
                             "Strong vehicle: high AR exposed AND low AR unexposed (RR >> 1) / Non-vehicle: similar AR in both groups (RR ≈ 1)",
                             "Strong vehicle: high absolute case count / Non-vehicle: low absolute case count",
                             "Strong vehicle: item eaten by more than 50% of attendees / Non-vehicle: eaten by fewer"],
                            key="ob1_aq4"
                        )
                        if aq4 == "Strong vehicle: high AR exposed AND low AR unexposed (RR >> 1) / Non-vehicle: similar AR in both groups (RR ≈ 1)":
                            st.success("""
✅ This is the core rule of foodborne outbreak analysis — and now you can see it clearly in the table:

| Item | AR exposed | AR unexposed | RR | Verdict |
|---|---|---|---|---|
| Salad bar | 87.2% | 14.3% | **6.1** | ✅ Strong vehicle |
| Caesar dressing | 88.4% | 19.6% | **4.51** | ✅ Strong vehicle |
| Hot entrée | 52.4% | 53.8% | **0.97** | ❌ Not a vehicle |
| Rolls/bread | 52.8% | 52.8% | **1.0** | ❌ Not a vehicle |
| Soft-serve | 51.3% | 54.0% | **0.95** | ❌ Not a vehicle |

The vehicles are not the most *popular* foods — they're the foods where eating them made a *difference*. Now identify the most likely specific vehicle below.
                            """)
                        elif aq4 != "— Select —":
                            st.error("❌ Absolute counts and overall popularity are misleading. The defining pattern of a vehicle: people who ate it got sick at a much higher rate than people who didn't eat it. High AR exposed + low AR unexposed = high RR = strong vehicle signal.")

            st.divider()
            q4a = st.radio("**Decision 4A:** Based on your analysis, which food item is the most likely specific vehicle?", [
                "— Select —",
                "Hot entrée (pasta) — most students ate it",
                "Caesar salad dressing — strong RR, biologically plausible, more specific than greens",
                "Salad bar (mixed greens) — highest RR in the table",
                "Soft-serve ice cream — high absolute case count",
            ], key="ob1_q4a")

            if q4a == "Caesar salad dressing — strong RR, biologically plausible, more specific than greens":
                st.success("""
✅ **Correct.** Caesar dressing is the most actionable specific vehicle. Both dressing and greens show strong signals — but the dressing is the more specific item (raw egg, handled by an ill food worker) and is the more testable hypothesis for environmental sampling and food handler investigation. Investigators narrow from a food category (salad bar) to the specific contaminated item (dressing) — this is how outbreak reports cite vehicles.
                """)
            elif q4a == "Salad bar (mixed greens) — highest RR in the table":
                st.warning("""
⚠️ **Statistically defensible, but not the most specific answer.** The salad bar greens do have the highest RR (6.1). However, greens and Caesar dressing were consumed together by nearly everyone who visited the salad bar. The dressing is the more specific vehicle — it narrows the hypothesis to a single contaminated item that has a clear biological mechanism (raw egg + ill food handler). In practice, investigators report the most specific vehicle they can identify.
                """)
            elif q4a != "— Select —":
                st.error("""
❌ Work through the analysis questions above if you haven't already. The key: identify items with high AR exposed AND low AR unexposed (high RR). High absolute case counts or overall popularity are not the right criteria.
                """)

            if q4a != "— Select —":
                st.divider()
                st.markdown("#### 🧮 Calculate the overall attack rate for this outbreak")
                st.markdown("The outbreak brief told you: **47 cases** among **89 students** who ate Tuesday dinner.")
                col_ar1, col_ar2 = st.columns(2)
                with col_ar1:
                    total_sick_input = st.number_input("Total sick (cases):", min_value=0, max_value=200, value=0, key="ob1_ar1")
                with col_ar2:
                    total_exposed_input = st.number_input("Total who ate Tuesday dinner:", min_value=0, max_value=500, value=0, key="ob1_ar2")

                if total_exposed_input > 0 and total_sick_input > 0:
                    overall_ar = round(total_sick_input / total_exposed_input * 100, 1)
                    st.metric("Overall attack rate", f"{overall_ar}%")

                    correct_sick, correct_total = 47, 89
                    correct_ar = round(correct_sick / correct_total * 100, 1)

                    if total_sick_input == correct_sick and total_exposed_input == correct_total:
                        st.success(f"✅ **Correct — {total_sick_input}/{total_exposed_input} = {overall_ar}%.** Just over half of all diners became ill. An attack rate above 50% is unusually high for a foodborne outbreak and is consistent with a widely consumed contaminated item (like a salad bar item served to most attendees).")
                    elif total_sick_input != correct_sick and total_exposed_input == correct_total:
                        st.error(f"❌ The denominator (89 diners) is correct, but check the numerator. The case count from the outbreak brief is {correct_sick}, not {total_sick_input}. AR = {correct_sick}/{correct_total} = {correct_ar}%.")
                    elif total_sick_input == correct_sick and total_exposed_input != correct_total:
                        st.error(f"❌ The case count ({correct_sick}) is correct, but check the denominator. The attack rate uses all people who were exposed to the meal — all {correct_total} students who ate Tuesday dinner, not just those who got sick. AR = {correct_sick}/{correct_total} = {correct_ar}%.")
                    elif total_exposed_input < total_sick_input:
                        st.error("❌ The denominator (total exposed) cannot be smaller than the numerator (total sick). The denominator is everyone who ate the meal — sick AND well.")
                    else:
                        st.warning(f"⚠️ Not quite. From the scenario: {correct_sick} cases among {correct_total} students who ate Tuesday dinner. AR = {correct_sick}/{correct_total} = **{correct_ar}%**. Check which numbers you used.")
                elif total_sick_input > 0 or total_exposed_input > 0:
                    st.info("Enter both values to calculate the attack rate.")


        # ── STEP 5 ──

            next_step_button(ob1_step, OB1_STEPS, "ob1_idx")

        elif ob1_step == "Step 5 — Control measures & resolution":
            st.subheader("Step 5 — Implement control measures")
            st.info("💡 **Steps 9–10 of 10:** Implement control measures → Communicate findings")

            st.markdown("""
**Lab results are in:** Norovirus GII.4 detected in 5 of 6 stool samples. Environmental swabs positive on the salad bar sneeze guard and Caesar dressing pump handle.

**Food handler interview reveals:** One dining hall employee worked a full shift Tuesday despite vomiting that morning. This employee prepared and handled the Caesar dressing.

**Current situation:** 47 cases, 3 hospitalizations (rehydration only, all recovered). No deaths. Two new cases reported Thursday from students who did not eat Tuesday but had contact with ill roommates.
            """)

            q5a = st.radio("**Decision 5A:** The two new Thursday cases (contact with ill roommates) indicate what?", [
                "— Select —",
                "The outbreak is over — these are unrelated",
                "Person-to-person transmission has begun — secondary spread",
                "The Caesar dressing is still being served — still point-source exposure",
            ], key="ob1_q5a")

            if q5a == "Person-to-person transmission has begun — secondary spread":
                st.success("""
✅ **Correct.** Norovirus is highly contagious person-to-person (fecal-oral, vomit aerosol). These two cases represent a secondary wave beginning. The outbreak has shifted from pure point-source to mixed. Control measures must now address both the food source and person-to-person transmission.
                """)
            elif q5a != "— Select —":
                st.error("❌ Two cases in direct contact with ill students, without dining hall exposure, indicates person-to-person transmission has begun. This is a critical inflection point requiring expanded control measures.")

            if q5a != "— Select —":
                st.divider()
                st.markdown("#### Select ALL appropriate control measures (check all that apply):")
                cm1 = st.checkbox("Remove Caesar dressing from service immediately", key="ob1_cm1")
                cm2 = st.checkbox("Close the entire university", key="ob1_cm2")
                cm3 = st.checkbox("Exclude ill food handlers from work until 48h symptom-free", key="ob1_cm3")
                cm4 = st.checkbox("Reinforce hand hygiene among all dining staff", key="ob1_cm4")
                cm5 = st.checkbox("Issue guidance to ill students on isolation and hygiene", key="ob1_cm5")
                cm6 = st.checkbox("Enhance cleaning and disinfection of dining surfaces", key="ob1_cm6")
                cm7 = st.checkbox("Test all food items in the dining hall", key="ob1_cm7")

                if st.button("Submit control measures", key="ob1_cm_submit"):
                    score = sum([cm1, cm3, cm4, cm5, cm6])
                    if cm2:
                        st.error("❌ Closing the university is not proportionate and would not be recommended at this case count. Targeted interventions are appropriate.")
                    if not cm1:
                        st.error("❌ Removing the identified vehicle (Caesar dressing) is the single most important immediate step.")
                    if score >= 4 and cm1 and not cm2:
                        st.success(f"""
✅ **Well done.** You selected {score+1}/5 appropriate measures. The key actions are: (1) remove the vehicle, (2) exclude ill food handlers, (3) reinforce hand hygiene, (4) isolate ill students and advise hygiene, (5) enhance disinfection. Testing all food items is low-yield at this stage — focus resources on the identified vehicle and secondary spread.
                        """)
                    elif cm1:
                        st.info(f"You selected {score+1} measures. Consider also: {'excluding ill food handlers, ' if not cm3 else ''}{'reinforcing hand hygiene, ' if not cm4 else ''}{'guidance to ill students, ' if not cm5 else ''}{'enhanced disinfection' if not cm6 else ''}")

            st.divider()
            with st.expander("📋 Resolution & What You Applied"):
                st.markdown("""
**Outcome:** The Caesar dressing was prepared using raw shell eggs contaminated with norovirus from the ill food handler. 47 primary cases. 8 secondary cases in the following 4 days. All recovered. No deaths.

**The 10 steps you applied:**
| Step | What you did |
|---|---|
| 1. Prepare | Reviewed clinical picture, incubation period, agent characteristics |
| 2. Establish outbreak | Compared 47 cases to baseline of 2–3/week → clear excess |
| 3. Verify diagnosis | Clinical criteria consistent with norovirus; lab confirmation |
| 4. Case definition | Person (student/staff) + place (dining hall) + time (72h of Tuesday dinner) + clinical (vomiting or ≥3 stools) |
| 5. Case finding | Interviewed all 89 Tuesday diners |
| 6. Descriptive epi | Epidemic curve (point source), person characteristics, place (salad bar cluster) |
| 7. Hypothesis | Caesar dressing as vehicle based on differential exposure rates |
| 8. Test hypothesis | Attack rates and RR confirmed Caesar dressing (RR > 5) |
| 9. Control | Removed vehicle, excluded ill worker, hygiene reinforcement, isolation guidance |
| 10. Communicate | Report to student health, dining services, and state health department |
                """)

                with st.expander("🦠 What is norovirus?"):
                    st.markdown("""
**Norovirus** is the leading cause of foodborne illness in the United States, responsible for approximately 19–21 million illnesses annually. Key features:
- **Transmission:** Fecal-oral (food, water, contaminated surfaces), person-to-person, vomit aerosol
- **Incubation:** 12–48 hours (typically 24–36h)
- **Symptoms:** Projectile vomiting, watery non-bloody diarrhea, nausea, low-grade fever, myalgias
- **Duration:** 1–3 days (self-limited)
- **Infectious dose:** Extremely low — as few as 18 viral particles can cause infection
- **Environmental stability:** Survives on surfaces for days; resistant to many standard disinfectants (requires bleach-based products)
- **High-risk settings:** Cruise ships, nursing homes, hospitals, schools, catered events
- **Key control:** Exclude ill food handlers for 48h after symptom resolution; hand hygiene (soap and water — alcohol gel less effective); bleach disinfection of surfaces
                    """)


            next_step_button(ob1_step, OB1_STEPS, "ob1_idx")

    # ════════════════════════════════════════════════════════════════
    # SCENARIO 2: MEASLES
    # ════════════════════════════════════════════════════════════════
    elif ob_scenario == "📚 Scenario 2: Measles in an Under-Vaccinated Elementary School":

        col_brief, col_stats = st.columns([2,1])
        with col_brief:
            st.markdown("""
### 🎯 Your Mission
A parent calls the county health department: their 7-year-old is home from school with a rash and high fever. The child returned from an international trip 12 days ago. Over the next 3 days, 6 more children at the same school report similar symptoms. The school has a 72% MMR vaccination rate. Your job: confirm the diagnosis, stop transmission, and determine whether the outbreak could have been prevented.
            """)
        with col_stats:
            st.markdown("""
<div style="background:#fef3c7;border-radius:8px;padding:14px;font-size:13px;">
<b>📋 Outbreak Brief</b><br><br>
🤒 <b>Cases reported:</b> 7 (growing)<br>
🏥 <b>Hospitalizations:</b> 1<br>
💀 <b>Deaths:</b> 0<br>
📍 <b>Location:</b> Elementary school<br>
🧒 <b>Population:</b> 450 students<br>
💉 <b>MMR coverage:</b> 72%
</div>
            """, unsafe_allow_html=True)

        ob2_step = st.radio("Jump to step:", [
            "Step 1 — Verify diagnosis & chain of infection",
            "Step 2 — Herd immunity & the math behind the outbreak",
            "Step 3 — Contact tracing & case finding",
            "Step 4 — Control measures",
            "Step 5 — Could this have been prevented?",
        ], index=st.session_state.get("ob2_idx", 0), horizontal=False)
        st.divider()

        if ob2_step == "Step 1 — Verify diagnosis & chain of infection":
            st.subheader("Step 1 — Confirm measles and trace the chain")
            st.markdown("""
**Index case (Patient Zero):** 7-year-old, unvaccinated, returned from international travel 12 days ago. Presents with: 3-day prodrome of high fever (104°F), cough, coryza (runny nose), conjunctivitis. Then: classic maculopapular rash starting at hairline, spreading downward. Koplik spots (white spots on buccal mucosa) noted by clinician.

**Lab:** IgM measles antibody positive (state lab). PCR confirmatory test sent to CDC.

**Exposure timeline:** Returned from trip → attended school for 3 days before rash appeared (highly infectious during prodrome).
            """)

            q1 = st.radio("**Decision 1A:** How long was the index case potentially infectious at school before diagnosis?", [
                "— Select —",
                "0 days — measles is only infectious after rash appears",
                "3 days — infectious during the prodrome (4 days before to 4 days after rash onset)",
                "Only on the day of rash — maximum infectiousness",
                "10 days — for the full incubation period",
            ], key="ob2_q1a")

            if q1 == "3 days — infectious during the prodrome (4 days before to 4 days after rash onset)":
                st.success("""
✅ **Correct.** Measles is infectious from 4 days before to 4 days after rash onset — the prodrome period when the child appears to have "just a cold" is the most dangerous period for transmission. The index case attended school for 3 days during this window, potentially exposing every susceptible student they encountered.

This is why outbreak control is so difficult: by the time measles is diagnosed (rash + Koplik spots), the infectious period is already partially over and secondary cases are incubating.
                """)
            elif q1 != "— Select —":
                st.error("❌ Measles is infectious from 4 days BEFORE rash onset through 4 days AFTER — the prodrome cough/fever/runny nose phase is peak infectiousness. Waiting for the rash to diagnose means exposure has already occurred.")

            if q1 != "— Select —":
                st.divider()
                st.markdown("""
**Chain of infection — measles:**
| Link | Details |
|---|---|
| **Agent** | Measles virus (Paramyxovirus, RNA) |
| **Reservoir** | Humans only (no animal reservoir) |
| **Portal of exit** | Respiratory tract (cough, sneeze) |
| **Transmission** | Airborne — virus survives in air for up to 2 hours after infectious person leaves the room |
| **Portal of entry** | Respiratory tract |
| **Susceptible host** | Unvaccinated or immunocompromised |
                """)
                st.warning("""
⚠️ **Airborne transmission critical point:** Measles is one of the most contagious pathogens known. Unlike respiratory droplets that fall within 1 meter, measles virus remains suspended in the air for up to 2 hours. A susceptible person entering the same room AFTER the index case has left can still be infected. This makes standard droplet precautions insufficient.
                """)


            next_step_button(ob2_step, OB2_STEPS, "ob2_idx")

        elif ob2_step == "Step 2 — Herd immunity & the math behind the outbreak":
            st.subheader("Step 2 — Why did this outbreak happen? The herd immunity calculation")

            st.markdown("""
The school has 72% MMR vaccination rate. Let's calculate whether this is enough to prevent an outbreak using the herd immunity threshold.
            """)

            st.markdown("#### 🧮 Calculate the herd immunity threshold")
            r0_measles = st.slider("R₀ for measles in this school setting:", 10, 18, 15, key="ob2_r0")
            hit = round((1 - 1/r0_measles) * 100, 1)
            current_immunity = 72
            effective_r = round(r0_measles * (1 - current_immunity/100), 2)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Herd Immunity Threshold", f"{hit}%")
            with col2:
                st.metric("Current MMR coverage", "72%", delta=f"{round(72-hit,1)}% below threshold")
            with col3:
                st.metric("Effective R (Rₑ)", effective_r, delta="epidemic growing" if effective_r > 1 else "epidemic declining")

            if effective_r > 1:
                st.error(f"""
**Why the outbreak is happening:** With R₀ = {r0_measles} and only 72% immune, Rₑ = {effective_r}. Each case is generating {effective_r} new cases on average. The school is {round(hit-72,1)} percentage points below the herd immunity threshold — a significant immunity gap that allows the virus to spread efficiently.
                """)

            st.divider()
            q2a = st.radio("**Decision 2A:** The school has 450 students. How many susceptible students are there?", [
                "— Select —",
                "28 students (only those with documented vaccine exemptions)",
                f"126 students (28% of 450 = those not vaccinated)",
                "72 students (the number not vaccinated this year)",
                "It depends on prior infection history — vaccine records alone are insufficient",
            ], key="ob2_q2a")

            if q2a == "It depends on prior infection history — vaccine records alone are insufficient":
                st.success("""
✅ **Correct — and the most sophisticated answer.** Susceptibility = unvaccinated + vaccine failures (2–5% of vaccinated) + immunocompromised vaccinated individuals + those with unknown status. 28% unvaccinated = at minimum 126 susceptibles, but true susceptibility is higher when you account for primary vaccine failure (~2–5% of MMR recipients). This is why the effective R can remain >1 even with seemingly high coverage.
                """)
            elif q2a == "126 students (28% of 450 = those not vaccinated)":
                st.warning("""
⚠️ **Partially correct.** 28% × 450 = 126 unvaccinated students is the minimum count of susceptibles. But some vaccinated students have primary vaccine failure (2–5%), so the true susceptible pool is larger. Additionally, students with unknown or undocumented status add uncertainty.
                """)
            elif q2a != "— Select —":
                st.error("❌ Susceptibility is not simply the number who didn't get vaccinated this year. It includes those with no prior vaccination, vaccine failures, undocumented status, and immunocompromised individuals regardless of vaccination.")

            if q2a != "— Select —":
                st.divider()
                st.markdown("#### 📈 Epidemic curve projection")
                st.markdown(f"""
With Rₑ = **{effective_r}**, project the wave pattern:
- **Generation 1** (index case): 1 case
- **Generation 2** (~10 days later): ~{round(effective_r)} cases
- **Generation 3** (~20 days later): ~{round(effective_r**2)} cases
- **Generation 4** (~30 days later): ~{round(effective_r**3)} cases

This exponential growth pattern continues until susceptibles are exhausted or vaccination coverage increases above the HIT of {hit}%.
                """)


            next_step_button(ob2_step, OB2_STEPS, "ob2_idx")

        elif ob2_step == "Step 3 — Contact tracing & case finding":
            st.subheader("Step 3 — Who was exposed? Contact tracing at scale")
            st.markdown("""
You now have 7 confirmed cases. The index case attended school for 3 days during the infectious period. Your team needs to identify all contacts and determine their immune status.
            """)
            st.info("💡 **Step 5 of 10:** Find cases systematically — active case finding")

            st.markdown("""
**Exposure settings to investigate:**
1. **Classrooms** — same class as index case (25 students + teacher)
2. **School bus** — 42 students rode the same bus
3. **Cafeteria** — shared lunch period with ~180 students
4. **Gymnasium** — PE class (30 students) in a poorly ventilated space
5. **Hallways and common areas** — indirect exposure, hard to quantify
            """)

            q3a = st.radio("**Decision 3A:** For each exposure setting, should you classify contacts as high, medium, or low risk?", [
                "— Select —",
                "All contacts are equal — anyone in the school is at equal risk",
                "Duration and proximity determine risk — classroom and gym (prolonged, enclosed) = highest",
                "Only direct face-to-face contact counts — hallway contacts are not at risk",
            ], key="ob2_q3a")

            if q3a == "Duration and proximity determine risk — classroom and gym (prolonged, enclosed) = highest":
                st.success("""
✅ **Correct.** For airborne transmission, risk is proportional to duration of exposure and ventilation quality. Prolonged shared air space (classroom, gym) = highest risk. Cafeteria (shorter exposure, more people, better ventilation) = moderate. Hallways (brief exposure) = lower risk, but not zero since measles can survive 2 hours in air.
                """)
            elif q3a != "— Select —":
                st.error("❌ For airborne pathogens, exposure duration and ventilation are critical determinants of risk. Not all contacts are equal.")

            if q3a != "— Select —":
                st.divider()
                st.markdown("#### 📋 Contact tracing matrix")
                contact_data = pd.DataFrame({
                    "Setting": ["Same classroom", "School bus", "Cafeteria (same period)", "Gymnasium (PE)", "General school"],
                    "Contacts identified": [25, 42, 180, 30, 173],
                    "Vaccination status known": [24, 38, 120, 28, 90],
                    "Confirmed vaccinated": [19, 30, 89, 22, 62],
                    "Unvaccinated/unknown": [6, 12, 91, 8, 111],
                })
                st.dataframe(contact_data, use_container_width=True, hide_index=True)

                total_unvax = int(contact_data["Unvaccinated/unknown"].sum())
                total_contacts = int(contact_data["Contacts identified"].sum())
                st.metric("Total contacts identified", total_contacts)
                st.metric("Unvaccinated or unknown status", total_unvax,
                          delta="require post-exposure vaccination or exclusion")

                q3b = st.radio("**Decision 3B:** What should happen to unvaccinated contacts?", [
                    "— Select —",
                    "Nothing unless they develop symptoms",
                    "Exclude from school for 21 days OR vaccinate within 72h of exposure",
                    "Require quarantine at home until PCR tested",
                    "Vaccinate everyone regardless of prior status",
                ], key="ob2_q3b")

                if q3b == "Exclude from school for 21 days OR vaccinate within 72h of exposure":
                    st.success("""
✅ **Correct.** This is the standard public health response for unvaccinated measles contacts. MMR given within 72 hours of exposure can prevent or attenuate illness. If vaccination is refused or >72 hours have passed, exclusion from school for 21 days (one incubation period) prevents further exposure. This is a legally authorized public health measure.
                    """)
                elif q3b != "— Select —":
                    st.error("❌ 'Wait and see' allows further transmission during the incubation period. Exclusion or post-exposure vaccination is the appropriate public health intervention.")


            next_step_button(ob2_step, OB2_STEPS, "ob2_idx")

        elif ob2_step == "Step 4 — Control measures":
            st.subheader("Step 4 — Emergency vaccination and outbreak control")

            st.markdown("""
**Current status:** 12 confirmed cases (Day 10 of outbreak). The outbreak is in its second generation. 3 hospitalizations (pneumonia complication in one immunocompromised child). 228 unvaccinated or unknown-status contacts identified.

**Available interventions:**
1. Emergency vaccination clinic at school (MMR)
2. School closure (partial or full)
3. Exclusion of unvaccinated students
4. Enhanced surveillance for new cases
5. Healthcare provider alert (notify ER, clinics to report suspect cases)
            """)

            st.markdown("#### 🧮 Calculate vaccination coverage needed")
            current_cov = st.slider("Current school MMR coverage:", 60, 95, 72, key="ob2_vax_slider")
            r0_val = 15
            hit_val = round((1 - 1/r0_val) * 100, 1)
            gap = round(hit_val - current_cov, 1)
            students_needed = round((gap/100) * 450)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("HIT for measles (R₀=15)", f"{hit_val}%")
                st.metric("Current coverage", f"{current_cov}%")
            with col2:
                st.metric("Coverage gap", f"{gap}%", delta=f"Need {students_needed} more students vaccinated")
                reff = round(r0_val * (1 - current_cov/100), 2)
                st.metric("Current Rₑ", reff, delta="outbreak growing" if reff > 1 else "outbreak slowing")

            if current_cov >= hit_val:
                st.success(f"✅ At {current_cov}% coverage, Rₑ = {reff} < 1. Herd immunity achieved — outbreak will decline.")
            else:
                st.warning(f"⚠️ At {current_cov}% coverage, Rₑ = {reff} > 1. Outbreak will continue to grow until coverage reaches {hit_val}%.")

            st.divider()
            q4a = st.radio("**Decision 4A:** Should the school be closed?", [
                "— Select —",
                "Yes, immediately close for 2 weeks",
                "No — targeted exclusion of unvaccinated students is more proportionate and maintains education",
                "Only close if cases exceed 25",
            ], key="ob2_q4a")

            if q4a == "No — targeted exclusion of unvaccinated students is more proportionate and maintains education":
                st.success("""
✅ **Correct.** Excluding only unvaccinated students (who are at risk and can transmit) allows vaccinated students to continue education without interruption. Full school closure is a higher-level intervention reserved for when targeted exclusion fails or when a large proportion of students are susceptible. Proportionality is a core principle of public health intervention.
                """)
            elif q4a == "Yes, immediately close for 2 weeks":
                st.warning("""
⚠️ **Premature.** School closure is a high-impact intervention that disrupts education for vaccinated students who are not at risk. Start with targeted exclusion of unvaccinated contacts. Full closure may become necessary if the outbreak grows and targeted exclusion proves insufficient.
                """)
            elif q4a != "— Select —":
                st.error("❌ Waiting for a specific case count threshold before acting allows exponential growth to occur. Act early with targeted measures.")


            next_step_button(ob2_step, OB2_STEPS, "ob2_idx")

        elif ob2_step == "Step 5 — Could this have been prevented?":
            st.subheader("Step 5 — Prevention and policy implications")
            st.markdown("""
The outbreak is now controlled after an emergency vaccination clinic raised coverage to 95%. Final case count: 23 cases, 4 hospitalizations, 0 deaths. 228 unvaccinated students excluded for varying periods.
            """)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Final outbreak summary")
                st.markdown("""
- **Total cases:** 23
- **Hospitalizations:** 4 (pneumonia x2, encephalitis x1, otitis media x1)
- **Deaths:** 0
- **School days missed by unvaccinated students:** 2,850 total (228 students × avg 12.5 days)
- **Cost of emergency response:** Estimated $280,000 (staff, vaccines, investigation)
                """)
            with col2:
                st.markdown("#### What would have prevented this")
                st.markdown("""
- **Vaccination coverage at 93%+:** Would have kept Rₑ < 1; one infectious case would not have sparked an outbreak
- **School vaccination requirement enforcement:** 28% unvaccinated is far above the level compatible with herd immunity
- **Pre-travel health consultation:** Index family could have been counseled on measles risk in destination country
- **Clinician recognition:** Earlier diagnosis (Koplik spots, prodrome) would have shortened exposure window
                """)

            st.divider()
            q5a = st.radio("**Decision 5A:** What policy change would most prevent future outbreaks?", [
                "— Select —",
                "Require vaccination of all staff but allow student exemptions to continue",
                "Enforce existing vaccination requirements and restrict non-medical exemptions",
                "Provide education campaigns — choice is sufficient",
                "Conduct annual screenings but take no policy action",
            ], key="ob2_q5a")

            if q5a == "Enforce existing vaccination requirements and restrict non-medical exemptions":
                st.success("""
✅ **Correct.** Mathematical modeling and real-world data both show that states with easier non-medical (philosophical/religious) exemptions have higher rates of vaccine-preventable disease outbreaks. Enforcement of existing requirements and restriction of non-medical exemptions is the most evidence-based policy intervention to maintain herd immunity.
                """)
            elif q5a != "— Select —":
                st.error("❌ Voluntary measures have consistently proven insufficient to maintain measles herd immunity (93%+). Policy enforcement is the most effective tool.")

            with st.expander("🦠 What is measles?"):
                st.markdown("""
**Measles** is one of the most contagious pathogens ever described. Key features:
- **R₀:** 12–18 in unvaccinated populations (one of the highest known)
- **Herd immunity threshold:** ~93–95%
- **Transmission:** Airborne — survives in air for up to 2 hours after patient leaves
- **Incubation:** 8–12 days to symptom onset; 14–18 days to rash
- **Infectious period:** 4 days before to 4 days after rash onset
- **Prodrome:** Fever, cough, coryza, conjunctivitis (the "3 Cs") — Koplik spots pathognomonic
- **Complications:** Pneumonia (leading cause of measles death), encephalitis, SSPE (rare, fatal, years later)
- **Vaccine:** MMR — 97% effective after 2 doses
- **Elimination:** United States achieved measles elimination in 2000; maintained with high vaccination coverage
- **Re-emergence:** Outbreaks occur in clusters of unvaccinated individuals; imported cases seed outbreaks in communities below HIT
                """)


            next_step_button(ob2_step, OB2_STEPS, "ob2_idx")

    # ════════════════════════════════════════════════════════════════
    # SCENARIO 3: SALMONELLA
    # ════════════════════════════════════════════════════════════════
    elif ob_scenario == "🥘 Scenario 3: Salmonellosis at a Community Church Potluck":

        col_brief, col_stats = st.columns([2,1])
        with col_brief:
            st.markdown("""
### 🎯 Your Mission
It's Sunday evening. The county health department receives 4 calls from individuals reporting severe diarrhea, fever, and abdominal cramps after attending a church potluck dinner earlier that day. By Monday morning, 23 people have called with similar symptoms. All attended the same event. The pastor reports approximately 120 people were present. Your job: identify the vehicle, establish the case definition, calculate attack rates, and implement control.
            """)
        with col_stats:
            st.markdown("""
<div style="background:#f0fdf4;border-radius:8px;padding:14px;font-size:13px;">
<b>📋 Outbreak Brief</b><br><br>
🤒 <b>Cases reported:</b> 23 (and growing)<br>
🏥 <b>Hospitalizations:</b> 2<br>
💀 <b>Deaths:</b> 0<br>
📍 <b>Location:</b> Community church<br>
👥 <b>Event attendees:</b> ~120<br>
🕐 <b>Meal time:</b> Sunday 12:30 PM
</div>
            """, unsafe_allow_html=True)

        ob3_step = st.radio("Jump to step:", [
            "Step 1 — Build the case definition & line list",
            "Step 2 — Epidemic curve & incubation period estimation",
            "Step 3 — Food-specific attack rates (calculate)",
            "Step 4 — Environmental investigation",
            "Step 5 — Control, report & prevent recurrence",
        ], index=st.session_state.get("ob3_idx", 0), horizontal=False)
        st.divider()

        if ob3_step == "Step 1 — Build the case definition & line list":
            st.subheader("Step 1 — Case definition and line list construction")
            st.markdown("""
You need to systematically characterize who is sick before you can analyze the data. The **line list** is the epidemiologist's most important tool — one row per case, one column per variable.
            """)
            st.info("💡 **Step 4 of 10:** Construct a working case definition")

            st.markdown("#### ✏️ Interactive case definition builder")
            col1, col2 = st.columns(2)
            with col1:
                cd_person = st.selectbox("Person:", [
                    "Any person in the county",
                    "Any person who attended the First Baptist Church potluck",
                    "Any church member",
                ], key="ob3_cd_person")
                cd_time = st.selectbox("Time:", [
                    "Any time in November",
                    "Symptom onset between Sunday noon and Tuesday midnight",
                    "Only Sunday attendees who got sick same day",
                ], key="ob3_cd_time")
            with col2:
                cd_clinical = st.selectbox("Clinical criteria:", [
                    "Any GI symptom",
                    "Diarrhea (≥3 loose stools/24h) AND/OR fever (≥38°C) within 72h of meal",
                    "Lab-confirmed Salmonella only",
                ], key="ob3_cd_clinical")
                cd_lab = st.selectbox("Lab classification:", [
                    "Confirmed (Salmonella isolated from stool)",
                    "Probable (clinical criteria met, no lab)",
                    "Use both confirmed AND probable",
                ], key="ob3_cd_lab")

            if cd_person and cd_time and cd_clinical and cd_lab:
                case_def = f"{cd_person}, with {cd_clinical.lower()}, {cd_time.lower()}"
                st.info(f"**Your case definition:** {case_def}")

                if "potluck" in cd_person and "72h" in cd_clinical and "Tuesday" in cd_time:
                    st.success("✅ Strong case definition — anchored to the exposure event, time-limited, uses appropriate clinical threshold.")
                elif "lab-confirmed" in cd_clinical:
                    st.error("❌ Lab-only case definitions miss the majority of cases and delay investigation. Use clinical criteria with lab as confirmation.")

            st.divider()
            st.markdown("#### 📋 Sample line list (first 10 cases)")
            line_list = pd.DataFrame({
                "Case #": range(1, 11),
                "Age": [34, 67, 8, 45, 52, 23, 71, 39, 14, 58],
                "Sex": ["F","M","M","F","F","M","F","M","F","M"],
                "Onset time": ["Sun 8pm","Sun 6pm","Sun 9pm","Mon 2am","Mon 1am","Sun 7pm","Mon 4am","Sun 11pm","Mon 3am","Mon 6am"],
                "Diarrhea": ["✅","✅","✅","✅","✅","✅","✅","✅","✅","✅"],
                "Fever": ["✅","✅","❌","✅","✅","❌","✅","✅","❌","✅"],
                "Vomiting": ["✅","❌","✅","❌","✅","✅","❌","✅","✅","❌"],
                "Chicken salad": ["✅","✅","❌","✅","✅","✅","✅","✅","❌","✅"],
                "Deviled eggs": ["✅","✅","✅","✅","❌","✅","✅","❌","✅","✅"],
                "Potato salad": ["✅","❌","✅","✅","✅","❌","✅","✅","✅","✅"],
            })
            st.dataframe(line_list, use_container_width=True, hide_index=True)

            q1a = st.radio("**Decision 1A:** What does the line list immediately suggest about the most likely vehicle?", [
                "— Select —",
                "Potato salad — appears frequently",
                "Chicken salad or deviled eggs — egg/poultry = Salmonella, and most cases ate one or both",
                "Vomiting pattern suggests norovirus, not Salmonella",
                "Cannot tell from this limited data",
            ], key="ob3_q1a")

            if q1a == "Chicken salad or deviled eggs — egg/poultry = Salmonella, and most cases ate one or both":
                st.success("""
✅ **Correct.** Salmonella is most commonly associated with poultry, eggs, and egg-containing dishes (chicken salad, deviled eggs, mayonnaise-based salads). The line list shows most cases ate chicken salad and/or deviled eggs. This generates the primary hypothesis to test with attack rates. Note also that fever + diarrhea (non-bloody at this stage) is consistent with non-typhoidal Salmonella.
                """)
            elif q1a != "— Select —":
                st.error("❌ Biological plausibility matters: Salmonella's primary vehicles are poultry, eggs, and egg-containing dishes. The line list shows these items prominently in cases.")


            next_step_button(ob3_step, OB3_STEPS, "ob3_idx")

        elif ob3_step == "Step 2 — Epidemic curve & incubation period estimation":
            st.subheader("Step 2 — Epidemic curve and incubation period")
            st.markdown("The meal was served at **12:30 PM Sunday**. Below are the onset times for all 23 confirmed cases.")
            st.info("💡 **Step 6 of 10:** Describe in terms of time — epidemic curve")

            onset_hours = [6, 7, 8, 8, 9, 9, 10, 10, 11, 12, 12, 13, 13, 14, 15, 16, 17, 18, 20, 22, 25, 28, 30]
            onset_labels = [f"+{h}h" for h in onset_hours]

            import collections
            onset_counts = collections.Counter(onset_hours)
            curve_df = pd.DataFrame([{"Hours after meal": h, "Cases": onset_counts.get(h, 0)} for h in range(0, 32)])
            curve_df = curve_df[curve_df["Cases"] > 0]
            st.bar_chart(curve_df.set_index("Hours after meal"))
            st.caption("X-axis: hours after meal (12:30 PM Sunday). Each bar = cases with that onset hour.")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
**Summary statistics:**
- First case: +6 hours after meal
- Last case: +30 hours after meal
- **Peak:** +10–14 hours
- **Median incubation:** ~12 hours
- **Range:** 6–30 hours
                """)
            with col2:
                st.markdown("""
**Salmonella incubation reference:**
- Typical: 6–72 hours
- Most common: 12–36 hours
- Median: ~18 hours
- Range varies by inoculum dose
                """)

            q2a = st.radio("**Decision 2A:** What does this epidemic curve confirm?", [
                "— Select —",
                "Propagated outbreak — cases still occurring 30 hours later indicates person-to-person spread",
                "Point-source outbreak — all cases within one incubation period of a single exposure",
                "Endemic pattern — stable ongoing transmission",
                "Mixed — early point source followed by secondary cases",
            ], key="ob3_q2a")

            if q2a == "Point-source outbreak — all cases within one incubation period of a single exposure":
                st.success("""
✅ **Correct.** All 23 cases occurred within 6–30 hours of a single exposure event (the potluck meal). The shape — single peak, rapid rise and fall — is a classic point-source curve. The "tail" of cases at +25–30 hours represents natural variation in incubation time, not a new wave.

The incubation range (6–30h) is consistent with Salmonella, narrowing the list of potential agents before lab confirmation.
                """)
            elif q2a != "— Select —":
                st.error("❌ All cases cluster within 30 hours of a single meal — this is a point-source pattern. For secondary spread (propagated), you would expect a gap of approximately one incubation period before a new wave appears.")

            if q2a != "— Select —":
                st.divider()
                st.markdown("#### 🧮 Use the curve to estimate the exposure time")
                st.markdown("""
**Forensic timing:** The median incubation for Salmonella is ~12–18 hours. Working backward from the peak onset (10–14 hours post-meal), the median case had an incubation of ~12 hours.

If you didn't know when the meal occurred, you could estimate it: find the median onset time, subtract the expected median incubation period for the suspected agent.

**Median onset:** approximately +12 hours after 12:30 PM = ~12:30 AM Monday
**Subtract median incubation (12h):** ~12:30 PM Sunday → consistent with the known meal time ✅

This technique is used in investigations where the exposure time is unknown.
                """)


            next_step_button(ob3_step, OB3_STEPS, "ob3_idx")

        elif ob3_step == "Step 3 — Food-specific attack rates (calculate)":
            st.subheader("Step 3 — Calculate food-specific attack rates")
            st.markdown("""
You have completed telephone interviews with 98 of the 120 attendees (82% response rate). 23 meet the case definition.

For each food item, the table below shows how many attendees ate it, and of those, how many became sick.
            """)
            st.info("💡 **Steps 7–8 of 10:** Develop hypotheses → Test hypotheses analytically")

            food_items_3 = {
                "Chicken salad": (18, 4, 5, 71),
                "Deviled eggs": (17, 5, 6, 70),
                "Potato salad (mayo-based)": (15, 15, 8, 60),
                "Green bean casserole": (8, 52, 15, 23),
                "Macaroni and cheese": (6, 54, 17, 21),
                "Lemonade": (12, 48, 11, 27),
                "Chocolate cake": (10, 50, 13, 25),
            }

            st.markdown("#### 🧮 Complete the table — calculate AR and RR for each food item")
            st.markdown("*AR = (Sick among those who ate) ÷ (Total who ate) × 100. RR = AR exposed ÷ AR unexposed.*")

            results3 = []
            for food, (ate_sick, ate_well, notate_sick, notate_well) in food_items_3.items():
                ate_total = ate_sick + ate_well
                notate_total = notate_sick + notate_well
                ar_exp = round(ate_sick / ate_total * 100, 1)
                ar_unexp = round(notate_sick / notate_total * 100, 1)
                rr = round(ar_exp / ar_unexp, 2) if ar_unexp > 0 else float("inf")
                results3.append({
                    "Food": food,
                    "Ate: sick/total": f"{ate_sick}/{ate_total}",
                    "AR exposed %": ar_exp,
                    "Didn't eat: sick/total": f"{notate_sick}/{notate_total}",
                    "AR unexposed %": ar_unexp,
                    "RR": rr
                })

            results3_df = pd.DataFrame(results3)
            st.dataframe(results3_df, use_container_width=True, hide_index=True)

            q3a = st.radio("**Decision 3A:** Which food item is the most likely vehicle?", [
                "— Select —",
                "Potato salad — high number ate it",
                "Chicken salad — highest RR with very low attack rate in unexposed",
                "Deviled eggs — high RR, egg-Salmonella association",
                "Both chicken salad AND deviled eggs — same cook, cross-contamination likely",
            ], key="ob3_q3a")

            if q3a == "Both chicken salad AND deviled eggs — same cook, cross-contamination likely":
                st.success("""
✅ **Correct — excellent epidemiologic reasoning.** Both chicken salad and deviled eggs show high RR and very low AR unexposed. In real investigations, when two items both show strong associations, look for a common source: the same cook, the same contaminated ingredient (raw chicken), the same utensils, or the same refrigerator. Here, the church member who brought both dishes used the same cutting board for raw chicken and egg preparation — a classic cross-contamination scenario.
                """)
            elif q3a == "Chicken salad — highest RR with very low attack rate in unexposed":
                st.warning("""
⚠️ **Partially correct.** Chicken salad does have the highest RR. But when two items show similar strong signals (chicken salad AND deviled eggs), consider a common source — same cook, same ingredient, cross-contamination. The best answer acknowledges both.
                """)
            elif q3a != "— Select —":
                st.error("❌ Focus on the items with the highest RR AND the lowest AR unexposed. Potato salad actually has similar attack rates in those who ate vs. didn't eat (suggesting no association with illness).")

            if q3a != "— Select —":
                st.divider()
                st.markdown("#### 🧮 Practice: Calculate the RR for chicken salad manually")
                st.markdown("Ate chicken salad: 18 sick, 4 well. Did not eat: 5 sick, 71 well.")

                ar_exp_input = st.number_input("AR exposed (ate chicken salad) %:", 0.0, 100.0, 0.0, 0.1, key="ob3_ar_exp")
                ar_unexp_input = st.number_input("AR unexposed (did not eat) %:", 0.0, 100.0, 0.0, 0.1, key="ob3_ar_unexp")
                rr_input = st.number_input("RR:", 0.0, 50.0, 0.0, 0.01, key="ob3_rr")

                if st.button("Check calculation", key="ob3_check_rr"):
                    correct_ar_exp = round(18/22*100, 1)
                    correct_ar_unexp = round(5/76*100, 1)
                    correct_rr = round(correct_ar_exp/correct_ar_unexp, 2)
                    st.markdown(f"""
**Correct values:**
- AR exposed = 18/22 = **{correct_ar_exp}%**
- AR unexposed = 5/76 = **{correct_ar_unexp}%**
- RR = {correct_ar_exp}/{correct_ar_unexp} = **{correct_rr}**

An RR of {correct_rr} means students who ate the chicken salad were {correct_rr}× more likely to become ill than those who did not. This is strong evidence for chicken salad as a vehicle.
                    """)
                    if abs(ar_exp_input - correct_ar_exp) < 1 and abs(rr_input - correct_rr) < 0.2:
                        st.success("✅ Your calculation is correct!")
                    else:
                        st.info("Check your arithmetic — divide sick ÷ total (not sick + well) to get the attack rate.")


            next_step_button(ob3_step, OB3_STEPS, "ob3_idx")

        elif ob3_step == "Step 4 — Environmental investigation":
            st.subheader("Step 4 — Environmental investigation and source tracing")
            st.markdown("""
The analytic study has identified chicken salad and deviled eggs as vehicles. Both were prepared by the same congregation member (Mrs. Johnson). Now you need to trace the contamination to its source.
            """)
            st.info("💡 **Step 8 continued:** Environmental sampling + source tracing")

            st.markdown("""
**Environmental investigation findings:**
- Mrs. Johnson prepared both dishes Saturday evening at home
- She purchased whole chickens from a local grocery store Saturday morning
- She used a wooden cutting board that had been used for raw chicken
- The same cutting board was used to chop celery and onions for the chicken salad
- Deviled eggs were prepared in the same kitchen, same surfaces
- Dishes were refrigerated Saturday night, transported to church in a cooler Sunday
- Temperature at time of service: chicken salad = 58°F (should be ≤41°F)
- Mrs. Johnson reports no illness herself

**Samples collected:**
- Leftover chicken salad: submitted to state lab
- Mrs. Johnson's cutting board: swab submitted
- Remaining whole chicken from grocery store (same purchase): submitted
- Stool samples from 8 cases
            """)

            q4a = st.radio("**Decision 4A:** The chicken salad temperature was 58°F at service. Why does this matter?", [
                "— Select —",
                "It doesn't matter — Salmonella only comes from contaminated animals, not temperature",
                "Temperatures between 41°F and 135°F allow Salmonella to multiply rapidly — the 'danger zone'",
                "58°F is only slightly above the 55°F threshold — minimal risk",
                "Temperature only matters for viruses, not bacteria",
            ], key="ob3_q4a")

            if q4a == "Temperatures between 41°F and 135°F allow Salmonella to multiply rapidly — the 'danger zone'":
                st.success("""
✅ **Correct.** The USDA "temperature danger zone" for bacterial growth is 41°F–135°F (5°C–57°C). At 58°F, Salmonella can double every 20–30 minutes. Even a small initial contamination can reach an infectious dose (10³–10⁶ organisms) within hours at this temperature. The combination of contamination (cross-contamination from raw chicken) AND temperature abuse (inadequate refrigeration/transport) created ideal conditions for a large outbreak.
                """)
            elif q4a != "— Select —":
                st.error("❌ Temperature is critical for bacterial foodborne illness. Unlike viruses (which don't replicate in food), bacteria like Salmonella multiply exponentially at temperatures between 41°F and 135°F.")

            if q4a != "— Select —":
                st.divider()
                q4b = st.radio("**Decision 4B:** Lab results show Salmonella Enteritidis in the leftover chicken salad and cutting board. The grocery store chicken is also positive. What do you do?", [
                    "— Select —",
                    "Issue press release blaming Mrs. Johnson",
                    "Contact the state health department and FDA/USDA to investigate the grocery store chicken supplier",
                    "Close the church for 2 weeks",
                    "No further action — the event is over",
                ], key="ob3_q4b")

                if q4b == "Contact the state health department and FDA/USDA to investigate the grocery store chicken supplier":
                    st.success("""
✅ **Correct.** When a contaminated commercial food product is implicated, investigation extends up the supply chain. This outbreak may be one of many — PulseNet (CDC's molecular surveillance network) may identify the same Salmonella strain in cases from other states linked to the same supplier. A voluntary recall or regulatory action may be needed to prevent further illness nationally.

This is how local foodborne investigations become national — the church potluck is the sentinel event that alerts the system to a broader contamination.
                    """)
                elif q4b != "— Select —":
                    st.error("❌ When a commercially distributed product is the source, the investigation extends beyond the local outbreak. Other communities may be at risk from the same supplier.")


            next_step_button(ob3_step, OB3_STEPS, "ob3_idx")

        elif ob3_step == "Step 5 — Control, report & prevent recurrence":
            st.subheader("Step 5 — Control, reporting, and prevention")
            st.info("💡 **Steps 9–10 of 10:** Implement control measures → Communicate findings")

            st.markdown("""
**Final outbreak profile:**
- 23 cases, 2 hospitalizations, 0 deaths
- Salmonella Enteritidis serotype confirmed in 7/8 stool samples
- Same strain found in chicken salad and cutting board
- PulseNet match: identical molecular fingerprint to 12 cases in 2 other counties from same grocery chain
- Grocery chain initiated voluntary recall of whole chickens from that distributor

**Lessons applied:**
            """)

            lessons = {
                "Cross-contamination prevention": "Use separate cutting boards for raw meat and ready-to-eat foods",
                "Temperature control": "Keep cold foods at ≤41°F during preparation, storage, and transport",
                "Potluck food safety": "Bring hot foods hot (≥135°F) and cold foods cold (≤41°F)",
                "Hand hygiene": "Wash hands thoroughly after handling raw poultry",
                "Food handler illness": "Exclude food handlers who are ill (though Mrs. Johnson was not ill herself)",
                "Supply chain surveillance": "PulseNet enables local outbreaks to trigger national investigations",
            }

            for lesson, detail in lessons.items():
                with st.expander(f"✅ {lesson}"):
                    st.markdown(detail)

            st.divider()
            st.markdown("#### 📝 Write your outbreak report")
            st.markdown("A complete outbreak investigation report includes:")

            report_sections = [
                ("Background", "When and where the outbreak was identified; who was affected"),
                ("Methods", "Case definition used; how cases were found; how data were collected"),
                ("Results", "Epidemic curve; case count; attack rates; most likely vehicle"),
                ("Conclusions", "Probable source; contributing factors; mechanism of contamination"),
                ("Recommendations", "Immediate control measures; long-term prevention"),
            ]

            for section, description in report_sections:
                st.markdown(f"**{section}:** {description}")

            with st.expander("🦠 What is Salmonella?"):
                st.markdown("""
**Salmonella** is a gram-negative bacteria and one of the most common causes of foodborne illness worldwide.

- **Species:** Salmonella enterica (>2,500 serotypes); most common in US = S. Typhimurium and S. Enteritidis
- **Sources:** Poultry, eggs, beef, pork, reptiles, contaminated produce
- **Transmission:** Fecal-oral; ingestion of contaminated food or water; contact with infected animals
- **Incubation:** 6–72 hours (typically 12–36h)
- **Symptoms:** Diarrhea (may be bloody), fever, abdominal cramps, vomiting
- **Duration:** 4–7 days (self-limited in healthy adults)
- **At-risk populations:** Infants, elderly, immunocompromised — may develop bacteremia, meningitis
- **Infectious dose:** As low as 10³ organisms (lower in high-fat vehicles like chocolate, peanut butter)
- **Treatment:** Usually supportive; antibiotics for severe cases or bacteremia (resistance emerging)
- **Prevention:** Cook poultry to 165°F; avoid cross-contamination; refrigerate properly; hand hygiene
- **Surveillance:** Nationally notifiable; PulseNet provides molecular fingerprinting for outbreak detection
                """)


            next_step_button(ob3_step, OB3_STEPS, "ob3_idx")

    elif ob_scenario == "— Choose an outbreak —":
        st.info("Select a scenario above to begin your investigation.")
        st.markdown("""
#### 🎯 What you'll practice in Outbreak Lab
Each scenario walks you through a real-style outbreak investigation, applying the **10-step framework** with:
- **Decision points** — choose the right investigative action and get immediate feedback
- **Interactive calculations** — calculate attack rates, RR, and herd immunity thresholds yourself
- **Epidemic curves** — read and interpret real outbreak patterns
- **Case definitions** — build your own and understand the tradeoffs
- **Control measures** — choose and justify interventions

**The three scenarios cover:**
| Scenario | Agent | Key skills |
|---|---|---|
| 🍽️ University Dining Hall | Norovirus | Attack rates, vehicle identification, secondary spread |
| 📚 Elementary School | Measles | Herd immunity math, contact tracing, vaccination policy |
| 🥘 Church Potluck | Salmonella | Case definition, incubation estimation, supply chain tracing |
        """)


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

**Ecological Study**
Unit of analysis is **groups or populations** (countries, cities, time periods), not individuals. Exposure and outcome are measured as group averages or rates. Useful for hypothesis generation and policy surveillance. Cannot establish individual-level causation.

**Ecological Fallacy (Aggregation Bias)**
The error of inferring individual-level relationships from group-level data. A correlation observed between country-level variables does not mean the same relationship holds within individuals. Classic example: countries with more TVs have lower infant mortality — TVs don't protect infants; wealth causes both.

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
Hospital patients are not representative of the general population — both the exposure and disease independently increase hospitalization probability. A type of collider bias — conditioning on hospitalization (a collider) opens a spurious path between exposure and disease.

**Healthy Worker Effect**
Employed workers are systematically healthier than the general population. Causes SMR < 1 in occupational studies even without protective effects.

**Healthy User Bias**
People who use health-promoting interventions (vitamins, screening, medications) are systematically different from those who don't — they tend to be wealthier, more health-conscious, and healthier overall. Confounds observational studies of preventive interventions. Classic example: hormone therapy and cardiovascular disease in the WHI era.

**Loss to Follow-Up Bias**
Dropout related to both exposure and outcome distorts results. **Differential loss** (more common in one exposure group) biases toward or away from null depending on direction. If the sickest exposed participants leave, the exposed group looks healthier than it is → RR biased toward null.

**Information Bias (Misclassification)**
Exposure or outcome is measured incorrectly.

**Non-Differential Misclassification**
Measurement error equal across all groups (same error rate regardless of outcome or exposure status). Always biases the measure of association **toward null** (attenuates the association). A null finding in the presence of non-differential misclassification may not mean no effect.

**Differential Misclassification**
Measurement error differs between groups (error rate depends on outcome or exposure status). Can bias in either direction — toward or away from null.

**Recall Bias**
Cases remember past exposures more carefully than controls after receiving a diagnosis. Differential misclassification. Common in case-control studies. Typically biases OR **away from null** (overestimates association).

**Reverse Causation**
The outcome actually causes the exposure, not the other way around. Common in cross-sectional studies where temporality cannot be established. Example: lower physical activity associated with depression — but depression may cause inactivity, not the reverse.

**Confounding by Indication**
The reason for receiving a treatment (the indication) is itself associated with the outcome, creating spurious associations. Classic example: sicker patients receive more aggressive treatment → treatment appears harmful in crude analyses. Controlled by adjusting for disease severity.
        """)

    with st.expander("🔀 Confounding & Effect Modification"):
        st.markdown("""
**Confounding**
A variable that distorts the apparent association between exposure and outcome. Must be: (1) associated with exposure in the source population, (2) independently associated with outcome, (3) not on the causal pathway between exposure and outcome.

**Confounder Control — Design:** Randomization, restriction, matching.

**Confounder Control — Analysis:** Stratification (Mantel-Haenszel), multivariable regression, propensity scores.

**Mantel-Haenszel Method**
Stratified analysis technique that produces a pooled (weighted average) estimate of RR or OR across strata of a confounder. Compares the crude pooled estimate to stratum-specific estimates to assess confounding. If the Mantel-Haenszel adjusted estimate differs meaningfully from the crude estimate (>10%), the stratification variable is a confounder.

**Residual Confounding**
Confounding that remains after adjustment, due to imperfect measurement of confounders or unmeasured confounders. Present in virtually all observational studies to some degree.

**Effect Modification (Interaction)**
The magnitude or direction of the association between exposure and outcome differs across levels of a third variable (the effect modifier). **A real biological or social phenomenon to be reported, not a bias to be removed.** The appropriate response is to present stratum-specific estimates, not a single adjusted estimate. Confounding is removed by adjustment; effect modification is revealed by stratification.

**10% Rule**
If adjusting for a variable changes RR/OR by >10%, it is a meaningful confounder worth controlling.

**DAG (Directed Acyclic Graph)**
A visual tool for representing causal assumptions. Nodes = variables, arrows = causal direction. Used to identify which variables to adjust for and which to leave alone.

**Confounder (DAG)**
A common cause of both exposure and outcome. Creates a backdoor path. Should be adjusted for.

**Mediator (DAG)**
A variable on the causal pathway between exposure and outcome (exposure causes mediator causes outcome). Adjusting for a mediator blocks the causal pathway — over-adjustment bias. Do NOT adjust for mediators when estimating total effect.

**Collider (DAG)**
A variable caused by both exposure and outcome (arrows collide into it). Colliders naturally block paths. Conditioning on a collider opens a spurious association between exposure and outcome — collider bias. Do NOT adjust for colliders.

**Moderator (DAG)**
A variable that modifies the strength of the exposure-outcome relationship. Synonymous with effect modifier.

**M-Bias**
Bias introduced by adjusting for a variable that is a collider between two unmeasured common causes of the exposure and outcome. Adjusting for such a variable opens a backdoor path that didn't previously exist.

**Proxy Variable**
A measured variable used as a substitute for an unmeasured variable of interest. Adjusting for a proxy provides only partial control for the underlying variable — residual confounding remains. Example: education as proxy for socioeconomic status.
        """)

    with st.expander("🔗 Causal Inference"):
        st.markdown("""
**Bradford Hill Criteria (1965)**
Nine criteria for evaluating whether an observed association is likely causal:
1. **Strength** — stronger associations less likely due to unmeasured confounding
2. **Consistency** — replicated across studies, populations, methods
3. **Specificity** — exposure associated with specific disease, not many outcomes
4. **Temporality** — exposure precedes outcome (the ONLY mandatory criterion)
5. **Biological gradient** — dose-response relationship
6. **Plausibility** — biologically plausible mechanism
7. **Coherence** — consistent with known biology and natural history
8. **Experiment** — removal of exposure reduces disease (natural experiment)
9. **Analogy** — similar relationships established for analogous exposures

**Temporality is the only mandatory criterion.** The others are supportive.

**Counterfactual Framework**
Causation requires asking: what would have happened to the same person if their exposure status had been different? The fundamental problem of causal inference is that we can never observe both potential outcomes for the same person at the same time.

**Reverse Causation**
See Bias section. The outcome causes the exposure — a particular threat in cross-sectional studies.
        """)

    with st.expander("📊 Disease Frequency Measures"):
        st.markdown("""
**Prevalence**
Proportion of population with condition at a point in time. Numerator: existing cases. No time unit. Denominator includes people who already have disease.

**Cumulative Incidence (Attack Rate)**
Proportion of disease-free population that develops disease during a specified period. Numerator: new cases only. Denominator: disease-free at start of period.

**Attack Rate**
Cumulative incidence in an outbreak context. Same formula, shorter time frame. Food-specific attack rates are used to identify the vehicle in foodborne outbreaks.

**Secondary Attack Rate (SAR)**
Proportion of susceptible contacts of an index case who develop disease within one incubation period. SAR = Secondary cases ÷ Susceptible contacts. The index case is excluded from both numerator and denominator. Measures household or close-contact transmissibility.

**Incidence Rate (Incidence Density)**
New cases per unit person-time at risk. Used when follow-up varies. Units: per 1,000 person-years. More precise than cumulative incidence when participants contribute different observation times.

**Case Fatality Rate (CFR)**
Deaths from disease ÷ total cases × 100. Measures disease lethality. Denominator is cases only (people with disease), not total population. A proportion, not a true rate.

**Mortality Rate**
Deaths ÷ total population at risk. Denominator is the whole population, not just cases. Measures death burden in a population.

**Prevalence-Incidence Relationship (P = I × D)**
At steady state: Prevalence ≈ Incidence × Average Duration. Rearranges to estimate any one value if the other two are known. Higher incidence or longer duration both increase prevalence. In the ART era, HIV prevalence rises despite falling incidence because duration has increased dramatically.

**Point-Source Epidemic**
All cases exposed to same source at same time. Sharp epidemic curve; width ≈ one incubation period. No secondary spread.

**Propagated Epidemic**
Person-to-person spread. Multiple waves in epidemic curve; each wave ≈ one incubation period apart. Cases spread across multiple settings over weeks.

**Mixed Epidemic**
Begins as a point source, followed by person-to-person transmission. Initial sharp peak followed by subsequent waves.

**Incubation Period**
Time from exposure to symptom onset. The range of onset times in a point-source outbreak approximates the plausible incubation period range for that pathogen.
        """)

    with st.expander("🔬 Screening & Diagnostic Tests"):
        st.markdown("""
**Sensitivity**
True positive rate: proportion of true cases that test positive. High sensitivity → few false negatives → rules OUT disease when negative. **SnNout.** Fixed property of the test — does not change with prevalence.

**Specificity**
True negative rate: proportion of true non-cases that test negative. High specificity → few false positives → rules IN disease when positive. **SpPin.** Fixed property of the test — does not change with prevalence.

**PPV (Positive Predictive Value)**
Probability that a positive test reflects true disease. Depends on prevalence — low in low-prevalence populations even with an excellent test.

**NPV (Negative Predictive Value)**
Probability that a negative test reflects true absence of disease. Increases as prevalence decreases — negative tests are more reassuring when disease is rare. In high-prevalence populations, a negative test cannot as confidently rule out disease.

**Accuracy**
(TP + TN) ÷ N. Proportion of all tests correct. Misleading in low-prevalence settings — predicting everyone negative gives high accuracy but is clinically useless.

**Sensitivity-Specificity Tradeoff**
Lowering the test cutpoint increases sensitivity but decreases specificity (more positives, more false positives). ROC curve plots this tradeoff across all possible cutpoints.

**Prevalence Effect on PPV**
Same sensitivity and specificity, different prevalence → dramatically different PPV. Even a 99% specific test has poor PPV at 0.1% prevalence because the enormous pool of non-cases generates many false positives in absolute numbers.

**LR+ (Positive Likelihood Ratio)**
LR+ = Sensitivity ÷ (1 − Specificity). How much more likely is a positive result in someone WITH disease vs. WITHOUT. LR+ > 10 = strong evidence for disease. Unlike PPV, LR+ is independent of prevalence and can be applied to individual patients with known pre-test probability.

**LR− (Negative Likelihood Ratio)**
LR− = (1 − Sensitivity) ÷ Specificity. How much more likely is a negative result in someone WITH disease vs. WITHOUT. LR− < 0.1 = strong evidence against disease.

**Pre-test Probability**
Probability of disease before the test result is known. Based on clinical history, demographics, and epidemiology. The starting point for Bayesian test interpretation.

**Post-test Probability**
Probability of disease after incorporating the test result. Calculated as:
Pre-test odds × LR = Post-test odds → convert back to probability.
Post-test probability = Post-test odds ÷ (1 + Post-test odds).

**Bayes' Theorem (clinical)**
The formal framework for updating probability with new evidence. In clinical testing: Post-test odds = Pre-test odds × Likelihood Ratio.
        """)

    with st.expander("📐 Measures of Association"):
        st.markdown("""
**Risk Ratio (RR)**
Risk in exposed ÷ risk in unexposed. Cohort studies. RR = 1: no difference; RR > 1: higher risk; RR < 1: protective.

**Prevalence Ratio (PR)**
Same formula as RR but used in cross-sectional studies where the outcome is prevalent (existing), not incident (new).

**Odds Ratio (OR)**
Odds of outcome in exposed ÷ odds in unexposed. Used in case-control studies and logistic regression. OR is always farther from 1 than RR for the same data when the outcome is common. When the outcome is uncommon (as a rough heuristic, often <10%), OR approximates RR reasonably well — but there is no hard cutoff, and the approximation degrades gradually as prevalence increases.

**Incidence Rate Ratio (IRR)**
Rate in exposed ÷ rate in unexposed using person-time denominators. Used when follow-up time varies across participants.

**Hazard Ratio (HR)**
Ratio of instantaneous event rates at any moment in time. Output of Cox proportional hazards regression. Used when follow-up varies and participants may be censored. Approximates RR under proportional hazards assumption.

**Risk Difference (Attributable Risk)**
Risk in exposed − risk in unexposed. Absolute excess risk. More clinically meaningful than RR for public health decisions.

**Confidence Interval (CI)**
Range of plausible values for the true effect estimate. 95% CI: if the study were repeated many times, 95% of CIs would contain the true value. CI excluding null value (1 for ratios, 0 for differences) → p < 0.05.
        """)

    with st.expander("📉 Advanced Epi Measures"):
        st.markdown("""
**Attributable Risk (AR) / Risk Difference**
Risk in exposed − risk in unexposed. Absolute excess risk per 100 exposed. Answers: how many extra cases occur because of exposure?

**Attributable Risk Percent (AR%)**
AR ÷ risk in exposed × 100. Fraction of disease in the exposed group attributable to exposure. Answers: of all disease in the exposed group, what proportion is due to exposure?

**Population Attributable Risk Percent (PAR%)**
Fraction of all disease in the **total population** attributable to exposure. Formula: Pe(RR−1) / [1+Pe(RR−1)] × 100. Accounts for both exposure prevalence (Pe) and strength of association (RR). Used for population-level prevention decisions. A common exposure with modest RR can have higher PAR% than a rare exposure with large RR.

**AR% vs PAR%:** AR% applies to the exposed group; PAR% applies to the whole population. PAR% will always be ≤ AR%.

**Standardized Mortality Ratio (SMR)**
Observed deaths ÷ Expected deaths (expected = reference rates × study group age structure). SMR > 1: excess mortality. SMR < 1: lower mortality (often healthy worker effect). Used in indirect standardization.

**Healthy Worker Effect**
Workers healthier than general population → SMR < 1 in occupational cohorts even without true protection.

**Number Needed to Treat (NNT)**
How many patients need treatment for one additional person to benefit. NNT = 1 ÷ |Risk Difference|. Smaller NNT = more effective intervention.

**Number Needed to Harm (NNH)**
How many patients need exposure for one additional person to be harmed. NNH = 1 ÷ |Risk Difference|. Same formula as NNT; direction of effect distinguishes benefit (NNT) from harm (NNH).

**Hazard Ratio (HR)**
See Measures of Association.
        """)

    with st.expander("🧪 Hypothesis Testing & Power"):
        st.markdown("""
**Null Hypothesis (H₀)**
Default: no association, no difference. Always an equality (RR = 1, μ₁ = μ₂). What you are trying to find evidence against.

**Alternative Hypothesis (H₁)**
States an association exists. Two-tailed (≠): no direction predicted. One-tailed (< or >): direction pre-specified.

**p-value**
Probability of observing a result as extreme as yours (or more extreme) if H₀ were true. NOT the probability H₀ is true. NOT the probability the result occurred by chance.

**Type I Error (α)**
Rejecting true H₀. False positive. Controlled by setting α = 0.05. Analogy: convicting an innocent person.

**Type II Error (β)**
Failing to reject false H₀. False negative. Conventional target: β ≤ 0.20. Analogy: acquitting a guilty person.

**Statistical Power**
Probability of correctly detecting a real effect when it exists. Power = 1 − β. Conventional minimum: 80%. Increased by: larger sample size, larger true effect, lower measurement error, lower α.

**Confidence Interval (CI)**
See Measures of Association. 95% CI excluding null → p < 0.05. CI provides more information than p-value alone — shows range of plausible effect sizes.

**Chi-Square (χ²)**
Tests whether observed counts differ from expected under independence. Always two-tailed. Larger χ² → smaller p-value.

**One-Tailed Test**
Tests effect in one specific direction. All 5% error tolerance in one tail. Only appropriate when directional hypothesis was pre-specified before data collection based on strong prior evidence.

**Two-Tailed Test**
Tests any difference regardless of direction. Default in epidemiology. Chi-square is nondirectional — it tests total departure from independence without distinguishing direction.
        """)

    with st.expander("📏 Standardization"):
        st.markdown("""
**Crude Rate**
Overall rate without adjusting for confounders. Can mislead when populations have different age structures — apparent rate differences may be entirely due to different age distributions.

**Direct Standardization**
Applies each population's age-specific rates to a single standard population. Produces age-adjusted rate. Best for comparing two populations when age-specific rates are stable.

**Indirect Standardization**
Applies reference population's age-specific rates to study group's age structure. Produces expected deaths → calculates SMR. Best when study group has small numbers making age-specific rates unstable.

**Confounding by Age**
Apparent rate difference due to different age structures, not true disease burden. Standardization removes this. The most common confounder in disease frequency comparisons across populations.
        """)

    with st.expander("🎯 Reliability & Validity"):
        st.markdown("""
**Validity**
Whether a measure captures what it intends to measure. A valid measure has minimal systematic error (bias). Distinct from reliability.

**Face Validity**
The measure *looks* like it measures the concept. Weakest form of validity. Example: asking "do you exercise?" has face validity for physical activity.

**Content Validity**
The measure covers all relevant dimensions of the construct. Example: a diet questionnaire asking only about fat has poor content validity for "overall diet quality."

**Criterion Validity**
How well the measure correlates with a gold standard.
- *Concurrent validity:* Measured at the same time as the gold standard
- *Predictive validity:* An earlier measure predicts a future outcome

**Construct Validity**
The measure behaves as theoretically expected — correlates with related constructs and doesn't correlate with unrelated ones.

**Internal Validity**
Study results accurately reflect the true relationship in the study population. Threatened by bias and confounding.

**External Validity (Generalizability)**
Study findings apply to populations and settings beyond the study. Requires internal validity first.

**Reliability**
Whether a measure produces consistent results under the same conditions. A reliable measure has minimal random error.

**Test-Retest Reliability**
Same measure applied to same subjects at two time points. Assumes the underlying construct hasn't changed.

**Inter-Rater Reliability**
Agreement between different observers measuring the same thing. Measured with **Kappa (κ)**.

**Intra-Rater Reliability**
Consistency of the same observer measuring the same thing at different times.

**Internal Consistency**
For multi-item scales: do all items measure the same construct? Measured with **Cronbach's alpha (α)**. Acceptable: α ≥ 0.70.

**Kappa (κ)**
Measure of inter-rater agreement beyond chance. κ = (P_observed − P_expected) ÷ (1 − P_expected). Benchmarks: <0.20 = slight; 0.21–0.40 = fair; 0.41–0.60 = moderate; 0.61–0.80 = substantial; >0.80 = almost perfect.

**Cronbach's Alpha (α)**
Measure of internal consistency for multi-item scales. Ranges 0–1. α ≥ 0.70 acceptable; α ≥ 0.90 excellent.

**The Reliability-Validity Relationship**
A measure can be reliable without being valid (consistently wrong). A measure cannot be valid without being reliable (random scatter cannot consistently hit the truth). Reliability is necessary but not sufficient for validity.

**Connection to Misclassification**
Low reliability → random measurement error → often non-differential misclassification in categorical variables (or random error in continuous ones). Non-differential misclassification of a binary exposure typically attenuates simple associations toward the null, though this is not guaranteed in all model structures or data configurations.
Invalid measurement → systematic error differing by group → differential misclassification → bias in either direction.
        """)

    with st.expander("🏛️ Foundations — Prevention, Infection & Causation"):
        st.markdown("""
**Natural History of Disease**
The progression of disease in an individual over time without intervention. Four stages: (1) Susceptibility — no disease, risk factors accumulate; (2) Subclinical — pathological changes underway, no symptoms, detectable by screening; (3) Clinical — signs and symptoms present; (4) Resolution — recovery, disability, or death.

**Primary Prevention**
Preventing disease before it occurs. Acts during the susceptibility stage. Examples: vaccination, seat belts, smoking cessation, water fluoridation. Measured by incidence reduction.

**Secondary Prevention**
Early detection and treatment before symptoms appear. Acts during the subclinical stage. Examples: mammography, Pap smear, blood pressure screening. Requires a detectable preclinical phase and effective early treatment.

**Tertiary Prevention**
Reducing disability and complications in those with established disease. Acts during clinical and resolution stages. Examples: cardiac rehabilitation, diabetes management, physical therapy after stroke.

**Quaternary Prevention**
Protecting patients from unnecessary or harmful interventions — overdiagnosis, overtreatment, and iatrogenic harm. Increasingly recognized as a fourth level.

**Lead-Time Bias**
When screening appears to extend survival only because disease is detected earlier, not because treatment is more effective. Total lifespan is unchanged — the clock starts earlier. Use mortality rates (not survival time) to evaluate screening effectiveness.

**Chain of Infection**
Six-link framework for infectious disease transmission: (1) Agent, (2) Reservoir, (3) Portal of Exit, (4) Mode of Transmission, (5) Portal of Entry, (6) Susceptible Host. Breaking any one link prevents transmission.

**Reservoir**
Where an infectious agent normally lives and multiplies. Can be human, animal (zoonosis), or environmental (soil, water).

**Portal of Exit / Portal of Entry**
How an agent leaves the reservoir and enters a new host. Common portals: respiratory tract, gastrointestinal tract, skin breaks, blood, mucous membranes.

**Mode of Transmission**
How an agent travels from reservoir to host. Direct (contact, droplet, vertical) or indirect (airborne, vehicle-borne, vector-borne, fomite).

**Airborne vs. Droplet Transmission**
Droplet: large particles (>5μm), travel <1m, fall quickly (classically influenza). Airborne: droplet nuclei (<5μm), remain suspended, travel >1m (TB, measles, chickenpox). Note: COVID-19 is now understood to transmit primarily via respiratory particles across a spectrum of sizes — short-range inhalation in poorly ventilated spaces is the dominant route, making the historic droplet/airborne binary inadequate for describing it.

**Vector-Borne Transmission**
Biological vector: pathogen replicates inside vector (malaria, dengue, Lyme). Mechanical vector: pathogen transported without replication (housefly + Salmonella).

**R₀ (Basic Reproduction Number)**
Average number of secondary cases from one infectious individual in a fully susceptible population with no intervention. R₀ = β × κ × D (transmission probability × contact rate × duration of infectiousness). R₀ < 1 → epidemic dies out; R₀ > 1 → epidemic grows.

**Effective Reproduction Number (Rₑ)**
R₀ adjusted for current population immunity: Rₑ = R₀ × (proportion susceptible). Rₑ < 1 → epidemic declining; Rₑ > 1 → epidemic growing.

**Herd Immunity**
Indirect protection of susceptibles when enough of the population is immune to break transmission chains. Herd Immunity Threshold (HIT) = 1 − 1/R₀. Above the HIT, Rₑ < 1 and epidemic declines.

**Incubation Period**
Time from exposure (infection) to onset of symptoms. Range of incubation periods approximates the width of the epidemic curve in a point-source outbreak.

**Latency Period**
Time between infection/initiation and detectable disease. In chronic disease, the latency period (subclinical stage) can span years to decades.

**Wilson & Jungner Criteria (1968)**
Ten WHO criteria for justifying a population-wide screening program: (1) important health problem, (2) natural history understood, (3) latent/early stage exists, (4) suitable test, (5) test acceptable to population, (6) effective treatment exists, (7) facilities available, (8) treatment policy agreed, (9) cost-effective, (10) ongoing program. All 10 must be met.

**PICO Framework**
Structured approach to forming research questions: P = Population/Patient, I = Intervention/Exposure, C = Comparison/Control, O = Outcome. Maps directly onto study design choice. Variants: PICOT (adds Time), PECO (uses Exposure instead of Intervention for observational studies).

**Rothman's Sufficient-Component Cause Model (Causal Pies)**
A causal framework where disease results from the joint action of multiple component causes forming a sufficient cause. Each "pie" is a complete causal mechanism; each "slice" is a component cause. Key concepts:
- **Component cause:** A factor that contributes to ≥1 sufficient cause but alone cannot produce disease
- **Sufficient cause:** A minimal set of components that inevitably produces disease
- **Necessary cause:** Appears in every sufficient cause — disease cannot occur without it
- **U (unknown):** Unknown components always present — explains why PAR%s don't sum to 100%

**Biologic Synergy (Rothman)**
Two component causes that appear in the same sufficient cause. Their joint effect exceeds the sum of their individual effects on the additive scale. Synergy is expected, not exceptional, under Rothman's model.

**Induction Period**
Time between the action of a component cause and the completion of the sufficient cause (initiation of disease). Different from latency period (disease initiation to detection).

**PAR% and the Causal Pies**
PAR% reflects both how often a component participates in causal mechanisms and how common it is in the population. A component appearing in many pathways but rare in the population can have modest PAR%. A common exposure with even modest RR can have high PAR% because it participates in many mechanisms across many people. Removing it disrupts all those pathways simultaneously.

**Outbreak Investigation — The 10 Steps**
Systematic framework: (1) Prepare for field work, (2) Establish outbreak exists, (3) Verify diagnosis, (4) Construct case definition, (5) Find cases systematically, (6) Describe by person/place/time, (7) Develop hypotheses, (8) Test hypotheses analytically, (9) Implement control measures, (10) Communicate findings. Steps 9 and 10 begin as early as possible — do not wait for complete investigation.

**Case Definition**
Criteria defining who counts as a case in an outbreak investigation. Components: person (who), place (where), time (when), clinical criteria (what). Levels: Confirmed (lab-proven), Probable (clinical + epi link), Suspected (some clinical criteria). Sensitive early in outbreak; refined as investigation proceeds.
        """)

    st.divider()
    st.markdown("*Return to any module to apply these concepts in context.*")
