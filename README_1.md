# 🧭 EpiLab — Epidemiology Decision Simulator

An interactive epidemiology learning suite built with Streamlit, designed for undergraduate and graduate public health students, clinical students, and faculty teaching epidemiology.

---

## Contents

### Module 1 — Study Design & Causation
- **Foundations** — Natural history of disease, levels of prevention (primary/secondary/tertiary/quaternary), chain of infection, R₀ & herd immunity, outbreak investigation (10 steps), PICO framework
- **Study Designs** — Cohort, case-control, cross-sectional, ecological, case-crossover, RCT; evidence hierarchy; Design Selector with data type guidance and confounding adjustment methods
- **Bias** — Selection bias, information bias, misclassification, reliability & validity, Kappa calculator
- **Confounding & Effect Modification** — DAG library (confounder, mediator, collider, moderator, M-bias, proxy), stratified analysis, Mantel-Haenszel
- **Causal Inference** — Bradford Hill criteria, Rothman's sufficient-component cause model (causal pies)

### Module 2 — Foundations of Measurement
- **Disease Frequency** — Prevalence, cumulative incidence, incidence rate, CFR, SAR, P=I×D; epidemic curves with four-pattern comparison (point-source, propagated, mixed, endemic) with baseline and epidemic threshold
- **Screening & Diagnostics** — 2×2 table, sensitivity, specificity, PPV, NPV, LR+/LR−, pre/post-test probability calculator, prevalence effect on PPV, Wilson & Jungner screening criteria

### Module 3 — Measures & Analysis
- **Measures of Association** — RR, OR, PR, IRR, chi-square with CI visualization; stratified analysis with Mantel-Haenszel
- **Advanced Epi Measures** — PAR%, SMR, AR/AR%, NNT, NNH, HR, IRR with interactive calculators
- **Standardization** — Direct, indirect, SMR, age-adjustment
- **Hypothesis Testing & Power** — One vs. two-tailed tests, p-value interpretation, Type I/II error visual, Hypothesis Builder (8 scenarios), power & sample size calculator

### Module 4 — Practice (35+ scenarios, randomized, locked feedback)
- **Study Design** — 8 scenarios covering all designs including ecological and retrospective cohort
- **Advanced Measures** — 8 scenarios; measure identification + interactive calculation
- **Confounding & Bias** — 8 scenarios covering misclassification, effect modification, loss to follow-up, confounding by indication
- **Screening & Frequency** — 11 scenarios covering LR+/LR−, SAR, P=I×D, CFR vs. attack rate, epidemic curves, NPV in high-risk populations

### Reference
- **Glossary** — All key terms across all modules, organized by topic

---

## Running Locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

**Default login credentials** (hardcoded in app.py — no secrets file needed):
- Username: `marymathis` / Password: `epilab2024`
- Username: `student1` / Password: `epilab2024`
- Username: `guest` / Password: `epilab2024`

To add or change users, edit the `USERS` dictionary near the top of `app.py`.

---

## Deploying to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file path to `app.py`
5. Click **Deploy**

**Optional — manage credentials via Streamlit Secrets** (overrides the built-in USERS dict):

In Streamlit Cloud → App Settings → Secrets, paste:
```toml
[users]
marymathis = "your_password"
student1 = "password1"
```

---

## File Structure

```
app.py                  # Main application (~7,300 lines)
requirements.txt        # Python dependencies
.gitignore              # Excludes secrets and cache files
README.md               # This file
```

## Dependencies

```
streamlit>=1.32.0
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.9.0
```

---

## Instructor Materials

Four instructor manuals (.docx) are available separately — one per module. Each contains learning objectives (CEPH-competency aligned), section-by-section teaching guidance, common student misconceptions, discussion questions, assessment ideas, answer keys (Module 4), and cross-module connections.

---

*EpiLab Interactive | Developed by Mary W. Mathis, DrPH*
