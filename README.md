# 🧭 EpiLab — Epidemiology Decision Simulator

An interactive epidemiology learning suite built with Streamlit, designed for undergraduate and graduate public health students, clinical students, and faculty teaching epidemiology.

---

## Contents

### Module 1 — Study Design & Causation
- **Foundations** — Natural history of disease (4-stage visual), levels of prevention (primary/secondary/tertiary/quaternary + lead-time bias), chain of infection (6-link diagram + modes of transmission), herd immunity & R₀ (interactive calculator), outbreak investigation (10 steps), PICO framework (interactive builder)
- **Study Designs** — Cohort, case-control, cross-sectional, ecological, case-crossover, RCT; evidence hierarchy (7-level); Design Selector with data type guidance (nominal/ordinal/discrete/continuous), statistical test recommendations, and confounding adjustment methods (design-stage and analysis-stage)
- **Bias** — Selection bias, information bias, misclassification, reliability & validity, interactive Kappa calculator
- **Confounding & Effect Modification** — DAG library (confounder, mediator, collider, moderator, M-bias, proxy), interactive stratified analysis, Mantel-Haenszel
- **Causal Inference** — Bradford Hill criteria (application exercise), Rothman's sufficient-component cause model (causal pies with letter-keyed visual and pedagogical notes)

### Module 2 — Foundations of Measurement
- **Disease Frequency** — Prevalence, cumulative incidence, incidence rate, CFR, SAR, P=I×D; epidemic curves with 8 named presets AND four-pattern side-by-side comparison (point-source, propagated, mixed, endemic) with baseline and epidemic threshold lines
- **Screening & Diagnostics** — 2×2 table, sensitivity, specificity, PPV, NPV, LR+/LR−, pre/post-test probability calculator, prevalence effect on PPV slider, Wilson & Jungner screening criteria (all 10 with cervical cancer model and PSA cautionary example)

### Module 3 — Measures & Analysis
- **Measures of Association** — RR, OR, PR, IRR, chi-square with CI visualization; stratified analysis with Mantel-Haenszel (3 presets)
- **Advanced Epi Measures** — PAR%, SMR, AR/AR%, NNT, NNH, HR, IRR with interactive scenario-based calculators
- **Standardization** — Direct, indirect, SMR, age-adjustment
- **Hypothesis Testing & Power** — One vs. two-tailed tests, p-value interpretation, Type I/II error visual, Hypothesis Builder (8 scenarios), power & sample size calculator

### Module 4 — Practice & Application
- **Study Design Practice** — 8 randomized scenarios with locked feedback
- **Advanced Measures Practice** — 8 scenarios; measure selection + interactive calculation
- **Confounding & Bias Practice** — 8 scenarios covering misclassification, effect modification, confounding by indication, loss to follow-up
- **Screening & Frequency Practice** — 11 scenarios covering LR+/LR−, SAR, P=I×D, CFR, epidemic curves, NPV
- **🔍 Outbreak Lab** — 3 complete outbreak investigations (EIS-style):
  - Scenario 1: Norovirus at a University Dining Hall (attack rates, vehicle identification, secondary spread)
  - Scenario 2: Measles in an Under-Vaccinated School (herd immunity math, contact tracing, vaccination policy)
  - Scenario 3: Salmonellosis at a Church Potluck (case definition, incubation estimation, supply chain tracing)
  - Built-in field reference: Compendium of Acute Foodborne GI Diseases (incubation × symptoms × food vehicle)

### Reference
- **Glossary** — All key terms across all modules, organized by topic (10 expandable sections)

---

## Running Locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

**Default login credentials** (hardcoded in app.py — no secrets file needed):
- `marymathis` / `epilab2024`
- `student1` / `epilab2024`
- `guest` / `epilab2024`

To add or change users, edit the `USERS` dictionary near the top of `app.py`.

---

## Deploying to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file path to `app.py`
5. Click **Deploy**

**Optional — manage credentials via Streamlit Secrets:**

In Streamlit Cloud → App Settings → Secrets:
```toml
[users]
marymathis = "your_password"
student1 = "password1"
```

---

## File Structure

```
app.py                  # Main application (~8,500 lines)
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

## Scientific Accuracy Notes

This app incorporates the following evidence-based nuances often absent from introductory epidemiology tools:
- COVID-19 described as respiratory particle transmission across a spectrum of sizes — not classified under the historic droplet/airborne binary
- Chi-square described as nondirectional (not "two-tailed") with the p-value correctly framed as conditional on the null
- Non-differential misclassification described as typically (not always) attenuating toward null
- Rare disease OR≈RR described as a teaching heuristic with no hard cutoff
- Epidemic threshold described as one common approach, not a universal standard
- Rothman's causal pies annotated with explicit notes that equal slice size does not imply equal causal strength

---

*EpiLab Interactive | Developed by Mary W. Mathis, DrPH*
