# EpiLab Interactive — Epidemiology Decision Simulator
### 2026 Edition

EpiLab Interactive is a web-based epidemiology decision simulator built with Streamlit, targeting undergraduate and graduate public health students, epidemiology faculty, practitioners, and clinical students. Licensed access required.

**Live deployments:**
- Course students → Streamlit Cloud (free tier, hibernation acceptable)
- Gumroad buyers → Railway ($5/mo, always-on, no hibernation)

---

## Full Content Inventory

### Module 1 — Study Design & Causation

**Foundations** (6 sections): Epidemiology Triangle (Agent/Host/Environment/TIME at center, SVG diagram, 3 scenarios) · Natural History & Levels of Prevention · Chain of Infection · Herd Immunity & R₀ (interactive calculator) · Outbreak Investigation 10 Steps · PICO Framework

**Study Designs** (3 sections): Design Overview + Ecologic Fallacy (3 practice scenarios) · Design Selector (interactive decision tree) · RCT & Evidence Hierarchy including Randomization, Blinding (single/double/triple), Intent-to-Treat analysis

**Bias** (4 sections): Selection Bias · Information Bias (differential vs. non-differential) · Bias Direction Exercise (8 scenarios) · Reliability & Validity (Cohen's Kappa calculator)

**Confounding & Effect Modification** (5 sections): What Is Confounding? · Controlling Confounding · Effect Modification · Interactive Stratified Analysis · DAG Library (6 SVG diagrams: Confounder, Mediator, Collider, Moderator, M-Bias, Proxy)

**Causal Inference** (4 sections): Bradford Hill Criteria · Application Exercise (5 classic pairs) · Rothman's Causal Pies · Web of Causation (MacMahon & Pugh, with 6-model comparison table)

---

### Module 2 — Foundations of Measurement

**Disease Frequency** (7 sections): Core Measures · Interactive Calculator · Prevalence-Incidence Relationship (P=I×D) · Epidemic Curves (4 types, interactive) · Person/Place/Time (5 time pattern types, 4 practice scenarios including migration study) · Public Health Surveillance (5 data source types, 5 practice scenarios) · Mortality Measures & YPLL (interactive calculator)

**Screening & Diagnostic Tests** (6 sections): Core Concepts (2×2 table) · Interactive 2×2 Calculator (3 evidence-based presets: Mammography, Rapid Strep, PSA) · Likelihood Ratios & Fagan Nomogram (dynamic nomogram with sliders, 2 practice scenarios) · Prevalence Effect on PPV · Wilson & Jungner Criteria · ROC Curve (distribution overlap visual, interactive ROC explorer, AUC table)

---

### Module 3 — Measures & Analysis

**Measures of Association**: RR, PR, OR, IRR with full plain-language interpretations, CI interpretation, magnitude framing, AR%/NNT/NNH expander, design-appropriate cautions, 3 presets

**Advanced Epi Measures** (5): PAR% · SMR · Attributable Risk & AR% · NNT/NNH (contextual impact scale with real-world comparators) · Hazard Ratio

**Standardization**: Direct and indirect, interactive calculator

**Hypothesis Testing & Power** (4 sections): One vs. Two-tailed · Null hypothesis interpretation · Hypothesis Builder (8 scenarios) · Power & Sample Size calculator

---

### Module 4 — Practice & Application

Practice sections (4): Study Design & Classification (8 scenarios) · Advanced Epi Measures (8) · Confounding & Bias (8) · Screening & Disease Frequency (11)

**Outbreak Lab**: Compendium of Acute Foodborne GI Diseases (12 pathogens) · Investigation 1: Norovirus/University Dining Hall · Investigation 2: Measles/Under-Vaccinated School · Investigation 3: Salmonellosis/Church Potluck — each with 5 progressive steps, guided questions, Next Step buttons

**Glossary**: 10 expandable reference sections

---

## Technical Stack

- **Framework:** Streamlit (Python 3.11)
- **Libraries:** pandas, numpy, scipy
- **Visualizations:** SVG via `st.components.v1.html()` — no external plotting dependencies
- **Deployment:** Streamlit Cloud (course) + Railway Hobby (buyers, $5/mo)

---

## Credential System

Priority order at startup:

1. `EPILAB_USERS` env var (JSON) — Railway buyer deployments
2. Streamlit Cloud `st.secrets["users"]` — course deployments
3. Hardcoded fallback — local dev only

**Streamlit Cloud secrets:**
```toml
[users]
marymathis = "your_admin_password"
student1 = "classpass2026"
```

**Railway env var (`EPILAB_USERS`):**
```json
{"buyer001": "Kx9mP2vL", "buyer002": "Rn4jQ8wT"}
```

See `CREDENTIALS_GUIDE.md` for full buyer management workflow, password generator, and receipt email template.

---

## Deployment

**Streamlit Cloud:** Push `app.py` to `marywmathis/epilab` main branch → auto-redeploys in ~60s

**Railway:** Requires `railway.toml`, `nixpacks.toml`, `.streamlit/config.toml` in repo root. Set `EPILAB_USERS` env var in Railway dashboard before first deploy.

Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`

---

## File Structure

```
/
├── app.py                    # Main application
├── requirements.txt
├── railway.toml              # Railway deployment config
├── nixpacks.toml             # Build config
├── CREDENTIALS_GUIDE.md      # Buyer credential management
├── README.md
└── .streamlit/
    └── config.toml           # Streamlit server config
```

---

## Instructor Manuals (included with Gumroad package)

| Manual | Content |
|--------|---------|
| Module 1 — Study Design & Causation | Foundations, Study Designs, Bias, Confounding, Causal Inference |
| Module 2 — Foundations of Measurement | Disease Frequency, Screening & Diagnostics |
| Module 3 — Measures & Analysis | Measures of Association, Advanced Measures, Standardization, Hypothesis Testing |
| Module 4 — Practice & Outbreak Lab | Practice sections, Outbreak Lab, Glossary |

Each manual: learning objectives, content summary, key tables, discussion questions, common misconceptions, facilitation notes.

---

EpiLab Interactive © 2026 | Licensed use only
