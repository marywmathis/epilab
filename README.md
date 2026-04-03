# 🧭 EpiLab — Epidemiology Decision Simulator

An interactive epidemiology learning suite built with Streamlit.

## Modules
- **Module 1:** Study Design & Causation (designs, bias, confounding, causal inference)
- **Module 2:** Foundations (disease frequency, epidemic curves, screening & diagnostics)
- **Module 3:** Measures & Analysis (RR, OR, PR, IRR, standardization, hypothesis testing)
- **Module 4:** Practice scenarios with locked feedback
- **Reference:** Glossary

## Running locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Default login:
- Username: `marymathis`
- Password: `epilab2024`

To add/change users, edit the `USERS` dictionary near the top of `app.py`.

## Deploying to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file to `app.py`
5. (Optional) Add credentials via App Settings → Secrets:

```toml
[users]
marymathis = "your_chosen_password"
student1 = "password1"
```

## File structure

```
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── .gitignore          # Excludes secrets and cache files
└── README.md           # This file
```
