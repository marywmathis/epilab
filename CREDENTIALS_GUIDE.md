# EpiLab Buyer Credentials — Railway Deployment Guide

## How credentials work

**Streamlit Cloud (course students)**
- Managed via Streamlit Cloud → App Settings → Secrets
- Format in the Secrets editor:
  ```toml
  [users]
  marymathis = "your_admin_password"
  student1 = "classpass2026"
  student2 = "classpass2026"
  ```
- Students all share a class password, or get individual ones

**Railway (Gumroad buyers)**
- Managed via Railway dashboard → Your service → Variables
- Single environment variable: `EPILAB_USERS`
- Value is a JSON string of all buyer credentials

---

## Adding a new buyer (Railway)

When someone purchases on Gumroad, you need to:

1. Go to **railway.app** → your EpiLab service → **Variables** tab
2. Find the `EPILAB_USERS` variable
3. Edit it to add the new buyer

### Format
```json
{"buyer001": "Kx9mP2vL", "buyer002": "Rn4jQ8wT", "buyer003": "Yh7bN1sA"}
```

Each key is the username, each value is the password.
**Use the password generator below** — never reuse passwords between buyers.

### Step-by-step
1. Copy the current value of `EPILAB_USERS`
2. Add the new buyer to the JSON object
3. Save — Railway redeploys automatically (takes ~30 seconds)
4. Email the buyer their username and password

---

## Password generator (Python — run locally)

```python
import secrets, string

def gen_password(length=10):
    chars = string.ascii_letters + string.digits
    return ''.join(secrets.choice(chars) for _ in range(length))

def gen_username(order_number):
    return f"buyer{str(order_number).zfill(3)}"

# Generate for a new buyer
order = 1  # increment for each new buyer
print(f"Username: {gen_username(order)}")
print(f"Password: {gen_password()}")
```

---

## Gumroad receipt email template

After adding credentials to Railway, send the buyer:

```
Subject: Your EpiLab Access Credentials

Thank you for purchasing EpiLab Interactive (2026 Edition).

Your instructor manuals are attached to this receipt.

To access the simulator:
→ https://YOUR-RAILWAY-URL.up.railway.app

Username: buyer001
Password: Kx9mP2vL

Log in with these credentials. The app loads immediately — no installation required.

Questions? Reply to this email.
```

---

## Revoking access

To revoke a buyer's access (e.g., fraudulent purchase or refund):
1. Go to Railway → Variables → `EPILAB_USERS`
2. Remove that buyer's entry from the JSON
3. Save

Their credentials will stop working within ~30 seconds.

---

## Initial setup — first time deploying to Railway

Set `EPILAB_USERS` to at least one credential before launch:

```json
{"marymathis": "YOUR_ADMIN_PASSWORD"}
```

Then add buyers as they purchase.

---

## Both deployments at a glance

| | Streamlit Cloud | Railway |
|---|---|---|
| **URL** | share.streamlit.io URL | railway.app URL |
| **Audience** | Course students | Gumroad buyers |
| **Credentials** | Streamlit Secrets (TOML) | `EPILAB_USERS` env var (JSON) |
| **Cost** | Free | ~$5/month |
| **Hibernation** | Yes (7 days inactive) | No — always on |
| **Code source** | GitHub main branch | GitHub main branch |
| **Updates** | Auto on git push | Auto on git push |
