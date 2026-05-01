# Credit Risk Early Warning System
## Intesa Sanpaolo CFO Office Demo | Powered by Verkko

---

## Architecture

```
Layer 1: Data
  1A  FMP API → Real quarterly financials (10 Italian/European corporates)
  1B  Synthetic generator → 200-client loan portfolio

Layer 2: ML Pipeline
  2A  Feature engineering (Altman Z-Score, revenue trends, ratio merging)
  2B  LightGBM PD model (probability of default scoring)
  2C  Output store → credit_risk_output.csv

Layer 3: Verkko
  Upload credit_risk_output.csv → Verkko Data Sources
  Run 4 demo queries live in Verkko Chat
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your FMP API key (optional — synthetic fallback if not set)
```bash
export FMP_API_KEY="your_key_here"
```

### 3. Run the full pipeline
```bash
python credit_risk_pipeline.py
```

This produces `credit_risk_output.csv` — upload this to Verkko.

---

## Output Schema (`credit_risk_output.csv`)

| Column | Description |
|---|---|
| `client_id` | ISP-CORP-XXXX internal identifier |
| `company_name` | Corporate borrower name |
| `ticker` | Exchange ticker (public companies only) |
| `sector` | Industry classification |
| `loan_type` | Term Loan / Revolving Credit / Trade Finance / Project Finance |
| `loan_amount_eur` | Facility size in EUR |
| `interest_rate_pct` | Contractual interest rate |
| `loan_to_value` | LTV ratio |
| `days_past_due` | Current DPD status |
| `debt_to_equity` | Leverage ratio |
| `current_ratio` | Liquidity ratio |
| `interest_coverage` | EBIT / Interest expense |
| `return_on_equity` | Profitability ratio |
| `revenue_qoq_change` | Quarterly revenue growth rate |
| `zscore_approx` | Altman Z-Score approximation |
| `pd_score_pct` | **Model PD score (0–100%)** |
| `risk_tier` | Low Risk / Watch List / Elevated / High Risk |
| `internal_rating` | AAA-AA / A / BBB / BB / B / CCC-D |
| `hidden_risk_flag` | 1 = revenue declining >15% but zero DPD |

---

## Verkko Demo Queries

### Query 1 — Portfolio Overview
> *"Show me a breakdown of our corporate loan portfolio by risk tier. How many clients are in each category and what is the total exposure?"*

### Query 2 — Hidden Risk (The Killer Demo)
> *"List all clients whose quarterly revenue declined by more than 15% but who are still current on their loan payments. Sort by loan exposure."*

**Why this matters:** These clients look fine in traditional DPD-based systems but their underlying business is deteriorating. Verkko surfaces this invisible risk instantly.

### Query 3 — Sector Concentration
> *"What is our total exposure to the Automotive sector? Show me the PD scores for each client in that sector."*

### Query 4 — Client Deep Dive
> *"Pull up the full credit profile for Stellantis. Show me their financial ratios, our model's PD score, and the key risk drivers."*

---

## What the CFO Sees

| What Verkko Shows | What It Proves |
|---|---|
| Natural language → complex portfolio analytics | Replaces manual report requests |
| Real FMP financials blended with internal loan data | Integrates external + proprietary sources |
| Predictive PD scores, not just historical DPD | Forward-looking risk management |
| Hidden risk query (declining revenue + current) | Finds what current systems miss |
| Instant answers, no analyst queue | Dramatically accelerates decisions |

---

## Connecting to Verkko

**Option A — File Upload (Prototype, fastest):**
1. Run pipeline → generates `credit_risk_output.csv`
2. Upload to Verkko → Data Sources → Upload File
3. Start querying immediately in Verkko Chat

**Option B — Database (Production):**
```python
# Push to PostgreSQL
from sqlalchemy import create_engine
engine = create_engine("postgresql://user:pass@host:5432/intesa_risk")
df_output.to_sql("credit_risk_scores", engine, if_exists="replace", index=False)
# Connect this DB to Verkko via Integrations panel
```
