#!/usr/bin/env python3
"""
Credit Risk Early Warning System — End-to-End Pipeline
Intesa Sanpaolo CFO Office Demo | Powered by Verkko

Layers:
  1A: Real corporate financials via FMP API
  1B: Synthetic loan portfolio generation
  2A: Feature engineering + Altman Z-Score
  2B: LightGBM credit risk model (PD scoring)
  2C: Output store for Verkko integration
"""

import os
import glob
import warnings
import requests
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

FMP_API_KEY = os.getenv("FMP_API_KEY", "demo")  # Set your key via env var
BASE_URL    = "https://financialmodelingprep.com/stable"

TICKERS = [
    "RACE", "STLAM.MI", "ENEL.MI", "TIT.MI", "LDO.MI",
    "UCG.MI", "ISP.MI", "MB.MI", "AMP.MI", "CPR.MI"
]

SECTOR_MAP = {
    "RACE":    "Automotive/Luxury",
    "STLAM.MI":"Automotive",
    "ENEL.MI": "Utilities",
    "TIT.MI":  "Telecom",
    "LDO.MI":  "Aerospace",
    "UCG.MI":  "Banking",
    "ISP.MI":  "Banking",
    "MB.MI":   "Financial Services",
    "AMP.MI":  "Healthcare",
    "CPR.MI":  "Consumer Goods",
}

COMPANY_NAMES = {
    "RACE":    "Ferrari NV",
    "STLAM.MI":"Stellantis NV",
    "ENEL.MI": "Enel SpA",
    "TIT.MI":  "Telecom Italia SpA",
    "LDO.MI":  "Leonardo SpA",
    "UCG.MI":  "UniCredit SpA",
    "ISP.MI":  "Intesa Sanpaolo SpA",
    "MB.MI":   "Mediobanca SpA",
    "AMP.MI":  "Amplifon SpA",
    "CPR.MI":  "Campari Group SpA",
}


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1A — FMP DATA ACQUISITION
# ─────────────────────────────────────────────────────────────────────────────

def _fmp_get(endpoint: str, ticker: str, period: str = "quarter") -> list:
    """Generic FMP API call with error handling."""
    url = f"{BASE_URL}/{endpoint}"
    params = {"symbol": ticker, "period": period, "apikey": FMP_API_KEY}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        # FMP sometimes returns {"Error Message": ...}
        print(f"  ⚠  {ticker} / {endpoint}: {data}")
        return []
    except Exception as exc:
        print(f"  ✗  {ticker} / {endpoint}: {exc}")
        return []


def get_income_statement(ticker: str) -> list:
    return _fmp_get("income-statement", ticker)

def get_balance_sheet(ticker: str) -> list:
    return _fmp_get("balance-sheet-statement", ticker)

def get_cash_flow(ticker: str) -> list:
    return _fmp_get("cash-flow-statement", ticker)

def get_financial_ratios(ticker: str) -> list:
    return _fmp_get("ratios", ticker)


def pull_fmp_financials(tickers: list) -> pd.DataFrame:
    """
    Pull quarterly financials for all tickers and merge into one DataFrame.
    Falls back to synthetic data if API key is 'demo' or calls fail.
    """
    print("\n[Layer 1A] Pulling FMP corporate financials...")
    all_records = []

    for ticker in tickers:
        print(f"  → {ticker}")
        income  = get_income_statement(ticker)
        balance = get_balance_sheet(ticker)
        cashflow= get_cash_flow(ticker)
        ratios  = get_financial_ratios(ticker)

        if not income:
            print(f"    No income data — skipping {ticker}")
            continue

        for i, inc in enumerate(income):
            bs  = balance[i]  if i < len(balance)  else {}
            cf  = cashflow[i] if i < len(cashflow) else {}
            rat = ratios[i]   if i < len(ratios)   else {}

            all_records.append({
                "ticker":            ticker,
                "date":              inc.get("date"),
                "period":            inc.get("period"),
                # Income Statement
                "revenue":           inc.get("revenue"),
                "gross_profit":      inc.get("grossProfit"),
                "operating_income":  inc.get("operatingIncome"),
                "net_income":        inc.get("netIncome"),
                "ebitda":            inc.get("ebitda"),
                "interest_expense":  inc.get("interestExpense"),
                # Balance Sheet
                "total_assets":      bs.get("totalAssets"),
                "total_debt":        bs.get("totalDebt"),
                "total_equity":      bs.get("totalStockholdersEquity"),
                "cash":              bs.get("cashAndCashEquivalents"),
                # Cash Flow
                "operating_cf":      cf.get("operatingCashFlow"),
                "free_cf":           cf.get("freeCashFlow"),
                # Ratios
                "debt_to_equity":    rat.get("debtEquityRatio"),
                "current_ratio":     rat.get("currentRatio"),
                "interest_coverage": rat.get("interestCoverage"),
                "return_on_equity":  rat.get("returnOnEquity"),
            })

    if not all_records:
        print("  ⚠  No live FMP data retrieved — generating synthetic financials.")
        return _synthetic_financials(tickers)

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    print(f"  ✓  {len(df)} quarterly records for {df['ticker'].nunique()} companies.")
    return df


def _synthetic_financials(tickers: list) -> pd.DataFrame:
    """Generate realistic synthetic financials when FMP is unavailable."""
    np.random.seed(42)
    records = []
    quarters = pd.date_range(end="2024-12-31", periods=16, freq="QE")

    # Sector-specific base revenue (€M)
    base_rev = {
        "RACE":    5_000_000_000,  "STLAM.MI": 45_000_000_000,
        "ENEL.MI": 25_000_000_000, "TIT.MI":    3_500_000_000,
        "LDO.MI":  4_000_000_000,  "UCG.MI":   12_000_000_000,
        "ISP.MI":  10_000_000_000, "MB.MI":     1_200_000_000,
        "AMP.MI":    450_000_000,  "CPR.MI":      700_000_000,
    }

    for ticker in tickers:
        rev = base_rev.get(ticker, 2_000_000_000)
        for q in quarters:
            noise = np.random.normal(1.0, 0.05)
            r = rev * noise
            records.append({
                "ticker":            ticker,
                "date":              q,
                "period":            "Q" + str(q.quarter),
                "revenue":           round(r, 0),
                "gross_profit":      round(r * np.random.uniform(0.25, 0.55), 0),
                "operating_income":  round(r * np.random.uniform(0.08, 0.22), 0),
                "net_income":        round(r * np.random.uniform(0.04, 0.15), 0),
                "ebitda":            round(r * np.random.uniform(0.12, 0.30), 0),
                "interest_expense":  round(r * np.random.uniform(0.01, 0.05), 0),
                "total_assets":      round(r * np.random.uniform(2.0, 6.0), 0),
                "total_debt":        round(r * np.random.uniform(0.5, 2.5), 0),
                "total_equity":      round(r * np.random.uniform(0.8, 2.0), 0),
                "cash":              round(r * np.random.uniform(0.1, 0.5), 0),
                "operating_cf":      round(r * np.random.uniform(0.06, 0.20), 0),
                "free_cf":           round(r * np.random.uniform(0.02, 0.12), 0),
                "debt_to_equity":    round(np.random.uniform(0.3, 3.5), 3),
                "current_ratio":     round(np.random.uniform(0.8, 2.8), 3),
                "interest_coverage": round(np.random.uniform(1.5, 18.0), 3),
                "return_on_equity":  round(np.random.uniform(0.02, 0.25), 4),
            })

    df = pd.DataFrame(records)
    print(f"  ✓  Synthetic: {len(df)} quarterly records for {len(tickers)} companies.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1B — SYNTHETIC LOAN PORTFOLIO
# ─────────────────────────────────────────────────────────────────────────────

def generate_loan_portfolio(tickers: list, n_clients: int = 200) -> pd.DataFrame:
    """
    Simulate Intesa Sanpaolo's corporate loan book.
    First len(tickers) clients are public companies linked to FMP data.
    Remaining are fictional private Italian corporates.
    """
    print("\n[Layer 1B] Generating synthetic loan portfolio...")
    np.random.seed(42)

    loan_types   = ["Term Loan", "Revolving Credit", "Trade Finance", "Project Finance"]
    private_sectors = ["Manufacturing", "Retail", "Hospitality",
                       "Real Estate", "Agriculture", "Technology"]
    italian_prefixes = ["Ital", "Med", "Alp", "Adri", "Toscana", "Lomb",
                        "Veneto", "Sicil", "Napol", "Roman"]
    entity_types = ["Srl", "SpA", "Sas", "Snc"]

    clients = []
    for i in range(n_clients):
        is_public = i < len(tickers)

        if is_public:
            ticker       = tickers[i]
            company_name = COMPANY_NAMES.get(ticker, ticker + " SpA")
            sector       = SECTOR_MAP.get(ticker, "Financial Services")
        else:
            ticker       = None
            pfx          = italian_prefixes[(i - len(tickers)) % len(italian_prefixes)]
            company_name = f"{pfx}Corp{i:03d} {np.random.choice(entity_types)}"
            sector       = np.random.choice(private_sectors)

        # Loan economics
        loan_amount   = np.round(np.random.lognormal(mean=16.5, sigma=1.8), 2)
        interest_rate = np.round(np.random.uniform(1.5, 8.0), 2)

        # Health score → default probability
        health = np.random.beta(5, 2)          # Right-skewed: most companies healthy
        if is_public:
            health = min(health * 1.1, 1.0)    # Public companies slightly healthier

        default_prob = np.clip(1 - health + np.random.normal(0, 0.08), 0.01, 0.99)
        has_defaulted = int(np.random.binomial(1, default_prob * 0.15))

        # Days past due — correlated with health
        dpd_choices = [0, 0, 0, 0, 30, 60, 90]
        dpd_probs   = [0.60, 0.15, 0.10, 0.05, 0.05, 0.03, 0.02]
        if health < 0.4:  # Distressed companies more likely to be past due
            dpd_probs = [0.30, 0.15, 0.15, 0.10, 0.15, 0.10, 0.05]

        clients.append({
            "client_id":            f"ISP-CORP-{i:04d}",
            "company_name":         company_name,
            "ticker":               ticker,
            "is_public":            is_public,
            "sector":               sector,
            "loan_type":            np.random.choice(loan_types),
            "loan_amount_eur":       np.round(loan_amount, 2),
            "interest_rate_pct":     interest_rate,
            "loan_to_value":         np.round(np.random.uniform(0.40, 0.95), 2),
            "years_as_client":       np.random.randint(1, 25),
            "num_facilities":        np.random.randint(1, 6),
            "days_past_due":         np.random.choice(dpd_choices, p=dpd_probs),
            "has_defaulted":         has_defaulted,
            "health_score_internal": np.round(health, 4),
        })

    df = pd.DataFrame(clients)
    print(f"  ✓  {len(df)} corporate clients | "
          f"{df['is_public'].sum()} public, {(~df['is_public']).sum()} private | "
          f"Default rate: {df['has_defaulted'].mean():.1%}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2A — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df_loans: pd.DataFrame,
                      df_financials: pd.DataFrame) -> pd.DataFrame:
    """
    Merge loan portfolio with FMP financials for public companies.
    Simulate equivalent ratios for private companies.
    Compute Altman Z-Score approximation and revenue trend.
    """
    print("\n[Layer 2A] Engineering features...")
    np.random.seed(99)

    # ── Revenue QoQ trend for public companies ──────────────────────────────
    df_fin_sorted = df_financials.sort_values(["ticker", "date"])
    df_fin_sorted["revenue_qoq_change"] = (
        df_fin_sorted.groupby("ticker")["revenue"].pct_change()
    )
    latest_trends = (
        df_fin_sorted.groupby("ticker")
        .last()[["revenue_qoq_change"]]
        .reset_index()
    )

    # ── Latest quarter financials per ticker ────────────────────────────────
    df_latest_fin = (
        df_financials.sort_values("date", ascending=False)
        .groupby("ticker")
        .first()
        .reset_index()
    )
    df_latest_fin = df_latest_fin.merge(latest_trends, on="ticker", how="left")

    # ── Public companies: merge real financials ──────────────────────────────
    df_public  = df_loans[df_loans["is_public"]].copy()
    df_private = df_loans[~df_loans["is_public"]].copy()

    df_enriched = df_public.merge(
        df_latest_fin[["ticker", "debt_to_equity", "current_ratio",
                        "interest_coverage", "return_on_equity",
                        "revenue", "revenue_qoq_change"]],
        on="ticker", how="left"
    )

    # ── Private companies: simulate financial ratios ─────────────────────────
    n_priv = len(df_private)
    df_private = df_private.copy()
    df_private["debt_to_equity"]    = np.random.uniform(0.3, 4.0, n_priv).round(3)
    df_private["current_ratio"]     = np.random.uniform(0.5, 3.0, n_priv).round(3)
    df_private["interest_coverage"] = np.random.uniform(0.5, 15.0, n_priv).round(3)
    df_private["return_on_equity"]  = np.random.uniform(-0.20, 0.40, n_priv).round(4)
    df_private["revenue"]           = np.random.lognormal(15.5, 1.5, n_priv).round(0)
    df_private["revenue_qoq_change"]= np.random.uniform(-0.20, 0.20, n_priv).round(4)

    # ── Combine ──────────────────────────────────────────────────────────────
    df_model = pd.concat([df_enriched, df_private], ignore_index=True)

    # Fill any remaining NaNs in financial ratios
    ratio_cols = ["debt_to_equity", "current_ratio", "interest_coverage",
                  "return_on_equity", "revenue_qoq_change"]
    for col in ratio_cols:
        median_val = df_model[col].median()
        df_model[col] = df_model[col].fillna(median_val)

    # ── Altman Z-Score approximation ─────────────────────────────────────────
    # Classic formula adapted for available features:
    # Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    df_model["zscore_approx"] = (
        1.2 * df_model["current_ratio"].clip(0, 5) +
        1.4 * df_model["return_on_equity"].clip(-1, 1) +
        3.3 * (df_model["interest_coverage"] / 20).clip(0, 1) +
        0.6 * (1 / df_model["debt_to_equity"].clip(0.1, 10)) +
        1.0 * df_model["revenue_qoq_change"].clip(-1, 1)
    ).round(4)

    # ── Stress indicator: revenue declining but current on payments ──────────
    df_model["revenue_declining"] = (
        df_model["revenue_qoq_change"] < -0.15
    ).astype(int)
    df_model["hidden_risk_flag"] = (
        (df_model["revenue_qoq_change"] < -0.15) &
        (df_model["days_past_due"] == 0)
    ).astype(int)

    print(f"  ✓  Feature matrix: {df_model.shape[0]} clients × {df_model.shape[1]} features")
    print(f"  ✓  Hidden risk candidates (declining revenue + current): "
          f"{df_model['hidden_risk_flag'].sum()} clients")
    return df_model


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2B — CREDIT RISK MODEL
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "loan_amount_eur", "interest_rate_pct", "loan_to_value",
    "years_as_client", "num_facilities", "days_past_due",
    "debt_to_equity", "current_ratio", "interest_coverage",
    "return_on_equity", "revenue_qoq_change", "zscore_approx",
]


def train_credit_model(df_model: pd.DataFrame):
    """
    Train a LightGBM classifier to predict probability of default (PD).
    Returns the trained model and evaluation metrics.
    """
    print("\n[Layer 2B] Training credit risk model...")

    try:
        import lightgbm as lgb
        USE_LGBM = True
    except ImportError:
        print("  ⚠  LightGBM not installed — falling back to XGBoost/sklearn.")
        USE_LGBM = False

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, classification_report

    X = df_model[FEATURE_COLS].fillna(0)
    y = df_model["has_defaulted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if USE_LGBM:
        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
    else:
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
                random_state=42, eval_metric="logloss", verbosity=0,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=4,
                random_state=42,
            )

    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred       = model.predict(X_test)
    auc          = roc_auc_score(y_test, y_pred_proba)

    print(f"  ✓  AUC-ROC: {auc:.4f}")
    print("  Classification Report:")
    report = classification_report(y_test, y_pred,
                                   target_names=["Performing", "Default"])
    for line in report.split("\n"):
        print(f"    {line}")

    return model, auc


def score_portfolio(df_model: pd.DataFrame, model) -> pd.DataFrame:
    """Apply the trained model to score all clients with PD scores."""
    print("\n[Layer 2B] Scoring full portfolio...")

    X_all = df_model[FEATURE_COLS].fillna(0)
    df_model = df_model.copy()
    df_model["pd_score"]     = model.predict_proba(X_all)[:, 1]
    df_model["pd_score_pct"] = (df_model["pd_score"] * 100).round(2)

    df_model["risk_tier"] = pd.cut(
        df_model["pd_score"],
        bins=[0, 0.05, 0.15, 0.30, 1.0],
        labels=["Low Risk", "Watch List", "Elevated", "High Risk"],
        include_lowest=True,
    )

    # Basel-style internal rating
    df_model["internal_rating"] = pd.cut(
        df_model["pd_score"],
        bins=[0, 0.02, 0.05, 0.10, 0.20, 0.35, 1.0],
        labels=["AAA-AA", "A", "BBB", "BB", "B", "CCC-D"],
        include_lowest=True,
    )

    tier_summary = df_model.groupby("risk_tier", observed=True).agg(
        n_clients=("client_id", "count"),
        total_exposure_eur=("loan_amount_eur", "sum"),
        avg_pd_pct=("pd_score_pct", "mean"),
    ).reset_index()
    tier_summary["total_exposure_eur"] = (
        tier_summary["total_exposure_eur"] / 1e6
    ).round(1)
    tier_summary["avg_pd_pct"] = tier_summary["avg_pd_pct"].round(1)

    print("\n  ┌─ Portfolio Risk Tier Summary ──────────────────────────────────┐")
    print(f"  {'Risk Tier':<14} {'# Clients':>10} {'Exposure (€M)':>15} {'Avg PD%':>10}")
    print("  " + "─" * 54)
    for _, row in tier_summary.iterrows():
        print(f"  {str(row['risk_tier']):<14} {row['n_clients']:>10} "
              f"{row['total_exposure_eur']:>15,.1f} {row['avg_pd_pct']:>9.1f}%")
    print("  └" + "─" * 53 + "┘")

    return df_model


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2C — OUTPUT STORE
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_COLS = [
    "client_id", "company_name", "ticker", "sector", "loan_type",
    "loan_amount_eur", "interest_rate_pct", "loan_to_value",
    "days_past_due", "years_as_client", "num_facilities",
    "debt_to_equity", "current_ratio", "interest_coverage",
    "return_on_equity", "revenue_qoq_change", "zscore_approx",
    "pd_score_pct", "risk_tier", "internal_rating",
    "hidden_risk_flag", "has_defaulted",
]


def save_output(df_scored: pd.DataFrame,
                csv_path: str = "credit_risk_output.csv",
                parquet_path: str = "credit_risk_output.parquet") -> pd.DataFrame:
    """Save the scored portfolio to CSV and Parquet for Verkko ingestion."""
    print("\n[Layer 2C] Saving output store...")

    df_output = df_scored[OUTPUT_COLS].copy()
    df_output = df_output.sort_values("pd_score_pct", ascending=False).reset_index(drop=True)

    # Convert categoricals to string for clean CSV export
    df_output["risk_tier"]       = df_output["risk_tier"].astype(str)
    df_output["internal_rating"] = df_output["internal_rating"].astype(str)

    df_output.to_csv(csv_path, index=False)
    try:
        df_output.to_parquet(parquet_path, index=False)
        print(f"  ✓  Parquet → {parquet_path}")
    except Exception:
        pass  # pyarrow optional

    print(f"  ✓  CSV    → {csv_path}")
    print(f"  ✓  {len(df_output)} clients ready for Verkko ingestion.")
    return df_output


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — VERKKO DEMO QUERIES
# ─────────────────────────────────────────────────────────────────────────────

def run_verkko_demo_queries(df_output: pd.DataFrame) -> None:
    """
    Simulate the 4 Verkko Chat demo queries for the Intesa CFO office.
    These replicate what Verkko would answer when connected to the output store.
    """
    print("\n" + "═" * 70)
    print("  LAYER 3 — VERKKO CHAT DEMO QUERIES")
    print("  (Simulating live Verkko responses for CFO office demo)")
    print("═" * 70)

    # ── Query 1: Portfolio Risk Overview ────────────────────────────────────
    print("\n💬 Query 1: 'Show me a breakdown of our corporate loan portfolio by "
          "risk tier. How many clients are in each category and what is the "
          "total exposure?'")
    print()

    q1 = df_output.groupby("risk_tier").agg(
        num_clients=("client_id", "count"),
        total_exposure_eur_m=("loan_amount_eur", lambda x: round(x.sum() / 1e6, 1)),
        avg_pd_score=("pd_score_pct", lambda x: round(x.mean(), 1)),
    ).reset_index()

    tier_order = ["Low Risk", "Watch List", "Elevated", "High Risk"]
    q1["risk_tier"] = pd.Categorical(q1["risk_tier"], categories=tier_order, ordered=True)
    q1 = q1.sort_values("risk_tier")

    print(f"  {'Risk Tier':<14} {'# Clients':>10} {'Exposure (€M)':>15} {'Avg PD%':>10}")
    print("  " + "─" * 54)
    for _, row in q1.iterrows():
        print(f"  {str(row['risk_tier']):<14} {row['num_clients']:>10} "
              f"{row['total_exposure_eur_m']:>15,.1f} {row['avg_pd_score']:>9.1f}%")

    total_exp = df_output["loan_amount_eur"].sum() / 1e6
    print(f"\n  Total portfolio exposure: €{total_exp:,.1f}M across "
          f"{len(df_output)} corporate clients.")

    # ── Query 2: Hidden Risk (The Killer Demo) ───────────────────────────────
    print("\n" + "─" * 70)
    print("💬 Query 2: 'List all clients whose quarterly revenue declined by "
          "more than 15% but who are still current on their loan payments. "
          "Sort by loan exposure.'")
    print("  [This finds risks current systems CANNOT see — the killer demo]")
    print()

    hidden = df_output[
        (df_output["revenue_qoq_change"] < -0.15) &
        (df_output["days_past_due"] == 0)
    ].sort_values("loan_amount_eur", ascending=False)

    if len(hidden) > 0:
        print(f"  ⚠  ALERT: {len(hidden)} clients with deteriorating revenue "
              f"but ZERO missed payments:")
        print()
        print(f"  {'Company':<28} {'Sector':<20} {'Exposure (€M)':>14} "
              f"{'Rev QoQ':>8} {'PD%':>7} {'Risk Tier':<12}")
        print("  " + "─" * 96)
        for _, row in hidden.head(10).iterrows():
            exp_m = row["loan_amount_eur"] / 1e6
            print(f"  {row['company_name']:<28} {row['sector']:<20} "
                  f"{exp_m:>14,.1f} "
                  f"{row['revenue_qoq_change']:>7.1%} "
                  f"{row['pd_score_pct']:>6.1f}% "
                  f"{str(row['risk_tier']):<12}")
        hidden_exp = hidden["loan_amount_eur"].sum() / 1e6
        print(f"\n  Total hidden-risk exposure: €{hidden_exp:,.1f}M — "
              f"INVISIBLE to traditional DPD-based monitoring.")
    else:
        print("  No hidden risk clients found in this portfolio.")

    # ── Query 3: Sector Concentration Risk ──────────────────────────────────
    print("\n" + "─" * 70)
    print("💬 Query 3: 'What is our total exposure to the Automotive sector? "
          "Show me the PD scores for each client in that sector.'")
    print()

    auto = df_output[
        df_output["sector"].str.contains("Automotive", case=False, na=False)
    ].sort_values("pd_score_pct", ascending=False)

    if len(auto) > 0:
        auto_exp = auto["loan_amount_eur"].sum() / 1e6
        pct_total = auto_exp / total_exp * 100
        print(f"  Automotive sector: {len(auto)} clients | "
              f"Total exposure: €{auto_exp:,.1f}M ({pct_total:.1f}% of portfolio)")
        print()
        print(f"  {'Company':<28} {'Loan Type':<18} {'Exposure (€M)':>14} "
              f"{'PD%':>7} {'Rating':<8} {'Risk Tier':<12}")
        print("  " + "─" * 94)
        for _, row in auto.iterrows():
            exp_m = row["loan_amount_eur"] / 1e6
            print(f"  {row['company_name']:<28} {row['loan_type']:<18} "
                  f"{exp_m:>14,.1f} "
                  f"{row['pd_score_pct']:>6.1f}% "
                  f"{str(row['internal_rating']):<8} "
                  f"{str(row['risk_tier']):<12}")
    else:
        print("  No Automotive sector clients found.")

    # ── Query 4: Client Deep Dive ────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("💬 Query 4: 'Pull up the full credit profile for Stellantis. "
          "Show me their financial ratios, our model's PD score, and the "
          "key risk drivers.'")
    print()

    stellantis = df_output[
        df_output["company_name"].str.contains("Stellantis", case=False, na=False)
    ]

    if len(stellantis) == 0:
        # Fallback: show highest-risk public company
        stellantis = df_output[
            df_output["ticker"].notna()
        ].sort_values("pd_score_pct", ascending=False).head(1)

    if len(stellantis) > 0:
        s = stellantis.iloc[0]
        exp_m = s["loan_amount_eur"] / 1e6
        print(f"  ╔══ CREDIT PROFILE: {s['company_name'].upper()} ══╗")
        print(f"  Client ID:          {s['client_id']}")
        print(f"  Sector:             {s['sector']}")
        print(f"  Ticker:             {s['ticker'] or 'Private'}")
        print()
        print(f"  ── Loan Facility ──────────────────────────────")
        print(f"  Type:               {s['loan_type']}")
        print(f"  Exposure:           €{exp_m:,.1f}M")
        print(f"  Interest Rate:      {s['interest_rate_pct']:.2f}%")
        print(f"  Loan-to-Value:      {s['loan_to_value']:.0%}")
        print(f"  Days Past Due:      {s['days_past_due']}")
        print(f"  Years as Client:    {s['years_as_client']}")
        print()
        print(f"  ── Financial Ratios ───────────────────────────")
        print(f"  Debt / Equity:      {s['debt_to_equity']:.2f}x")
        print(f"  Current Ratio:      {s['current_ratio']:.2f}x")
        print(f"  Interest Coverage:  {s['interest_coverage']:.1f}x")
        print(f"  Return on Equity:   {s['return_on_equity']:.1%}")
        print(f"  Revenue QoQ Δ:      {s['revenue_qoq_change']:+.1%}")
        print(f"  Altman Z-Score:     {s['zscore_approx']:.2f}")
        print()
        print(f"  ── Model Output ───────────────────────────────")
        print(f"  PD Score:           {s['pd_score_pct']:.2f}%")
        print(f"  Internal Rating:    {s['internal_rating']}")
        print(f"  Risk Tier:          {s['risk_tier']}")
        print(f"  Hidden Risk Flag:   {'⚠ YES' if s['hidden_risk_flag'] else '✓ No'}")
        print()

        # Key risk drivers narrative
        drivers = []
        if s["debt_to_equity"] > 2.5:
            drivers.append(f"High leverage (D/E: {s['debt_to_equity']:.1f}x)")
        if s["interest_coverage"] < 3.0:
            drivers.append(f"Thin interest coverage ({s['interest_coverage']:.1f}x)")
        if s["revenue_qoq_change"] < -0.10:
            drivers.append(f"Revenue contraction ({s['revenue_qoq_change']:+.1%} QoQ)")
        if s["current_ratio"] < 1.0:
            drivers.append(f"Liquidity stress (current ratio: {s['current_ratio']:.2f}x)")
        if s["days_past_due"] > 0:
            drivers.append(f"Payment delinquency ({s['days_past_due']} DPD)")
        if s["zscore_approx"] < 1.8:
            drivers.append(f"Distress zone Z-Score ({s['zscore_approx']:.2f})")

        if drivers:
            print(f"  ── Key Risk Drivers ───────────────────────────")
            for d in drivers:
                print(f"  ⚠  {d}")
        else:
            print(f"  ✓  No major risk flags identified.")

    print("\n" + "═" * 70)
    print("  END OF VERKKO DEMO QUERIES")
    print("═" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    use_fmp: bool = True,
    n_clients: int = 200,
    csv_output: str = "credit_risk_output.csv",
) -> pd.DataFrame:
    """
    Full end-to-end pipeline:
      1A → Pull FMP financials (or synthetic fallback)
      1B → Generate synthetic loan portfolio
      2A → Feature engineering
      2B → Train LightGBM PD model
      2C → Score portfolio & save output
      3  → Run Verkko demo queries
    """
    print("\n" + "═" * 70)
    print("  CREDIT RISK EARLY WARNING SYSTEM")
    print("  Intesa Sanpaolo CFO Office Demo | Powered by Verkko")
    print(f"  Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 70)

    # ── Layer 1 ──────────────────────────────────────────────────────────────
    if use_fmp and FMP_API_KEY != "demo":
        df_financials = pull_fmp_financials(TICKERS)
    else:
        if FMP_API_KEY == "demo":
            print("\n[Layer 1A] FMP_API_KEY not set — using synthetic financials.")
        df_financials = _synthetic_financials(TICKERS)

    df_loans = generate_loan_portfolio(TICKERS, n_clients=n_clients)

    # ── Layer 2 ──────────────────────────────────────────────────────────────
    df_model  = engineer_features(df_loans, df_financials)
    model, _  = train_credit_model(df_model)
    df_scored = score_portfolio(df_model, model)
    df_output = save_output(df_scored, csv_path=csv_output)

    # ── Layer 3 ──────────────────────────────────────────────────────────────
    run_verkko_demo_queries(df_output)

    print(f"\n✅ Pipeline complete. Upload '{csv_output}' to Verkko to enable "
          f"live natural-language queries on the full portfolio.")
    return df_output


if __name__ == "__main__":
    df = run_pipeline(use_fmp=True, n_clients=200)
