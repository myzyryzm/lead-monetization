"""
The tool layer: turn the trained model into an offer-ranking engine.

Flow for one lead:
  lead  -->  score against EVERY offer  -->  EV = P(convert) * payout
        -->  rank  -->  top-K offers to send

This module exposes:
  train_model()                 -> fits LR, returns the fitted pipeline
  recommend(model, lead, offers, k)  -> ranked offers with prob, EV, rationale

The `rationale` here is a simple rule-based stub. In the full build this
seam is where the LLM writes the human explanation + the email/SMS copy.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

CATEGORICAL = ["source_platform", "intent_category", "geo_state",
               "device", "offer_category", "intent_match"]
# payout is NOT a model feature (not causal to consumer) — used only for EV.
NUMERIC = ["days_since_signup", "total_opens", "days_since_last_open", "commitment_level"]

# Match matrix (same as the generator) so we can compute intent_match
# for any lead x offer pairing at prediction time.
MATCH = {
    "finance":       {"finance":"match","insurance":"related","health":"mismatch","sweepstakes":"related","home_services":"related"},
    "insurance":     {"finance":"related","insurance":"match","health":"mismatch","sweepstakes":"related","home_services":"related"},
    "health":        {"finance":"mismatch","insurance":"mismatch","health":"match","sweepstakes":"related","home_services":"mismatch"},
    "sweepstakes":   {"finance":"mismatch","insurance":"mismatch","health":"mismatch","sweepstakes":"match","home_services":"mismatch"},
    "home_services": {"finance":"related","insurance":"related","health":"mismatch","sweepstakes":"related","home_services":"match"},
}


def train_model(pairs_path="pairs.csv"):
    df = pd.read_csv(pairs_path)
    y = df["converted"]
    X = df[CATEGORICAL + NUMERIC]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    base = Pipeline([
        ("prep", ColumnTransformer([
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ])),
        ("clf", LogisticRegression(max_iter=1000)),   # no class_weight -> calibratable
    ])
    # isotonic calibration -> predict_proba returns honest probabilities,
    # so expected_value (prob * payout) is in real dollars. Ranking preserved.
    model = CalibratedClassifierCV(base, method="isotonic", cv=5)
    model.fit(X_train, y_train)
    return model


def _score_leads_vectorized(model, offer, leads_df):
    """Score EVERY lead against one fixed `offer` in a single predict_proba call.

    The offer's category and commitment_level are constant across leads, so we
    build one feature matrix and score in a single pass instead of looping one
    lead at a time. Returns (probs, matches): a numpy array of P(convert) and a
    pandas Series of intent_match labels, positionally aligned to leads_df rows.
    """
    cat = offer["category"]
    matches = leads_df["intent_category"].map(lambda ic: MATCH[ic][cat])
    feat = pd.DataFrame({
        "source_platform": leads_df["source_platform"].values,
        "intent_category": leads_df["intent_category"].values,
        "geo_state": leads_df["geo_state"].values,
        "device": leads_df["device"].values,
        "offer_category": cat,
        "intent_match": matches.values,
        "days_since_signup": leads_df["days_since_signup"].values,
        "total_opens": leads_df["total_opens"].values,
        "days_since_last_open": leads_df["days_since_last_open"].values,
        "commitment_level": offer["commitment_level"],
    })
    probs = model.predict_proba(feat[CATEGORICAL + NUMERIC])[:, 1]
    return probs, matches


def _rationale(lead, offer, match):
    """Rule-based placeholder. The LLM replaces this in the full build."""
    bits = [f"{match} (lead intent '{lead['intent_category']}' vs offer '{offer['category']}')"]
    if lead["days_since_last_open"] <= 7:
        bits.append("recently engaged")
    elif lead["days_since_last_open"] > 90:
        bits.append("dormant — re-engagement risk")
    if lead["source_platform"] == "google_search":
        bits.append("high-intent source")
    return "; ".join(bits)


def recommend(model, lead: dict, offers: pd.DataFrame, k=3):
    """Score `lead` against all `offers`, return top-k ranked by expected value."""
    rows = []
    for _, offer in offers.iterrows():
        match = MATCH[lead["intent_category"]][offer["category"]]
        feat = {
            "source_platform": lead["source_platform"],
            "intent_category": lead["intent_category"],
            "geo_state": lead["geo_state"],
            "device": lead["device"],
            "offer_category": offer["category"],
            "intent_match": match,
            "days_since_signup": lead["days_since_signup"],
            "total_opens": lead["total_opens"],
            "days_since_last_open": lead["days_since_last_open"],
            "commitment_level": offer["commitment_level"],
        }
        prob = model.predict_proba(pd.DataFrame([feat]))[0, 1]
        rows.append({
            "offer_id": offer["offer_id"],
            "offer_name": offer["offer_name"],
            "category": offer["category"],
            "payout": offer["payout"],
            "prob_convert": round(float(prob), 4),
            "expected_value": round(float(prob) * offer["payout"], 2),
            "rationale": _rationale(lead, offer, match),
        })
    ranked = pd.DataFrame(rows).sort_values("expected_value", ascending=False)
    return ranked.head(k).reset_index(drop=True)


if __name__ == "__main__":
    model = train_model()
    leads = pd.read_csv("leads.csv")
    offers = pd.read_csv("offers.csv")

    # Demo on a few contrasting leads
    samples = {
        "finance lead": leads[leads.intent_category == "finance"].iloc[0],
        "sweepstakes lead": leads[leads.intent_category == "sweepstakes"].iloc[0],
        "health lead": leads[leads.intent_category == "health"].iloc[0],
    }
    for label, lead in samples.items():
        print("\n" + "=" * 70)
        print(f"{label.upper()}  | source={lead.source_platform} "
              f"opted_into={lead.opted_into} "
              f"days_since_last_open={lead.days_since_last_open}")
        print("=" * 70)
        recs = recommend(model, lead.to_dict(), offers, k=3)
        print(recs[["offer_name","category","payout","prob_convert","expected_value"]].to_string(index=False))
        print("top pick rationale:", recs.iloc[0]["rationale"])
