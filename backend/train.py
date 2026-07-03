"""
Train a conversion model on pairs.csv.

Stage 1: Logistic Regression (interpretable baseline).
We deliberately inspect the learned weights to confirm the model
recovered the hidden dynamics we planted in the generator.

Target (y):   converted  (0/1)
Features (X): lead + offer + interaction columns
Dropped:      ids, true_p (scaffolding), and raw text we don't model
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# ----------------------------------------------------------------------
# 1. LOAD + SELECT COLUMNS
# ----------------------------------------------------------------------
df = pd.read_csv("pairs.csv")

# y = the label. Everything the model is allowed to "see" goes in X.
y = df["converted"]

CATEGORICAL = ["source_platform", "intent_category", "geo_state",
               "device", "offer_category", "intent_match"]
# payout REMOVED from model features (not causal to the consumer; lives in EV only).
# commitment_level added as the causal offer feature. Treated as numeric here
# so logistic regression can use its ordinal structure directly.
NUMERIC = ["days_since_signup", "total_opens", "days_since_last_open", "commitment_level"]

X = df[CATEGORICAL + NUMERIC]

# NOTE: lead_id, offer_id, email, true_p, converted are NOT in X.
# true_p especially would be cheating — it's the answer key.

# ----------------------------------------------------------------------
# 2. TRAIN / TEST SPLIT
# ----------------------------------------------------------------------
# stratify=y keeps the ~6% conversion rate balanced across both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ----------------------------------------------------------------------
# 3. PREPROCESS + MODEL  (one Pipeline so nothing leaks)
# ----------------------------------------------------------------------
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
    ("num", StandardScaler(), NUMERIC),
])

model = Pipeline([
    ("prep", preprocess),
    # class_weight handles the imbalance: a "convert" example is rare,
    # so we tell the model to care about it ~16x more than a non-convert.
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

model.fit(X_train, y_train)

# ----------------------------------------------------------------------
# 4. EVALUATE  (accuracy is useless at 6% base rate -> use ranking metrics)
# ----------------------------------------------------------------------
proba = model.predict_proba(X_test)[:, 1]

auc   = roc_auc_score(y_test, proba)
ap    = average_precision_score(y_test, proba)   # area under precision-recall
brier = brier_score_loss(y_test, proba)          # calibration (lower=better)

print("=" * 55)
print("LOGISTIC REGRESSION — test metrics")
print("=" * 55)
print(f"ROC AUC          : {auc:.3f}   (0.5=random, 1.0=perfect)")
print(f"Avg Precision    : {ap:.3f}   (base rate = {y.mean():.3f})")
print(f"Brier score      : {brier:.4f}  (calibration, lower better)")

# ----------------------------------------------------------------------
# 5. INSPECT WEIGHTS  (did we recover the planted dynamics?)
# ----------------------------------------------------------------------
feat_names = model.named_steps["prep"].get_feature_names_out()
coefs = model.named_steps["clf"].coef_[0]
weights = pd.Series(coefs, index=feat_names).sort_values()

print("\n" + "=" * 55)
print("LEARNED WEIGHTS — strongest negative -> strongest positive")
print("=" * 55)
print("\nTop 8 POSITIVE (push conversion UP):")
print(weights.tail(8).iloc[::-1].to_string())
print("\nTop 8 NEGATIVE (push conversion DOWN):")
print(weights.head(8).to_string())

# ----------------------------------------------------------------------
# 6. PRECISION@K  (the metric that matters for ranking offers)
# ----------------------------------------------------------------------
# If we act on the model's top-ranked predictions, how many convert?
order = np.argsort(proba)[::-1]
y_sorted = y_test.values[order]
for k in [100, 500, 1000]:
    print(f"\nPrecision@{k:<5}: {y_sorted[:k].mean():.3f}  "
          f"(vs {y.mean():.3f} baseline -> {y_sorted[:k].mean()/y.mean():.1f}x lift)")
