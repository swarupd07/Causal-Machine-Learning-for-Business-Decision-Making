"""
causal_model.py
-----------------
This module contains three things:

  1. A TRADITIONAL predictive model  -> "Will this customer purchase?"
     (On its own it's misleading for targeting decisions, because it just finds customers who were going to buy anyway.)

  2. A CAUSAL / UPLIFT model (T-learner)  -> "How much MORE likely is this
     customer to purchase BECAUSE of the better coupon?"
     This is the number a business actually needs for targeting.

  3. Business decision functions -> turn the uplift number into a
     recommendation (send the better coupon or not) using simple ROI math,
     plus a Propensity Score Matching cross-check of the overall effect.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------
# 1. Traditional predictive model (baseline - ignores treatment)
# ---------------------------------------------------------------------
def train_baseline_model(X, Y):
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.25, random_state=42, stratify=Y
    )
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"Baseline model (predicts purchase, ignores treatment) AUC: {test_auc:.3f}")
    return model


# ---------------------------------------------------------------------
# 2. Causal uplift model: T-learner
# ---------------------------------------------------------------------
def train_uplift_model(X, T, Y):
    
    X_treated, Y_treated = X[T == 1], Y[T == 1]
    X_control, Y_control = X[T == 0], Y[T == 0]

    model_treated = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model_control = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)

    model_treated.fit(X_treated, Y_treated)
    model_control.fit(X_control, Y_control)

    return model_treated, model_control


def predict_uplift(model_treated, model_control, X):
    
    p1 = model_treated.predict_proba(X)[:, 1]
    p0 = model_control.predict_proba(X)[:, 1]
    uplift = p1 - p0
    return p1, p0, uplift


# ---------------------------------------------------------------------
# Propensity Score Matching -> statistical cross-check of the AVERAGE
# effect (sanity check that the T-learner's average uplift is believable)
# ---------------------------------------------------------------------
def propensity_score_ate(X, T, Y):
    model = LogisticRegression(max_iter=2000)
    model.fit(X, T)
    ps = np.clip(model.predict_proba(X)[:, 1], 0.01, 0.99).reshape(-1, 1)

    T = np.asarray(T)
    Y = np.asarray(Y)
    treated_ps, control_ps = ps[T == 1], ps[T == 0]
    treated_Y, control_Y = Y[T == 1], Y[T == 0]

    nn = NearestNeighbors(n_neighbors=1).fit(control_ps)
    _, match_idx = nn.kneighbors(treated_ps)
    matched_control_Y = control_Y[match_idx.flatten()]

    return float(np.mean(treated_Y - matched_control_Y))


# ---------------------------------------------------------------------
# 3. Business metrics: turn uplift into a $ recommendation
# ---------------------------------------------------------------------
def business_recommendation(uplift, avg_purchase_value=15.0, coupon_cost=2.0):
    
    expected_incremental_revenue = uplift * avg_purchase_value
    net_value = expected_incremental_revenue - coupon_cost
    recommend = net_value > 0
    return {
        "uplift": uplift,
        "expected_incremental_revenue": expected_incremental_revenue,
        "net_value": net_value,
        "recommend": recommend,
    }


def portfolio_summary(business_result, coupon_cost=2.0):
    """
    Aggregate business metrics across all customers:
      how many customers should get the better coupon,
      total incremental revenue, total cost, overall ROI.
    """
    recommend = business_result["recommend"]
    n_recommend = int(recommend.sum())
    total_incremental_revenue = float(
        business_result["expected_incremental_revenue"][recommend].sum()
    )
    total_cost = n_recommend * coupon_cost
    roi = (total_incremental_revenue - total_cost) / total_cost if total_cost > 0 else 0.0

    return {
        "customers_targeted": n_recommend,
        "total_incremental_revenue": total_incremental_revenue,
        "total_cost": total_cost,
        "roi": roi,
    }
