# Causal estimators and business rules for coupon targeting.

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


PROPENSITY_CLIP = (0.01, 0.99)


def causal_train_test_split(X, T, Y, test_size=0.25, seed=42):
    # Spliting data into train/evaluation sets and also preserving treatment outcome proportions.
    joint = pd.Series(T).astype(str) + "_" + pd.Series(Y).astype(str)
    indices = np.arange(len(Y))
    train_idx, eval_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=joint
    )
    return np.sort(train_idx), np.sort(eval_idx)


def train_baseline_model(X_train, Y_train, X_eval=None, Y_eval=None):
    # Fiting baseline model to predict purchase outcome without treatment information.
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train, Y_train)
    if X_eval is not None and Y_eval is not None:
        auc = roc_auc_score(Y_eval, model.predict_proba(X_eval)[:, 1])
        print(f"Held-out baseline purchase AUC: {auc:.3f}")
    return model


def train_uplift_model( X, T, Y, n_estimators=100, max_depth=6, random_state=42):
    # Fiting separate treated/control outcome models ( T-learner jise hum kehte hain ) to estimate individual uplift.
    T_array = np.asarray(T)
    Y_array = np.asarray(Y)

    model_treated = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1,
    )
    model_control = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=10,
        random_state=random_state + 1,
        n_jobs=-1,
    )
    model_treated.fit(X.iloc[T_array == 1], Y_array[T_array == 1])
    model_control.fit(X.iloc[T_array == 0], Y_array[T_array == 0])
    return model_treated, model_control


def _positive_probability(model, X):
    # model ne predicted probability for the positive class 1
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    return probabilities[:, classes.index(1)]


def predict_uplift(model_treated, model_control, X):
    # Predict individual uplift by subtracting control from treated predicted probabilities.
    p1 = _positive_probability(model_treated, X)
    p0 = _positive_probability(model_control, X)
    return p1, p0, p1 - p0


def fit_propensity_model(X, T):
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X, T)
    return model


def propensity_diagnostics(propensity_model, X, clip=PROPENSITY_CLIP):
    # Measure how much estimation depends on propensity clipping.
    raw = propensity_model.predict_proba(X)[:, 1]
    low, high = clip
    clipped_mask = (raw < low) | (raw > high)
    return {
        "raw_scores": raw,
        "n_clipped": int(clipped_mask.sum()),
        "fraction_clipped": float(clipped_mask.mean()),
        "min_propensity": float(raw.min()),
        "max_propensity": float(raw.max()),
    }


def propensity_score_ate(X, T, Y, propensity_model=None, clip=PROPENSITY_CLIP, return_diagnostics=False):
    # Estimating ATT by nearest-neighbour propensity-score matching.
    if propensity_model is None:
        propensity_model = fit_propensity_model(X, T)
    diagnostics = propensity_diagnostics(propensity_model, X, clip=clip)
    ps = np.clip(diagnostics["raw_scores"], *clip).reshape(-1, 1)
    T_array = np.asarray(T)
    Y_array = np.asarray(Y)

    treated_ps, control_ps = ps[T_array == 1], ps[T_array == 0]
    treated_y, control_y = Y_array[T_array == 1], Y_array[T_array == 0]
    if len(treated_ps) == 0 or len(control_ps) == 0:
        raise ValueError("PSM requires treated and control rows in the evaluation set.")

    nearest = NearestNeighbors(n_neighbors=1).fit(control_ps)
    _, match_idx = nearest.kneighbors(treated_ps)
    estimate = float(np.mean(treated_y - control_y[match_idx.ravel()]))
    if return_diagnostics:
        clean_diagnostics = {
            key: value for key, value in diagnostics.items() if key != "raw_scores"
        }
        return estimate, clean_diagnostics
    return estimate


def aipw_ate(
    X, T, Y, model_treated, model_control, propensity_model, clip=PROPENSITY_CLIP):
   # Compute the doubly-robust AIPW average treatment effect
    T_array = np.asarray(T, dtype=float)
    Y_array = np.asarray(Y, dtype=float)
    mu1, mu0, _ = predict_uplift(model_treated, model_control, X)
    ps = np.clip(propensity_model.predict_proba(X)[:, 1], *clip)
    scores = ( T_array * (Y_array - mu1) / ps - (1.0 - T_array) * (Y_array - mu0) / (1.0 - ps) + mu1 - mu0 )
    return float(np.mean(scores))


def coupon_expiration_overlap(df):
    # Returns within-coupon expiration proportions and >90% warning rows
    table = pd.crosstab(df["coupon"], df["expiration"], normalize="index")
    dominant = table.max(axis=1)
    warnings = pd.DataFrame(
        {
            "coupon": dominant.index,
            "dominant_share": dominant.values,
            "flag_over_90pct": dominant.values > 0.90,
        }
    )
    return table, warnings


def treatment_balance(T):
    shares = pd.Series(T).value_counts(normalize=True).sort_index()
    return {
        "control_share": float(shares.get(0, 0.0)),
        "treated_share": float(shares.get(1, 0.0)),
    }


def _stratified_bootstrap_indices(T, rng):
    # Resampling each treatment arm separately, preserving arm sizes
    T_array = np.asarray(T)
    sampled = []
    for arm in (0, 1):
        arm_idx = np.flatnonzero(T_array == arm)
        if len(arm_idx) == 0:
            raise ValueError("A stratified bootstrap requires both treatment arms.")
        sampled.append(rng.choice(arm_idx, size=len(arm_idx), replace=True))
    combined = np.concatenate(sampled)
    rng.shuffle(combined)
    return combined


def bootstrap_causal_estimates(X_train, T_train, Y_train, X_eval, T_eval, Y_eval, n_boot=100, seed=42, bootstrap_trees=50):
    # Arm-stratified CIs for held-out T-learner, PSM, and AIPW estimates
    rng = np.random.default_rng(seed)
    estimates = {"t_learner": [], "psm": [], "aipw": []}

    for iteration in range(n_boot):
        train_idx = _stratified_bootstrap_indices(T_train, rng)
        eval_idx = _stratified_bootstrap_indices(T_eval, rng)
        X_b = X_train.iloc[train_idx].reset_index(drop=True)
        T_b = pd.Series(T_train).iloc[train_idx].reset_index(drop=True)
        Y_b = pd.Series(Y_train).iloc[train_idx].reset_index(drop=True)
        X_e = X_eval.iloc[eval_idx].reset_index(drop=True)
        T_e = pd.Series(T_eval).iloc[eval_idx].reset_index(drop=True)
        Y_e = pd.Series(Y_eval).iloc[eval_idx].reset_index(drop=True)

        model_t, model_c = train_uplift_model(
            X_b,
            T_b,
            Y_b,
            n_estimators=bootstrap_trees,
            random_state=seed + iteration * 2,
        )
        ps_model = fit_propensity_model(X_b, T_b)
        _, _, uplift = predict_uplift(model_t, model_c, X_e)
        estimates["t_learner"].append(float(np.mean(uplift)))
        estimates["psm"].append(propensity_score_ate(X_e, T_e, Y_e, ps_model))
        estimates["aipw"].append(
            aipw_ate(X_e, T_e, Y_e, model_t, model_c, ps_model)
        )

    return {
        name: tuple(np.percentile(values, [2.5, 97.5]).astype(float))
        for name, values in estimates.items()
    }


def _bootstrap_wrapper(X, T, Y, estimator_name, n_boot=100, seed=42):
    train_idx, eval_idx = causal_train_test_split(X, T, Y, seed=seed)
    result = bootstrap_causal_estimates(
        X.iloc[train_idx],
        pd.Series(T).iloc[train_idx],
        pd.Series(Y).iloc[train_idx],
        X.iloc[eval_idx],
        pd.Series(T).iloc[eval_idx],
        pd.Series(Y).iloc[eval_idx],
        n_boot=n_boot,
        seed=seed,
    )
    return np.asarray(result[estimator_name])


def bootstrap_ate_ps(X, T, Y, n_boot=100, seed=42):
    return _bootstrap_wrapper(X, T, Y, "psm", n_boot=n_boot, seed=seed)


def bootstrap_ate_T(X, T, Y, n_boot=100, seed=42):
    return _bootstrap_wrapper(X, T, Y, "t_learner", n_boot=n_boot, seed=seed)


def business_recommendation(
    uplift, avg_purchase_value=15.0, coupon_cost=2.0, budget=None
):
    # Computing expected incremental revenue, net value, and whether to recommend coupon targeting
    uplift = np.asarray(uplift, dtype=float)
    expected_incremental_revenue = uplift * avg_purchase_value
    net_value = expected_incremental_revenue - coupon_cost
    profitable = net_value > 0

    if budget is None:
        recommend = profitable
    else:
        recommend = np.zeros(len(uplift), dtype=bool)
        capacity = max(0, int(np.floor(float(budget) / coupon_cost)))
        order = np.argsort(-net_value)
        selected = order[profitable[order]][:capacity]
        recommend[selected] = True

    return {
        "uplift": uplift,
        "expected_incremental_revenue": expected_incremental_revenue,
        "net_value": net_value,
        "recommend": recommend,
    }


def portfolio_summary(business_result, coupon_cost=2.0):
    recommend = np.asarray(business_result["recommend"], dtype=bool)
    n_recommend = int(recommend.sum())
    total_incremental_revenue = float(
        np.asarray(business_result["expected_incremental_revenue"])[recommend].sum()
    )
    total_cost = n_recommend * coupon_cost
    roi = (total_incremental_revenue - total_cost) / total_cost if total_cost else 0.0
    return {
        "customers_targeted": n_recommend,
        "total_incremental_revenue": total_incremental_revenue,
        "total_cost": total_cost,
        "roi": roi,
    }


def qini_curve(Y, T, uplift, propensity_scores=None):
    # To build a held-out inverse-propensity-weighted cumulative gain curve
    Y_array = np.asarray(Y, dtype=float)
    T_array = np.asarray(T, dtype=float)
    uplift_array = np.asarray(uplift, dtype=float)
    if propensity_scores is None:
        propensity_scores = np.repeat(T_array.mean(), len(T_array))
    ps = np.clip(np.asarray(propensity_scores, dtype=float), *PROPENSITY_CLIP)
    transformed = T_array * Y_array / ps - (1.0 - T_array) * Y_array / (1.0 - ps)
    order = np.argsort(-uplift_array)
    cumulative_gain = np.cumsum(transformed[order]) / len(Y_array)
    fraction = np.arange(1, len(Y_array) + 1) / len(Y_array)
    return pd.DataFrame(
        {
            "fraction_targeted": np.concatenate([[0.0], fraction]),
            "model_gain": np.concatenate([[0.0], cumulative_gain]),
            "random_gain": np.concatenate([[0.0], fraction * cumulative_gain[-1]]),
        }
    )


def subgroup_uplift(df, group_col="coupon_type", min_size=30):
    # Summarize held-out model uplift by a human-readable segment
    summary = (
        df.groupby(group_col, observed=True)["uplift"]
        .agg(mean_uplift="mean", customers="size")
        .reset_index()
    )
    return summary[summary["customers"] >= min_size].sort_values(
        "mean_uplift", ascending=False
    )
