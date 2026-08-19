# Reusable orchestration helpers for the Streamlit dashboard

from __future__ import annotations

import pandas as pd

from causal_model import (
    aipw_ate,
    business_recommendation,
    causal_train_test_split,
    fit_propensity_model,
    portfolio_summary,
    predict_uplift,
    propensity_score_ate,
    train_baseline_model,
    train_uplift_model,
)
from data_prep import (
    fit_feature_encoder,
    load_and_prepare,
    transform_causal_features,
)


def get_data():
    return load_and_prepare()


def prepare_holdout(df_clean, T, Y, test_size=0.25, seed=42):
    train_idx, eval_idx = causal_train_test_split(
        T, Y, test_size=test_size, seed=seed
    )

    # Split raw rows before fitting the encoder
    df_train = df_clean.iloc[train_idx].reset_index(drop=True)
    df_eval = (
        df_clean.iloc[eval_idx]
        .copy()
        .reset_index(names="original_row")
    )

    T_train = T.iloc[train_idx].reset_index(drop=True)
    Y_train = Y.iloc[train_idx].reset_index(drop=True)
    T_eval = T.iloc[eval_idx].reset_index(drop=True)
    Y_eval = Y.iloc[eval_idx].reset_index(drop=True)

    # Learn categorical schema from training data only
    encoder = fit_feature_encoder(df_train)

    X_train = transform_causal_features(
        df_train, encoder
    ).reset_index(drop=True)

    X_eval = transform_causal_features(
        df_eval, encoder
    ).reset_index(drop=True)

    baseline = train_baseline_model(
        X_train, Y_train, X_eval, Y_eval
    )
    model_t, model_c = train_uplift_model(
        X_train, T_train, Y_train
    )
    propensity_model = fit_propensity_model(X_train, T_train)
    p1, p0, uplift = predict_uplift(
        model_t, model_c, X_eval
    )

    return {
        "train_idx": train_idx,
        "eval_idx": eval_idx,
        "df_train": df_train,
        "df_eval": df_eval,
        "X_train": X_train,
        "T_train": T_train,
        "Y_train": Y_train,
        "X_eval": X_eval,
        "T_eval": T_eval,
        "Y_eval": Y_eval,
        "encoder": encoder,
        "baseline": baseline,
        "model_treated": model_t,
        "model_control": model_c,
        "propensity_model": propensity_model,
        "p1": p1,
        "p0": p0,
        "uplift": uplift,
    }


def build_results_df(df_eval, p0, p1, uplift, avg_purchase_value, coupon_cost):
    # Combining held-out raw data, causal predictions, and business value
    business = business_recommendation(uplift, avg_purchase_value, coupon_cost)
    return pd.DataFrame(
        {
            "original_row": df_eval["original_row"].to_numpy(),
            "coupon_type": df_eval["coupon"].to_numpy(),
            "destination": df_eval["destination"].to_numpy(),
            "income": df_eval["income"].to_numpy(),
            "p_purchase_weak_coupon": p0,
            "p_purchase_better_coupon": p1,
            "uplift": uplift,
            "expected_incremental_revenue": business[
                "expected_incremental_revenue"
            ],
            "net_value": business["net_value"],
            "recommend_better_coupon": business["recommend"],
        }
    )


def filter_results(results_df, selected_coupons, selected_destinations, selected_incomes):
    return results_df[
        results_df["coupon_type"].isin(selected_coupons)
        & results_df["destination"].isin(selected_destinations)
        & results_df["income"].isin(selected_incomes)
    ].copy()


def apply_targeting(filtered_df, avg_purchase_value, coupon_cost, strategy="Value threshold", budget=None):
    # Applying threshold targeting or top-uplift targeting under a budget cap
    budget_value = budget if strategy == "Budget cap" else None
    business = business_recommendation(
        filtered_df["uplift"].to_numpy(),
        avg_purchase_value,
        coupon_cost,
        budget=budget_value,
    )
    result = filtered_df.copy()
    result["expected_incremental_revenue"] = business[
        "expected_incremental_revenue"
    ]
    result["net_value"] = business["net_value"]
    result["recommend_better_coupon"] = business["recommend"]
    return result


def get_portfolio_summary(targeted_df, coupon_cost):
    business = {
        "expected_incremental_revenue": targeted_df[
            "expected_incremental_revenue"
        ].to_numpy(),
        "recommend": targeted_df["recommend_better_coupon"].to_numpy(),
    }
    return portfolio_summary(business, coupon_cost)


def get_causal_diagnostics(bundle):
    # Compute held-out T-learner, PSM, and doubly-robust AIPW estimates
    ate_psm, ps_diagnostics = propensity_score_ate(
    bundle["X_eval"],
    bundle["T_eval"],
    bundle["Y_eval"],
    bundle["propensity_model"],
    caliper=0.20,
    return_diagnostics=True,)

    ate_aipw = aipw_ate(
        bundle["X_eval"],
        bundle["T_eval"],
        bundle["Y_eval"],
        bundle["model_treated"],
        bundle["model_control"],
        bundle["propensity_model"],)
    
    return {
        "avg_uplift": float(bundle["uplift"].mean()),
        "ate_psm": ate_psm,
        "ate_aipw": ate_aipw,
        "propensity": ps_diagnostics,
    }


def get_top_customers(filtered_df, n_show):
    return filtered_df.sort_values("uplift", ascending=False).head(n_show)


def get_recommended_csv_bytes(filtered_df):
    return filtered_df[filtered_df["recommend_better_coupon"]].to_csv(
        index=False
    ).encode("utf-8")


def run_whatif(model_treated, model_control, X_eval, row_id, what_if_age, what_if_income):
    X_whatif = X_eval.loc[[row_id]].copy()
    X_whatif["age"] = what_if_age
    X_whatif["income"] = what_if_income
    return predict_uplift(model_treated, model_control, X_whatif)
