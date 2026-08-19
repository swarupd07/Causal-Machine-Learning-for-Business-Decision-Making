# Streamlit dashboard for held-out causal coupon targeting.

import pandas as pd
import plotly.express as px
import streamlit as st

from backend import (
    apply_targeting,
    build_results_df,
    filter_results,
    get_causal_diagnostics,
    get_data,
    get_portfolio_summary,
    get_recommended_csv_bytes,
    prepare_holdout,
    run_whatif,
)
from causal_model import (
    bootstrap_causal_estimates,
    coupon_expiration_overlap,
    qini_curve,
    subgroup_uplift,
    treatment_balance,
)


st.set_page_config(page_title="Causal Coupon Targeting", layout="wide")
st.title("Causal ML for Coupon Targeting")
st.caption(
    "Which customers should receive a 1-day coupon instead of a 2-hour coupon? "
    "Every reported causal estimate and targeting result below is computed on "
    "a held-out evaluation set, not the rows used to train the models."
)


@st.cache_data
def load_cached_data():
    return get_data()


@st.cache_resource
def fit_cached_holdout(df_clean, T, Y):
    return prepare_holdout(
        df_clean, T, Y, test_size=0.25, seed=42
    )


@st.cache_data(show_spinner=False)
def get_cached_bootstrap_cis(X_train, T_train, Y_train, X_eval, T_eval, Y_eval):
    return bootstrap_causal_estimates(
        X_train,
        T_train,
        Y_train,
        X_eval,
        T_eval,
        Y_eval,
        n_boot=100,
        seed=42,
        bootstrap_trees=50,
    )


df_clean, T, Y = load_cached_data()
bundle = fit_cached_holdout(df_clean, T, Y)
diagnostics = get_causal_diagnostics(bundle)

with st.spinner("Computing cached, stratified bootstrap confidence intervals..."):
    confidence_intervals = get_cached_bootstrap_cis(
        bundle["X_train"],
        bundle["T_train"],
        bundle["Y_train"],
        bundle["X_eval"],
        bundle["T_eval"],
        bundle["Y_eval"],
    )


st.sidebar.header("Business assumptions")
avg_purchase_value = st.sidebar.slider(
    "Average purchase value ($)", 5.0, 50.0, 15.0, 1.0
)
coupon_cost = st.sidebar.slider(
    "Cost of sending the better coupon ($)", 0.5, 10.0, 2.0, 0.5
)
targeting_strategy = st.sidebar.radio(
    "Targeting rule", ["Value threshold", "Budget cap"]
)

base_results = build_results_df(
    bundle["df_eval"],
    bundle["p0"],
    bundle["p1"],
    bundle["uplift"],
    avg_purchase_value,
    coupon_cost,
)

st.sidebar.header("Filter held-out customers")
coupon_options = sorted(base_results["coupon_type"].unique())
destination_options = sorted(base_results["destination"].unique())
income_options = sorted(base_results["income"].unique())
selected_coupons = st.sidebar.multiselect(
    "Coupon type", coupon_options, default=coupon_options
)
selected_destinations = st.sidebar.multiselect(
    "Destination", destination_options, default=destination_options
)
selected_incomes = st.sidebar.multiselect(
    "Income bracket", income_options, default=income_options
)

filtered_df = filter_results(
    base_results, selected_coupons, selected_destinations, selected_incomes
)
budget = None
if targeting_strategy == "Budget cap":
    max_budget = max(coupon_cost, float(len(filtered_df) * coupon_cost))
    budget = st.sidebar.number_input(
        "Campaign budget ($)",
        min_value=0.0,
        max_value=max_budget,
        value=min(1000.0, max_budget),
        step=float(coupon_cost),
    )

targeted_df = apply_targeting(
    filtered_df,
    avg_purchase_value,
    coupon_cost,
    strategy=targeting_strategy,
    budget=budget,
)
summary = get_portfolio_summary(targeted_df, coupon_cost)


tab_overview, tab_top, tab_lookup = st.tabs(
    ["Overview & Diagnostics", "Top Customers", "Customer Lookup"]
)

with tab_overview:
    st.caption(
        f"Showing {len(targeted_df):,} filtered customers from a "
        f"{len(bundle['X_eval']):,}-row held-out evaluation set."
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers to target", f"{summary['customers_targeted']:,}")
    col2.metric(
        "Incremental revenue", f"${summary['total_incremental_revenue']:,.0f}"
    )
    col3.metric("Campaign cost", f"${summary['total_cost']:,.0f}")
    col4.metric("Campaign ROI", f"{summary['roi'] * 100:,.1f}%")

    st.divider()
    st.subheader("Held-out causal estimates")
    m1, m2, m3 = st.columns(3)
    t_ci = confidence_intervals["t_learner"]
    psm_ci = confidence_intervals["psm"]
    aipw_ci = confidence_intervals["aipw"]
    m1.metric(
        "T-learner average uplift",
        f"{diagnostics['avg_uplift']:+.3f}",
        f"95% CI [{t_ci[0]:+.3f}, {t_ci[1]:+.3f}]",
    )
    m2.metric(
        "Propensity matching ATT",
        f"{diagnostics['ate_psm']:+.3f}",
        f"95% CI [{psm_ci[0]:+.3f}, {psm_ci[1]:+.3f}]",
    )
    m3.metric(
        "Doubly-robust AIPW ATE",
        f"{diagnostics['ate_aipw']:+.3f}",
        f"95% CI [{aipw_ci[0]:+.3f}, {aipw_ci[1]:+.3f}]",
    )
    st.caption(
        "The estimators use different assumptions. Agreement is reassuring, "
        "but it does not by itself prove that all confounding has been removed."
    )

    with st.expander("Positivity, overlap, and treatment-balance checks", expanded=True):
        balance = treatment_balance(T)
        b1, b2, b3 = st.columns(3)
        b1.metric("Control share (2h)", f"{balance['control_share']:.1%}")
        b2.metric("Treated share (1d)", f"{balance['treated_share']:.1%}")
        b3.metric(
            "Propensities clipped",
            f"{diagnostics['propensity']['n_clipped']:,} "
            f"({diagnostics['propensity']['fraction_clipped']:.1%})",
        )

        overlap_table, overlap_flags = coupon_expiration_overlap(df_clean)
        flagged = overlap_flags[overlap_flags["flag_over_90pct"]]
        if len(flagged):
            names = ", ".join(flagged["coupon"].astype(str))
            st.warning(
                "Potential positivity concern: these coupon types exceed 90% in "
                f"one expiration arm: {names}. Interpret or target them separately."
            )
        else:
            st.success(
                "No coupon type exceeds 90% concentration in a single expiration arm."
            )
        if diagnostics["propensity"]["n_clipped"]:
            st.warning(
                "Some held-out rows reached the 0.01/0.99 propensity clipping "
                "bounds and therefore have weak empirical counterfactual support."
            )

        overlap_long = (
            overlap_table.reset_index()
            .melt(id_vars="coupon", var_name="expiration", value_name="share")
        )
        fig_overlap = px.bar(
            overlap_long,
            x="coupon",
            y="share",
            color="expiration",
            barmode="group",
            title="Expiration mix within each coupon type",
            labels={"share": "Within-coupon share"},
        )
        fig_overlap.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_overlap, use_container_width=True)

        st.markdown("#### Propensity-score matching quality")

        psm_diag = diagnostics["propensity"]

        p1, p2, p3, p4 = st.columns(4)

        p1.metric(
            "PSM match rate",
            f"{psm_diag['match_rate']:.1%}",
        )

        p2.metric(
            "Treated unmatched",
            f"{psm_diag['n_unmatched']:,}",
        )

        p3.metric(
            "Max |SMD| before",
            f"{psm_diag['max_abs_smd_before']:.3f}",
        )

        p4.metric(
            "Max |SMD| after",
            f"{psm_diag['max_abs_smd_after']:.3f}",
        )

        st.caption(
            f"Caliper: 0.20 SD of logit propensity score "
            f"(width={psm_diag['caliper_width_logit']:.3f})."
        )

        if psm_diag["max_abs_smd_after"] > 0.10:
            st.warning(
                f"{psm_diag['imbalanced_features_after']} features still "
                "have |SMD| > 0.10 after matching."
            )
        else:
            st.success(
                "All encoded features have |SMD| <= 0.10 after matching."
            )

        balance_table = (
            psm_diag["balance_table"]
            .sort_values("abs_smd_after", ascending=False)
            .head(15)
        )

        st.dataframe(
            balance_table[
                [
                    "feature",
                    "smd_before",
                    "smd_after",
                    "abs_smd_after",
                ]
            ].round(3),
            use_container_width=True,
        )

    fig_dist = px.histogram(
        base_results,
        x="uplift",
        nbins=40,
        title="Held-out distribution of estimated individual uplift",
        labels={"uplift": "Estimated causal uplift"},
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    raw_ps = bundle["propensity_model"].predict_proba(bundle["X_eval"])[:, 1]
    qini = qini_curve(
        bundle["Y_eval"], bundle["T_eval"], bundle["uplift"], raw_ps
    ).melt(
        id_vars="fraction_targeted",
        value_vars=["model_gain", "random_gain"],
        var_name="ranking",
        value_name="cumulative_gain",
    )
    fig_qini = px.line(
        qini,
        x="fraction_targeted",
        y="cumulative_gain",
        color="ranking",
        title="Held-out uplift/Qini-style cumulative gain",
        labels={
            "fraction_targeted": "Fraction targeted",
            "cumulative_gain": "IPW cumulative incremental response",
        },
    )
    fig_qini.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig_qini, use_container_width=True)

    subgroup = subgroup_uplift(base_results, "coupon_type")
    fig_subgroup = px.bar(
        subgroup,
        x="coupon_type",
        y="mean_uplift",
        hover_data=["customers"],
        title="Held-out average uplift by coupon type",
        labels={"mean_uplift": "Average estimated uplift"},
    )
    st.plotly_chart(fig_subgroup, use_container_width=True)


with tab_top:
    st.subheader("Customers ranked by held-out estimated uplift")
    if targeted_df.empty:
        st.warning("No customers match the current sidebar filters.")
    else:
        n_show = st.slider("How many customers to show", 5, 50, 15, 5)
        top_customers = targeted_df.sort_values("uplift", ascending=False).head(n_show)
        fig_bar = px.bar(
            top_customers,
            x="original_row",
            y="uplift",
            color="recommend_better_coupon",
            hover_data=[
                "coupon_type",
                "destination",
                "income",
                "expected_incremental_revenue",
                "net_value",
            ],
            title=f"Top {min(n_show, len(top_customers))} customers by uplift",
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.dataframe(targeted_df.round(3), use_container_width=True)
        st.download_button(
            "Download recommended customers (CSV)",
            data=get_recommended_csv_bytes(targeted_df),
            file_name="recommended_customers.csv",
            mime="text/csv",
        )


with tab_lookup:
    st.subheader("Look up a held-out customer")
    if targeted_df.empty:
        st.warning("No customers match the current sidebar filters.")
    else:
        options = targeted_df.index.tolist()
        row_id = st.selectbox(
            "Choose a customer",
            options,
            format_func=lambda i: (
                f"Original row {int(base_results.loc[i, 'original_row'])} - "
                f"{base_results.loc[i, 'coupon_type']}"
            ),
        )
        row = targeted_df.loc[row_id]
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "P(purchase) - 2h coupon", f"{row['p_purchase_weak_coupon']:.2f}"
        )
        c2.metric(
            "P(purchase) - 1d coupon", f"{row['p_purchase_better_coupon']:.2f}"
        )
        c3.metric("Estimated uplift", f"{row['uplift']:+.2f}")

        if row["recommend_better_coupon"]:
            st.success(
                "Recommendation: send the 1-day coupon under the selected "
                f"{targeting_strategy.lower()} rule."
            )
        elif row["net_value"] <= 0:
            st.warning("Do not upgrade: expected incremental value does not cover cost.")
        else:
            st.info("Profitable estimate, but this customer falls outside the budget cap.")

        st.divider()
        st.subheader("What-if: change this customer's age or income")
        w1, w2 = st.columns(2)
        what_if_age = w1.slider(
            "Age", 18, 60, int(bundle["X_eval"].loc[row_id, "age"])
        )
        what_if_income = w2.slider(
            "Income level (1=lowest, 9=highest)",
            1,
            9,
            int(bundle["X_eval"].loc[row_id, "income"]),
        )
        _, _, uplift_whatif = run_whatif(
            bundle["model_treated"],
            bundle["model_control"],
            bundle["X_eval"],
            row_id,
            what_if_age,
            what_if_income,
        )
        st.metric(
            "What-if estimated uplift",
            f"{uplift_whatif[0]:+.2f}",
            delta=f"{(uplift_whatif[0] - row['uplift']):+.2f} vs. original",
        )
