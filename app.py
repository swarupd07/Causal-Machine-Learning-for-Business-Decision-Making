"""
app.py
------
Streamlit dashboard for the causal ML coupon project.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from data_prep import load_and_prepare
from causal_model import (
    train_baseline_model,
    train_uplift_model,
    predict_uplift,
    propensity_score_ate,
    business_recommendation,
    portfolio_summary,
)

st.set_page_config(page_title="Causal Coupon Targeting", layout="wide")
st.title("Causal ML for Coupon Targeting")
st.caption(
    "Business question: which customers should receive the BETTER coupon "
    "(1-day redemption window) instead of the weaker one (2-hour window), "
    "based on the estimated CAUSAL effect on purchase - not just who is "
    "predicted to buy anyway?"
)


@st.cache_data
def get_data():
    return load_and_prepare()


@st.cache_resource
def get_models(X, T, Y):
    baseline = train_baseline_model(X, Y)
    model_t, model_c = train_uplift_model(X, T, Y)
    return baseline, model_t, model_c


df_clean, X, T, Y = get_data()
baseline_model, model_treated, model_control = get_models(X, T, Y)
p1, p0, uplift = predict_uplift(model_treated, model_control, X)

# --- Sidebar: business assumptions + filters ------------------------------
st.sidebar.header("Business assumptions")
avg_purchase_value = st.sidebar.slider("Average purchase value ($)", 5.0, 50.0, 15.0, 1.0)
coupon_cost = st.sidebar.slider("Cost of sending the better coupon ($)", 0.5, 10.0, 2.0, 0.5)

st.sidebar.header("Filter customers")
coupon_options = sorted(df_clean["coupon"].unique())
destination_options = sorted(df_clean["destination"].unique())
income_options = sorted(df_clean["income"].unique())

selected_coupons = st.sidebar.multiselect("Coupon type", coupon_options, default=coupon_options)
selected_destinations = st.sidebar.multiselect("Destination", destination_options, default=destination_options)
selected_incomes = st.sidebar.multiselect("Income bracket", income_options, default=income_options)

biz = business_recommendation(uplift, avg_purchase_value, coupon_cost)

results_df = pd.DataFrame({
    "coupon_type": df_clean["coupon"],
    "destination": df_clean["destination"],
    "income": df_clean["income"],
    "p_purchase_weak_coupon": p0.round(3),
    "p_purchase_better_coupon": p1.round(3),
    "uplift": uplift.round(3),
    "expected_incremental_revenue": biz["expected_incremental_revenue"].round(2),
    "recommend_better_coupon": biz["recommend"],
})

# Apply sidebar filters
filtered_df = results_df[
    results_df["coupon_type"].isin(selected_coupons)
    & results_df["destination"].isin(selected_destinations)
    & results_df["income"].isin(selected_incomes)
]

summary = portfolio_summary(
    business_recommendation(filtered_df["uplift"].values, avg_purchase_value, coupon_cost),
    coupon_cost,
)

tab_overview, tab_top, tab_lookup = st.tabs(["Overview", "Top Customers", "Customer Lookup"])

# ---------------------------------------------------------------------
# TAB 1: Overview
# ---------------------------------------------------------------------
with tab_overview:
    st.caption(f"Showing results for {len(filtered_df):,} customers matching your sidebar filters.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers to target", f"{summary['customers_targeted']:,}")
    col2.metric("Total incremental revenue", f"${summary['total_incremental_revenue']:,.0f}")
    col3.metric("Total campaign cost", f"${summary['total_cost']:,.0f}")
    col4.metric("Campaign ROI", f"{summary['roi'] * 100:,.1f}%")

    st.divider()

    st.subheader("Why causal targeting beats plain prediction")
    st.write(
        "The baseline model only predicts *who buys*, so it would rank "
        "already-loyal customers highest even if a coupon wouldn't change "
        "their behavior at all. The uplift model instead ranks customers by "
        "how much MORE likely a coupon makes them to buy."
    )

    ate_psm = propensity_score_ate(X, T, Y)
    avg_uplift = uplift.mean()
    m1, m2 = st.columns(2)
    m1.metric("Average uplift (T-learner)", f"{avg_uplift:+.3f}")
    m2.metric("Average effect (Propensity Score Matching cross-check)", f"{ate_psm:+.3f}")
    st.caption(
        "These two independent methods should roughly agree - that agreement "
        "is a sanity check that the causal estimate is trustworthy."
    )

    fig_dist = px.histogram(
        results_df, x="uplift", nbins=40,
        title="Distribution of estimated uplift across all customers",
        labels={"uplift": "Estimated uplift (causal effect)"},
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 2: Top Customers
# ---------------------------------------------------------------------
with tab_top:
    st.subheader("Customers ranked by estimated uplift")
    n_show = st.slider("How many customers to show", 5, 50, 15, 5)
    top_customers = filtered_df.sort_values("uplift", ascending=False).head(n_show)

    fig_bar = px.bar(
        top_customers.reset_index(),
        x="index", y="uplift", color="recommend_better_coupon",
        hover_data=["coupon_type", "destination", "income", "expected_incremental_revenue"],
        labels={"index": "Customer row", "uplift": "Estimated uplift"},
        title=f"Top {n_show} customers by uplift",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.dataframe(top_customers, use_container_width=True)

    csv_bytes = filtered_df[filtered_df["recommend_better_coupon"]].to_csv(index=True).encode("utf-8")
    st.download_button(
        "Download recommended customers (CSV)",
        data=csv_bytes,
        file_name="recommended_customers.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------
# TAB 3: Customer Lookup (with live what-if sliders)
# ---------------------------------------------------------------------
with tab_lookup:
    st.subheader("Look up a single customer")

    if len(filtered_df) == 0:
        st.warning("No customers match the current sidebar filters.")
    else:
        options = filtered_df.index.tolist()
        row_id = st.selectbox(
            "Choose a customer (row number - coupon type)",
            options,
            format_func=lambda i: f"Row {i} - {results_df.loc[i, 'coupon_type']}",
        )
        row = results_df.loc[row_id]

        c1, c2, c3 = st.columns(3)
        c1.metric("P(purchase) - weak coupon", f"{row['p_purchase_weak_coupon']:.2f}")
        c2.metric("P(purchase) - better coupon", f"{row['p_purchase_better_coupon']:.2f}")
        c3.metric("Estimated uplift", f"{row['uplift']:+.2f}")

        if row["recommend_better_coupon"]:
            st.success(
                f"Recommendation: SEND the better coupon. Expected incremental "
                f"revenue ${row['expected_incremental_revenue']:.2f} exceeds the "
                f"${coupon_cost:.2f} cost."
            )
        else:
            st.warning(
                "Recommendation: DO NOT upgrade this coupon - the expected "
                "incremental revenue does not cover the cost."
            )

        st.divider()
        st.subheader("What-if: change this customer's age or income")
        st.caption("Move the sliders to see how the estimated uplift changes in real time.")

        w1, w2 = st.columns(2)
        what_if_age = w1.slider("Age", 18, 60, int(X.loc[row_id, "age"]))
        what_if_income = w2.slider("Income level (1=lowest, 9=highest)", 1, 9, int(X.loc[row_id, "income"]))

        X_whatif = X.loc[[row_id]].copy()
        X_whatif["age"] = what_if_age
        X_whatif["income"] = what_if_income

        p1_whatif, p0_whatif, uplift_whatif = predict_uplift(model_treated, model_control, X_whatif)
        st.metric("What-if estimated uplift", f"{uplift_whatif[0]:+.2f}",
                   delta=f"{(uplift_whatif[0] - row['uplift']):+.2f} vs. original")