# Causal ML for Coupon Targeting 🎯

An interactive Streamlit dashboard that answers a real business question with **causal inference**, not just prediction:

> **Which customers should receive the better coupon (1-day redemption) instead of the weaker one (2-hour redemption) — based on how much it actually changes their behavior?**

Built on the [UCI In-Vehicle Coupon Recommendation dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00603/in-vehicle-coupon-recommendation.csv).

---

## Why This Isn't Just "Predict Who Buys"

A standard ML model answers *"who is likely to purchase?"* — and ranks already-loyal customers highest, even if a coupon wouldn't change their behavior at all. That wastes marketing spend on people who would have bought anyway.

This project instead estimates the **causal treatment effect** of the coupon upgrade: *"how much MORE likely does the better coupon make this specific customer to buy?"* That's the number a business actually needs to make a targeting decision.

| Approach | Question it answers | Business use |
|---|---|---|
| Baseline model | "Will this customer buy?" | ❌ Misleading for targeting |
| Uplift / causal model | "How much does the coupon *change* their behavior?" | ✅ Correct basis for targeting |

---

## Key Highlights

- **🎯 Individual-level targeting** — every customer gets a personalized recommendation on whether to upgrade their coupon.
- **📈 Treatment Effect Estimation (Uplift Modeling)** — a **T-learner** trains separate models on treated vs. control groups to isolate the causal effect of the coupon, rather than mere correlation.
- **🔬 Causal Inference Cross-Check** — a **Propensity Score Matching (PSM)** estimate of the Average Treatment Effect (ATE) validates the uplift model's average result. Agreement between the two independent methods is a built-in sanity check on trustworthiness.
- **💰 Business Optimization Layer** — converts uplift into **Expected Incremental Revenue** and **Net Value**, and only recommends sending the better coupon when it's profitable (revenue gain > coupon cost).
- **📊 Portfolio-Level ROI** — aggregates individual decisions into total customers targeted, total incremental revenue, total cost, and overall **campaign ROI**.
- **🖥️ Interactive Dashboard** — filter by coupon type, destination, and income; explore top-uplift customers; adjust business assumptions (purchase value, coupon cost) live; and run "what-if" simulations on individual customers.

---

## Statistics + Decision Science + ML — How It Fits Together

```
Raw survey data
      │
      ▼
┌─────────────────┐   Clean, encode, split into
│  Data Prep       │   confounders (X), treatment (T), outcome (Y)
└─────────────────┘
      │
      ▼
┌─────────────────────────────┐     ┌──────────────────────────────┐
│  Baseline Model (ML)        │     │  Uplift Model / T-learner    │
│  "Who buys?"                │     │  (Causal Inference)          │
│  Logistic Regression        │     │  Two Random Forests:         │
└─────────────────────────────┘     │  P(buy | treated) vs         │
                                     │  P(buy | control)            │
                                     └──────────────────────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────────┐
                                  │  Uplift = P(treated) −      │
                                  │           P(control)        │
                                  └────────────────────────────┘
                                              │
                          ┌───────────────────┼───────────────────┐
                          ▼                                       ▼
              ┌───────────────────────┐                ┌────────────────────────┐
              │  PSM Cross-Check      │                │  Business Decision      │
              │  (Statistics)         │                │  (Decision Science)     │
              │  Nearest-neighbor     │                │  Incremental Revenue,   │
              │  matched ATE          │                │  Net Value, ROI, Send/  │
              │                       │                │  Don't-Send rule        │
              └───────────────────────┘                └────────────────────────┘
```

---

## Project Structure

```
.
├── app.py            # Streamlit dashboard (3 tabs: Overview, Top Customers, Customer Lookup)
├── causal_model.py   # Baseline model, T-learner uplift model, PSM ATE, business logic
├── data_prep.py       # Data loading, cleaning, and feature engineering (X, T, Y)
└── requirements.txt   # Python dependencies
```

## Methodology

1. **Data Prep** (`data_prep.py`) — loads the UCI dataset, cleans missing/duplicate rows, encodes ordinal features (age, income, frequency-of-visit fields), and one-hot encodes nominal features. Defines:
   - **X** — customer/context confounders
   - **T** — treatment indicator (`1` = 1-day coupon, `0` = 2-hour coupon)
   - **Y** — outcome (coupon redeemed / purchase made)

2. **Causal Modeling** (`causal_model.py`)
   - `train_baseline_model` — logistic regression predicting purchase, ignoring treatment (the "naive" comparison).
   - `train_uplift_model` / `predict_uplift` — T-learner: two Random Forests fit separately on treated and control groups; uplift = difference in predicted purchase probability.
   - `propensity_score_ate` — logistic regression propensity model + nearest-neighbor matching to independently estimate the average treatment effect.
   - `business_recommendation` / `portfolio_summary` — turn uplift into dollar-denominated incremental revenue, net value, and aggregate ROI.

3. **Dashboard** (`app.py`)
   - **Overview** — portfolio KPIs (customers targeted, incremental revenue, cost, ROI), uplift distribution, and the T-learner vs. PSM sanity check.
   - **Top Customers** — ranks and visualizes customers by estimated uplift; exports the recommended list as CSV.
   - **Customer Lookup** — inspect any customer's probabilities and recommendation, plus a live what-if slider to see how age/income shift their estimated uplift.

## Getting Started

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

`Python` · `pandas` / `numpy` · `scikit-learn` (Logistic Regression, Random Forest, Nearest Neighbors) · `Streamlit` · `Plotly`

---

*This project demonstrates an end-to-end causal ML workflow — from raw data to a statistically cross-validated, business-optimized targeting decision — packaged as a deployable interactive dashboard.*
