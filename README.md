# Causal ML for Coupon Targeting 🎯

An interactive Streamlit dashboard that answers a real business question with **causal inference**, not just prediction:

> **Which customers should receive the better coupon (1-day redemption) instead of the weaker one (2-hour redemption) — based on how much it actually changes their behavior?**

Every causal estimate and targeting number in the dashboard is computed on a **held-out evaluation split** — never on the rows the models were trained on.

Built on the [UCI In-Vehicle Coupon Recommendation dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00603/in-vehicle-coupon-recommendation.csv).

---

## Why This Isn't Just "Predict Who Buys"

A standard ML model answers *"who is likely to purchase?"* — and ranks already-loyal customers highest, even if a coupon wouldn't change their behavior at all. That wastes marketing spend on people who would have bought anyway.

This project instead estimates the **causal treatment effect** of the coupon upgrade: *"how much MORE likely does the better coupon make this specific customer to buy?"* — and reports that estimate the way a rigorous analysis should: cross-checked by three independent estimators, with confidence intervals, and validated for common causal-inference failure modes (positivity, overlap, treatment-arm balance).

| Approach | Question it answers | Business use |
|---|---|---|
| Baseline model | "Will this customer buy?" | ❌ Misleading for targeting |
| Uplift / causal model | "How much does the coupon *change* their behavior?" | ✅ Correct basis for targeting |

---

## Key Concepts

- **🎯 Individual-level targeting** — every held-out customer gets a personalized recommendation on whether to upgrade their coupon.
- **📈 Treatment Effect Estimation (Uplift Modeling)** — a **T-learner** trains separate Random Forests on treated vs. control groups, fit only on the training split, to isolate the causal effect of the coupon rather than mere correlation.
- **🔬 Two independent cross-checks** —
  - **Propensity Score Matching (PSM/ATT)**: a logistic-regression propensity model + nearest-neighbor matching, evaluated on the held-out split.
  - **Doubly-robust AIPW (Augmented Inverse Propensity Weighting)**: combines the outcome models and the propensity model so the estimate stays consistent even if one of the two is misspecified.
- **📏 Uncertainty quantification** — seeded, arm-stratified bootstrap resampling produces 95% confidence intervals for all three estimators, instead of reporting bare point estimates.
- **🧪 Positivity & overlap diagnostics** — within-coupon expiration proportions, an explicit flag for any coupon type exceeding 90% concentration in one treatment arm, treatment-arm balance (% control vs. % treated), and a count of propensity scores that hit the 0.01/0.99 clipping bounds (i.e., customers with weak counterfactual support).
- **💰 Business optimization layer** — converts uplift into **Expected Incremental Revenue** and **Net Value**, with two selectable targeting rules:
  - **Value threshold** — recommend whenever expected incremental revenue exceeds the coupon cost.
  - **Budget cap** — rank customers by net value and target as many as a fixed campaign budget allows.
- **📊 Portfolio-level ROI** — aggregates individual decisions into total customers targeted, total incremental revenue, total cost, and overall **campaign ROI**.
- **📉 Qini-style gain curve** — an inverse-propensity-weighted cumulative gain curve comparing model-ranked targeting against random targeting.
- **🗂️ Subgroup uplift** — average estimated uplift broken out by coupon type, to see where the causal effect is strongest.
- **🖥️ Interactive dashboard** — filter by coupon type, destination, and income; explore top-uplift customers; adjust business assumptions (purchase value, coupon cost, targeting rule, budget) live; and run "what-if" simulations on individual customers' age/income.
- **🧱 Inference-safe encoding** — a persisted `OneHotEncoder(handle_unknown="ignore")` fit only on training data, so unseen categories at evaluation/inference time are handled explicitly instead of silently breaking column alignment.

---

## Results (Held-Out Evaluation Set)

The figures below come from a real run of the dashboard (`localhost:8501`) on the full cleaned dataset, with a 75/25 train/held-out split, default business assumptions ($15 avg. purchase value, $2 coupon cost), and the **Value threshold** targeting rule with no sidebar filters applied. A full screenshot walkthrough is included in the repo as [`Causal_Coupon_Targeting.pdf`](./Causal_Coupon_Targeting.pdf).

**Portfolio summary** — 3,153 held-out customers:

| Metric | Value |
|---|---|
| Customers to target | **1,790** |
| Incremental revenue | **$5,382** |
| Campaign cost | **$3,580** |
| Campaign ROI | **50.3%** |

**Held-out causal estimates** (three independent estimators, all evaluated on the same held-out split):

| Estimator | Point estimate | 95% CI (stratified bootstrap) |
|---|---|---|
| T-learner average uplift | **+0.149** | [+0.129, +0.159] |
| Propensity matching ATT | **+0.159** | [+0.102, +0.233] |
| Doubly-robust AIPW ATE | **+0.168** | [+0.131, +0.199] |

All three intervals sit clearly above zero and overlap with each other — reassuring, but agreement across estimators sharing the same confounder set and ignorability assumption does not by itself prove all confounding has been removed.

**Positivity, overlap, and treatment-balance checks:**

| Check | Result |
|---|---|
| Control share (2h coupon) | 44.2% |
| Treated share (1d coupon) | 55.8% |
| Propensities clipped at 0.01/0.99 | 0 (0.0%) |
| Coupon types exceeding 90% concentration in one expiration arm | None flagged |

No coupon type is a near-deterministic predictor of the treatment, treatment arms are reasonably balanced, and no held-out customer needed propensity clipping — together, mild evidence against a severe positivity violation.

**Held-out average uplift by coupon type** (highest to lowest): Restaurant(&lt;20) → Restaurant(20-50) → Carry out & Take away → Coffee House — the model finds the coupon upgrade matters most for the cheaper, more impulse-driven restaurant coupons.

---

## Why the Numbers Differ From an Earlier In-Sample Version

If you've seen an earlier version of this project reporting +0.143 / +0.139 with no confidence intervals: those numbers came from an **in-sample** implementation (trained and evaluated on the same rows, no held-out split, no bootstrap CIs, no overlap check). The current pipeline fits every nuisance model on a training partition only and reports all causal estimates, targeting decisions, and diagnostics on a genuinely held-out partition — so the numbers above are not directly comparable to that earlier version, and are the ones to cite going forward.

---

## Statistics + Decision Science + ML — How It Fits Together

```
Raw survey data
      │
      ▼
┌─────────────────────┐   Clean, encode (persisted OneHotEncoder),
│  data_prep.py         │   split into confounders (X), treatment (T), outcome (Y)
└─────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  causal_train_test_split     │   Joint treatment/outcome-stratified 75/25 split
└─────────────────────────────┘
      │
      ├────────────────────────────┬──────────────────────────────┐
      ▼                            ▼                              ▼
┌─────────────────────┐  ┌──────────────────────┐   ┌──────────────────────────┐
│  Baseline model       │  │  T-learner            │   │  Propensity model         │
│  "Who buys?"          │  │  (Causal Inference)    │   │  Logistic Regression      │
│  Logistic Regression  │  │  Two Random Forests:  │   │  P(T=1 | X)               │
│  (train, held-out AUC)│  │  P(buy|treated) vs.   │   └──────────────────────────┘
└─────────────────────┘  │  P(buy|control)       │              │
                          └──────────────────────┘              │
                                     │                           │
                                     ▼                           ▼
                          ┌───────────────────────┐   ┌──────────────────────────┐
                          │ Uplift = P(treated) −  │   │ PSM ATT (matching) +      │
                          │          P(control)    │   │ Doubly-robust AIPW ATE    │
                          └───────────────────────┘   └──────────────────────────┘
                                     │                           │
                                     └─────────────┬─────────────┘
                                                    ▼
                                   ┌─────────────────────────────────┐
                                   │  Stratified bootstrap → 95% CIs   │
                                   │  Positivity/overlap/balance checks│
                                   └─────────────────────────────────┘
                                                    │
                                                    ▼
                                   ┌─────────────────────────────────┐
                                   │  Business decision layer          │
                                   │  Incremental revenue, net value,  │
                                   │  Value-threshold or Budget-cap     │
                                   │  targeting, portfolio ROI          │
                                   └─────────────────────────────────┘
```

---

## Project Structure

```
.
├── app.py                                    # Streamlit dashboard (Overview & Diagnostics, Top Customers, Customer Lookup)
├── backend.py                                 # Orchestration layer: holdout prep, results building, filtering, targeting
├── causal_model.py                            # T-learner, PSM/ATT, AIPW, bootstrap CIs, overlap & balance diagnostics, Qini, business rules
├── data_prep.py                               # Data loading, cleaning, and OneHotEncoder-based feature engineering (X, T, Y)
├── requirements.txt                           # Python dependencies (pinned)
├── EDA.ipynb                                  # Exploratory data analysis notebook
├── in-vehicle-coupon-recommendation.csv        # Local backup of the UCI dataset (used if the remote URL is unreachable)
├── Causal_Coupon_Targeting.pdf                 # Screenshot walkthrough of the live dashboard and results
└── README.md
```

## Methodology

1. **Data Prep** (`data_prep.py`) — loads the UCI dataset (falling back to the local `in-vehicle-coupon-recommendation.csv` if the remote URL is unreachable), cleans missing/duplicate rows, ordinal-encodes age/income/visit-frequency fields, and one-hot encodes nominal features via a persisted `OneHotEncoder(handle_unknown="ignore")` so the schema is inference-safe against unseen categories. Defines:
   - **X** — customer/context confounders
   - **T** — treatment indicator (`1` = 1-day coupon, `0` = 2-hour coupon)
   - **Y** — outcome (coupon redeemed / purchase made)

2. **Causal Modeling** (`causal_model.py`)
   - `causal_train_test_split` — a joint treatment/outcome-stratified 75/25 split, so every downstream estimate is computed on genuinely held-out rows.
   - `train_baseline_model` — logistic regression predicting purchase, ignoring treatment, evaluated on held-out AUC (the "naive" comparison).
   - `train_uplift_model` / `predict_uplift` — T-learner: two Random Forests fit only on the training partition's treated and control rows; uplift = difference in predicted purchase probability on held-out rows.
   - `fit_propensity_model` / `propensity_score_ate` — logistic regression propensity model + nearest-neighbor matching, returning both the ATT estimate and propensity diagnostics (clipping counts, min/max scores).
   - `aipw_ate` — doubly-robust AIPW estimator combining the T-learner outcome models and the propensity model.
   - `bootstrap_causal_estimates` / `bootstrap_ate_T` / `bootstrap_ate_ps` — seeded, treatment-arm-stratified bootstrap resampling for 95% confidence intervals on all three estimators.
   - `coupon_expiration_overlap` / `treatment_balance` — positivity/overlap diagnostics and treatment-arm balance checks.
   - `qini_curve` / `subgroup_uplift` — IPW-weighted cumulative gain curve and per-coupon-type average uplift.
   - `business_recommendation` / `portfolio_summary` — turn uplift into dollar-denominated incremental revenue, net value, and aggregate ROI, under either a value-threshold or a fixed-budget targeting rule.

3. **Backend** (`backend.py`) — orchestrates data loading, holdout preparation, results-table construction, sidebar filtering, and targeting so `app.py` only handles UI state and rendering.

4. **Dashboard** (`app.py`)
   - **Overview & Diagnostics** — portfolio KPIs, all three held-out causal estimates with bootstrap CIs, positivity/overlap/balance checks, the uplift distribution, the Qini-style gain curve, and per-coupon-type subgroup uplift.
   - **Top Customers** — ranks and visualizes held-out customers by estimated uplift; exports the recommended list as CSV.
   - **Customer Lookup** — inspect any held-out customer's probabilities and recommendation, plus a live what-if slider to see how age/income shift their estimated uplift.

## Getting Started

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

`Python` · `pandas` / `numpy` · `scikit-learn` (Logistic Regression, Random Forest, OneHotEncoder, Nearest Neighbors) · `Streamlit` · `Plotly`

---

*This project demonstrates an end-to-end causal ML workflow — from raw data to held-out, uncertainty-quantified, diagnostic-checked, business-optimized targeting decisions — packaged as a deployable interactive dashboard.*
