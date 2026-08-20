# 🎯 Causal ML for Coupon Targeting
### Held-out causal inference for business decision making

A causal machine learning pipeline that answers a real targeting question — **not** "will this customer buy?", but **"will this customer buy *because* we gave them a better coupon?"** — using three held-out causal estimators, evaluated entirely on a held-out set, and surfaced through an interactive Streamlit dashboard.

---

## The Business Question

> **Should a customer receive a 1-day coupon redemption window instead of a 2-hour window — and is the extra acceptance it causes worth more than it costs?**

A plain predictive model would rank customers by *purchase probability*, which over-targets people who were already going to accept the coupon anyway. This project instead estimates each customer's **incremental** (causal) response — the number a business actually needs to spend a targeting budget efficiently — and turns it into a concrete send/don't-send decision.

## Key Highlights

| Highlight | What it means here |
|---|---|
| 🎯 **Decide who should receive a coupon** | Every customer gets an individual causal effect estimate — not a purchase probability — used to drive the targeting decision |
| 💰 **ROI** | Targeting is evaluated against a real cost-benefit rule, with both a value-threshold mode and a budget-capped mode |
| 📈 **Incremental Revenue** | Measures the *extra* revenue the better coupon generates, not total revenue |
| 🔬 **Treatment Effect** | Estimated three ways — T-learner, caliper-matched PSM, and doubly-robust AIPW — all reported with bootstrap confidence intervals |
| 📊 **Statistics + Decision Science + ML** | Combines logistic regression, random forests, propensity-score matching, covariate balance diagnostics, and ROI decision rules |
| 🧠 **Causal Inference** | Explicit treatment/control split, confounder adjustment, positivity/overlap checks, and covariate balance testing — not just correlation |
| 🏢 **Business Optimization** | Outputs a targeting list, campaign ROI, and a Qini-style gain curve — the same diagnostics used in real uplift-marketing teams |

---

## Real Results (Held-Out Evaluation Set)

All numbers below come from the dashboard, computed **entirely on a held-out evaluation split** — none of these rows were used to train any model.

| Metric | Value |
|---|---|
| Held-out evaluation set size | 3,153 customers |
| **T-learner** average uplift | **+0.148**, 95% CI [+0.129, +0.159] |
| **Propensity-matching ATT** (0.20 SD caliper) | **+0.171**, 95% CI [+0.120, +0.229] |
| **Doubly-robust AIPW** ATE | **+0.170**, 95% CI [+0.132, +0.200] |
| Treated / control split (held-out) | 55.8% (1-day) / 44.2% (2-hour) |
| Propensity scores requiring clipping | 1 (0.03%; displayed as 0.0%) |
| Coupon types exceeding 90% concentration in one expiration arm | None — no severe positivity violation detected |
| PSM match rate | 96.0% |
| Unmatched treated observations | 71 |
| Maximum \|SMD\| before / after matching | 0.331 / 0.237 |
| Features with post-match \|SMD\| > 0.10 | 11 |
| Customers recommended (value-threshold rule, $15/$2 assumptions) | 1,832 |
| Projected incremental revenue | $5,433 |
| Projected campaign cost | $3,664 |
| **Projected campaign ROI** | **48.3%** |

**All three causal estimators agree in direction and rough magnitude**, and each interval excludes zero — the strongest evidence this project can offer that the effect isn't just sampling noise. (It does not, on its own, rule out confounding — see [Limitations](#honest-limitations).)

**Held-out uplift by coupon type** (customers with ≥30 held-out observations): Restaurant(<20) shows the largest average uplift (~0.19), followed by Restaurant(20-50) (~0.165) and Carry out & Take away (~0.14) — useful for prioritizing which coupon categories benefit most from the longer redemption window.

---

## Methodology

### 1. Held-out evaluation (not in-sample)
The data is split **once**, up front, into a 75% training set and a 25% evaluation set, stratified jointly on treatment × outcome to preserve both the treated/control ratio and the acceptance rate in both splits (`causal_train_test_split`). **Every causal estimate, every diagnostic, and every dashboard number is computed on the held-out 25%** — the models never see these rows during training. The categorical feature encoder is also **fit only on the training split** and then applied to the evaluation split, preventing category-leakage from evaluation rows influencing the feature schema. The logistic-regression baseline and propensity models use `StandardScaler` → `LogisticRegression` pipelines fitted only on training data.

### 2. Three causal estimators

- **T-learner** — two separate `RandomForestClassifier`s (`min_samples_leaf=10`, trained in parallel via `n_jobs=-1`), one fit on treated training rows, one on control training rows. Individual uplift = difference in their held-out predictions.
- **Propensity-Score Matching (ATT)** — a logistic regression propensity model, followed by 1-nearest-neighbor matching **on the logit propensity score**, restricted by a **0.20 standard-deviation caliper**: treated customers without an acceptably close control match are excluded rather than force-matched to a poor counterfactual. Reports the **Average Treatment effect on the Treated**, since matching only estimates effects for treated units with usable matches.
- **Doubly-robust AIPW (Augmented Inverse Propensity Weighting)** — combines the T-learner's outcome models with the propensity model into a single estimator that stays consistent even if *one* of the two underlying models is misspecified. Serves as the third, most theoretically robust cross-check.

### 3. Covariate balance diagnostics for PSM
Beyond just matching, the pipeline computes **standardized mean differences (SMD)** for every encoded feature, before and after matching (`standardized_mean_differences`). The dashboard reports the match rate, number of unmatched treated customers, and the maximum |SMD| before/after — with a warning if any feature still exceeds the conventional 0.10 balance threshold post-matching. This is a real, textbook propensity-score-matching quality check, not just a point estimate.

### 4. Positivity & overlap checks
- `coupon_expiration_overlap` cross-tabulates coupon type against expiration window and flags any coupon type where one expiration arm exceeds 90% share — a proxy for a positivity violation.
- Propensity scores are clipped to `[0.01, 0.99]`, and the fraction of held-out rows actually requiring clipping is reported directly on the dashboard (1 of 3,153 rows, or 0.03%, in the current run).

### 5. Arm-stratified bootstrap confidence intervals
Instead of a plain bootstrap, each resample draws **separately, with replacement, from the treated and control arms** (`_stratified_bootstrap_indices`), preserving the treatment split in every resample. All three estimators (T-learner, PSM, AIPW) are recomputed on 100 such resamples (with a reduced 50-tree forest per resample for tractability), and the 2.5th/97.5th percentiles form each 95% CI. Both the bootstrap and the main model fits are wrapped in Streamlit caching (`@st.cache_data`/`@st.cache_resource`) so they run once per session, not on every UI interaction.

### 6. Qini-style uplift curve
An inverse-propensity-weighted cumulative gain curve (`qini_curve`) ranks held-out customers by predicted uplift and compares the model's cumulative incremental response against a random-targeting baseline — the standard diagnostic for whether an uplift model's *ranking*, not just its average estimate, is useful for targeting.

### 7. Business translation with two targeting modes
- **Value threshold** — recommend the better coupon whenever `uplift × avg_purchase_value − coupon_cost > 0`.
- **Budget cap** — rank customers by net value and greedily select the top customers until a user-specified campaign budget is exhausted, a real constrained-targeting mode rather than an unconstrained threshold rule.

---

## Dashboard

Built with **Streamlit** + **Plotly**, in three tabs:

- **Overview & Diagnostics** — headline ROI metrics, all three causal estimates with confidence intervals, an expandable positivity/overlap/balance panel (treatment split, clipped-propensity count, expiration-mix chart, PSM match rate and SMD balance table), the held-out uplift distribution, the Qini gain curve, and average uplift by coupon type.
- **Top Customers** — filterable, sortable table of held-out customers ranked by uplift, with a CSV export of the recommended targeting list.
- **Customer Lookup** — inspect any held-out customer's predicted outcomes with/without the better coupon, plus **live what-if sliders** (age, income) that recompute uplift for that customer in real time.

Sidebar controls adjust the average purchase value, coupon cost, targeting strategy (value threshold vs. budget cap), and filters by coupon type, destination, and income bracket — every metric and chart updates accordingly.

---

## Tech Stack

Python · pandas · NumPy · scikit-learn · Streamlit · Plotly

## Repo Structure

```
causal-ml-coupon-targeting/
├── app.py                                  # Streamlit dashboard (3 tabs, cached pipeline)
├── backend.py                              # Orchestration layer: holdout split, diagnostics, targeting, what-if
├── causal_model.py                         # T-learner, caliper-matched PSM + SMD balance, AIPW, bootstrap CI, Qini, overlap checks
├── data_prep.py                            # Load/clean dataset, train-only OneHotEncoder fitting, feature transforms
├── requirements.txt                        # Pinned dependencies
├── EDA.ipynb                               # Exploratory analysis notebook (missingness, duplicates, expiration/coupon distributions)
├── in-vehicle-coupon-recommendation.csv    # Local backup of the dataset (used if the UCI URL is unreachable)
└── Causal Coupon Targeting1.pdf            # Exported dashboard screenshots showing the real held-out results above
```

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app downloads the dataset automatically (falling back to the local CSV backup if offline), fits the held-out pipeline, computes all three causal estimates plus bootstrap confidence intervals, and renders the full dashboard.

---

## Dataset

**UCI "In-Vehicle Coupon Recommendation" dataset** — real survey responses covering driving context, customer demographics, coupon details, and acceptance outcomes.

- Source: https://archive.ics.uci.edu/dataset/603/in+vehicle+coupon+recommendation
- **Treatment (T):** `expiration` — 1 = 1-day redemption window, 0 = 2-hour window
- **Outcome (Y):** `Y` — 1 = customer accepted the coupon
- **Confounders (X):** age, income, education, occupation, marital status, gender, destination, passenger, weather, time, temperature, coupon type, visit frequency (bar/coffee house/carry-away/restaurants), driving-distance and direction flags — one-hot encoded (nominal) or numerically mapped (ordinal), with the encoder fit strictly on the training split.

## Statistics & ML Concepts Applied

- Potential outcomes framework, confounding, ignorability, positivity/overlap
- Held-out (train/eval) causal model evaluation, stratified splitting
- Propensity scores, caliper matching, standardized mean differences (covariate balance)
- Meta-learners (T-learner) and doubly-robust estimation (AIPW)
- Arm-stratified bootstrap confidence intervals
- Qini-style uplift-ranking evaluation
- Budget-constrained targeting as a greedy allocation rule
- Cost-benefit / ROI decision rules

## <a name="honest-limitations"></a>Honest Limitations

- This is **observational data** — the redemption window was not randomly assigned, so all causal claims rest on the **ignorability assumption** (no unmeasured confounding), which no amount of diagnostics can fully verify.
- Confidence intervals reflect **sampling variability**, not the risk of confounding bias — a tight interval means the estimate is precise, not that it's unbiased.
- The overlap check flags severe (>90%) positivity violations by coupon type, but does not rule out subtler imbalance across combinations of features.
- Caliper matching improved balance but did not achieve complete covariate balance: 11 encoded features retained |SMD| > 0.10, with maximum post-match |SMD| = 0.237. Therefore, the PSM ATT should be interpreted cautiously.
- `avg_purchase_value` and `coupon_cost` are illustrative, adjustable assumptions, not measured business data — the ROI/revenue figures move accordingly and do not currently carry their own confidence interval.

## Why This Project Stands Out

| Standard ML Project | This Project |
|---|---|
| Predict who will buy | Decide who *should* receive the better coupon |
| Accuracy, F1-score | ROI, incremental revenue, treatment effect, Qini gain |
| In-sample evaluation | Held-out train/eval split throughout |
| One point estimate | Three independent estimators, each with a bootstrap CI |
| Correlation | Causal inference, with explicit positivity and covariate-balance diagnostics |

## Resume-Ready Description

> **Causal ML for Coupon Targeting:** Built a held-out causal inference pipeline estimating individual and average treatment effects via a T-learner, caliper-matched Propensity Score Matching (with covariate balance diagnostics), and a doubly-robust AIPW estimator, each validated with arm-stratified bootstrap confidence intervals. Translated results into ROI-based, budget-constrained targeting recommendations and Qini-curve-validated customer rankings, delivered through an interactive Streamlit dashboard.

## Possible Extensions

- Randomized A/B validation of the model's targeting recommendations
- X-learner or causal-forest estimator as a fourth cross-check
- Propagate bootstrap resamples into the ROI/revenue figures for a business-level confidence interval
- Model calibration checks for the propensity and outcome models
- Subgroup uplift and balance diagnostics broken out by additional segments (destination, income)
