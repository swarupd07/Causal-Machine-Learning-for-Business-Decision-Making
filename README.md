# Causal ML for Coupon Targeting

An end-to-end causal machine-learning system for deciding which customers should receive a **1-day coupon instead of a 2-hour coupon**, based on the estimated incremental change in purchase probability—not merely the customer's baseline likelihood of purchasing.

The project combines held-out uplift estimation, propensity-score matching, doubly robust estimation, uncertainty quantification, covariate-balance diagnostics and business-value optimization in an interactive Streamlit dashboard.

> **Important:** This is an observational causal analysis. The estimates rely on consistency, positivity and conditional exchangeability given the measured confounders. Agreement among estimators strengthens confidence but does not prove that all confounding has been removed.

## Why causal targeting?

A standard prediction model answers:

> Who is likely to purchase?

That can waste campaign budget on customers who would purchase even without the stronger offer. This project instead asks:

> For which customers does upgrading the coupon from 2 hours to 1 day increase purchase probability enough to justify its cost?

| Approach | Estimated quantity | Business use |
|---|---|---|
| Purchase prediction | \(P(Y=1\mid X)\) | Identifies likely buyers |
| Causal uplift modeling | \(P(Y=1\mid T=1,X)-P(Y=1\mid T=0,X)\) | Identifies customers whose behavior may change because of treatment |

## Main features

- **Train-only feature encoding:** raw rows are split before categorical encoding. The `OneHotEncoder(handle_unknown="ignore")` learns its schema only from training rows and applies the same schema to held-out evaluation rows.
- **T-learner uplift modeling:** separate Random Forest outcome models estimate purchase probability under the 1-day and 2-hour coupon conditions.
- **Propensity-score matching:** logistic-regression propensity scores are matched using nearest-neighbor matching with a caliper of 0.20 standard deviations of the logit propensity score.
- **Balance diagnostics:** standardized mean differences are reported before and after matching, together with match rate, unmatched treated observations and maximum residual imbalance.
- **Doubly robust AIPW:** combines propensity and outcome models so the ATE remains consistent if either nuisance-model family is correctly specified, under the required causal assumptions.
- **Held-out evaluation:** nuisance models are fitted on the 75% training partition; estimates, ranking and targeting outputs are evaluated on the remaining 25%.
- **Bootstrap confidence intervals:** seeded, treatment-arm-stratified bootstrap resampling provides 95% intervals for the T-learner, PSM ATT and AIPW ATE.
- **Overlap diagnostics:** treatment balance, propensity-score clipping and coupon-expiration concentration checks flag weak counterfactual support.
- **Business optimization:** converts uplift into expected incremental revenue, net value, campaign cost and ROI under value-threshold and budget-cap strategies.
- **Interactive analysis:** Streamlit views cover diagnostics, uplift distributions, cumulative gain, subgroup uplift, customer ranking, CSV export and individual what-if analysis.

## Causal setup

| Component | Definition |
|---|---|
| Treatment \(T=1\) | 1-day coupon expiration |
| Control \(T=0\) | 2-hour coupon expiration |
| Outcome \(Y\) | Coupon accepted/redeemed |
| Covariates \(X\) | Customer characteristics, context, coupon type and behavioral-frequency variables measured before the outcome |
| Individual score | T-learner conditional uplift estimate |
| Population estimands | Plug-in average uplift, PSM ATT and AIPW ATE |

## Leakage-safe data flow

```text
Raw UCI survey data
        |
        v
Clean rows and map treatment/outcome
        |
        v
Joint T/Y-stratified 75/25 split on raw rows
        |
        +---------------------------+
        |                           |
        v                           v
Training raw rows              Held-out raw rows
        |                           |
Fit encoder on training only        |
        |                           |
        +------ same encoder -------+
                    |
                    v
        Aligned train/evaluation features
                    |
                    v
    Fit outcome and propensity models on train
                    |
                    v
   Estimate effects and targeting value on holdout
```

Unknown evaluation categories are ignored without changing the number or order of model inputs. The encoder is retained in the fitted in-memory analysis bundle. For an external deployment, save the encoder together with the trained models using `joblib`.

## Methods

### 1. T-learner

Two Random Forest classifiers are trained independently:

\[
\hat\mu_1(x)=\hat P(Y=1\mid T=1,X=x), \qquad
\hat\mu_0(x)=\hat P(Y=1\mid T=0,X=x)
\]

The estimated uplift is:

\[
\hat\tau(x)=\hat\mu_1(x)-\hat\mu_0(x)
\]

### 2. Propensity-score matching ATT

A logistic-regression model estimates:

\[
\hat e(x)=P(T=1\mid X=x)
\]

Treated observations are matched to their nearest controls using logit propensity scores. A match is accepted only when its distance is within:

\[
0.20\times SD(\operatorname{logit}(\hat e(X)))
\]

The dashboard reports:

- number and percentage of treated observations matched;
- unmatched treated observations;
- mean and maximum match distance;
- mean and maximum absolute SMD before and after matching;
- number of covariates with \(|SMD|>0.10\) after matching;
- a feature-level balance table.

### 3. Doubly robust AIPW

The AIPW score combines outcome regression and inverse propensity weighting:

\[
\hat\tau_{AIPW}=\frac{1}{n}\sum_i\left[
\frac{T_i(Y_i-\hat\mu_1(X_i))}{\hat e(X_i)}-
\frac{(1-T_i)(Y_i-\hat\mu_0(X_i))}{1-\hat e(X_i)}+
\hat\mu_1(X_i)-\hat\mu_0(X_i)
\right]
\]

Propensity scores are clipped to `[0.01, 0.99]` for numerical stability, and the number of clipped observations is displayed.

### 4. Uncertainty quantification

The pipeline uses seeded, treatment-arm-stratified bootstrap samples. In every iteration it resamples the training and evaluation partitions, refits the nuisance models and recomputes all three causal estimators.

## Results

The following results were recorded from the updated held-out pipeline after introducing train-only encoding, 0.20-SD caliper matching and post-match balance diagnostics. The dashboard was run with no customer filters, a `$15` average purchase value, a `$2` incremental coupon cost and the value-threshold targeting rule.

### Held-out causal estimates

| Estimator | Point estimate | 95% bootstrap CI |
|---|---:|---:|
| T-learner average uplift | **+0.148** | **[+0.129, +0.159]** |
| Caliper-matched PSM ATT | **+0.194** | **[+0.117, +0.230]** |
| Doubly robust AIPW ATE | **+0.169** | **[+0.131, +0.199]** |

### Matching quality

The dashboard calculates matching quality directly from the held-out evaluation partition and displays:

- treated observations matched and unmatched;
- match rate under the 0.20-SD logit-propensity caliper;
- maximum absolute SMD before and after matching;
- number of encoded covariates with \(|SMD|>0.10\) after matching;
- a feature-level balance table sorted by residual imbalance.

These diagnostics are intentionally generated from the current run rather than copied as permanent constants, so balance can be rechecked whenever the split, covariates or matching specification changes.

### Business result

Using an average purchase value of `$15`, an incremental coupon cost of `$2` and the value-threshold targeting strategy:

| Metric | Result |
|---|---:|
| Held-out customers | **3,153** |
| Customers targeted | **1,832** |
| Expected incremental revenue | **$5,433** |
| Campaign cost | **$3,664** |
| Estimated campaign ROI | **48.3%** |

These are model-based decision estimates, not realized revenue from a randomized production campaign.

## Business decision rules

For each held-out customer:

\[
\text{Expected incremental revenue}=\hat\tau(x)\times\text{purchase value}
\]

\[
\text{Net value}=\text{expected incremental revenue}-\text{coupon cost}
\]

- **Value threshold:** target customers with positive estimated net value.
- **Budget cap:** rank profitable customers by net value and select as many as the budget permits.

## Project structure

```text
.
├── app.py
├── backend.py
├── causal_model.py
├── data_prep.py
├── EDA.ipynb
├── requirements.txt
├── in-vehicle-coupon-recommendation.csv
└── Causal Coupon Targeting1.pdf
```

## Running the project

```bash
pip install -r requirements.txt
streamlit run app.py
```

The application downloads the UCI dataset and falls back to the included CSV when remote access is unavailable.

## Limitations

- The treatment was not assigned by this project; causal interpretation depends on no important unmeasured confounding after conditioning on the included covariates.
- Matching balance reduces observed imbalance but cannot establish balance on unobserved variables.
- The T-learner's customer-level effects are heterogeneous model estimates, not directly observed individual causal effects.
- The what-if controls can create combinations with limited empirical support and should be interpreted as model sensitivity, not guaranteed intervention outcomes.
- The ROI calculation depends on user-specified purchase-value and coupon-cost assumptions.
- A randomized controlled campaign or prospective A/B test would provide stronger validation.

## Technology

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Streamlit` · `Plotly`

## Suggested next steps

- Persist the encoder, outcome models and propensity model for external inference.
- Add automated tests for split isolation, feature alignment, caliper matching and business rules.
- Add sensitivity analysis for unmeasured confounding.
- Compare against an S-learner or causal forest using the same held-out protocol.
- Validate targeting policy value in a randomized experiment.
