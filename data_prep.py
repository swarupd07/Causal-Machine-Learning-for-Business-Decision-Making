# data_prep.py

import pandas as pd

# ------------------------------ Constants --------------------------------------

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00603/in-vehicle-coupon-recommendation.csv"
)
LOCAL_BACKUP = "in-vehicle-coupon-recommendation.csv"

NOMINAL_COLS = [
    "destination", "passanger", "weather", "time",
    "coupon", "gender", "maritalStatus", "education", "occupation",
]
ORDINAL_MAPS = {
    "age": {
        "below21": 18, "21": 21, "26": 26, "31": 31,
        "36": 36, "41": 41, "46": 46, "50plus": 50,
    },
    "income": {
        "Less than $12500": 1, "$12500 - $24999": 2, "$25000 - $37499": 3,
        "$37500 - $49999": 4, "$50000 - $62499": 5, "$62500 - $74999": 6,
        "$75000 - $87499": 7, "$87500 - $99999": 8, "$100000 or More": 9,
    },
    "Bar": {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4},
    "CoffeeHouse": {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4},
    "CarryAway": {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4},
    "RestaurantLessThan20": {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4},
    "Restaurant20To50": {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4},
}
NUMERIC_COLS = ["temperature", "has_children", "toCoupon_GEQ15min", "direction_same"]

#------------------------------ Functions --------------------------------------
# Load, clean, and prepare the dataset for causal analysis. Returns X, T, Y.
# 1. Load the dataset from the UCI repository (or local backup if offline)
def load_data():
    df = pd.read_csv(DATA_URL)

    # Download the dataset, or fall back to a local copy if offline

    """
    try:
        df = pd.read_csv(DATA_URL)
    except Exception:
        df = pd.read_csv(LOCAL_BACKUP)
    
    """
    
    return df

# 2. Clean the dataset by dropping unusable columns and rows, filling missing values, and removing duplicates
def clean_data(df):
    # Dropping unusable columns/rows so the rest of the pipeline is simple
    df = df.copy()
    if "car" in df.columns:
        df = df.drop(columns=["car"])

    freq_cols = ["Bar", "CoffeeHouse", "CarryAway",
                 "RestaurantLessThan20", "Restaurant20To50"]
    for col in freq_cols:
        df[col] = df[col].fillna("never")

    df = df.dropna()
    df = df.drop_duplicates()
    return df

# 3. Prepare the dataset for causal analysis by building X (confounders), T (treatment), and Y (outcome)
def prepare_causal_data(df):
    """
    Build X (confounders), T (treatment), Y (outcome) for causal analysis.
    Also returns df_clean (aligned raw rows) so the dashboard can show
    human-readable info (coupon type, destination, etc.) for each customer.
    """
    df = df.copy()

    T = (df["expiration"] == "1d").astype(int)
    Y = df["Y"].astype(int)

    ordinal_df = pd.DataFrame({
        col: df[col].map(mapping) for col, mapping in ORDINAL_MAPS.items()
    })
    numeric_df = df[NUMERIC_COLS].astype(float)
    nominal_df = pd.get_dummies(df[NOMINAL_COLS], drop_first=True)

    X = pd.concat([numeric_df, ordinal_df, nominal_df], axis=1)
    X = X.dropna()

    T = T.loc[X.index].reset_index(drop=True)
    Y = Y.loc[X.index].reset_index(drop=True)
    df_clean = df.loc[X.index].reset_index(drop=True)
    X = X.reset_index(drop=True)

    return df_clean, X, T, Y

# 4. Convenience wrapper: load -> clean -> build X, T, Y in one call
def load_and_prepare():
    # Convenience wrapper: load -> clean -> build X, T, Y in one call
    df = load_data()
    df = clean_data(df)
    return prepare_causal_data(df)
