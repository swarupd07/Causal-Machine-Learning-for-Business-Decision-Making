# Loading, cleaning, and encoding the coupon dataset for causal analysis

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00603/in-vehicle-coupon-recommendation.csv"
)
LOCAL_BACKUP = Path(__file__).with_name("in-vehicle-coupon-recommendation.csv")

NOMINAL_COLS = [
    "destination",
    "passanger",
    "weather",
    "time",
    "coupon",
    "gender",
    "maritalStatus",
    "education",
    "occupation",
]
ORDINAL_MAPS = {
    "age": {
        "below21": 18,
        "21": 21,
        "26": 26,
        "31": 31,
        "36": 36,
        "41": 41,
        "46": 46,
        "50plus": 50,
    },
    "income": {
        "Less than $12500": 1,
        "$12500 - $24999": 2,
        "$25000 - $37499": 3,
        "$37500 - $49999": 4,
        "$50000 - $62499": 5,
        "$62500 - $74999": 6,
        "$75000 - $87499": 7,
        "$87500 - $99999": 8,
        "$100000 or More": 9,
    },
    "Bar": {"never": 0, "less1": 1, "1~3": 2, "4~8": 6, "gt8": 10},
    "CoffeeHouse": {"never": 0, "less1": 1, "1~3": 2, "4~8": 6, "gt8": 10},
    "CarryAway": {"never": 0, "less1": 1, "1~3": 2, "4~8": 6, "gt8": 10},
    "RestaurantLessThan20": {
        "never": 0,
        "less1": 1,
        "1~3": 2,
        "4~8": 6,
        "gt8": 10,
    },
    "Restaurant20To50": {
        "never": 0,
        "less1": 1,
        "1~3": 2,
        "4~8": 6,
        "gt8": 10,
    },
}
NUMERIC_COLS = [
    "temperature",
    "has_children",
    "toCoupon_GEQ15min",
    "direction_same",
]


def load_data(data_url=DATA_URL, local_backup=LOCAL_BACKUP):
    try:
        return pd.read_csv(data_url)
    except Exception as remote_error:
        backup_path = Path(local_backup)
        if backup_path.exists():
            return pd.read_csv(backup_path)
        raise RuntimeError(
            "Could not download the UCI dataset and no local backup was found at "
            f"{backup_path}."
        ) from remote_error


def clean_data(df):
    df = df.copy()
    if "car" in df.columns:
        df = df.drop(columns=["car"])

    frequency_cols = [
        "Bar",
        "CoffeeHouse",
        "CarryAway",
        "RestaurantLessThan20",
        "Restaurant20To50",
    ]
    for column in frequency_cols:
        df[column] = df[column].fillna("never")
    return df.dropna().drop_duplicates()


def make_encoder():
    return OneHotEncoder(
        handle_unknown="ignore",
        drop="first",
        sparse_output=False,
        dtype=float,
    )


def _base_features(df):
    ordinal_df = pd.DataFrame(
        {column: df[column].map(mapping) for column, mapping in ORDINAL_MAPS.items()},
        index=df.index,
    )
    numeric_df = df[NUMERIC_COLS].astype(float)
    return pd.concat([numeric_df, ordinal_df], axis=1)


def transform_causal_features(df, encoder):
    
    base = _base_features(df)
    valid_index = base.dropna().index
    base = base.loc[valid_index]
    nominal = df.loc[valid_index, NOMINAL_COLS]
    encoded = encoder.transform(nominal)
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(NOMINAL_COLS),
        index=valid_index,
    )
    return pd.concat([base, encoded_df], axis=1).astype(float)


def prepare_causal_data(df, encoder=None):
    df = df.copy()
    base = _base_features(df)
    valid_index = base.dropna().index
    df_valid = df.loc[valid_index]

    if encoder is None:
        encoder = make_encoder()
        encoder.fit(df_valid[NOMINAL_COLS])
    X = transform_causal_features(df_valid, encoder)
    aligned_index = X.index

    T = (df.loc[aligned_index, "expiration"] == "1d").astype(int).reset_index(drop=True)
    Y = df.loc[aligned_index, "Y"].astype(int).reset_index(drop=True)
    df_clean = df.loc[aligned_index].reset_index(drop=True)
    X = X.reset_index(drop=True)
    return df_clean, X, T, Y, encoder


def load_and_prepare():
    return prepare_causal_data(clean_data(load_data()))
