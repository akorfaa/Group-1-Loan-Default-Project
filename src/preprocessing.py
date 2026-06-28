import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from .config import (
    DATA_PATH,
    TARGET_COL,
    DROP_COLS,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    TEST_SIZE,
    RANDOM_STATE,
)


## LOAD DATA
def load_data(path=DATA_PATH):
    """Load the Nigeria loan default dataset."""
    # TODO: Read the CSV file and return a DataFrame
    # Hint: pd.read_csv(path)
    df = pd.read_csv(path)
    return df


## CLEAN DATA
def clean_data(df):
    """
    Clean the dataset.
    Steps:
    1. Drop columns in DROP_COLS
    2. Fill missing numerical values with the column median
    3. Fill missing categorical values with 'Unknown'
    """
    df = df.copy()

    # TODO: Step 1 — drop columns in DROP_COLS
    # Hint: df = df.drop(columns=DROP_COLS, errors='ignore')
    df = df.drop(columns=DROP_COLS, errors="ignore")

    # TODO: Step 2 — fill missing numerical values with median
    # Hint: for col in NUMERICAL_COLS:
    #           if col in df.columns:
    #               df[col] = df[col].fillna(df[col].median())
    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # TODO: Step 3 — fill missing categorical values with 'Unknown'
    # Hint: for col in CATEGORICAL_COLS:
    #           if col in df.columns:
    #               df[col] = df[col].fillna('Unknown')
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    return df


## ENCODE DATA
def encode_features(df):
    """
    Encode categorical columns using LabelEncoder.
    Loop through CATEGORICAL_COLS and label encode each one.
    """
    df = df.copy()
    le = LabelEncoder()

    # TODO: Loop through CATEGORICAL_COLS and encode each one
    # Hint: for col in CATEGORICAL_COLS:
    #           if col in df.columns:
    #               df[col] = le.fit_transform(df[col].astype(str))
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    return df


## PREPARE DATA - SUMMARY
def prepare_features(df):
    """
    Full pipeline — returns X and y ready for modelling.
    Steps:
    1. Call clean_data(df)
    2. Call encode_features(df)
    3. Separate X and y
    4. Split into train and test sets
    5. Return X_train, X_test, y_train, y_test
    """
    # TODO: Step 1 — clean
    df = clean_data(df)

    # TODO: Step 2 — encode
    df = encode_features(df)

    # TODO: Step 3 — X = df.drop(columns=[TARGET_COL])
    #                y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # TODO: Step 4 — train_test_split with stratify=y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # TODO: Step 5 — return X_train, X_test, y_train, y_test
    return X_train, X_test, y_train, y_test
