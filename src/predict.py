import pickle
import pandas as pd
from src.preprocessing import clean_data, encode_features
from src.config import MODEL_PATH, TARGET_COL


def load_model(path=MODEL_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)


def align_columns(df, model):
    """Reorder columns to match model training order."""
    if hasattr(model, 'feature_names_in_'):
        cols = [c for c in model.feature_names_in_ if c in df.columns]
        return df[cols]
    return df


def predict_single(model, input_dict):
    """
    Predict default risk for a single loan application.
    Returns dict with prediction (0/1), label (Repaid/Default), probability.
    """
    # TODO: pd.DataFrame([input_dict])
    # TODO: clean_data and encode_features
    # TODO: drop TARGET_COL if present
    # TODO: align_columns
    # TODO: model.predict and model.predict_proba
    # TODO: return {'prediction': int, 'label': str, 'probability': float}
    pass


def predict_batch(model, df):
    """
    Predict default risk for a batch of loan applications.
    Returns original df with Prediction, Probability, Label columns added.
    """
    # TODO: copy and process df
    # TODO: align_columns
    # TODO: model.predict and model.predict_proba
    # TODO: add columns and return
    pass
