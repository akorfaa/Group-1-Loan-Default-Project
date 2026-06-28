import pickle
import pandas as pd
from .preprocessing import clean_data, encode_features
from .config import MODEL_PATH, TARGET_COL


def load_model(path=MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def align_columns(df, model):
    """Reorder columns to match model training order."""
    if hasattr(model, "feature_names_in_"):
        cols = [c for c in model.feature_names_in_ if c in df.columns]
        return df[cols]
    return df


def predict_single(model, input_dict):
    """
    Predict default risk for a single loan application.
    Returns dict with prediction (0/1), label (Repaid/Default), probability.
    """
    # TODO: pd.DataFrame([input_dict])
    df = pd.DataFrame([input_dict])
    # TODO: clean_data and encode_features
    df = clean_data(df)
    df = encode_features(df)
    # TODO: drop TARGET_COL if present
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])
    # TODO: align_columns
    df = align_columns(df, model)
    # TODO: model.predict and model.predict_proba
    prediction = model.predict(df)
    probability = model.predict_proba(df)[:, 1]
    # TODO: return {'prediction': int, 'label': str, 'probability': float}
    return {
        "prediction": int(prediction[0]),
        "label": "Default" if prediction[0] == 1 else "Repaid",
        "probability": float(probability[0]),
    }


def predict_batch(model, df):
    """
    Predict default risk for a batch of loan applications.
    Returns original df with Prediction, Probability, Label columns added.
    """
    # TODO: copy and process df
    result = df.copy()
    df_copy = clean_data(result.copy())
    df_copy = encode_features(df_copy)
    if TARGET_COL in df_copy.columns:
        df_copy = df_copy.drop(columns=[TARGET_COL], errors="ignore")
    # TODO: align_columns
    df_copy = align_columns(df_copy, model)
    # TODO: model.predict and model.predict_proba
    # Get predictions first before adding any new columns
    predictions = model.predict(df_copy)
    probabilities = model.predict_proba(df_copy)[:, 1]
    result["Prediction"] = predictions
    result["Probability"] = probabilities
    result["Label"] = result["Prediction"].apply(
        lambda x: "Default" if x == 1 else "Repaid"
    )
    # TODO: add columns and return
    return result
