import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import lightgbm as lgb
from scipy.stats import randint, loguniform, uniform

from .preprocessing import load_data, prepare_features
from .config import MODEL_PATH, RANDOM_STATE


def get_models_and_params(spw):
    """
    Define all models and their hyperparameter search spaces.
    spw = scale_pos_weight for handling class imbalance.

    TODO: Add LightGBM and one more model of your choice.
    The Decision Tree and Logistic Regression are provided as examples.
    """
    models = {
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {
                "max_depth": randint(3, 15),
                "min_samples_leaf": randint(1, 20),
                "criterion": ["gini", "entropy"],
            },
        ),
        "Logistic Regression": (
            LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
            ),
            {"C": loguniform(0.01, 100)},
        ),
        # TODO: Add Random Forest
        "Random Forest": (
            RandomForestClassifier(
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
            ),
            {
                "n_estimators": randint(100, 400),
                "max_depth": randint(3, 12),
                "max_features": ["sqrt", "log2"],
            },
        ),
        # TODO: Add XGBoost
        "XGBoost": (
            XGBClassifier(
                scale_pos_weight=spw,
                eval_metric="logloss",
                verbosity=0,
                random_state=RANDOM_STATE,
            ),
            {
                "n_estimators": randint(100, 400),
                "learning_rate": loguniform(1e-3, 3e-1),
                "max_depth": randint(2, 8),
            },
        ),
        # TODO: Add LightGBM
        "LightGBM": (
            lgb.LGBMClassifier(
                scale_pos_weight=spw, random_state=RANDOM_STATE, verbose=-1
            ),
            {
                "n_estimators": randint(100, 400),
                "learning_rate": loguniform(1e-3, 3e-1),
                "num_leaves": randint(20, 80),
            },
        ),
    }
    return models


def tune_and_compare(models, X_train, y_train, X_test, y_test, n_iter=20, cv=5):
    """
    Run RandomizedSearchCV for each model and return comparison results.
    """
    results = []
    best_models = {}

    for name, (model, params) in models.items():
        print(f"Tuning {name}...")

        # TODO: Create RandomizedSearchCV with n_iter, cv=5, scoring='roc_auc'
        # Hint: search = RandomizedSearchCV(
        #           estimator=model,
        #           param_distributions=params,
        #           n_iter=n_iter, cv=cv, scoring='roc_auc',
        #           random_state=RANDOM_STATE, n_jobs=-1
        #       )
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=n_iter,
            cv=cv,
            scoring="roc_auc",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        # TODO: Fit the search on X_train and y_train
        search.fit(X_train, y_train)
        # TODO: Get predictions and compute roc_auc_score on X_test and y_test
        y_pred = search.predict(X_test)
        y_prob = search.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)

        # TODO: Append results dict to results list
        results.append(
            {
                "Model": name,
                "CV ROC-AUC": search.best_score_,
                "Test ROC-AUC": roc_auc,
                "Recall (Default)": classification_report(
                    y_test, y_pred, output_dict=True
                )["1"]["recall"],
            }
        )

        # TODO: Store best model in best_models[name]
        best_models[name] = search.best_estimator_

    results_df = pd.DataFrame(results).sort_values("Test ROC-AUC", ascending=False)
    return results_df, best_models


def save_model(model, path=MODEL_PATH):
    """Save the best model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved: {path}")


if __name__ == "__main__":
    df = load_data()
    print(f"Data loaded: {df.shape}")

    X_train, X_test, y_train, y_test = prepare_features(df)
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / pos
    print(f"scale_pos_weight: {spw:.2f}")

    models = get_models_and_params(spw)

    print("\nRunning RandomizedSearchCV...\n")
    results_df, best_models = tune_and_compare(models, X_train, y_train, X_test, y_test)

    print("\n--- Model Comparison ---")
    print(
        results_df[
            ["Model", "CV ROC-AUC", "Test ROC-AUC", "Recall (Default)"]
        ].to_string(index=False)
    )

    best_name = results_df.iloc[0]["Model"]
    best_model = best_models[best_name]
    print(f"\nBest model: {best_name}")

    save_model(best_model)
    print("Training complete.")
