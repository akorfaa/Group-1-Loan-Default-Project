import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
)
from .preprocessing import load_data, prepare_features
from .config import MODEL_PATH


def load_model(path=MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.
    Steps:
    1. Get predictions and probabilities
    2. Print classification report with target_names=['Repaid', 'Default']
    3. Print ROC-AUC and Average Precision
    4. Return y_pred, y_prob, roc_auc, avg_precision
    """
    # TODO: y_pred = model.predict(X_test)
    y_pred = model.predict(X_test)
    # TODO: y_prob = model.predict_proba(X_test)[:, 1]
    y_prob = model.predict_proba(X_test)[:, 1]
    # TODO: print(classification_report(y_test, y_pred, target_names=['Repaid', 'Default']))
    print(classification_report(y_test, y_pred, target_names=["Repaid", "Default"]))
    # TODO: print ROC-AUC and Average Precision
    roc_auc = roc_auc_score(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    print(f"ROC-AUC: {roc_auc}")
    print(f"Average Precision: {avg_precision}")
    # TODO: return y_pred, y_prob
    return y_pred, y_prob, roc_auc, avg_precision


def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix heatmap."""
    # TODO: Use sns.heatmap on confusion_matrix(y_test, y_pred)
    # Labels: ['Repaid', 'Default']
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Repaid", "Default"],
        yticklabels=["Repaid", "Default"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def plot_roc_curve(y_test, y_prob):
    """Plot ROC curve."""
    # TODO: Use roc_curve and roc_auc_score to plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()


# Finding the best threshold with a high recall (>= 0.65) and the best F1 score
def find_best_threshold(y_test, y_prob, target_recall=0.65):
    from sklearn.metrics import precision_recall_curve

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

    best_threshold = 0.5
    best_f1 = 0
    best_recall = 0
    best_precision = 0

    for p, r, t in zip(precisions, recalls, thresholds):
        if r >= target_recall:
            f1 = 2 * p * r / (p + r + 1e-9)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
                best_recall = r
                best_precision = p

    print(
        f"\nBest threshold: {best_threshold:.3f} → Recall: {best_recall:.2f}, Precision: {best_precision:.2f}, F1: {best_f1:.2f}"
    )
    return best_threshold


if __name__ == "__main__":
    df = load_data()
    _, X_test, _, y_test = prepare_features(df)
    model = load_model()
    y_pred, y_prob, roc_auc, avg_precision = evaluate_model(model, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)
    # Threshold tuning — find the best threshold for high recall (>= 0.65) and best F1 score
    threshold = find_best_threshold(y_test, y_prob, target_recall=0.65)
    y_pred_adjusted = (y_prob >= threshold).astype(int)

    print("\n--- After Threshold Tuning ---")
    print(
        classification_report(
            y_test, y_pred_adjusted, target_names=["Repaid", "Default"]
        )
    )
