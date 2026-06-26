import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, roc_curve
)
from src.preprocessing import load_data, prepare_features
from src.config import MODEL_PATH


def load_model(path=MODEL_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.
    Steps:
    1. Get predictions and probabilities
    2. Print classification report with target_names=['Repaid', 'Default']
    3. Print ROC-AUC and Average Precision
    4. Return y_pred and y_prob
    """
    # TODO: y_pred = model.predict(X_test)
    # TODO: y_prob = model.predict_proba(X_test)[:, 1]
    # TODO: print(classification_report(y_test, y_pred, target_names=['Repaid', 'Default']))
    # TODO: print ROC-AUC and Average Precision
    # TODO: return y_pred, y_prob
    pass


def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix heatmap."""
    # TODO: Use sns.heatmap on confusion_matrix(y_test, y_pred)
    # Labels: ['Repaid', 'Default']
    pass


def plot_roc_curve(y_test, y_prob):
    """Plot ROC curve."""
    # TODO: Use roc_curve and roc_auc_score to plot
    pass


if __name__ == '__main__':
    df = load_data()
    _, X_test, _, y_test = prepare_features(df)
    model = load_model()
    y_pred, y_prob = evaluate_model(model, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)
    plt.show()
