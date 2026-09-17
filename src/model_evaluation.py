import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(model, X_test, y_test, y_pred=None):
    if y_pred is None:
        y_pred = model.predict(X_test)

    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    except (AttributeError, IndexError, ValueError):
        roc_auc = None

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }
    return results


def plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix",
                          save_path=None):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_classification_report(y_test, y_pred, save_path=None):
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        report_df.iloc[:2, :4],
        annot=True, fmt=".2f", cmap="YlGnBu"
    )
    plt.title("Classification Report")
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curve(model, X_test, y_test, save_path=None):
    from sklearn.metrics import roc_curve, auc

    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except (AttributeError, IndexError):
        return None

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    import pandas as pd

    tree_models = (
        "RandomForestClassifier",
        "DecisionTreeClassifier",
        "GradientBoostingClassifier",
        "XGBClassifier",
    )
    model_name = type(model).__name__

    if model_name in tree_models:
        importances = model.feature_importances_
    else:
        try:
            importances = model.coef_[0]
        except (AttributeError, IndexError):
            return None

    feature_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=feature_imp, x="importance", y="feature")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
    return feature_imp
