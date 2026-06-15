import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Sem diabetes", "Com diabetes"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Matriz de confusao - {model_name}")
    fig.tight_layout()
    plt.show()


def plot_roc_curves(results: list[dict]):
    fig, ax = plt.subplots(figsize=(7, 6))
    for result in results:
        fpr, tpr, _ = roc_curve(result["y_true"], result["y_proba"])
        ax.plot(
            fpr,
            tpr,
            label=f"{result['model_name']} (AUC = {result['metrics']['roc_auc']:.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Classificador aleatorio")
    ax.set_xlabel("Taxa de falso positivo")
    ax.set_ylabel("Taxa de verdadeiro positivo")
    ax.set_title("Curvas ROC - KNN vs SVM")
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_metrics_comparison(metrics_df: pd.DataFrame):
    melted = metrics_df.melt(
        id_vars="modelo",
        value_vars=["accuracy", "precision", "recall", "f1_score", "roc_auc"],
        var_name="metrica",
        value_name="valor",
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=melted, x="metrica", y="valor", hue="modelo", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Comparacao de metricas entre modelos")
    ax.set_xlabel("Metrica")
    ax.set_ylabel("Valor")
    fig.tight_layout()
    plt.show()


def plot_knn_k_search(grid_search: GridSearchCV):
    results = pd.DataFrame(grid_search.cv_results_)
    k_col = next(c for c in results.columns if c.endswith("n_neighbors"))
    grouped = (
        results.groupby(k_col)["mean_test_score"]
        .mean()
        .reset_index()
        .sort_values(k_col)
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(grouped[k_col], grouped["mean_test_score"], marker="o")
    ax.set_title("KNN - validacao cruzada por valor de k")
    ax.set_xlabel("Numero de vizinhos (k)")
    ax.set_ylabel("F1-score medio (validacao cruzada)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
