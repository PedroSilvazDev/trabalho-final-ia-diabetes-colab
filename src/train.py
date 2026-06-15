from dataclasses import dataclass

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import KNN_BEST_CONFIG, RANDOM_STATE, SVM_BEST_CONFIG
from .evaluate import compute_metrics
from .preprocess import SplitData


@dataclass
class ModelResult:
    model_name: str
    estimator: Pipeline
    best_params: dict
    y_pred: np.ndarray
    y_proba: np.ndarray
    metrics: dict
    preprocessing_summary: str


def build_knn_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=KNN_BEST_CONFIG["imputer_strategy"])),
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=KNN_BEST_CONFIG["n_neighbors"],
                    weights=KNN_BEST_CONFIG["weights"],
                    metric=KNN_BEST_CONFIG["metric"],
                ),
            ),
        ]
    )


def build_svm_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=SVM_BEST_CONFIG["imputer_strategy"])),
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    C=SVM_BEST_CONFIG["C"],
                    kernel=SVM_BEST_CONFIG["kernel"],
                    gamma=SVM_BEST_CONFIG["gamma"],
                    class_weight=SVM_BEST_CONFIG["class_weight"],
                    probability=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _knn_summary() -> str:
    return (
        f"KNN: imputacao {KNN_BEST_CONFIG['imputer_strategy']}, StandardScaler, "
        f"k={KNN_BEST_CONFIG['n_neighbors']}, "
        f"weights={KNN_BEST_CONFIG['weights']}, "
        f"metric={KNN_BEST_CONFIG['metric']}"
    )


def _svm_summary() -> str:
    return (
        f"SVM: imputacao {SVM_BEST_CONFIG['imputer_strategy']}, StandardScaler, "
        f"kernel={SVM_BEST_CONFIG['kernel']}, C={SVM_BEST_CONFIG['C']}, "
        f"gamma={SVM_BEST_CONFIG['gamma']}, "
        f"class_weight={SVM_BEST_CONFIG['class_weight']}"
    )


def train_knn(data: SplitData) -> ModelResult:
    model = build_knn_pipeline()
    model.fit(data.x_train, data.y_train)

    y_pred = model.predict(data.x_test)
    y_proba = model.predict_proba(data.x_test)[:, 1]
    metrics = compute_metrics(data.y_test, y_pred, y_proba)

    return ModelResult(
        model_name="KNN",
        estimator=model,
        best_params=dict(KNN_BEST_CONFIG),
        y_pred=y_pred,
        y_proba=y_proba,
        metrics=metrics,
        preprocessing_summary=_knn_summary(),
    )


def train_svm(data: SplitData) -> ModelResult:
    model = build_svm_pipeline()
    model.fit(data.x_train, data.y_train)

    y_pred = model.predict(data.x_test)
    y_proba = model.predict_proba(data.x_test)[:, 1]
    metrics = compute_metrics(data.y_test, y_pred, y_proba)

    return ModelResult(
        model_name="SVM",
        estimator=model,
        best_params=dict(SVM_BEST_CONFIG),
        y_pred=y_pred,
        y_proba=y_proba,
        metrics=metrics,
        preprocessing_summary=_svm_summary(),
    )


def print_result_summary(result: ModelResult, data: SplitData) -> None:
    train_accuracy = accuracy_score(data.y_train, result.estimator.predict(data.x_train))

    print(f"\n=== {result.model_name} ===")
    print(f"Pre-processamento: {result.preprocessing_summary}")
    print(f"Configuracao utilizada: {result.best_params}")
    print(f"Acuracia no treino: {train_accuracy:.4f}")
    print(f"Acuracia no teste: {result.metrics['accuracy']:.4f}")
    print(f"Precisao: {result.metrics['precision']:.4f}")
    print(f"Revocacao: {result.metrics['recall']:.4f}")
    print(f"F1-score: {result.metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {result.metrics['roc_auc']:.4f}")
