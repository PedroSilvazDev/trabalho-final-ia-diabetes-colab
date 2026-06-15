from dataclasses import dataclass

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC

from .config import RANDOM_STATE
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
    grid_search: GridSearchCV | None = None


def _cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def build_knn_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier()),
        ]
    )


def build_svm_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("svm", SVC(probability=True, random_state=RANDOM_STATE)),
        ]
    )


def train_knn(data: SplitData) -> ModelResult:
    pipeline = build_knn_pipeline()
    param_grid = {
        "imputer__strategy": ["median"],
        "scaler": [StandardScaler(), RobustScaler(), MinMaxScaler()],
        "knn__n_neighbors": list(range(3, 22, 2)),
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"],
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=_cv(),
        n_jobs=-1,
    )
    grid_search.fit(data.x_train, data.y_train)

    model = grid_search.best_estimator_
    y_pred = model.predict(data.x_test)
    y_proba = model.predict_proba(data.x_test)[:, 1]
    metrics = compute_metrics(data.y_test, y_pred, y_proba)

    scaler_name = type(model.named_steps["scaler"]).__name__
    summary = (
        f"KNN: imputacao mediana, {scaler_name}, "
        f"k={model.named_steps['knn'].n_neighbors}, "
        f"weights={model.named_steps['knn'].weights}, "
        f"metric={model.named_steps['knn'].metric}"
    )

    return ModelResult(
        model_name="KNN",
        estimator=model,
        best_params=grid_search.best_params_,
        y_pred=y_pred,
        y_proba=y_proba,
        metrics=metrics,
        preprocessing_summary=summary,
        grid_search=grid_search,
    )


def train_svm(data: SplitData) -> ModelResult:
    pipeline = build_svm_pipeline()
    param_grid = {
        "imputer__strategy": ["median", "mean"],
        "scaler": [StandardScaler()],
        "svm__C": [0.1, 1, 10, 100],
        "svm__kernel": ["linear", "rbf"],
        "svm__gamma": ["scale", "auto"],
        "svm__class_weight": [None, "balanced"],
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=_cv(),
        n_jobs=-1,
    )
    grid_search.fit(data.x_train, data.y_train)

    model = grid_search.best_estimator_
    y_pred = model.predict(data.x_test)
    y_proba = model.predict_proba(data.x_test)[:, 1]
    metrics = compute_metrics(data.y_test, y_pred, y_proba)

    svm = model.named_steps["svm"]
    summary = (
        f"SVM: imputacao {model.named_steps['imputer'].strategy}, StandardScaler, "
        f"kernel={svm.kernel}, C={svm.C}, gamma={svm.gamma}, "
        f"class_weight={svm.class_weight}"
    )

    return ModelResult(
        model_name="SVM",
        estimator=model,
        best_params=grid_search.best_params_,
        y_pred=y_pred,
        y_proba=y_proba,
        metrics=metrics,
        preprocessing_summary=summary,
        grid_search=grid_search,
    )


def print_result_summary(result: ModelResult, data: SplitData) -> None:
    train_accuracy = accuracy_score(data.y_train, result.estimator.predict(data.x_train))

    print(f"\n=== {result.model_name} ===")
    print(f"Pre-processamento: {result.preprocessing_summary}")
    print(f"Melhores hiperparametros: {result.best_params}")
    print(f"Acuracia no treino: {train_accuracy:.4f}")
    print(f"Acuracia no teste: {result.metrics['accuracy']:.4f}")
    print(f"Precisao: {result.metrics['precision']:.4f}")
    print(f"Revocacao: {result.metrics['recall']:.4f}")
    print(f"F1-score: {result.metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {result.metrics['roc_auc']:.4f}")
