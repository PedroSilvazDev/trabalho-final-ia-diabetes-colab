from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

DATASET_PATH = DATA_DIR / "diabetes.csv"
DATASET_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
)

TARGET_COLUMN = "Outcome"
FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2

KNN_BEST_CONFIG = {
    "imputer_strategy": "median",
    "n_neighbors": 13,
    "weights": "uniform",
    "metric": "euclidean",
}

SVM_BEST_CONFIG = {
    "imputer_strategy": "mean",
    "C": 1,
    "kernel": "rbf",
    "gamma": "scale",
    "class_weight": "balanced",
}

INTEGRANTES = [
    "Pedro Henrique da Silva - RA: 23021607-2",
    "Victor Hugo Rodrigues de Oliveira - RA: 23418156-2",
    "Victor Hungo Silva Garcia - RA: 23030968-2",
]
