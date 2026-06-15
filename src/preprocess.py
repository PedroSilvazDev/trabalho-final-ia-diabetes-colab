from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


@dataclass
class SplitData:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


def replace_invalid_zeros(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    zero_as_missing = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]
    for column in zero_as_missing:
        cleaned[column] = cleaned[column].replace(0, np.nan)
    return cleaned


def split_data(df: pd.DataFrame) -> SplitData:
    cleaned = replace_invalid_zeros(df)
    x = cleaned[FEATURE_COLUMNS]
    y = cleaned[TARGET_COLUMN].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return SplitData(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train.to_numpy(),
        y_test=y_test.to_numpy(),
        feature_names=FEATURE_COLUMNS,
    )
