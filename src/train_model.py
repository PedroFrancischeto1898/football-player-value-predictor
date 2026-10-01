from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =========================================================
# CAMINHOS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "players_model.csv"
)

FIGURE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "actual_vs_predicted_log_target.png"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "market_value_model.joblib"
)

# =========================================================
# FEATURES E TARGET
# =========================================================

NUMERIC_FEATURES = [
    "age",
    "minutes",
    "goals",
    "assists",
]

CATEGORICAL_FEATURES = [
    "position",
]

TARGET = "market_value_in_eur"


# =========================================================
# CARREGAMENTO
# =========================================================

def load_dataset():

    df = pd.read_csv(
        DATASET_PATH
    )

    return df


# =========================================================
# MODELO
# =========================================================

def build_model():

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "position",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="passthrough",
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "regression",
                LinearRegression(),
            ),
        ]
    )

    return model


# =========================================================
# TREINAMENTO
# =========================================================

def train_model(
    df: pd.DataFrame,
):

    features = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    X = df[
        features
    ]

    # Target original em euros
    y = df[
        TARGET
    ]

    # Transformação logarítmica
    y_log = np.log1p(
        y
    )

    (
        X_train,
        X_test,
        y_train_log,
        y_test_log,
        y_train_euro,
        y_test_euro,
    ) = train_test_split(
        X,
        y_log,
        y,
        test_size=0.20,
        random_state=42,
    )

    model = build_model()

    model.fit(
        X_train,
        y_train_log,
    )

    predictions_log = model.predict(
        X_test
    )

    # Retorno das previsões para euros
    predictions_euro = np.expm1(
        predictions_log
    )

    # Valor de mercado negativo não faz sentido
    predictions_euro = np.maximum(
        predictions_euro,
        0,
    )

    return (
        model,
        y_test_euro,
        predictions_euro,
        len(X_train),
        len(X_test),
    )


# =========================================================
# AVALIAÇÃO
# =========================================================

def evaluate_model(
    y_test,
    predictions,
):

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    print(
        "\n=== RESULTADOS ==="
    )

    print(
        "Target: log1p(market_value_in_eur)"
    )

    print(
        f"MAE: €{mae:,.0f}"
    )

    print(
        f"R² em euros: {r2:.4f}"
    )


# =========================================================
# GRÁFICO
# =========================================================

def create_plot(
    y_test,
    predictions,
):

    FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        y_test,
        predictions,
        alpha=0.5,
    )

    minimum = min(
        y_test.min(),
        predictions.min(),
    )

    maximum = max(
        y_test.max(),
        predictions.max(),
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
    )

    plt.xlabel(
        "Valor de mercado real (€)"
    )

    plt.ylabel(
        "Valor de mercado previsto (€)"
    )

    plt.title(
        "Valor real × valor previsto — target log"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_PATH,
        dpi=150,
    )

    plt.close()

    print(
        f"\nGráfico salvo em:\n{FIGURE_PATH}"
    )

# =========================================================
# SALVAR MODELO
# =========================================================

def save_model(model):

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print(
        f"\nModelo salvo em:\n{MODEL_PATH}"
    )

# =========================================================
# EXECUÇÃO
# =========================================================

if __name__ == "__main__":

    dataset = load_dataset()

    (
        model,
        y_test,
        predictions,
        train_size,
        test_size,
    ) = train_model(
        dataset
    )

    print(
        "\n=== DATASET ==="
    )

    print(
        f"Jogadores: {len(dataset):,}"
    )

    print(
        f"Treino: {train_size:,}"
    )

    print(
        f"Teste: {test_size:,}"
    )

    evaluate_model(
        y_test,
        predictions,
    )

    create_plot(
        y_test,
        predictions,
    )
    
    save_model(
        model
    )