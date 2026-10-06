import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from train_model import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_model,
    load_dataset,
)


# =========================================================
# PREPARAÇÃO DA AVALIAÇÃO
# =========================================================

def prepare_test_data(
    df: pd.DataFrame,
):

    features = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    X = df[
        features
    ]

    y = df[
        TARGET
    ]

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# =========================================================
# MODELO
# =========================================================

def get_model_predictions(
    X_train,
    X_test,
    y_train,
):

    model = build_model()

    y_train_log = np.log1p(
        y_train
    )

    model.fit(
        X_train,
        y_train_log,
    )

    predictions_log = model.predict(
        X_test
    )

    predictions = np.expm1(
        predictions_log
    )

    predictions = np.maximum(
        predictions,
        0,
    )

    return predictions


# =========================================================
# BASELINE
# =========================================================

def get_baseline_predictions(
    y_train,
    test_size,
):

    baseline_value = y_train.median()

    predictions = np.full(
        test_size,
        baseline_value,
    )

    return (
        baseline_value,
        predictions,
    )


# =========================================================
# MÉTRICAS GERAIS
# =========================================================

def evaluate_general_metrics(
    y_test,
    model_predictions,
    baseline_predictions,
    baseline_value,
):

    model_mae = mean_absolute_error(
        y_test,
        model_predictions,
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions,
    )

    model_r2 = r2_score(
        y_test,
        model_predictions,
    )

    improvement = (
        (
            baseline_mae
            - model_mae
        )
        / baseline_mae
        * 100
    )

    print(
        "\n=== COMPARAÇÃO COM BASELINE ==="
    )

    print(
        "Baseline:"
        " prever sempre a mediana"
        " do conjunto de treino"
    )

    print(
        f"Valor previsto pelo baseline: "
        f"€{baseline_value:,.0f}"
    )

    print(
        f"\nMAE do baseline: "
        f"€{baseline_mae:,.0f}"
    )

    print(
        f"MAE do modelo: "
        f"€{model_mae:,.0f}"
    )

    print(
        f"Melhoria sobre o baseline: "
        f"{improvement:.1f}%"
    )

    print(
        f"R² do modelo: "
        f"{model_r2:.4f}"
    )


# =========================================================
# DISTRIBUIÇÃO DOS ERROS
# =========================================================

def evaluate_error_distribution(
    y_test,
    predictions,
):

    absolute_errors = np.abs(
        y_test.to_numpy()
        - predictions
    )

    p50 = np.percentile(
        absolute_errors,
        50,
    )

    p80 = np.percentile(
        absolute_errors,
        80,
    )

    p90 = np.percentile(
        absolute_errors,
        90,
    )

    print(
        "\n=== DISTRIBUIÇÃO DOS ERROS ==="
    )

    print(
        f"Erro absoluto mediano: "
        f"€{p50:,.0f}"
    )

    print(
        "\n50% das previsões têm erro "
        f"de até €{p50:,.0f}"
    )

    print(
        "80% das previsões têm erro "
        f"de até €{p80:,.0f}"
    )

    print(
        "90% das previsões têm erro "
        f"de até €{p90:,.0f}"
    )

    return absolute_errors


# =========================================================
# DESEMPENHO POR FAIXA DE VALOR
# =========================================================

def evaluate_by_market_value(
    y_test,
    predictions,
    absolute_errors,
):

    results = pd.DataFrame(
        {
            "actual_value": (
                y_test.to_numpy()
            ),
            "prediction": predictions,
            "absolute_error": (
                absolute_errors
            ),
        }
    )

    results[
        "market_value_range"
    ] = pd.cut(
        results[
            "actual_value"
        ],
        bins=[
            0,
            5_000_000,
            20_000_000,
            50_000_000,
            np.inf,
        ],
        labels=[
            "< €5M",
            "€5M–€20M",
            "€20M–€50M",
            "≥ €50M",
        ],
        right=False,
    )

    grouped = (
        results
        .groupby(
            "market_value_range",
            observed=True,
        )
        .agg(
            players=(
                "absolute_error",
                "size",
            ),
            mae=(
                "absolute_error",
                "mean",
            ),
            median_error=(
                "absolute_error",
                "median",
            ),
        )
    )

    print(
        "\n=== DESEMPENHO POR FAIXA ==="
    )

    for market_range, row in (
        grouped.iterrows()
    ):

        print(
            f"\n{market_range}"
        )

        print(
            f"Jogadores: "
            f"{int(row['players']):,}"
        )

        print(
            f"MAE: "
            f"€{row['mae']:,.0f}"
        )

        print(
            f"Erro mediano: "
            f"€{row['median_error']:,.0f}"
        )


# =========================================================
# EXECUÇÃO
# =========================================================

if __name__ == "__main__":

    dataset = load_dataset()

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = prepare_test_data(
        dataset
    )

    model_predictions = (
        get_model_predictions(
            X_train,
            X_test,
            y_train,
        )
    )

    (
        baseline_value,
        baseline_predictions,
    ) = get_baseline_predictions(
        y_train,
        len(y_test),
    )

    print(
        "\n=== AVALIAÇÃO DO MODELO ==="
    )

    print(
        f"Jogadores no teste: "
        f"{len(y_test):,}"
    )

    evaluate_general_metrics(
        y_test,
        model_predictions,
        baseline_predictions,
        baseline_value,
    )

    absolute_errors = (
        evaluate_error_distribution(
            y_test,
            model_predictions,
        )
    )

    evaluate_by_market_value(
        y_test,
        model_predictions,
        absolute_errors,
    )