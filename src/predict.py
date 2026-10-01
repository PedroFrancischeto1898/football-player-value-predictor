from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =========================================================
# CAMINHO DO MODELO
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "market_value_model.joblib"
)


# =========================================================
# POSIÇÕES ACEITAS
# =========================================================

POSITIONS = [
    "Attack",
    "Midfield",
    "Defender",
    "Goalkeeper",
]


# =========================================================
# CARREGAMENTO
# =========================================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Modelo não encontrado. "
            "Execute primeiro: python src\\train_model.py"
        )

    return joblib.load(
        MODEL_PATH
    )


# =========================================================
# ENTRADA DO USUÁRIO
# =========================================================

def get_player_data():

    print("\n=== FOOTBALL PLAYER VALUE PREDICTOR ===")

    age = float(
        input("Idade: ")
    )

    print(
        "\nPosições disponíveis:"
    )

    for position in POSITIONS:
        print(
            f"- {position}"
        )

    position = input(
        "\nPosição: "
    ).strip().title()

    if position not in POSITIONS:
        raise ValueError(
            "Posição inválida. "
            f"Use uma destas: {', '.join(POSITIONS)}"
        )

    minutes = float(
        input("Minutos jogados: ")
    )

    goals = float(
        input("Gols: ")
    )

    assists = float(
        input("Assistências: ")
    )

    player = pd.DataFrame(
        [
            {
                "age": age,
                "minutes": minutes,
                "goals": goals,
                "assists": assists,
                "position": position,
            }
        ]
    )

    return player


# =========================================================
# PREVISÃO
# =========================================================

def predict_value(
    model,
    player,
):

    predicted_log_value = model.predict(
        player
    )[0]

    predicted_value = np.expm1(
        predicted_log_value
    )

    predicted_value = max(
        predicted_value,
        0,
    )

    return predicted_value


# =========================================================
# EXECUÇÃO
# =========================================================

if __name__ == "__main__":

    model = load_model()

    player = get_player_data()

    predicted_value = predict_value(
        model,
        player,
    )

    print(
        "\n=== RESULTADO ==="
    )

    print(
        "Valor de mercado estimado: "
        f"€{predicted_value:,.0f}"
    )