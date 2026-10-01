from pathlib import Path

import pandas as pd


# =========================================================
# CAMINHOS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PLAYERS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "players.csv.gz"
)

APPEARANCES_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "appearances.csv.gz"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "players_model.csv"
)


# =========================================================
# PERÍODO ANALISADO
# =========================================================

START_DATE = pd.Timestamp("2025-07-01")
REFERENCE_DATE = pd.Timestamp("2026-06-12")


# =========================================================
# CARREGAMENTO
# =========================================================

def load_data():

    players = pd.read_csv(
        PLAYERS_PATH,
        low_memory=False,
    )

    appearances = pd.read_csv(
        APPEARANCES_PATH,
        low_memory=False,
    )

    return players, appearances


# =========================================================
# ESTATÍSTICAS DOS JOGADORES
# =========================================================

def aggregate_appearances(
    appearances: pd.DataFrame,
):

    appearances = appearances.copy()

    appearances["date"] = pd.to_datetime(
        appearances["date"],
        errors="coerce",
    )

    appearances = appearances[
        appearances["date"].between(
            START_DATE,
            REFERENCE_DATE,
        )
    ].copy()

    stats = (
        appearances
        .groupby("player_id", as_index=False)
        .agg(
            minutes=(
                "minutes_played",
                "sum",
            ),
            goals=(
                "goals",
                "sum",
            ),
            assists=(
                "assists",
                "sum",
            ),
        )
    )

    return stats


# =========================================================
# DATASET FINAL
# =========================================================

def build_dataset(
    players: pd.DataFrame,
    stats: pd.DataFrame,
):

    players = players[
        [
            "player_id",
            "name",
            "date_of_birth",
            "position",
            "market_value_in_eur",
        ]
    ].copy()

    players["date_of_birth"] = pd.to_datetime(
        players["date_of_birth"],
        errors="coerce",
    )

    # Idade na data de referência
    players["age"] = (
        (
            REFERENCE_DATE
            - players["date_of_birth"]
        ).dt.days
        / 365.2425
    )

    df = players.merge(
        stats,
        on="player_id",
        how="inner",
    )

    df = df.rename(
        columns={
            "name": "player_name",
        }
    )

    # Apenas jogadores com informações necessárias
    df = df.dropna(
        subset=[
            "player_name",
            "age",
            "position",
            "market_value_in_eur",
            "minutes",
            "goals",
            "assists",
        ]
    ).copy()

    # Valor de mercado precisa ser positivo
    df = df[
        df["market_value_in_eur"] > 0
    ].copy()

    # Precisa ter jogado no período
    df = df[
        df["minutes"] > 0
    ].copy()

    # Idade arredondada apenas para facilitar leitura
    df["age"] = df["age"].round(2)

    df = df[
        [
            "player_id",
            "player_name",
            "age",
            "position",
            "minutes",
            "goals",
            "assists",
            "market_value_in_eur",
        ]
    ]

    df = df.sort_values(
        "market_value_in_eur",
        ascending=False,
    ).reset_index(
        drop=True
    )

    return df


# =========================================================
# SALVAMENTO
# =========================================================

def save_dataset(
    df: pd.DataFrame,
):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n=== DATASET CRIADO ===")

    print(
        f"Jogadores: {len(df):,}"
    )

    print(
        f"Arquivo: {OUTPUT_PATH}"
    )

    print(
        "\nColunas:"
    )

    print(
        df.columns.tolist()
    )

    print(
        "\nPrimeiros jogadores:"
    )

    print(
        df.head(10).to_string(
            index=False
        )
    )


# =========================================================
# EXECUÇÃO
# =========================================================

if __name__ == "__main__":

    players, appearances = load_data()

    stats = aggregate_appearances(
        appearances
    )

    dataset = build_dataset(
        players,
        stats,
    )

    save_dataset(
        dataset
    )