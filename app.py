from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import (
    load_model,
    predict_value,
)


# =========================================================
# CAMINHOS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

FIGURE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "actual_vs_predicted_log_target.png"
)


# =========================================================
# RESULTADOS DA AVALIAÇÃO
# =========================================================

MODEL_MAE = 4_276_655
MODEL_R2 = 0.2867

BASELINE_IMPROVEMENT = 17.7

MEDIAN_ERROR = 1_082_526
P80_ERROR = 4_735_271
P90_ERROR = 11_976_072


# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================

st.set_page_config(
    page_title="Football Player Value Predictor",
    page_icon="⚽",
    layout="centered",
)


# =========================================================
# TÍTULO
# =========================================================

st.title(
    "⚽ Football Player Value Predictor"
)

st.write(
    """
    Estimate a football player's market value using
    a simple Machine Learning model.
    """
)

st.caption(
    """
    Educational portfolio project built with Python,
    scikit-learn and Streamlit.
    """
)


# =========================================================
# MODELO
# =========================================================

model = load_model()


# =========================================================
# FORMULÁRIO
# =========================================================

with st.form(
    "player_form"
):

    age = st.number_input(
        "Age",
        min_value=15,
        max_value=45,
        value=21,
        step=1,
    )

    position = st.selectbox(
        "Position",
        [
            "Attack",
            "Midfield",
            "Defender",
            "Goalkeeper",
        ],
    )

    minutes = st.number_input(
        "Minutes played",
        min_value=0,
        value=3000,
        step=100,
    )

    goals = st.number_input(
        "Goals",
        min_value=0,
        value=18,
        step=1,
    )

    assists = st.number_input(
        "Assists",
        min_value=0,
        value=10,
        step=1,
    )

    submitted = st.form_submit_button(
        "Predict market value"
    )


# =========================================================
# PREVISÃO
# =========================================================

if submitted:

    player = pd.DataFrame(
        [
            {
                "age": float(age),
                "position": position,
                "minutes": float(minutes),
                "goals": float(goals),
                "assists": float(assists),
            }
        ]
    )

    predicted_value = predict_value(
        model,
        player,
    )

    st.subheader(
        "Estimated market value"
    )

    st.metric(
        label="Prediction",
        value=f"€{predicted_value:,.0f}",
    )

    st.caption(
        """
        This estimate is produced by an educational
        Linear Regression model and should not be
        interpreted as a professional player valuation.
        """
    )


# =========================================================
# DESEMPENHO DO MODELO
# =========================================================

st.divider()

st.header(
    "Model performance"
)

st.write(
    """
    The model was evaluated on 1,638 players that were
    not used to fit the regression.
    """
)

column_1, column_2, column_3 = st.columns(
    3
)

with column_1:

    st.metric(
        label="Test MAE",
        value="€4.28M",
    )

with column_2:

    st.metric(
        label="Better than baseline",
        value="17.7%",
    )

with column_3:

    st.metric(
        label="Median absolute error",
        value="€1.08M",
    )

st.write(
    f"""
    **R² on the test set:** {MODEL_R2:.4f}

    - 50% of predictions had an absolute error of up to
      **€{MEDIAN_ERROR / 1_000_000:.2f}M**
    - 80% had an error of up to
      **€{P80_ERROR / 1_000_000:.2f}M**
    - 90% had an error of up to
      **€{P90_ERROR / 1_000_000:.2f}M**
    """
)


# =========================================================
# GRÁFICO
# =========================================================

st.subheader(
    "Actual vs predicted values"
)

if FIGURE_PATH.exists():

    st.image(
        str(FIGURE_PATH),
        caption=(
            "Market value in the test set: "
            "actual values compared with model predictions."
        ),
        use_container_width=True,
    )

else:

    st.info(
        "Evaluation figure not found."
    )


# =========================================================
# LIMITAÇÕES
# =========================================================

st.subheader(
    "How should this model be interpreted?"
)

st.info(
    """
    The model performs substantially better for players
    valued below €5 million.

    Prediction errors increase considerably for high-value
    players, which tend to be underestimated by this simple
    model.
    """
)

with st.expander(
    "Model limitations"
):

    st.write(
        """
        The model uses only five features:

        - age
        - position
        - minutes played
        - goals
        - assists

        It does not consider factors such as club,
        league strength, contract duration, reputation,
        injury history, future potential or transfer
        negotiation context.

        The output should therefore be understood as an
        educational statistical estimate rather than a
        professional valuation.
        """
    )