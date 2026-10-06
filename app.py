import pandas as pd
import streamlit as st

from src.predict import (
    load_model,
    predict_value,
)


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