"""
Módulo para classificar vinhos como sendo de
baixa/média qualidade ou alta qualidade.

A classe positiva corresponde a quality >= 7.
"""

from pathlib import Path
import argparse
import joblib
import pandas as pd


VARIAVEIS_MODELO = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def carregar_modelo(caminho_modelo):
    """Carrega o pipeline treinado."""

    caminho_modelo = Path(caminho_modelo)

    if not caminho_modelo.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {caminho_modelo}"
        )

    return joblib.load(caminho_modelo)


def prever_qualidade(
    dados,
    modelo,
    limiar=0.50
):
    """
    Calcula a probabilidade de alta qualidade
    e gera a classificação binária.
    """

    colunas_ausentes = [
        coluna
        for coluna in VARIAVEIS_MODELO
        if coluna not in dados.columns
    ]

    if colunas_ausentes:
        raise ValueError(
            "Colunas ausentes na base: "
            + ", ".join(colunas_ausentes)
        )

    caracteristicas = dados[
        VARIAVEIS_MODELO
    ].copy()

    probabilidades = (
        modelo.predict_proba(
            caracteristicas
        )[:, 1]
    )

    previsoes = (
        probabilidades
        >= limiar
    ).astype(int)

    resultado = dados.copy()

    resultado[
        "probabilidade_alta_qualidade"
    ] = probabilidades

    resultado[
        "classe_prevista"
    ] = previsoes

    resultado[
        "descricao_classe_prevista"
    ] = (
        resultado[
            "classe_prevista"
        ]
        .map({
            0: "Baixa ou média qualidade",
            1: "Alta qualidade"
        })
    )

    return resultado


def executar():
    """Executa as previsões por linha de comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Classificação da qualidade "
            "de vinhos."
        )
    )

    parser.add_argument(
        "--entrada",
        required=True,
        help="Caminho do arquivo CSV de entrada."
    )

    parser.add_argument(
        "--modelo",
        default=(
            "models/"
            "regressao_logistica_vinhos.joblib"
        ),
        help="Caminho do modelo treinado."
    )

    parser.add_argument(
        "--saida",
        default=(
            "results/"
            "previsoes_novas_amostras.csv"
        ),
        help="Caminho do arquivo de saída."
    )

    parser.add_argument(
        "--limiar",
        type=float,
        default=0.50,
        help="Limiar de classificação."
    )

    argumentos = parser.parse_args()

    dados = pd.read_csv(
        argumentos.entrada
    )

    modelo = carregar_modelo(
        argumentos.modelo
    )

    resultado = prever_qualidade(
        dados=dados,
        modelo=modelo,
        limiar=argumentos.limiar
    )

    caminho_saida = Path(
        argumentos.saida
    )

    caminho_saida.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    resultado.to_csv(
        caminho_saida,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"Previsões salvas em: "
        f"{caminho_saida}"
    )


if __name__ == "__main__":
    executar()
