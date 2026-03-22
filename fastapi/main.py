from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="API - Previsão de Preço de Carros")

# =========================
# LOAD MODEL
# =========================
model = joblib.load("../modelo/lgbm_best_model_pipeline.pkl")

# =========================
# LOAD DATASET
# =========================
df_carros = pd.read_csv("../dados/dataset_carros_brasil.csv")

# =========================
# SCHEMA (entrada)
# =========================
class CarroInput(BaseModel):
    Marca: str
    Modelo: str
    Ano: int
    Quilometragem: int
    Cor: str
    Cambio: str
    Combustivel: str
    Portas: int

# =========================
# FUNÇÃO RECOMENDAÇÃO
# =========================
def recomendar(valor_estimado):
    limite_inferior = valor_estimado * 0.90
    limite_superior = valor_estimado * 1.10

    carros_abaixo = df_carros[
        (df_carros['Valor_Venda'] >= limite_inferior) &
        (df_carros['Valor_Venda'] < valor_estimado)
    ]

    carros_acima = df_carros[
        (df_carros['Valor_Venda'] >= valor_estimado) &
        (df_carros['Valor_Venda'] <= limite_superior)
    ]

    recomendados = pd.concat([
        carros_abaixo.sample(min(2, len(carros_abaixo))) if not carros_abaixo.empty else pd.DataFrame(),
        carros_acima.sample(min(2, len(carros_acima))) if not carros_acima.empty else pd.DataFrame()
    ])

    if recomendados.empty:
        return []

    colunas = ['Marca', 'Modelo', 'Ano', 'Cor', 'Combustivel', 'Valor_Venda']
    recomendados = recomendados[colunas].sort_values(by='Valor_Venda')

    return recomendados.to_dict(orient="records")

# =========================
# ENDPOINT
# =========================
@app.post("/prever")
def prever_preco(carro: CarroInput):
    
    # Feature engineering
    idade_carro = 2026 - carro.Ano

    df_entrada = pd.DataFrame([{
        "Marca": carro.Marca,
        "Modelo": carro.Modelo,
        "Ano": carro.Ano,
        "Quilometragem": carro.Quilometragem,
        "Cor": carro.Cor,
        "Cambio": carro.Cambio,
        "Combustivel": carro.Combustivel,
        "Portas": carro.Portas,
        "idade_carro": idade_carro
    }])

    try:
        previsao = model.predict(df_entrada)[0]

    except:
        # fallback (caso não seja pipeline)
        df_encoded = pd.get_dummies(df_entrada)
        df_encoded = df_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
        previsao = model.predict(df_encoded)[0]

    recomendacoes = recomendar(previsao)

    return {
        "valor_estimado": round(float(previsao), 2),
        "recomendacoes": recomendacoes
    }