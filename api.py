import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# =================================================================
# 1. CONFIGURAÇÃO DA API E MOCK MODEL (CORREÇÃO DO NAMEOERROR)
# =================================================================

# CORREÇÃO: Define a classe MockModel globalmente
class MockModel:
    def predict(self, df): return np.array([0])
    def predict_proba(self, df): return np.array([[0.5, 0.5]])

app = FastAPI(title="API de Previsão de Doença Cardíaca")

# Configuração CORS 
origins = ["*"] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carregar o modelo uma única vez ao iniciar a API
MODELO_ARQUIVO = 'modelo_classificacao_doenca_cardiaca.pkl'

try:
    MODELO_CLASSIFICADOR = joblib.load(MODELO_ARQUIVO)
    print("STATUS: Modelo carregado com sucesso!")
except FileNotFoundError:
    print(f"ERRO CRÍTICO: Arquivo {MODELO_ARQUIVO} não encontrado. Usando MockModel.")
    MODELO_CLASSIFICADOR = MockModel()


# Lista de colunas esperadas pelo modelo (13 FEATURES - Ordem CRÍTICA)
MODEL_COLUMNS = [
    "age", "sex", "chest_pain_type", "resting_blood_pressure", "cholesterol", "fasting_blood_sugar", 
    "resting_electrocardiogram", "max_heart_rate_achieved", "exercise_induced_angina", 
    "st_depression", "st_slope", "num_major_vessels", "thalassemia"
]

# =================================================================
# 2. DEFINIÇÃO DA ESTRUTURA DE DADOS (Pydantic Model)
# =================================================================

class DadosPaciente(BaseModel):
    age: float
    sex: float
    chest_pain_type: float
    resting_blood_pressure: float
    cholesterol: float
    fasting_blood_sugar: float
    resting_electrocardiogram: float
    max_heart_rate_achieved: float
    exercise_induced_angina: float
    st_depression: float
    st_slope: float
    num_major_vessels: float
    thalassemia: float 

# =================================================================
# 3. ENDPOINT DE PREVISÃO
# =================================================================

@app.post("/predict")
def predict_heart_disease(dados: DadosPaciente):
    """
    Recebe os dados do paciente e retorna a previsão de risco cardíaco.
    """
    # A verificação agora funciona, pois MockModel está definida globalmente
    if MODELO_CLASSIFICADOR is None or isinstance(MODELO_CLASSIFICADOR, MockModel):
        return {"erro": "Modelo de classificação não está operacional."}, 500
        
    try:
        dados_dict = dados.model_dump()
        df_paciente = pd.DataFrame([dados_dict])
        
        # --- LOG 1: VERIFICAÇÃO DOS DADOS BRUTOS RECEBIDOS ---
        print("\n--- [LOG API] Dados Recebidos do Frontend (Ordem Pydantic) ---")
        print(df_paciente.head(1).to_dict('records'))
        
        # 4. PRÉ-PROCESSAMENTO (GARANTIR ORDEM E COLUNAS)
        df_paciente = df_paciente[MODEL_COLUMNS] 

        # --- LOG 2: VERIFICAÇÃO DA ORDEM FINAL (CRÍTICO) ---
        print("\n--- [LOG API] Colunas FINAIS ENVIADAS ao Modelo ---")
        print(df_paciente.columns.tolist())
        print(df_paciente.values[0].tolist()) # Os valores reais que o modelo está vendo
        print("-----------------------------------------------------")
        
        # 5. PREVISÃO
        previsao = MODELO_CLASSIFICADOR.predict(df_paciente)[0]
        prob_positiva = MODELO_CLASSIFICADOR.predict_proba(df_paciente)[:, 1][0] 
        
        # 6. RESPOSTA
        resultado = {
            "previsao_binaria": int(previsao),
            "probabilidade_doenca": round(prob_positiva * 100, 2),
            "diagnostico": "Doença Cardiáca Presente" if previsao == 1 else "Doença Cardiáca Ausente"
        }
        
        return resultado
        
    except Exception as e:
        # Retorna o erro real para o frontend
        return {"erro": f"Erro interno na previsão: {str(e)}"}, 500

# =================================================================
# 4. ENDPOINT DE TESTE (ROOT)
# =================================================================

@app.get("/")
def home():
    return {"status": "API de Previsão Cardíaca está ativa. Use /predict com método POST."}