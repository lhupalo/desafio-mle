from fastapi.testclient import TestClient
import pytest
from src.main import app  # Substitua por onde seu app FastAPI está definido
from src.models.prediction_database import (
    PredictionInMemoryDatabase,
)  # Certifique-se de importar o seu modelo de banco de dados
from src.schemas.modelmanager import ModelManager

client = TestClient(app)


@pytest.fixture
def setup_in_memory_databases():
    # Configuração para limpar o banco de dados de predições
    db = PredictionInMemoryDatabase()
    db.predictions.delete_many({})  # Limpar antes dos testes

    # Garantir que o modelo esteja 'descarregado'
    model_manager = ModelManager()
    model_manager.model = None  # Descarregar explicitamente

    yield
    db.predictions.delete_many({})  # Limpar após os testes


def test_predict_without_loading_model(setup_in_memory_databases):
    flight_data = {
        "dep_time": 1345,
        "dep_delay": 10.5,
        "origin": "JFK",
        "dest": "LAX",
        "carrier": "AA",
        "distance": 2475,
        "month": 7,
    }
    response = client.post("/model/predict/", json=flight_data)
    assert response.status_code == 400
    assert response.json() == {"detail": "O modelo não foi carregado."}


def test_load_and_predict_integration(setup_in_memory_databases):
    # Carregar um modelo válido
    MODEL_FILE_PATH = "./notebook/modelo_linearReg.pkl"
    with open(MODEL_FILE_PATH, "rb") as f:
        load_response = client.post(
            "/model/load/",
            files={"file": ("modelo_linearReg.pkl", f, "application/octet-stream")},
        )
    assert load_response.status_code == 200

    # Fazer uma predição
    flight_data = {
        "dep_time": 1345,
        "dep_delay": 10.5,
        "origin": "JFK",
        "dest": "LAX",
        "carrier": "AA",
        "distance": 2475,
        "month": 7,
    }
    predict_response = client.post("/model/predict/", json=flight_data)
    assert predict_response.status_code == 200
    assert "predicted_arrival_delay" in predict_response.json()

    # Verificar se a predição foi salva na base de dados
    prediction_db = PredictionInMemoryDatabase()
    prediction_entry = prediction_db.predictions.find_one()
    assert prediction_entry is not None


def test_prediction_history_integration(setup_in_memory_databases):
    # Carregar um modelo válido
    MODEL_FILE_PATH = "./notebook/modelo_linearReg.pkl"
    with open(MODEL_FILE_PATH, "rb") as f:
        client.post(
            "/model/load/",
            files={"file": ("modelo_linearReg.pkl", f, "application/octet-stream")},
        )

    # Fazer uma predição
    flight_data = {
        "dep_time": 1345,
        "dep_delay": 10.5,
        "origin": "JFK",
        "dest": "LAX",
        "carrier": "AA",
        "distance": 2475,
        "month": 7,
    }
    client.post("/model/predict/", json=flight_data)

    # Recuperar o histórico
    history_response = client.get("/model/history")
    assert history_response.status_code == 200
    history = history_response.json()

    # Verificar se a predição está no histórico
    assert len(history) > 0
    assert any(
        entry["payload"] == flight_data for entry in history
    ), "A predição não foi encontrada no histórico"
