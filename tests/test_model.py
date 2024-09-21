import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from src.main import app
import os
from src.models.model_database import ModelInMemoryDatabase
from src.models.prediction_database import PredictionInMemoryDatabase
import mongomock

import re

client = TestClient(app)


@pytest.fixture(scope="module")
def setup_in_memory_databases():
    """
    Configura as bases de dados em memória antes dos testes e limpa após.
    """
    with patch("src.models.model_database.MongoClient", new=mongomock.MongoClient):
        with patch(
            "src.models.prediction_database.MongoClient", new=mongomock.MongoClient
        ):
            yield


@pytest.fixture(autouse=True)
def clear_databases(setup_in_memory_databases):
    """
    Limpa as coleções de modelos e predições antes de cada teste.
    """
    model_db = ModelInMemoryDatabase()
    prediction_db = PredictionInMemoryDatabase()
    model_db.models.delete_many({})
    prediction_db.predictions.delete_many({})


def test_load_model_success(setup_in_memory_databases):
    """
    Testa o carregamento bem-sucedido de um modelo e a criação de uma entrada de auditoria.
    """
    # Definir o caminho do modelo válido
    MODEL_FILE_PATH = "./notebook/modelo_linearReg.pkl"

    # Assegurar que o modelo de teste existe
    assert os.path.exists(
        MODEL_FILE_PATH
    ), f"Modelo de teste não encontrado em {MODEL_FILE_PATH}"

    with open(MODEL_FILE_PATH, "rb") as f:
        response = client.post(
            "/model/load/",
            files={"file": ("modelo_linearReg.pkl", f, "application/octet-stream")},
        )

    assert response.status_code == 200
    assert response.json() == {"message": "Modelo carregado com sucesso"}

    # Verificar se a entrada foi adicionada à base de dados de modelos
    model_db = ModelInMemoryDatabase()
    model_entry = model_db.models.find_one()
    assert model_entry is not None
    assert "model_id" in model_entry
    assert "load_time" in model_entry
    assert len(model_entry["model_id"]) == 8
    # Verificar o formato do load_time
    pattern = r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}"
    assert re.match(pattern, model_entry["load_time"]), "Formato de load_time inválido"


def test_load_model_invalid_file(setup_in_memory_databases):
    """
    Testa o carregamento de um arquivo inválido (não .pkl) e espera uma resposta de erro.
    """
    response = client.post(
        "/model/load/",
        files={"file": ("invalid_file.txt", b"conteudo do arquivo", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "O arquivo deve ser um arquivo .pkl"}


def test_load_model_corrupted_file(setup_in_memory_databases):
    """
    Testa o carregamento de um arquivo .pkl corrompido e espera uma resposta de erro.
    """
    # Criar um arquivo .pkl corrompido
    corrupted_file_path = "corrupted_model.pkl"
    with open(corrupted_file_path, "wb") as f:
        f.write(b"arquivo corrompido")

    with open(corrupted_file_path, "rb") as f:
        response = client.post(
            "/model/load/",
            files={"file": ("corrupted_model.pkl", f, "application/octet-stream")},
        )

    assert response.status_code == 500
    assert "Erro ao carregar o modelo" in response.json()["detail"]

    # Remover o arquivo corrompido após o teste
    os.remove(corrupted_file_path)


def test_predict_flight_delay(setup_in_memory_databases):
    """
    Testa a previsão de atraso de voo com dados válidos.
    """
    # Primeiro, carregar um modelo válido
    MODEL_FILE_PATH = "./notebook/modelo_linearReg.pkl"
    with open(MODEL_FILE_PATH, "rb") as f:
        response = client.post(
            "/model/load/",
            files={"file": ("modelo_linearReg.pkl", f, "application/octet-stream")},
        )
    assert response.status_code == 200, "Falha ao carregar o modelo"

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

    response = client.post(
        "/model/predict/",
        json=flight_data,
    )

    assert response.status_code == 200, "Falha ao fazer a predição"
    assert (
        "predicted_arrival_delay" in response.json()
    ), "Campo predicted_arrival_delay não encontrado na resposta"
    assert isinstance(
        response.json()["predicted_arrival_delay"], float
    ), "predicted_arrival_delay não é um float"

    # Verificar se a predição foi salva na base de dados de predições
    prediction_db = PredictionInMemoryDatabase()
    prediction_entry = prediction_db.predictions.find_one()

    assert (
        "payload" in prediction_entry
    ), "Payload não encontrado na entrada de predição"
    assert (
        prediction_entry["payload"] == flight_data
    ), "Payload armazenado não corresponde ao payload original"

    assert (
        prediction_entry is not None
    ), "A entrada de predição não foi encontrada na base de dados"
    assert (
        "model_id" in prediction_entry
    ), "model_id não encontrado na entrada de predição"
    assert (
        "prediction_value" in prediction_entry
    ), "prediction_value não encontrado na entrada de predição"
    assert (
        "prediction_time" in prediction_entry
    ), "prediction_time não encontrado na entrada de predição"
    assert len(prediction_entry["model_id"]) == 8, "model_id deve ter 8 caracteres"
    assert isinstance(
        prediction_entry["prediction_value"], float
    ), "prediction_value não é um float"

    # Verificar o formato do prediction_time
    pattern = r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}"
    assert re.match(
        pattern, prediction_entry["prediction_time"]
    ), "Formato de prediction_time inválido"


def test_predict_flight_delay_missing_field(setup_in_memory_databases):
    """
    Testa a previsão de atraso de voo com um campo obrigatório faltando.
    """
    # Fazer uma predição sem o campo 'dep_time'
    flight_data = {
        "dep_delay": 10.5,
        "origin": "JFK",
        "dest": "LAX",
        "carrier": "AA",
        "distance": 2475,
        "month": 7,
    }

    response = client.post(
        "/model/predict/",
        json=flight_data,
    )

    assert response.status_code == 422  # Código de erro para dados de entrada inválidos


def test_predict_flight_month_invalid_value(setup_in_memory_databases):
    """
    Testa a previsão de atraso de voo com um valor inválido para o mês (negativo).
    """
    # Fazer uma predição com mês inválido
    flight_data = {
        "dep_time": 100,
        "dep_delay": 10.5,
        "origin": "JFK",
        "dest": "LAX",
        "carrier": "AA",
        "distance": 2475,
        "month": -7,
    }

    response = client.post(
        "/model/predict/",
        json=flight_data,
    )

    assert response.status_code == 422


def test_predict_flight_dep_time_invalid_value(setup_in_memory_databases):
    """
    Testa a previsão de atraso de voo com um valor inválido para o horário de decolagem (negativo).
    """
    # Fazer uma predição com dep_time inválido
    flight_data = {
        "dep_time": -2230,
        "dep_delay": 10.5,
        "origin": "JFK",
        "dest": "LAX",
        "carrier": "AA",
        "distance": 2475,
        "month": 2,
    }

    response = client.post(
        "/model/predict/",
        json=flight_data,
    )

    assert response.status_code == 422


def test_predict_flight_distance_invalid_value(setup_in_memory_databases):
    """
    Testa a previsão de atraso de voo com um valor inválido para a distância (negativo).
    """
    # Fazer uma predição com distance inválido
    flight_data = {
        "dep_time": 100,
        "dep_delay": 10.5,
        "origin": "JFK",
        "dest": "LAX",
        "carrier": "AA",
        "distance": -1241,
        "month": 4,
    }

    response = client.post(
        "/model/predict/",
        json=flight_data,
    )

    assert response.status_code == 422


def test_get_prediction_history():
    """
    Testa a rota /model/history para garantir que retorna o histórico de predições.
    """
    # Primeiro, carregar um modelo válido
    MODEL_FILE_PATH = "./notebook/modelo_gBoost.pkl"
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

    # Testar a rota /model/history
    response = client.get("/model/history")
    assert response.status_code == 200
    history = response.json()

    assert isinstance(history, list), "A resposta deve ser uma lista"
    assert len(history) > 0, "A lista de predições deve conter pelo menos uma entrada"

    # Verificar se os campos estão presentes
    for entry in history:
        assert "model_id" in entry
        assert "prediction_time" in entry
        assert "payload" in entry
        assert "prediction_value" in entry
        assert (
            entry["payload"] == flight_data
        ), "O payload deve corresponder aos dados enviados"
