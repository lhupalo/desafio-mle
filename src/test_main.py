from fastapi.testclient import TestClient
from .main import app

client = TestClient(app)


def test_predict_flight_delay():
    response = client.post(
        "/model/predict/",
        json={
            "dep_time": 1345,
            "dep_delay": 10.5,
            "origin": "JFK",
            "dest": "LAX",
            "carrier": "AA",
            "distance": 2475,
            "month": 7,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "predicted_arrival_delay" in data
    assert isinstance(data["predicted_arrival_delay"], float)


def test_predict_flight_delay_missing_field():
    # Teste com um campo obrigatório faltando (por exemplo, 'dep_time')
    response = client.post(
        "/model/predict/",
        json={
            "dep_delay": 10.5,
            "origin": "JFK",
            "dest": "LAX",
            "carrier": "AA",
            "distance": 2475,
            "month": 7,
        },
    )

    assert response.status_code == 422  # Código de erro para dados de entrada inválidos


def test_predict_flight_month_invalid_value():
    # Teste com um valor inválido (exemplo: dep_time negativo)
    response = client.post(
        "/model/predict/",
        json={
            "dep_time": 100,
            "dep_delay": 10.5,
            "origin": "JFK",
            "dest": "LAX",
            "carrier": "AA",
            "distance": 2475,
            "month": -7,
        },
    )

    assert response.status_code == 422


def test_predict_flight_dep_time_invalid_value():
    # Teste com um valor inválido (exemplo: dep_time negativo)
    response = client.post(
        "/model/predict/",
        json={
            "dep_time": -2230,
            "dep_delay": 10.5,
            "origin": "JFK",
            "dest": "LAX",
            "carrier": "AA",
            "distance": 2475,
            "month": 2,
        },
    )

    assert response.status_code == 422


def test_predict_flight_distance_invalid_value():
    # Teste com um valor inválido (exemplo: dep_time negativo)
    response = client.post(
        "/model/predict/",
        json={
            "dep_time": 100,
            "dep_delay": 10.5,
            "origin": "JFK",
            "dest": "LAX",
            "carrier": "AA",
            "distance": -1241,
            "month": 4,
        },
    )

    assert response.status_code == 422
