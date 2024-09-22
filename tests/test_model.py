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
    Configura as bases de dados em memória utilizando mongomock para simular o comportamento
    do MongoDB durante os testes de integração.

    Esta fixture:
    - Substitui o cliente do MongoDB real pelo mongomock.MongoClient, permitindo que os testes sejam executados sem uma conexão real ao banco de dados.
    - Utiliza patch para modificar o comportamento do MongoClient nas bases de dados de modelos e predições.
    - Garante que a base de dados esteja limpa antes dos testes e simula a conexão ao MongoDB.

    Escopo:
        module: Esta fixture será aplicada uma vez por módulo de teste, garantindo que o ambiente de banco de dados
        em memória seja consistente durante todos os testes dentro do mesmo módulo.

    Yields:
        None: A fixture apenas modifica o comportamento dos clientes do MongoDB, não retorna nenhum valor.
    """

    with patch("src.models.model_database.MongoClient", new=mongomock.MongoClient):
        with patch(
            "src.models.prediction_database.MongoClient", new=mongomock.MongoClient
        ):
            yield


@pytest.fixture(autouse=True)
def clear_databases(setup_in_memory_databases):
    """
    Limpa as coleções de modelos e predições em memória antes de cada teste.

    Esta fixture é executada automaticamente (autouse=True) antes de cada teste individual,
    garantindo que as bases de dados em memória estejam vazias, evitando interferências entre os testes.

    - A coleção de modelos é limpa ao remover todos os documentos armazenados.
    - A coleção de predições é igualmente limpa, garantindo que cada teste comece com um estado de banco de dados vazio.

    Args:
        setup_in_memory_databases (fixture): Dependente da fixture setup_in_memory_databases, que configura o ambiente de banco de dados em memória.
    """

    model_db = ModelInMemoryDatabase()
    prediction_db = PredictionInMemoryDatabase()
    model_db.models.delete_many({})
    prediction_db.predictions.delete_many({})


def test_load_model_success(setup_in_memory_databases):
    """
    Testa o carregamento bem-sucedido de um modelo e verifica a criação de uma entrada de auditoria no banco de dados.

    Este teste garante que:
    1. Um arquivo de modelo válido pode ser carregado corretamente usando o endpoint /model/load/.
    2. A resposta do endpoint é bem-sucedida (código 200) e contém a mensagem esperada.
    3. Uma entrada de auditoria é adicionada ao banco de dados de modelos, contendo o model_id e o load_time.
    4. O model_id tem 8 caracteres e o load_time segue o formato DD/MM/AAAA HH:MM:SS.

    Passos do teste:
    - O teste verifica se o arquivo do modelo existe no caminho especificado antes de prosseguir.
    - Envia o arquivo para o endpoint /model/load/ e verifica a resposta.
    - Verifica se uma nova entrada foi adicionada ao banco de dados de modelos em memória, contendo um model_id de 8 caracteres e um load_time válido.

    Args:
        setup_in_memory_databases (fixture): Configura as bases de dados em memória antes de cada teste.

    Asserts:
        - Verifica se o arquivo do modelo existe no caminho especificado.
        - Verifica se a resposta do carregamento do modelo é bem-sucedida (código 200).
        - Verifica se a mensagem de sucesso é retornada na resposta.
        - Verifica se uma nova entrada foi adicionada ao banco de dados de modelos, com model_id e load_time.
        - Valida o formato do campo load_time conforme o padrão DD/MM/AAAA HH:MM:SS.

    Raises:
        AssertionError: Se alguma das verificações falhar, indicando problemas no fluxo de carregamento do modelo ou na auditoria.
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

    Este teste verifica que ao tentar carregar um arquivo que não possui a extensão .pkl,
    o sistema deve retornar uma mensagem de erro apropriada. O comportamento esperado é que
    o servidor rejeite a solicitação com um código de status 400 e uma mensagem de erro clara.

    Passos do teste:
    1. Envia um arquivo com extensão inválida (neste caso, um arquivo de texto).
    2. Verifica se a resposta do servidor tem o status 400.
    3. Verifica se a resposta contém a mensagem de erro correta.

    Args:
        setup_in_memory_databases (fixture): Configura as bases de dados em memória antes de cada teste.

    Asserts:
        - Verifica se o código de status da resposta é 400.
        - Verifica se a mensagem de erro retornada indica que o arquivo deve ser um .pkl.

    Raises:
        AssertionError: Se as verificações falharem, indicando problemas na validação de arquivos do modelo.
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

    Este teste verifica que o sistema lida corretamente com um arquivo de modelo que está corrompido.
    O comportamento esperado é que o servidor retorne um código de status 500, indicando um erro ao
    tentar carregar o modelo. A mensagem de erro deve especificar que houve um problema ao carregar o modelo.

    Passos do teste:
    1. Cria um arquivo .pkl corrompido com dados inválidos.
    2. Tenta carregar o arquivo corrompido usando o endpoint /model/load/.
    3. Verifica se a resposta do servidor tem o status 500.
    4. Verifica se a mensagem de erro contém a frase "Erro ao carregar o modelo".

    Args:
        setup_in_memory_databases (fixture): Configura as bases de dados em memória antes de cada teste.

    Asserts:
        - Verifica se o código de status da resposta é 500.
        - Verifica se a mensagem de erro indica que houve um problema ao carregar o modelo.

    Raises:
        AssertionError: Se as verificações falharem, indicando problemas na manipulação de arquivos corrompidos.
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

    Este teste assegura que o sistema possa:
    1. Carregar um modelo válido.
    2. Realizar uma predição de atraso de voo com dados de entrada adequados.
    3. Armazenar corretamente a entrada de predição na base de dados de predições.

    Passos do teste:
    1. Carrega um modelo válido a partir de um arquivo .pkl.
    2. Envia dados de voo válidos para o endpoint /model/predict/ e verifica a resposta.
    3. Assegura que a resposta contenha o campo predicted_arrival_delay e que seu tipo seja float.
    4. Verifica se a predição foi corretamente armazenada na base de dados, incluindo a correspondência do payload original e outros campos de auditoria.

    Args:
        setup_in_memory_databases (fixture): Configura as bases de dados em memória antes de cada teste.

    Asserts:
        - Verifica se o modelo foi carregado com sucesso.
        - Verifica se a resposta da predição é bem-sucedida (código 200).
        - Verifica a presença e o tipo do campo predicted_arrival_delay.
        - Assegura que a predição foi armazenada corretamente na base de dados, validando todos os campos necessários.
        - Verifica se o model_id tem 8 caracteres e que o prediction_time segue o formato correto.

    Raises:
        AssertionError: Se qualquer uma das verificações falhar, indicando problemas na previsão de atraso de voo ou na persistência da predição.
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
        "dep_time": "1345",
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

    Este teste verifica que o sistema retorna um erro adequado quando um dos campos obrigatórios
    necessários para a previsão de atraso de voo está ausente. O comportamento esperado é que
    o servidor retorne um código de status 422, indicando que os dados de entrada são inválidos
    devido à falta de um campo necessário.

    Passos do teste:
    1. Define um conjunto de dados de voo com o campo 'dep_time' ausente.
    2. Envia os dados para o endpoint /model/predict/.
    3. Verifica se a resposta do servidor tem o status 422.

    Args:
        setup_in_memory_databases (fixture): Configura as bases de dados em memória antes de cada teste.

    Asserts:
        - Verifica se o código de status da resposta é 422, indicando um erro de validação.

    Raises:
        AssertionError: Se a verificação do código de status falhar, indicando que a validação de entrada não está funcionando corretamente.
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

    Este teste verifica que o sistema retorna um erro apropriado quando um valor inválido é fornecido
    para o campo 'month', neste caso, um valor negativo. O comportamento esperado é que o servidor
    retorne um código de status 422, indicando que os dados de entrada são inválidos devido à
    violação das regras de validação.

    Passos do teste:
    1. Define um conjunto de dados de voo com o campo 'month' definido como -7.
    2. Envia os dados para o endpoint /model/predict/.
    3. Verifica se a resposta do servidor tem o status 422.

    Args:
        setup_in_memory_databases (fixture): Configura as bases de dados em memória antes de cada teste.

    Asserts:
        - Verifica se o código de status da resposta é 422, indicando um erro de validação.

    Raises:
        AssertionError: Se a verificação do código de status falhar, indicando que a validação de entrada não está funcionando corretamente.
    """

    # Fazer uma predição com mês inválido
    flight_data = {
        "dep_time": "0100",
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

    Este teste verifica que o sistema retorna um erro apropriado quando um valor inválido é fornecido
    para o campo 'dep_time', neste caso, um valor negativo. O comportamento esperado é que o servidor
    retorne um código de status 422, indicando que os dados de entrada são inválidos devido à
    violação das regras de validação.

    Passos do teste:
    1. Define um conjunto de dados de voo com o campo 'dep_time' definido como -2230.
    2. Envia os dados para o endpoint /model/predict/.
    3. Verifica se a resposta do servidor tem o status 422.

    Args:
        setup_in_memory_databases (fixture): Configura as bases de dados em memória antes de cada teste.

    Asserts:
        - Verifica se o código de status da resposta é 422, indicando um erro de validação.

    Raises:
        AssertionError: Se a verificação do código de status falhar, indicando que a validação de entrada não está funcionando corretamente.
    """
    # Fazer uma predição com dep_time inválido
    flight_data = {
        "dep_time": "-2230",
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

    Este teste verifica que o sistema retorna um erro apropriado quando um valor inválido é fornecido
    para o campo 'distance', neste caso, um valor negativo. O comportamento esperado é que o servidor
    retorne um código de status 422, indicando que os dados de entrada são inválidos devido à
    violação das regras de validação.

    Passos do teste:
    1. Define um conjunto de dados de voo com o campo 'distance' definido como -1241.
    2. Envia os dados para o endpoint /model/predict/.
    3. Verifica se a resposta do servidor tem o status 422.

    Args:
        setup_in_memory_databases (fixture): Configura as bases de dados em memória antes de cada teste.

    Asserts:
        - Verifica se o código de status da resposta é 422, indicando um erro de validação.

    Raises:
        AssertionError: Se a verificação do código de status falhar, indicando que a validação de entrada não está funcionando corretamente.
    """
    # Fazer uma predição com distance inválido
    flight_data = {
        "dep_time": "0100",
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

    Este teste verifica se a rota que fornece o histórico de predições está funcionando corretamente.
    O comportamento esperado é que o servidor retorne um código de status 200 e uma lista contendo
    pelo menos uma entrada, onde cada entrada deve incluir os campos 'model_id', 'prediction_time',
    'payload' e 'prediction_value'.

    Passos do teste:
    1. Carrega um modelo válido através da rota /model/load/.
    2. Realiza uma predição válida usando a rota /model/predict/.
    3. Faz uma requisição GET para a rota /model/history.
    4. Verifica se a resposta tem um código de status 200.
    5. Verifica se a resposta é uma lista e se contém pelo menos uma entrada.
    6. Para cada entrada no histórico, verifica se os campos obrigatórios estão presentes e se o
       payload corresponde aos dados enviados.

    Asserts:
        - Verifica se o código de status da resposta é 200.
        - Verifica se a resposta é uma lista.
        - Verifica se a lista contém pelo menos uma entrada.
        - Verifica a presença dos campos obrigatórios em cada entrada.

    Raises:
        AssertionError: Se qualquer uma das verificações falhar, indicando que a rota /model/history
        não está retornando os dados conforme esperado.
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
        "dep_time": "1345",
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
