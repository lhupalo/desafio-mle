from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from src.schemas.flightinfo import FlightInfo
from src.schemas.modelmanager import ModelManager
import os
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
from src.models.prediction_database import PredictionInMemoryDatabase

router = APIRouter()

model_manager = ModelManager()


@router.post("/model/predict/")
async def predict_flight_delay(
    flight_info: FlightInfo, model=Depends(model_manager.get_model)
):
    """Faz a previsão de atraso de chegada de um voo com base nas informações do voo e no modelo preditivo atual.

    Args:
        flight_info (FlightInfo): Um objeto contendo informações detalhadas sobre o voo, como horário de partida, atraso de partida, origem, destino, companhia aérea, distância e mês.
        model (Callable, optional): O modelo preditivo que será usado para fazer a previsão. Por padrão, utiliza o modelo fornecido pelo `model_manager`.

    Returns:
        dict: Um dicionário contendo a previsão de atraso de chegada do voo com a chave "predicted_arrival_delay" e o valor float da previsão.

    Raises:
        HTTPException: Caso o modelo não seja encontrado (status 500).
        ValueError: Caso o modelo não consiga realizar a previsão com os dados fornecidos.
        DatabaseError: Caso ocorra um erro ao salvar a predição na base de dados.
    """

    # flight_info.dep_time = str(flight_info.dep_time).lstrip("0")

    # Transformar os dados de entrada no formato que o modelo espera
    input_data = {
        "dep_time": str(flight_info.dep_time),
        "dep_delay": flight_info.dep_delay,
        "origin": flight_info.origin,
        "dest": flight_info.dest,
        "carrier": flight_info.carrier,
        "distance": flight_info.distance,
        "month": flight_info.month,
    }

    input_df = pd.DataFrame([input_data])
    # input_df["dep_time"] = int(str(input_df["dep_time"][0]).lstrip("0"))

    try:
        # Fazer a previsão
        prediction = model.predict(input_df)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error while attempting to predict: {str(e)}"
        )

    # Salvar a predição na base de dados
    prediction_db = PredictionInMemoryDatabase()
    prediction_entry = {
        "model_id": model_manager.current_model_id,  # O ID do modelo
        "prediction_value": prediction[0],
        "prediction_time": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "payload": input_data,  # Armazenando o payload da predição
    }
    prediction_db.predictions.insert_one(prediction_entry)

    return {"predicted_arrival_delay": float(prediction[0])}


@router.post("/model/load/")
async def load_model(file: UploadFile = File(...)):
    """
    Faz o upload e carrega um modelo de previsão a partir de um arquivo .pkl enviado.

    O arquivo é salvo temporariamente no servidor, o modelo é carregado no gerenciador de modelos
    e o arquivo temporário é removido após o carregamento.

    Args:
        file (UploadFile): Um arquivo enviado pelo usuário, que deve ser um arquivo .pkl contendo o modelo a ser carregado.

    Returns:
        dict: Um dicionário contendo uma mensagem de sucesso se o modelo for carregado corretamente.

    Raises:
        HTTPException:
            - Se o arquivo enviado não for um arquivo .pkl (status 400).
            - Se ocorrer um erro ao carregar o modelo (status 500).
    """
    # Verificar se o arquivo enviado é um arquivo .pkl
    if not file.filename.endswith(".pkl"):
        raise HTTPException(
            status_code=400, detail="O arquivo deve ser um arquivo .pkl"
        )

    # Salvar o arquivo temporariamente no servidor
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Carregar o modelo
    try:
        model_manager.load_model(file_path)
        os.remove(file_path)  # Remover o arquivo temporário após carregar o modelo
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erro ao carregar o modelo: {str(e)}"
        )

    return {"message": "Modelo carregado com sucesso"}


@router.get("/model/history/", response_model=List[Dict[str, Any]])
async def get_prediction_history():
    """
    Retorna o histórico de previsões armazenadas na base de dados em memória.

    A função recupera todas as previsões armazenadas, incluindo detalhes como o ID do modelo,
    o momento da previsão, o payload de entrada utilizado e o valor predito.

    Returns:
        List[Dict[str, Any]]: Uma lista de dicionários contendo o histórico das previsões. Cada dicionário contém:
            - model_id (str): O ID do modelo utilizado para a previsão.
            - prediction_time (str): O horário em que a previsão foi realizada.
            - payload (dict): O conjunto de dados de entrada utilizados para fazer a previsão.
            - prediction_value (float): O valor de previsão gerado pelo modelo.

    Raises:
        HTTPException: Pode levantar exceções de banco de dados em casos de falha ao recuperar os dados (se aplicável).
    """

    prediction_db = PredictionInMemoryDatabase()
    predictions = prediction_db.predictions.find()

    history = []
    for prediction in predictions:
        history.append(
            {
                "model_id": prediction["model_id"],
                "prediction_time": prediction["prediction_time"],
                "payload": prediction["payload"],
                "prediction_value": prediction["prediction_value"],
            }
        )

    return history
