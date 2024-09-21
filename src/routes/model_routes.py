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
    # Transformar os dados de entrada no formato que o modelo espera
    input_data = {
        "dep_time": flight_info.dep_time,
        "dep_delay": flight_info.dep_delay,
        "origin": flight_info.origin,
        "dest": flight_info.dest,
        "carrier": flight_info.carrier,
        "distance": flight_info.distance,
        "month": flight_info.month,
    }

    input_df = pd.DataFrame([input_data])

    # Fazer a previsão
    prediction = model.predict(input_df)

    # Salvar a predição na base de dados
    prediction_db = PredictionInMemoryDatabase()
    prediction_entry = {
        "model_id": model_manager.current_model_id,  # O ID do modelo
        "prediction_value": prediction[0],
        "prediction_time": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "payload": input_data,  # Armazenando o payload da predição
    }
    prediction_db.predictions.insert_one(prediction_entry)

    return {"predicted_arrival_delay": prediction[0]}


@router.post("/model/load/")
async def load_model(file: UploadFile = File(...)):
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
    Retorna a lista de todas as predições feitas.
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
