from fastapi import FastAPI
from schemas.flightinfo import FlightInfo
from src.models.database import InMemoryDatabase
import joblib
import pandas as pd

import uvicorn


app = FastAPI()

model = joblib.load("./notebook/modelo.pkl")


@app.get("/health", status_code=200, tags=["health"], summary="Health check")
async def health():
    return {"status": "ok"}


@app.post("/model/predict/")
async def predict_flight_delay(flight_info: FlightInfo):
    # Transformar os dados de entrada no formato que o modelo espera
    input_data = {
        "dep_time": [flight_info.dep_time],
        "dep_delay": [flight_info.dep_delay],
        "origin": [flight_info.origin],
        "dest": [flight_info.dest],
        "carrier": [flight_info.carrier],
        "distance": [flight_info.distance],
        "month": [flight_info.month],
    }

    input_df = pd.DataFrame(input_data)

    # Fazer a previsão
    prediction = model.predict(input_df)
    print(prediction)
    return {"predicted_arrival_delay": prediction[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug")
