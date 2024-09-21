from fastapi import HTTPException
import joblib
import string
import random
from datetime import datetime
from src.models.model_database import ModelInMemoryDatabase
from src.models.prediction_database import PredictionInMemoryDatabase


class ModelManager:
    def __init__(self):
        self.model = None
        self.current_model_id = None
        self.model_db = ModelInMemoryDatabase()
        self.prediction_db = PredictionInMemoryDatabase()

    def generate_hash(self, length=8):
        """Gera um hash aleatório de 8 caracteres alfanuméricos."""
        characters = string.ascii_letters + string.digits
        return "".join(random.choices(characters, k=length))

    def get_current_time(self):
        """Retorna a hora atual no formato 'dd/mm/yyyy hh:mm:ss'."""
        return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    def load_model(self, file_path: str):
        try:
            # Carregar o modelo
            self.model = joblib.load(file_path)

            # Gerar hash e capturar horário
            model_id = self.generate_hash()
            load_time = self.get_current_time()

            # Inserir no banco de dados de modelos
            model_entry = {"model_id": model_id, "load_time": load_time}
            self.model_db.models.insert_one(model_entry)

            # Armazenar o ID do modelo atual
            self.current_model_id = model_id

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Erro ao carregar o modelo: {str(e)}"
            )

    def get_model(self):
        if self.model is None:
            raise HTTPException(status_code=400, detail="O modelo não foi carregado.")
        return self.model

    def save_prediction(self, prediction_value: float):
        if self.current_model_id is None:
            raise HTTPException(
                status_code=400, detail="Nenhum modelo carregado para fazer a predição."
            )

        prediction_entry = {
            "model_id": self.current_model_id,
            "prediction_value": prediction_value,
            "prediction_time": self.get_current_time(),
        }
        self.prediction_db.predictions.insert_one(prediction_entry)
