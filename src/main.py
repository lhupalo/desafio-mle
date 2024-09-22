from fastapi import FastAPI
import uvicorn
from src.routes.model_routes import router as model_router
from src.routes.health_routes import router as health_router

from src.models.model_database import ModelInMemoryDatabase
from src.models.prediction_database import PredictionInMemoryDatabase

from fastapi.middleware.cors import CORSMiddleware

# Configurações de CORS
origins = [
    "http://localhost",
    "http://localhost:8000",
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir as rotas dos arquivos separados
app.include_router(model_router)
app.include_router(health_router)

model_db = ModelInMemoryDatabase()
prediction_db = PredictionInMemoryDatabase()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug")
