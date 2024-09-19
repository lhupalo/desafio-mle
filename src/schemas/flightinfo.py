from pydantic import BaseModel
from pydantic.functional_validators import field_validator


class FlightInfo(BaseModel):
    dep_time: int  # Horário de partida, por exemplo, em formato 24 horas (HHMM)
    dep_delay: float  # Atraso na partida (em minutos), opcional
    origin: str  # Aeroporto de origem (código IATA)
    dest: str  # Aeroporto de destino (código IATA)
    carrier: str  # Companhia aérea (código da companhia aérea)
    distance: float  # Distância em milhas
    month: int  # Mês do voo (1-12)

    @field_validator("dep_time")
    def dep_time_must_be_valid(cls, value):
        if value < 0:
            raise ValueError("Departure time must be positive")
        return value

    @field_validator("month")
    def month_must_be_valid(cls, value):
        if value < 1 or value > 12:
            raise ValueError("Month must be between 1 and 12")
        return value

    @field_validator("distance")
    def distance_must_be_valid(cls, value):
        if value < 0:
            raise ValueError("Distance must be positive")
        return value

    class ConfigDict:
        schema_extra = {
            "example": {
                "dep_time": 1345,
                "dep_delay": 10.5,
                "origin": "JFK",
                "dest": "LAX",
                "carrier": "AA",
                "distance": 2475,
                "month": 7,
            }
        }
