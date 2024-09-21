from mongomock import MongoClient


class PredictionInMemoryDatabase:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            client = MongoClient()
            cls._instance = client.get_database("prediction_db")
        return cls._instance
