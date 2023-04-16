import os
import joblib
from django.apps import AppConfig
from django.conf import settings


class RestedConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'rested'
    MODEL_FILE = os.path.join(settings.MODELS, "WeightPredictionModel.joblib")
    model = joblib.load(MODEL_FILE)
