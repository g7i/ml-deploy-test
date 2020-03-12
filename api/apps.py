import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from django.apps import AppConfig


class ApiConfig(AppConfig):
    name = 'api'
