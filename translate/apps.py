from django.apps import AppConfig
import html
import pathlib
import os
from translate.prediction import *

class TranslateConfig(AppConfig):
    name = 'translate'
    predictor = translate