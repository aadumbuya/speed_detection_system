from django.urls import path
from .views import *

urlpatterns = [
    path("train", trainModel),
    path("retrain", retrainModel),
    path("predict", Predicting),
]

