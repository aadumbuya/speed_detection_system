from django.shortcuts import render

from .retraining import *
from .model import *
from .prediction import *
from rest_framework.response import Response
from rest_framework.decorators import api_view

# Create your views here.

def trainmodel(data):
    attrs = data["attrs"]
    training_data = data["train"]
    response = model(loader_type=attrs["loader_type"],data=training_data,data_type=attrs["data_type"],scaler_type=attrs["scaler_type"],model_name=attrs["model_name"],model_type=attrs["model_type"],analysis_type=attrs["analysis_type"],train_column_limit=attrs["train_column_limit"],classes=attrs["classes"],label=attrs["label"],test_ratio=attrs["test_ratio"])
    return Response(response,status=200)

@api_view(['POST'])
def trainModel(request):
    data = request.data
    if request.method == 'POST' and request.FILES['csv_file']:
        csv_file = request.FILES['csv_file']
        # Read the uploaded CSV file
        decoded_file = csv_file.read().decode('utf-8').splitlines()
        data["train"] = decoded_file
    else:
        return Response({"error":"invalid request"},status=406)
    response = trainmodel(data)
    return response


def retrainmodel(data):
    attrs = data["attrs"]
    train = data["train"]
    scaled_train_x,scaled_test_x,label_x,test_y = process(loader_type=attrs["loader_type"],train=train,label=attrs["label"],train_column_limit=attrs["train_column_limit"],scaler_type=attrs["scaler_type"],data_type=attrs["data_type"],test_ratio=attrs["test_ratio"])
    response = retrained(train=train,label=attrs["label"],test=attrs["test"],test_label=attrs["test_label"],classes=attrs["classes"],typ_=attrs["typ_"],model_name=attrs["model_name"],scaler_type=attrs["scaler_type"],analysis_type=attrs["analysis_type"])
    return Response(response,status=200)

@api_view(['POST'])
def retrainModel(request):
    data = request.data
    if request.method == 'POST' and request.FILES['csv_file']:
        csv_file = request.FILES['csv_file']
        # Read the uploaded CSV file
        decoded_file = csv_file.read().decode('utf-8').splitlines()
        data["train"] = decoded_file
    else:
        return Response({"error":"invalid request"},status=406)
    response = retrainmodel(data)
    return response


def predicting(request):
    attrs = request.GET.dict()
    response = prediction(data=attrs["features"],model_name=attrs["model_name"],scaler_type=attrs["scaler_type"])
    return Response({"result":response},status=200)

@api_view(['GET'])
def Predicting(request):
    response = predicting(request)
    return response



