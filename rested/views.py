import numpy as np
import pandas as pd
from .apps import RestedConfig
from rest_framework.views import APIView
from rest_framework.response import Response


class WeightPrediction(APIView):
    def post(self, request):
        data = request.data
        height = data['Height']
        gender = data['Gender']
        if gender == 'Male':
            gender = 0
        elif gender == 'Female':
            gender = 1
        else:
            return Response("Gender field is invalid", status=400)
        model_linear_regression = RestedConfig.model
        weight_predicted = model_linear_regression.predict([[gender, height]])[0][0]
        weight_predicted = np.round(weight_predicted, 1)
        response_dict = {"Predicted Weight (kg)": weight_predicted}
        return Response(response_dict, status=200)
