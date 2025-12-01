from django.shortcuts import render

# Create your views here.

from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from .ml.loader import predict_image


@api_view(['POST'])
def predict(request):
    if 'image' not in request.FILES:
        return Response({"error": "Image file required"}, status=status.HTTP_400_BAD_REQUEST)

    result = predict_image(request.FILES['image'])
    return Response(result)

