from django.shortcuts import render
from .apps import TranslateConfig

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class call_model(APIView):
    def post(self, request):
        if request.method == 'POST':
            sentence = request.GET.get('sentence')
            response = TranslateConfig.predictor(sentence)
            context = {'response':response}
            return render(request, 'translate/index.html', context)
        else:
            return render(request, 'translate/index.html')

def index(request):
    return render(request, 'translate/index.html')
