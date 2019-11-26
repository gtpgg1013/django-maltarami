from django.urls import path
from . import views

app_name = 'translate'

urlpatterns = [
    path('', views.index, name='index'),
    path('model/', views.call_model.as_view(), name='model')
]