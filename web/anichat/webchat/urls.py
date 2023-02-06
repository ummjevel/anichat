from django.urls import path

from . import views

urlpatterns = [
    # path('', views.index, name='index'),
    path('', views.chatbot, name='chatbot'),
    path('model', views.model, name='model'),
]