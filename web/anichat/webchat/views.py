from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse

def index(request):
    return HttpResponse("Hello, world. You're at the webchat index.")

def chatbot(request):
    return render(request, 'webchat/index.html') 

def model(request):
    data = request.POST.get('message','')  
    # send to tts
    return JsonResponse({"message": data}, status=200)