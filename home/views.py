from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.conf import settings
from requests import session
from .forms import *
  
# Create your views here.
def home(request):
  
    if request.method == 'POST':
        form = ElementForm(request.POST, request.FILES)
  
        if form.is_valid():
            form.save()
            print("Form is valid")
            image_path = 'static/images/' + form.cleaned_data['element_Img'].name


        return render(request, 'home.html', {'image_path': image_path})

    else:
        print('Vou Pedir o form')
        form = ElementForm()
        return render(request, 'home.html', {'form' : form, 'image_path': None})
