from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.conf import settings
from requests import session
from .forms import *
from .utils import resnet
  
# Create your views here.
def home(request):
    
    data = {'image_path': None}

    if request.method == 'POST':
        form = ElementForm(request.POST, request.FILES)
  
        if form.is_valid():
            form.save()
            data['image_path'] = 'static/images/' + form.cleaned_data['element_Img'].name
            data['prediction'] = resnet.make_prediction('home/' + data['image_path'])
            #data['confussion_matrix'] = 



        return render(request, 'home.html', data)

    else:
        data['form'] = ElementForm()
        return render(request, 'home.html', data)
