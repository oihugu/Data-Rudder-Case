from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import *
  
# Create your views here.
def home(request):
  
    if request.method == 'POST':
        form = ElementForm(request.POST, request.FILES)
  
        if form.is_valid():
            form.save()
            return redirect('success')
    else:
        form = ElementForm()
    return render(request, 'home.html', {'form' : form})
  
  
def success(request):
    return HttpResponse('successfully uploaded')