from django import forms
from .models import *
  
class ElementForm(forms.ModelForm):
  
    class Meta:
        model = Element
        fields = ['element_Img']