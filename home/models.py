from .utils import mfs, resnet18
from django.db import models
import os
# Create your models here.



class Element(models.Model):
    element_Img = models.ImageField(upload_to='images/', storage=mfs)
