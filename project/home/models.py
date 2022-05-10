from .utils import mfs
from django.db import models
import os
# Create your models here.



class Element(models.Model):
    element_Img = models.ImageField(upload_to='home/static/images/', storage=mfs)
