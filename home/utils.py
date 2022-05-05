from django.core.files.storage import FileSystemStorage
from django.conf import settings
import torch.nn as nn
import torch
import os

class MyFileStorage(FileSystemStorage):
    def get_available_name(self, name, max_length):
        if self.exists(name):
            os.remove(os.path.join(settings.MEDIA_ROOT, name))
        return name

resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
resnet18.fc = nn.Sequential(nn.Linear(resnet18.fc.in_features,512),nn.ReLU(), nn.Dropout(), nn.Linear(512, 2))
resnet18.load_state_dict(torch.load('./home/model_resnet18.pth', map_location=torch.device('cpu')))
resnet18.eval()

mfs = MyFileStorage()