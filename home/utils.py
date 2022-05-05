from django.core.files.storage import FileSystemStorage
from torchvision import transforms
from django.conf import settings
from PIL import Image
import torch.nn as nn
import numpy as np
import torch
import os

class MyFileStorage(FileSystemStorage):
    def get_available_name(self, name, max_length):
        if self.exists(name):
            path = os.path.join(settings.MEDIA_ROOT, name)
            os.remove(path)
        return name

class ResNet():
    def __init__(self) -> None:
        self.resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.resnet18.fc = nn.Sequential(nn.Linear(self.resnet18.fc.in_features,512),nn.ReLU(), nn.Dropout(), nn.Linear(512, 2))
        self.resnet18.load_state_dict(torch.load('./home/model_resnet18.pth', map_location=torch.device('cpu')))
        self.resnet18.eval()
        self.img_test_transforms = transforms.Compose([
            transforms.Resize((244,244)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )
            ])
    
    def make_prediction(self, image_path):
        im = Image.open(image_path).convert('RGB')
        im = self.img_test_transforms(im)
        im = im.unsqueeze(0)
        pred = self.resnet18(im)
        pred = pred.argmax()
        return "Cat" if pred == 1 else "Dog"

        

mfs = MyFileStorage()
resnet = ResNet()