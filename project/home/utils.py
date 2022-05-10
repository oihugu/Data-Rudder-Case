from django.core.files.storage import FileSystemStorage
from torchvision import transforms
import plotly.graph_objects as go
import torchvision
from django.conf import settings
import plotly.express as px
import plotly.io as pio
import torch.nn as nn
from PIL import Image
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
            
        test_data_path = "../test/"
        test_data = torchvision.datasets.ImageFolder(root=test_data_path,transform=self.img_test_transforms, is_valid_file=self.check_image)
        batch_size=32
        num_workers = 6
        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        labels, predictions = self.test(os.listdir(settings.MEDIA_ROOT))
        
    
    def make_prediction(self, image_path):
        im = Image.open(image_path).convert('RGB')
        im = self.img_test_transforms(im)
        im = im.unsqueeze(0)
        pred = self.resnet18(im)
        pred = pred.argmax()
        return "Cat" if pred == 1 else "Dog"

    def test(self, test_data):
        labels = []
        predictions = []
        for image_path in test_data:
            labels.append(self.make_prediction(image_path))
            predictions.append(self.make_prediction(image_path))
        return labels, predictions
    
    def plot_confusion_matrix(self, labels, predictions):
        cm = px.imshow(
        px.confusion_matrix(
            labels,
            predictions,
            color_discrete_sequence=['#1f77b4', '#ff7f0e']
        ),
        title='Confusion Matrix',
        x='Predicted',
        y='True'
        )
        return cm
    
    def check_image(self, path):
        try:
            im = Image.open(path)
            return True
        except:
            return False
        

mfs = MyFileStorage()
resnet = ResNet()