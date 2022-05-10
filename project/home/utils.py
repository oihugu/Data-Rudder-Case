import random, os, requests, shutil
from venv import create
from django.core.files.storage import FileSystemStorage
from torchvision import transforms
import plotly.graph_objects as go
import plotly.figure_factory as ff
import torchvision, torch
from django.conf import settings
import plotly.express as px
import torch.nn as nn
from PIL import Image 
import tweepy

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
        self.test_model(test_data_loader)
        
    
    def make_prediction(self, image_path):
        im = Image.open(image_path).convert('RGB')
        im = self.img_test_transforms(im)
        im = im.unsqueeze(0)
        pred = self.resnet18(im)
        pred = pred.argmax()
        return "Cat" if pred == 1 else "Dog"

    def test_model(self, test_data_loader):
        correct = 0
        total = 0
        self._labels = []
        self._predictions = []
        with torch.no_grad():
            for data in test_data_loader:
                images, labels = data[0], data[1]
                outputs = self.resnet18(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                self._labels.append(labels)
                self._predictions.append(predicted)
    
    def confusion_matrix(self, labels, predictions):
        cm = torch.zeros(2, 2)
        for i in range(len(labels)):
            cm[labels[i], predictions[i]] += 1
        return cm

    def plot_confusion_matrix(self):
        labels = self._labels[0]
        predictions = self._predictions[0]
        cm = self.confusion_matrix(labels, predictions)
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Cat', 'Dog'],
            y=['Cat', 'Dog'],
            colorscale='Greys',
            reversescale=True,
            showscale=False,
        ))
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='True',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)')
    
        return fig.write_html("home/static/confusion_matrix.html", full_html=False, default_height=250, default_width=250, )


    def check_image(self, path):
        try:
            im = Image.open(path)
            return True
        except:
            return False
        
class twiiter_api():
    def __init__(self):
        auth = tweepy.OAuthHandler(settings.TWITTER_CONSUMER_KEY, settings.TWITTER_CONSUMER_SECRET)
        self.api = tweepy.API(auth)
    
    def get_cats_and_dogs(self, count):
        min_images = round(count-(.2 * count))
        cat_tweets = self.api.search_tweets(q='#cat', count=random.randint(min_images,count))
        dog_tweets = self.api.search_tweets(q='#dog', count=random.randint(min_images,count))
        cat_images = [self.download_image_from_tweet(tweet, 'cat') for tweet in cat_tweets]
        dog_images = [self.download_image_from_tweet(tweet, 'dog') for tweet in dog_tweets]
        return cat_images, dog_images

    def get_tweets(self, query, count):
        tweets = self.api.search(q=query + '#photo filter:images', count=count)
        return tweets
    
    def download_image(self, url, file_path, file_name):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as file:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, file)

    def download_image_from_tweet(self, tweet, type):
        try:
            url = tweet.entities['media'][0]['media_url_https']
            file_name = url.split('/')[-1]
            file_path = os.path.join(f'../test/{type}/', file_name)
            self.download_image(url, file_path, file_name)
        except Exception as e:
            print(e)
    
    
    def run(self):
        if not os.path.exists('../test'):
            os.makedirs('../test')
            os.makedirs('../test/dog')
            os.makedirs('../test/cat')
            self.get_cats_and_dogs(100)
    
    def clear_folder(self):
        if os.path.exists('../test'):
            shutil.rmtree('../test')

twitter = twiiter_api()
mfs = MyFileStorage()
resnet = ResNet()
