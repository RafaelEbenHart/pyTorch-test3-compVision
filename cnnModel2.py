import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader # diperlukan untuk membuat batch

import matplotlib.pyplot as plt
from helper_function import accuracy_fn
from function import evalModel,printTrainTime,trainStep,testStep
from timeit import default_timer as timer
from tqdm.auto import tqdm

# Convolutional Neural Network (CNN)
# CNN biasa dikenal dengan ConvNets
# CNN memiliki kemampuan untuk mengenal pola pada data visual

device = "cpu"

trainData = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=True, # train data,test data
    download=True, #download data or no
    transform=ToTensor(), # parse into tensor
    target_transform=None # how do we want to transform the labels
)

testData = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)
className = trainData.classes

# membuat CNN model
class FashionMNISTCNN(nn.Module):
    """Model architecture that replicate the Tiny VGG(salah satu jenis CNN)"""
    def __init__(self, input:int, neurons:int, output:int ):
        super().__init__()
        # layer blocks:
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=input,
                      out_channels=neurons,
                      # hyperparameters:
                      kernel_size=3, # adalah bagian yang akan melewati tiap bagian pada gambar
                      stride=1, # adalah berapa jauh langkah yang akan dilakukan kernel
                      padding=1), # adalah cara agar input dan output tetap konsisten,dengan menjaga dari hilangnya informasi dari input ke output
            nn.ReLU(),
            nn.Conv2d(in_channels=neurons,
                      out_channels=neurons,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            ## nn.maxPool,dimensi nya di tentukan pada conv yang di pakai
            ## maxPool berguna untuk mengambil nilai tertinggi dari area kecil(kernel hyperparameter)
            ),
        self.convBlock2=nn.Sequential(
            nn.Conv2d(in_channels=neurons,
                      out_channels=neurons,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=neurons,
                      out_channels=neurons,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Classifier layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=neurons * 0, # untuk menemukan in_features ini cukup sulit(sementara gunakan * 0)
                      out_features=output)
            ## note:
            # trick untuk menentukan in_features pada classifier
        )
    def forward(self,x):
        x = self.convBlock1(x)
        print(f"x pada convBlock 1: {x.shape}")
        x = self.convBlock2(x)
        print(f"x pada convBlock 2: {x.shape}")
        x = self.classifier(x)
        return x

torch.manual_seed(42)
model2 = FashionMNISTCNN(input=1, # input shape untuk model CNN tergantung pada color channel
                         neurons=10,
                         output=len(className)).to(device)





BATCH_SIZE = 32
trainDataLoader = DataLoader(dataset=trainData,
                             batch_size =BATCH_SIZE,
                             shuffle=True)
testDataLoader = DataLoader(dataset=testData,
                            batch_size = BATCH_SIZE,
                            shuffle=False)

print(model2.parameters())