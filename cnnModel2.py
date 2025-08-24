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
flatten = nn.Flatten()

# membuat CNN model
# note:
# saat membuat block2 jangan gunakan koma
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
            )
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
            nn.Linear(in_features=neurons * 7 * 7,
                      out_features=output)
            # untuk menemukan in_features ini cukup sulit(sementara gunakan * 0)
            ## note:
            # trick untuk menentukan in_features pada classifier:
            # gunakan print pada fungsi forward untuk melihat shape dari output convBlock terakhir
            # hal ini menentukan jumlah neurons / berapa value dari in_features untuk classifier layer
            # karena shpae dari convBlock2 adalah (10, 7, 7) maka
            # untuk mengatasi hal ini juga bisa gunakan lazyLinear,hal ini akan membantu mengkalkulasi sendiri in_features yang akan digunakan
        )
    def forward(self,x):
        x = self.convBlock1(x)
        print(f"x pada convBlock 1: {x.shape}")
        x = self.convBlock2(x)
        print(f"x pada convBlock 2: {x.shape}")
        print(f"Maxpool2d: {flatten(x).shape}")
        x = self.classifier(x)
        print(f"x pada classifier: {x.shape}") # print digunakan untuk troublesh0ot
        return x

torch.manual_seed(42)
model2 = FashionMNISTCNN(input=1, # input shape untuk model CNN tergantung pada color channel
                         neurons=10,
                         output=len(className)).to(device)

## membuat data dummy
torch.manual_seed(42)
images = torch.randn(32, 3, 64, 64)
testImage = images[0]

# print(f"image batch shape: {images.shape}")
# print(f"single image shape: {testImage.shape}")
# print(f"test image: \n{testImage}")

# conv layer
torch.manual_seed(42)
convLayer = nn.Conv2d(in_channels=3,
                      out_channels=10,
                      kernel_size=3,
                      stride=1,
                      padding=1)
convLayerOutput = convLayer(testImage)
print(convLayerOutput.shape)

## maxpool layer
maxPoolLayer = nn.MaxPool2d(kernel_size=2)
maxPoolLayerOutput = maxPoolLayer(convLayerOutput)
print(maxPoolLayerOutput.shape)
# saat kernel maxpool melewati data maka akan mengambil value terbesar pada area hyperparameter kernel (2x2) pada layer ini
# akibatnya shape berubah menjadi 32, 32 karena hanya diambil value tersebesarnya saja

# test model dengan data
randImgtensor = torch.randn(1, 1, 28, 28)
# print(randImgtensor.shape)
print(model2(randImgtensor).to(device))


BATCH_SIZE = 32
trainDataLoader = DataLoader(dataset=trainData,
                             batch_size =BATCH_SIZE,
                             shuffle=True)
testDataLoader = DataLoader(dataset=testData,
                            batch_size = BATCH_SIZE,
                            shuffle=False)

# print(model2.parameters())

