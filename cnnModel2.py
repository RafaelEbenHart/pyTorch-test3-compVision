import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader # diperlukan untuk membuat batch

import matplotlib.pyplot as plt
import torchmetrics
import mlxtend
from helper_function import accuracy_fn
from function import evalModel,printTrainTime,trainStep,testStep,makePredictions
from timeit import default_timer as timer
from tqdm.auto import tqdm

# Convolutional Neural Network (CNN)
# CNN biasa dikenal dengan ConvNets
# CNN memiliki kemampuan untuk mengenal pola pada data visual

device = "cuda"

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
        # print(f"x pada convBlock 1: {x.shape}")
        x = self.convBlock2(x)
        # print(f"x pada convBlock 2: {x.shape}")
        # print(f"Maxpool2d: {flatten(x).shape}")
        x = self.classifier(x)
        # print(f"x pada classifier: {x.shape}") # print digunakan untuk troublesh0ot
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
# print(model2(randImgtensor).to(device))


BATCH_SIZE = 32
trainDataLoader = DataLoader(dataset=trainData,
                             batch_size =BATCH_SIZE,
                             shuffle=True)
testDataLoader = DataLoader(dataset=testData,
                            batch_size = BATCH_SIZE,
                            shuffle=False)

# print(model2.parameters())

## loss function and optimizer

lossFn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model2.parameters(),
                            lr=0.1)
# train and test loop

torch.manual_seed(42)
startTime = timer()
epochs = 3

for epoch in tqdm(range(epochs)):
    print(f"\nEpoch: {epoch+1}/{epochs} \n------")
    trainStep(model=model2,
              dataLoader=trainDataLoader,
              lossFn=lossFn,
              optimizer=optimizer,
              accFn=accuracy_fn,
              perBatch=4000)
    testStep(model=model2,
             dataLoader=testDataLoader,
             lossFn=lossFn,
             accFn=accuracy_fn)
endTime = timer()
totalTrainTimeModel2 = printTrainTime(start=startTime,
                                    end=endTime,
                                    device=str(next(model2.parameters()).device))

model2Result = evalModel(model=model2,
                          dataLoader=testDataLoader,
                          lossFn=lossFn,
                          accuracy_fn=accuracy_fn)

print(model2Result)

###### prediction ######

import random
# random.seed(42)
testSamples = []
testLabel = []
for sample,label in random.sample(list(testData), k=12):
    testSamples.append(sample)
    testLabel.append(label)

# view the first sample shape
print(testSamples[0].shape)

# plt.imshow(testSamples[0].squeeze(), cmap="gray")
# plt.title(className[testLabel[0]])
# plt.show()

# make predictions
predProbs = makePredictions(model=model2,
                            data=testSamples)

# view first two predictions probabilities
print(predProbs[:2])
print(testLabel)

# convert prediciton probabilities to labels
predClasses = predProbs.argmax(dim=1)
print(predClasses)

# plot predictions
plt.figure(figsize=(9,9))
nrows = 4
ncols = 3
for i, sample in enumerate(testSamples):
    # create subplot
    plt.subplot(nrows,ncols, i+1)

    # plot the target image
    plt.imshow(sample.squeeze(), cmap="gray")
    plt.axis(False)

    # find the prediction (in text form)
    predLabel = className[predClasses[i]]

    # Get the truth label (in text form)
    truthLabel = className[testLabel[i]]

    # create a tittle for the plot
    titleText = f"pred: {predLabel} | Truth: {truthLabel}"

    # Check for equality between pred and truth and change color of title text
    if predLabel == truthLabel:
        plt.title(titleText, fontsize=10, c="g" ) # green text if prediction same as truth
    else:
        plt.title(titleText, fontsize=10, c="r" ) # red text if predictions different as truth

plt.show()

# membuat confusion metrics
# confusiion metric adalah salah satu cara untuk menyajikan data mengenai
# seberapa bingung model terhadap test data

# 1. make predictions with our trained model on the test dataset
# 2. make a confusion matrix torchmetrics.ConfusionMatrix
# 3. plot the confusion matix using mlxtend.plotting.plot_confuion_matrix()

# make  prediction with trained model
yPreds = []
model2.eval()
with torch.inference_mode():
    for X,y in tqdm(testDataLoader, desc="Making prediction"):
        # send the data and target to target deive
        X,y = X.to(device),y.to(device)
        # forward pass
        yLogit = model2(X)
        # turn predictions from logits to predictions probabilities to prediciton label
        yPred = torch.softmax(yLogit.squeeze(), dim=0).argmax(dim=1) # pastikan cek input dan output sebelum set dim disini
        # put prediction on CPU for eval
        yPreds.append(yPred.cpu())

# concatenante list of predictions into tensor
print(yPreds)
yPredTensor = torch.cat(yPreds)
print(yPredTensor[:10])


