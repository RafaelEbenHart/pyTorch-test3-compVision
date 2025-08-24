
# torch computer vision library
# 1.torchvision.dateset - dataset dan data loading untuk computer vision
# 2.torchvision.model -pretrained computer vision model
# 3.torchvision.transforms - function untuk manipulasi vision data agar bisa di gunakan oleh ML model
# 4.torch.utils.data.Dataset - base dataset class for python
# 5.torch.utils.data.DataLoader - creates python iterable over a dataset

import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader # diperlukan untuk membuat batch

import matplotlib.pyplot as plt
from helper_function import accuracy_fn
from timeit import default_timer as timer
from tqdm.auto import tqdm

import requests
from pathlib import Path
if Path("Helper_function.py").is_file():
    print("Exist")
else:
    print("download")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
    with open("helper_function.py","wb") as f:
        f.write(request.content)

# dataset yang akan digunakan adalah FashionMNIST dari torchvision.dataset

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

print(len(trainData),len(testData))
print(className) # cek class
print(trainData.class_to_idx) # cek label dari class
print(testData.targets)
images,label = trainData[0]
print(images.shape, label) # [1, 28, 28] 1 color channel karena hitam putih 28x28 height dan width

# smarter ml = color channel last
# CNN adalah NN yang membagi kerjanya dari kecil ke besar

# visualisasi dataset
# images, label =trainData[0]
# print(f"image shape: {images.shape}")
# plt.title(label)
# plt.imshow(images.squeeze()) # lakukan squeeze
# # plt.show()
# plt.imshow(images.squeeze(), cmap="gray")
# plt.title(className[label])
# plt.axis(False)
# # plt.show()

#menampilkan random images dari dataset
# torch.manual_seed(42)
fig = plt.figure(figsize=(5,5))
rows,cols = 4,4
# for i in range (1,rows*cols+1):
#     randomIdx = torch.randint(0, len(trainData),size=[1]).item()
#     img, label = trainData[randomIdx]
#     fig.add_subplot(rows,cols,i)
#     plt.imshow(img.squeeze(),cmap="gray")
#     plt.title(className[label])
#     plt.axis(False)
# plt.show()
# note:
# untuk beberapa kasus color channel first tidak di perbolehkan
# dan untuk gambar yang hitam puith biasanya tidak memakai color channel

# menyiapkan data loader
# dataloader mengubah dataset kita menjadi python iterable
# dalam artian kita akan mengubah data kita ke batches atau mini-batches
# mengapa harus di lakukan?
# 1. karena untuk keefektifan pada komputasi,karena hardware mungkin tidak sanggup untuk menampilkan data sebanyak 60000 images dalam satu percobaan,
# maka harus di percah menjadi 32 images dalam satu waktu (batch size 32)
# 2. memudahkan NN neural network untuk mengupdate gradients per epoch

# membuat batches
# hyperparameter gunakan capital
BATCH_SIZE = 32
trainDataLoader = DataLoader(dataset=trainData,
                             batch_size =BATCH_SIZE,
                             shuffle=True)
testDataLoader = DataLoader(dataset=testData,
                            batch_size = BATCH_SIZE,
                            shuffle=False)
# untuk test data bisa di di shuffle namun akan lebih mudah pada saat sesi test jika tidak di shuffle

# visualisai data batch
print(f"length batch trainData: {len(trainDataLoader)}")
print(f"length batch testData: {len(testDataLoader)}")

trainFeaturesBatch,trainLabelsBatch = next(iter(trainDataLoader))
# membuat traindata batch menjadi iterable
print(trainFeaturesBatch.shape,trainLabelsBatch.shape)

torch.manual_seed(42)
randomIdx = torch.randint(0,len(trainFeaturesBatch),size=[1]).item()
img,label = trainFeaturesBatch[randomIdx],trainLabelsBatch[randomIdx]
plt.imshow(img.squeeze(),cmap="gray")
plt.title(className[label])
plt.axis(False)
# plt.show()

# membuat model

# device = "cuda" if torch.cuda.is_available else "cpu"

# Basline mdoel
# basline model adalah versi simple dari model yang akan dikembangkan tergantung pada kebutuhan
# untuk mengatasi kompleksitas

# membuat flatten layer
flattenModel = nn.Flatten()

X = trainFeaturesBatch[0]
# flatten the sample
output = flattenModel(X)

print(f"shape before flattening: {X.shape}")
print(f"shape after flattening: {output.shape}")
# flatten membuat output menjadi vector dengan melakukan perkalian antar tiap dimensi pada tensor (1 x 28 x 28)


class FashionMNISTModelV0(nn.Module):
    def __init__(self, input:int,
                 neurons:int,
                 output:int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input,
                      out_features=neurons),
            nn.Linear(in_features=neurons,
                      out_features=output),
        )
    def forward(self,x):
        return self.layers(x)

model0 = FashionMNISTModelV0(
    input=784,
    neurons=10,
    output=len(className)
)
# input ditentukan oleh flatten
# output len() menentukan satu 1 dari setiap kelas

print(model0)

# membuat dummy forward pass
dummyX = torch.rand(1, 1, 28, 28)
print(model0(dummyX))

# setup loss,optimizer and evaluation metrics

lossFn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model0.parameters(),
                            lr=0.1)




# membuat function untuk mengukur waktu eksperimen


def printTrainTime(start:float,
                   end:float,
                   device: torch.device = None):
    """print perbandingan antara start dan end time"""
    totalTime = end-start
    print(f"Train time on {device}: {totalTime:.3f} seconds")
    return totalTime

startTime = timer()
# code ...
endTime = timer()
printTrainTime(start=startTime,end=endTime,device="cpu")

# membuat train dan test loop
# 1. loop through epochs
# 2. loop through training batches,perform training step,calculate the train loss per batch
# 3. loop through testing batches,perform testing step,calculate teh test loss per batch
# 4. print
# 5. time it

# import tqdm - progress bar


# set seed dan timer
torch.manual_seed(42)
trainTimeStartCpu = timer()

# set epochs
epochs = 3

# membuat
for epoch in tqdm(range(epochs)):
    print(f"\nEpoch: {epoch}\n------")

    trainLoss = 0
    # add a loop to loop through the training batches
    for batch , (X,y) in enumerate(trainDataLoader):
        model0.train()
        # forward()
        yPred = model0(X)
        #loss calculate per batch
        loss = lossFn(yPred,y)
        trainLoss += loss # akumulasi train loss
        # optimizer
        optimizer.zero_grad()
        # loss backward
        loss.backward()
        #step
        optimizer.step()

        # print()
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(trainDataLoader.dataset)} samples")
        # devide total train loss by len of train data looader
        trainLoss /= len(trainDataLoader)

    ## testing
    testLoss,testAcc = 0,0
    model0.eval()
    with torch.inference_mode():
        for X,y in testDataLoader:
            # forward
            testPred = model0(X)
            # calculate loss
            testLoss += lossFn(testPred,y)
            # calculate acc
            testAcc += accuracy_fn(y_true=y, y_pred=testPred.argmax(dim=1))
        # calculate test loss average per batch
        testLoss /= len(testDataLoader)

        # calculate test acc avg per batch
        testAcc /= len(testDataLoader)

    # print()
    print(f"\n| Train Loss: {trainLoss:.4f} | Test Loss: {testLoss:.4f} | Test Acc: {testAcc:.4f} |")

trainTimeEndCpu = timer()
totalTrainTimeMOdel0 = printTrainTime(start=trainTimeStartCpu,
                                      end=trainTimeEndCpu,
                                      device=str(next(model0.parameters()).device))

# membuat prediksi antara beberapa model

torch.manual_seed(42)
def evalModel(model: torch.nn.Module,
             dataLoader: torch.utils.data.DataLoader,
             lossFn: torch.nn.Module,
             accuracy_fn):
    """Returns a dictionary containing the results of model predicting on data_loader"""
    loss, acc = 0,0
    model.eval()
    with torch.inference_mode():
        for X,y in tqdm(dataLoader):
            # prediksi
            yPred = model(X)
            # akumulasi loss dan Acc per batch
            loss += lossFn(yPred,y)
            acc += accuracy_fn(y_true=y,
                               y_pred=yPred.argmax(dim=1))

        # scale loss and acc to find avg per batch
        loss /= len(dataLoader)
        acc /= len(dataLoader)
    return {"modelName" : model.__class__.__name__,# hanya bisa berkerja jika mmodel dibuat dengan class
            "ModelLoss" : f"{loss.item():.5f}",
            "ModelAcc" : f"{acc:.2f}%"}

model0Result = evalModel(model=model0,
                         dataLoader=testDataLoader,
                         lossFn=lossFn,
                         accuracy_fn=accuracy_fn)
print(model0Result)

