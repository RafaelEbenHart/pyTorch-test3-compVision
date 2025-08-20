
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