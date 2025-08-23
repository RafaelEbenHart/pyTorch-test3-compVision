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

device = "cuda" if torch.cuda.is_available else "cpu"

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

class FashinMNISTNonLienar(nn.Module):
    def __init__(self, input : int, neuron : int, output : int):
        super().__init__()
        self.layerStack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input, out_features=neuron),
            nn.ReLU(),
            nn.Linear(in_features=neuron,out_features=output),
            nn.ReLU()
        )
    def forward(self, x: torch.tensor):
        return self.layerStack(x)

model1 = FashinMNISTNonLienar(input=784,
                              neuron=10,
                              output=len(className)).to(device)

# print(next(model1.parameters()).device)
# dummyX = torch.rand(1, 1, 28, 28)
# print(model1(dummyX.to(device)))

# loader
BATCH_SIZE = 32
trainDataLoader = DataLoader(dataset=trainData,
                             batch_size =BATCH_SIZE,
                             shuffle=True)
testDataLoader = DataLoader(dataset=testData,
                            batch_size = BATCH_SIZE,
                            shuffle=False)

# loss function and optimizer

lossfn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model1.parameters(),
                            lr=0.1)

torch.manual_seed(42)
timeStartCuda = timer()

epochs = 3

# for epoch in tqdm(range(epochs)):
#     print(f"\nEpoch: {epoch+1}/{epochs}\n----------")
#     trainLoss = 0
#     for batch, (X,y) in enumerate(trainDataLoader):
#         X,y = X.to(device),y.to(device)
#         model1.train()
#         yPred = model1(X)
#         loss = lossfn(yPred,y)
#         trainLoss += loss
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()



#         if batch % 400 == 0:
#             print(f"Batch : {batch * len(X)}/{len(trainDataLoader.dataset)}")
#         trainLoss /= len(trainDataLoader)



#     testLoss,testAcc = 0,0
#     model1.eval()
#     with torch.inference_mode():
#         for X,y in testDataLoader:
#             X,y = X.to(device),y.to(device)
#             testPred = model1(X)
#             testLoss += lossfn(testPred,y)
#             testAcc += accuracy_fn(y_true=y,
#                                   y_pred=testPred.argmax(dim=1))
#         testLoss /= len(testDataLoader)
#         testAcc /= len(testDataLoader)

#     print(f"| Train Loss : {trainLoss:.5f} | Test Loss: {testLoss:.5f} | Test Acc: {testAcc:.2f}% |")

## dengan function:
for epoch in tqdm(range(epochs)):
    print(f"\nEpoch: {epoch+1}/{epochs}")
    trainStep(model=model1,
              dataLoader=trainDataLoader,
              lossFn=lossfn,
              optimizer=optimizer,
              accFn=accuracy_fn,
              perBatch=1000)
    testStep(model=model1,
             dataLoader=testDataLoader,
             lossFn=lossfn,
             accFn=accuracy_fn)


timeEndCuda = timer()
model1TotalTime = printTrainTime(start=timeStartCuda,
                                 end=timeEndCuda,
                                 device=str(next(model1.parameters()).device))
torch.manual_seed(42)
model1Result = evalModel(model=model1,
                         dataLoader=testDataLoader,
                         lossFn=lossfn,
                         accuracy_fn=accuracy_fn)
print(model1Result)