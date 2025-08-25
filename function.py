import torch
from torch import nn
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available else "cpu"
# prediction
# torch.manual_seed(42)
def evalModel(model: torch.nn.Module,
             dataLoader: torch.utils.data.DataLoader,
             lossFn: torch.nn.Module,
             accuracy_fn):
    """Returns a dictionary containing the results of model predicting on data_loader"""
    loss, acc = 0,0
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        for X,y in dataLoader:
            X,y = X.to(device),y.to(device)
            # prediksi
            yPred = model(X)
            # akumulasi loss dan Acc per batch
            loss += lossFn(yPred,y)
            acc += accuracy_fn(y_true=y,
                               y_pred=yPred.argmax(dim=1))

        # scale loss and acc to find avg per batch
        loss /= len(dataLoader)
        acc /= len(dataLoader)
    result = {"ModelName" : model.__class__.__name__,# hanya bisa berkerja jika mmodel dibuat dengan class
            "ModelLoss" : loss.item(),
            "ModelAcc" : acc}
    return result

# time
def printTrainTime(start:float,
                   end:float,
                   device: torch.device = None):
    """print perbandingan antara start dan end time"""
    totalTime = end-start
    print(f"Train time on {device}: {totalTime:.3f} seconds")
    return totalTime

# train loop
def trainStep(model: torch.nn.Module,
              dataLoader: torch.utils.data.DataLoader,
              lossFn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              accFn,
              perBatch: int):
    """Performs training with model trying to learn on dataLoader"""
    trainLoss,trainAcc = 0, 0
    for batch, (X,y) in enumerate(dataLoader):
        device = next(model.parameters()).device
        model.train()
        X,y = X.to(device),y.to(device) # put tu target device
        yPred = model(X) # forward pass
        # Calculate loss and acc
        loss = lossFn(yPred,y)
        trainLoss += loss
        trainAcc += accFn(y_true=y,
                               y_pred=yPred.argmax(dim=1))
        # optimizer zero grad,loss backward,optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # show batch
        if batch % perBatch == 0:
            print(f"Looked at: {(batch * len(X)) + perBatch}/{len(dataLoader.dataset)} samples")

    # calculate avg
    trainLoss /= len(dataLoader)
    trainAcc /= len(dataLoader)
    print(f"|Train Loss: {trainLoss:.5f} | Train Acc: {trainAcc:.2f}%|")

# test loop

def testStep(model: torch.nn.Module,
             dataLoader: torch.utils.data.DataLoader,
             lossFn: torch.nn.Module,
             accFn):
    """Performs testing with model trying to test on dataLoader"""
    testLoss,testAcc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X,y in dataLoader:
            device = next(model.parameters()).device
            X,y = X.to(device),y.to(device)
            testPred = model(X)
            testLoss += lossFn(testPred,y)
            testAcc += accFn(y_true=y,
                             y_pred=testPred.argmax(dim=1))

        testLoss /= len(dataLoader)
        testAcc /= len(dataLoader)
        print(f"|Test Loss: {testLoss:.5f} | Test Acc: {testAcc:.2f}%|")

def makePredictions(model:torch.nn.Module,
                    data:list):
    predProbs = []
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # prepare the sample (add a batch dimensio and pass to target device)
            sample = torch.unsqueeze(sample,dim=0).to(device)

            # forward pass (model output raw logits)
            predLogits = model(sample)

            # get prediction probability (logit -> preediction probability)
            predProb = torch.softmax(predLogits.squeeze(),dim=0)

            # get predprobs off the CUDA for further calculations
            predProbs.append(predProb.cpu())

    # stack the predProbs to turn list into a tensor
    return torch.stack(predProbs)