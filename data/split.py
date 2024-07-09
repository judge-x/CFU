import numpy as np
import torch
from torch.utils.data import Subset
from torch.distributions import Dirichlet
from .utils import *
torch.manual_seed(42)

def splitDataset(dataset, datasetName, numClient, byclass, iid, alpha, train):
    if byclass:
        return splitByClass(dataset, numClient)
    else:
        if iid:
            return splitIID(dataset, numClient, datasetName, train)
        else:
            return splitDirichlet(dataset, numClient, alpha, datasetName, train)


def splitIID(dataset, numClient, datasetName, train):

    dataLenClient = int(len(dataset) / numClient)
    clientsDataset=[]

    # labels = dataset.train_labels
    # numClass = labels.max() + 1
    # clientIndex = [[] for _ in range(numClient)]
    # sampleSizes = [0] * (numClass * numClient)

    for i in range(numClient):
        startIndex = i*dataLenClient
        endIndex = (i+1)*dataLenClient if i < (numClient-1) else len(dataset)

        datasetClient = Subset(dataset, list(range(startIndex,endIndex)))
        clientsDataset.append(datasetClient)
        
        # for idx in range(startIndex, endIndex):
        #     label = labels[idx].item()
        #     sampleSizes[i * numClass + label] += 1

    # if train=='train':
    #     plotbubble(datasetName, 0, numClient, numClass, sampleSizes, train)


    return clientsDataset, None

def splitDirichlet(dataset, numClient, alpha, datasetName, train):
    if datasetName == "cifar10":
        labels = dataset.targets
        numClass = 10

        labelDistribution = Dirichlet(torch.full((numClient,), alpha)).sample((numClass, 1))

        classIndex = []
        for y in range(numClass):
            classIndex.append(torch.nonzero(torch.tensor(labels) == y).flatten())

        clientIndex = [[] for _ in range(numClient)]
        sampleSizes = [0] * (numClass *numClient)
        classBatch = 0
        for z, frac in zip(classIndex, labelDistribution):
            frac = (frac * len(z)).int().tolist()[0]
            frac[-1] = len(z) - sum(frac[:-1])
            idcs = torch.split(z, frac)
            for i in range(len(clientIndex)):
                clientIndex[i]+=idcs[i].tolist()
                sampleSizes[i*numClass+classBatch] = frac[i]
            classBatch += 1
        if train=='train':
            plotbubble(datasetName, alpha, numClient, numClass, sampleSizes, train)

        clientsDataset = []
        for i in range(len(clientIndex)):
            datasetClient = Subset(dataset, clientIndex[i])
            clientsDataset.append(datasetClient)
    else:
        labels = dataset.train_labels
        numClass = labels.max() + 1

        labelDistribution = Dirichlet(torch.full((numClient,), alpha)).sample((numClass,1))

        classIndex = []
        for y in range(numClass):
            classIndex.append(torch.nonzero(labels == y).flatten())

        # Split index
        clientIndex = [[] for _ in range(numClient)]
        sampleSizes = [0] * (numClass *numClient)
        classBatch = 0
        for z, frac in zip(classIndex, labelDistribution):
            frac = (frac * len(z)).int().tolist()[0]
            frac[-1] = len(z) - sum(frac[:-1])
            idcs = torch.split(z, frac)
            for i in range(len(clientIndex)):
                clientIndex[i]+=idcs[i].tolist()
                sampleSizes[i*numClass+classBatch] = frac[i]
            classBatch += 1

        # draw split result
        if train=='train':
            plotbubble(datasetName, alpha, numClient, numClass, sampleSizes, train)

        # generate dataset
        clientsDataset = []
        for i in range(len(clientIndex)):
            datasetClient = Subset(dataset, clientIndex[i])
            clientsDataset.append(datasetClient)



    return clientsDataset, None

def splitByClass(dataset, numClient):
   

    labels = dataset.train_labels
    numClass = labels.max() + 1

    if numClient > numClass:
        assert "The number of class is not enough for split!"
    
    valiationIndices = []
    clientsDataset=[]

    for classLabel in range(numClass):
        indices = [idx for idx, (_, label) in enumerate(dataset) if label == classLabel]
        valiationIndices.append(indices[0:50])
        subDataset = Subset(dataset, indices)
        clientsDataset.append(subDataset)

    return clientsDataset, Subset(dataset, valiationIndices)
