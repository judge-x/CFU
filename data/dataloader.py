import os
import numpy as np

import torchvision.datasets as D
# from .utils import TwoCropTransform
from torch.utils.data import DataLoader
from torchvision import transforms
from .split import *

seed_value = 20
torch.manual_seed(seed_value)


class Dataset(object):
    def __init__(self, args) -> None:
        self.args=args

        self.dataset = args.dataset
        self.numClient = args.numClient

        self.dataTrain = None
        self.dataTest = None

        self.validationData = None
        
        self.dataSplitedTrain = []
        self.dataSplitedTest = []

        self.byclass = args.byclass
        self.IID = args.iid
        self.alpha = args.alpha

        self.get_dataset()
        self.totalTrain = len(self.dataTrain)


    def get_dataset(self):
        if self.dataset == "emnist":
            return self.get_EMNIST()
        elif self.dataset == "fmnist":
            return self.get_FMNIST()
        elif self.dataset == "cifar10":
            return self.get_CIFAR10()
        else:
            assert "no such dataset"

    def get_EMNIST(self):
        path = os.path.join(self.args.path,"data/emnist")
        
        self.dataTrain = D.EMNIST(root=path, train=True, transform=transforms.ToTensor(), download=True, split='letters')
        self.dataTest = D.EMNIST(root=path, train=False, transform=transforms.ToTensor(), download=True, split = 'letters')

        # if not self.args.backdoor:
        self.dataSplitedTrain, _ = splitDataset(self.dataTrain, self.dataset, self.numClient, self.byclass, self.IID, self.alpha, 'train')
        self.dataSplitedTest, _ = splitDataset(self.dataTest, self.dataset, self.numClient, self.byclass, True, self.alpha, 'test')
        # else:
        #     self.poisonList = [0]
        #     self.dataSplitedTrain = splitBackdoorDataset(self.dataTrain, self.numClient, self.IID, self.alpha, self.poisonList, True, self.dataset, self.args.backRate)
        #     self.dataSplitedTest = splitBackdoorDataset(self.dataTest, self.numClient, True, self.alpha, self.poisonList, False, self.dataset, 1)

    def get_FMNIST(self):
        path = os.path.join(self.args.path,"data/fmnist")
        
        transform =  transforms.Compose([transforms.ToTensor()])
        self.dataTrain = D.FashionMNIST(root=path, train=True, transform=transform, download=True)
        self.dataTest = D.FashionMNIST(root=path, train=False, transform=transform, download=True)

        # if not self.args.backdoor:
        #     self.dataSplitedTrain, _ = splitDataset(self.dataTrain, self.dataset, self.numClient, self.byclass, self.IID, self.alpha)
        #     self.dataSplitedTest, _ = splitDataset(self.dataTest, self.dataset, self.numClient, self.byclass, True, self.alpha)
        # else:
        #     self.poisonList = [0]
        #     self.dataSplitedTrain = splitBackdoorDataset(self.dataTrain, self.numClient, self.IID, self.alpha, self.poisonList, True, self.dataset, self.args.backRate)
        #     self.dataSplitedTest = splitBackdoorDataset(self.dataTest, self.numClient, True, self.alpha, self.poisonList, False, self.dataset, 1)

        self.dataSplitedTrain, sampleIndex = splitDataset(self.dataTrain, self.dataset, self.numClient, self.byclass, self.IID, self.alpha, 'train')
        self.recoverData = Subset(self.dataTrain, sampleIndex)
        self.dataSplitedTest, _ = splitDataset(self.dataTest, self.dataset, self.numClient, self.byclass, self.IID, self.alpha, 'test')
        # self.getValidation()
        # self.validationData = self.dataset


    def get_CIFAR10(self):
        path = os.path.join(self.args.path, "data/cifar10")

        train_transforms_cifar10 =  transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        test_transforms_cifar10 =  transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])


        self.dataTrain = D.CIFAR10(root=path, train=True, transform=train_transforms_cifar10, download=True)
        self.dataTest = D.CIFAR10(root=path, train=False, transform=test_transforms_cifar10, download=True)

        # if not self.args.backdoor:
        self.dataSplitedTrain, _ = splitDataset(self.dataTrain, self.dataset, self.numClient, self.byclass, self.IID, self.alpha, 'train')
        self.dataSplitedTest, _ = splitDataset(self.dataTest, self.dataset, self.numClient, self.byclass, self.IID, self.alpha, 'test')
        # else:
        #     self.poisonList = [0]
        #     self.dataSplitedTrain = splitBackdoorDataset(self.dataTrain, self.numClient, self.IID, self.alpha, self.poisonList, True, self.dataset,  self.args.backRate)
        #     self.dataSplitedTest = splitBackdoorDataset(self.dataTest, self.numClient, True, self.alpha, self.poisonList, False, self.dataset, 1)



        # self.getValidation()

    # def getValidation(self):
    #     class_indices = [[] for _ in range(10)]
    #     for idx, (_, label) in enumerate(self.dataTrain):
    #         class_indices[label].append(idx)

    #     subset_indices = []
    #     for indices in class_indices:
    #         subset_indices.extend(indices[:50])

    #     self.validationData = Subset(self.dataTrain, subset_indices)