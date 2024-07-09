import torch
import numpy as np
import os

import argparse
from utils.argparse import get_args
from core.model.CNN import *
from core.model.Resnet import *
from core.model.generator import Generative
from data.dataloader import *
from core.server import *
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

import warnings

if __name__ == '__main__':
    args=get_args()
    setup_seed(3407)
    warnings.filterwarnings("ignore")

    if args.device=="gpu":
        print("Let's use GPU!")
        device = torch.device(f'cuda:{args.gpuIndex}')
    else:
        device = "cpu"

    available_device_count = torch.cuda.device_count()
    map_location = {f'cuda:{3}': f'cuda:{0}'} if available_device_count >= 3 else 'cpu'

    # construct Net
    elif args.dataset == "fmnist":
        model = CNN_FMNIST_SUP(embeddingLen=args.embeddingLen)
        model_name = "CNN"
    elif args.dataset == "cifar10":
        # model = SupCEResNet(name= "resnet18", num_classes=10, embeddingLen=args.embeddingLen)
        model = CNN_CIFAR10_SUP()
        model_name = "CNN"
    elif args.dataset == "emnist":
        model = CNN_FMNIST_SUP(num_classes=37, embeddingLen=args.embeddingLen)
        model_name = "CNN" 
    else:
        assert "No such dataset, please check it."

    # get data
    dataloader = Dataset(args)

    
    if args.iid:
        if args.byclass:
            loadPath = os.path.join(args.path,"savedmodel/NormalModel_{}_{}_E_{}_byclass.pt".format(args.dataset, model_name, args.globalEpochs))
        else:
            loadPath = os.path.join(args.path,"savedmodel/NormalModel_{}_{}_E_{}.pt".format(args.dataset, model_name, args.globalEpochs))
    else:
        loadPath = os.path.join(args.path,"savedmodel/NormalModel_{}_{}_E_{}_al_{}.pt".format(args.dataset, model_name, args.globalEpochs, args.alpha))

    # train
    server = Server(args, loadPath, model, dataloader, device)
    server.train()

    # save model
    if args.unlearningChoice == False:
        # savePath = os.path.join(args.path, "savedmodel/NormalModel_{}_{}_E_{}.pt".format(args.dataset, model_name, args.globalEpochs))
        if not args.unlearningChoice:
            print("save model!")
            torch.save(server.globalModel.state_dict(), loadPath)


    # if args.model == "ResNet":
    #     savePath = os.path.join(args.path, "savedmodel/Classifier_{}_{}_E_{}.pt".format(args.dataset, args.model, args.globalEpochs))
    #     torch.save(server.classifier.state_dict(), savePath)

    
