import torch
import numpy as np
import os
import argparse
from utils.argparse import get_args
from core.model.CNN import *
from core.model.Resnet import *
from data.dataloader import *
from core.serverBackdoor import *

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args=get_args()
    

    if args.device=="gpu":
        print("Let's use GPU!")
        device = torch.device(f'cuda:{args.gpuIndex}')
    else:
        device = "cpu"


    setup_seed(20)

    # construct Net
    if args.dataset == "mnist":
        model = MLP_MNIST_SUP(embeddingLen=args.embeddingLen)
        model_name = "MLP"  
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
        assert "No such dataset, please check it"

    print(args)

    # get data
    dataloader = Dataset(args)

    loadPath = os.path.join(args.path, "savedmodel/Backdoor_{}_{}_E_{}.pt".format(args.dataset, model_name, args.globalEpochs))

    # train
    server = ServerBackdoor(args, model, loadPath, dataloader, device)
    server.train()

    # save model
    if not args.unlearningChoice:
        print("save model!")
        torch.save(server.globalModel.state_dict(), loadPath)

    
