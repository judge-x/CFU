import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import copy
from sklearn import manifold
import pandas as pd
import torch.nn


def getParameter(model):

    return model.state_dict()

def getDeltaNorm(model1, model2):
    
    # params1 = torch.cat([p.view(-1) for p in model1.parameters()])
    # params2 = torch.cat([p.view(-1) for p in model2.parameters()])

    params1 = torch.nn.utils.parameters_to_vector(copy.deepcopy(model1).parameters())
    params2 = torch.nn.utils.parameters_to_vector(copy.deepcopy(model2).parameters())

    param_diff = params1 - params2

    l2_norm = torch.norm(param_diff, p=2)

    return l2_norm

def noiseModel(model, sigma, device=torch.device("cpu")):
    model.to(device)
    if sigma > 0:
        for w_k in model.parameters():
            noise = torch.normal(0.0, sigma, size=w_k.shape).to(device)
            w_k.data.add_(noise)


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def getconloss(z, z_p, z_n, temperature):

    cos=torch.nn.CosineSimilarity(dim=-1)

    positive = cos(z, z_p)
    logits = positive.reshape(-1,1)

    negetive = cos(z, z_n)
    logits = torch.cat((logits, negetive.reshape(-1,1)), dim=1)
    logits /= temperature

    return logits

def getcon(z, z_p, z_n, temperature):

    cos=torch.nn.CosineSimilarity(dim=-1)

    positive = cos(z, z_p).reshape(-1,1)

    negetive = cos(z, z_n).reshape(-1,1)
    logits = torch.cat((positive, negetive), dim=1)
    logits /= temperature

    return logits

def plotTNSE(args, globalModel, validationDataloader, round, device):
    model = copy.deepcopy(globalModel).to(device)
    model.eval()
    embeddings = []
    labelss =[]

    with torch.no_grad():
        for _, (datas, labels) in enumerate(validationDataloader):
            datas = datas.to(device)
            labels = labels.to(device)
            _, embedding = model(datas)
            embeddings.append(embedding.cpu().numpy())
            labelss.append(labels.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labelss, axis=0)

    tsne = manifold.TSNE(n_components=2, random_state=0)
    embeddings_tsne = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    for i in range(10):
        plt.scatter(embeddings_tsne[labels == i, 0], embeddings_tsne[labels == i, 1], label=f'Class {i}', cmap='tab10')
    plt.title("t-SNE Embedding of {} Samples".format(args.dataset))
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    # plt.show()
    if args.unlearningChoice:
        plt.savefig("tnse_unlearning_{}_{}.pdf".format(args.dataset, round+1))
    else:
        plt.savefig("tnse_{}_{}.pdf".format(args.dataset, round+1))

def plotBackTNSE(args, globalModel, validationDataloader, round, device):
    model = copy.deepcopy(globalModel).to(device)
    model.eval()
    embeddings = []
    labelss =[]
    count = 0

    with torch.no_grad():
        for _, (datas, labels) in enumerate(validationDataloader):

            # add the trigger
            datasBackdoorLen = int(args.backRate * len(labels))
            for index in range(datasBackdoorLen):
                data = datas[index]
                datas[index] = addTrigger(data, args.dataset)
                labels[index] = 1
            datas = datas.to(device)
            labels = labels.to(device)

            _, embedding = model(datas)
            embeddings.append(embedding.cpu().numpy())
            labelss.append(labels.cpu().numpy())
            count+=1

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labelss, axis=0)

    tsne = manifold.TSNE(n_components=2, random_state=0)
    embeddings_tsne = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))

    for i in range(count):
        start = i * args.batchsize
        backend = start + int(args.backRate * args.batchsize)
        end = (i+1) * args.batchsize
        # if i == 0:
        #     plt.scatter(embeddings_tsne[start:backend, 0], embeddings_tsne[start:backend, 1], label='Backdoor Data', color='orange')
        #     plt.scatter(embeddings_tsne[backend:end, 0], embeddings_tsne[backend:end, 1], label='Non-Backdoor Data', color='green')
        # else:
        #     plt.scatter(embeddings_tsne[start:backend, 0], embeddings_tsne[start:backend, 1], color='orange')
        #     plt.scatter(embeddings_tsne[backend:end, 0], embeddings_tsne[backend:end, 1], color='green')

        if i == 0:
            plt.scatter(embeddings_tsne[start:backend, 0], embeddings_tsne[start:backend, 1], label='Backdoor Data',  alpha=1, s=10, c='red', marker='o')
            plt.scatter(embeddings_tsne[backend:end, 0], embeddings_tsne[backend:end, 1], label='Non-Backdoor Data',  alpha=1, s=10, c='blue', marker='x')
            # embeddings_tsne_ = embeddings_tsne[backend:end]
            # labels_ = labels[backend:end]
            # for i in range(10):
            #     plt.scatter(embeddings_tsne_[labels_ == i, 0], embeddings_tsne_[labels_ == i, 1], label=f'Class {i}', cmap='tab10')
        else:
            plt.scatter(embeddings_tsne[start:backend, 0], embeddings_tsne[start:backend, 1], alpha=1, s=10, c='red', marker='o')
            plt.scatter(embeddings_tsne[backend:end, 0], embeddings_tsne[backend:end, 1], alpha=1, s=10, c='blue', marker='x')


    # plt.title('t-SNE Embedding of FashionMNIST Samples')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    # plt.show()
    if args.unlearningChoice:
        plt.savefig("tnse_back_unlearning_{}_{}.pdf".format(args.dataset, round))
    else:
        plt.savefig("tnse_back_{}_{}.pdf".format(args.dataset, round))

def pltOverlapTNSE(args, globalModel, unlearningDataloader, remainDataloader, device):
    model = copy.deepcopy(globalModel).to(device)
    model.eval()
    embeddings1 = []
    embeddings2 = []

    with torch.no_grad():
        for _, (datas, labels) in enumerate(unlearningDataloader):
            datas = datas.to(device)
            labels = labels.to(device)
            _, embedding = model(datas)
            embeddings1.append(embedding.cpu().numpy())

    with torch.no_grad():
        for _, (datas, labels) in enumerate(remainDataloader):
            datas = datas.to(device)
            labels = labels.to(device)
            _, embedding = model(datas)
            embeddings2.append(embedding.cpu().numpy())

    
    # Convert lists of numpy arrays to a single numpy array
    embeddings1 = np.concatenate(embeddings1, axis=0)
    embeddings2 = np.concatenate(embeddings2, axis=0)

    # Create labels for embeddings
    labels1 = np.zeros(embeddings1.shape[0])
    labels2 = np.ones(embeddings2.shape[0])

    # Combine embeddings and labels
    embeddings = np.concatenate((embeddings1, embeddings2), axis=0)
    labels = np.concatenate((labels1, labels2), axis=0)

    # Perform t-SNE
    tsne = manifold.TSNE(n_components=2, random_state=0)
    embeddings_tsne = tsne.fit_transform(embeddings)

    # Plot the t-SNE results
    plt.figure(figsize=(10, 8))
    # plt.scatter(embeddings_tsne[labels == 0, 0], embeddings_tsne[labels == 0, 1], label='Unlearning Data', alpha=1, color='black', marker='+')
    # plt.scatter(embeddings_tsne[labels == 1, 0], embeddings_tsne[labels == 1, 1], label='Remaining Data', alpha=0.3, color='green', marker='*')

    plt.scatter(embeddings_tsne[labels == 0, 0], embeddings_tsne[labels == 0, 1], 
                label='Unlearning Data', alpha=1, s=10, c='red', marker='o')
    plt.scatter(embeddings_tsne[labels == 1, 0], embeddings_tsne[labels == 1, 1], 
                label='Remaining Data', alpha=1, s=10, c='blue', marker='x')
    plt.legend()
    plt.title('t-SNE of Overlap')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    # Save the plot as a PDF
    plt.savefig('overlap.pdf')
    plt.close()


def getModelName(args):
    if args.dataset == "emnist":
        model_name = "MLP"  
    elif args.dataset == "fmnist":
        model_name = "CNN"
    elif args.dataset == "cifar10":
        model_name = "ResNet"
    else:
        assert "No such dataset, please check it."

    return model_name


def addTrigger(data, datasetName):
    bd_img = torch.tensor(add_pattern_bd(data, datasetName, -1))
    return bd_img

def add_pattern_bd(x, dataset='cifar10', agent_idx=-1):
    """
    adds a trojan pattern to the image
    """
    
    # if cifar is selected, we're doing a distributed backdoor attack (i.e., portions of trojan pattern is split between agents, only works for plus)
    if dataset == 'cifar10':
        pattern_type='plus'
        x = np.transpose(x, (1, 2, 0))
        x = np.array(x.squeeze())
        agent_idx = -1

        if pattern_type == 'plus':
            start_idx = 5
            size = 4
            if agent_idx == -1:
                # vertical line
                for d in range(0, 3):  
                    for i in range(start_idx, start_idx+size+1):
                        x[i, start_idx+size//2][d] = 255
                # horizontal line
                for d in range(0, 3):  
                    for i in range(start_idx, start_idx+size + 1):
                        x[start_idx+size//2, i][d] = 255
        elif pattern_type == 'DBA':# DBA attack
            #upper part of vertical 
            start_idx = 0
            size = 4
            if agent_idx % 4 == 0:
                for d in range(0, 3):  
                    for i in range(start_idx, start_idx+(size//2)+1):
                        x[i, start_idx][d] = 0
                        
            #lower part of vertical
            elif agent_idx % 4 == 1:
                for d in range(0, 3):  
                    for i in range(start_idx+(size//2)+1, start_idx+size+1):
                        x[i, start_idx][d] = 0
                        
            #left-part of horizontal
            elif agent_idx % 4 == 2:
                for d in range(0, 3):  
                    for i in range(start_idx-size//2, start_idx+size//4 + 1):
                        x[start_idx+size//2, i][d] = 0
                        
            #right-part of horizontal
            elif agent_idx % 4 == 3:
                for d in range(0, 3):  
                    for i in range(start_idx-size//4+1, start_idx+size//2 + 1):
                        x[start_idx+size//2, i][d] = 0
        else:
            pattern_diffusion = 0
            change_range = 4
            height=32
            width=32
            diffusion = int(random.random() * pattern_diffusion * change_range)
            x[height-(6+diffusion), width-(6+diffusion)][0] = 255
            x[height-(6+diffusion), width-(6+diffusion)][1] = 255
            x[height-(6+diffusion), width-(6+diffusion)][2] = 255

            diffusion = 0
            x[height-(5+diffusion), width-(5+diffusion)][0] = 255
            x[height-(5+diffusion), width-(5+diffusion)][1] = 255
            x[height-(5+diffusion), width-(5+diffusion)][2] = 255

            diffusion = int(random.random() * pattern_diffusion * change_range)
            x[height-(4-diffusion), width-(6+diffusion)][0] = 255
            x[height-(4-diffusion), width-(6+diffusion)][1] = 255
            x[height-(4-diffusion), width-(6+diffusion)][2] = 255

            diffusion = int(random.random() * pattern_diffusion * change_range)
            x[height-(6+diffusion), width-(4-diffusion)][0] = 255
            x[height-(6+diffusion), width-(4-diffusion)][1]= 255
            x[height-(6+diffusion), width-(4-diffusion)][2] = 255

            diffusion = int(random.random() * pattern_diffusion * change_range)
            x[height-(4-diffusion), width-(4-diffusion)][0]= 255
            x[height-(4-diffusion), width-(4-diffusion)][1] = 255
            x[height-(4-diffusion), width-(4-diffusion)][2] = 255         


        x = np.transpose(x, (2, 0, 1))


    elif dataset == 'fmnist': 
        pattern_type='plus'  
        x = np.array(x.squeeze())
        if pattern_type == 'square':
            for i in range(21, 26):
                for j in range(21, 26):
                    x[i, j] = 0
        
        # elif pattern_type == 'copyright':
        #     trojan = cv2.imread('../watermark.png', cv2.IMREAD_GRAYSCALE)
        #     trojan = cv2.bitwise_not(trojan)
        #     trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
        #     x = x + trojan
            
        # elif pattern_type == 'apple':
        #     trojan = cv2.imread('../apple.png', cv2.IMREAD_GRAYSCALE)
        #     trojan = cv2.bitwise_not(trojan)
        #     trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
        #     x = x + trojan
            
        elif pattern_type == 'plus':
            start_idx = 0
            size = 3
            # vertical line  
            for i in range(start_idx, start_idx+size):
                x[i, start_idx+size//2] = 1
            
            # horizontal line
            for i in range(start_idx, start_idx+size):
                x[start_idx+size//2, i] = 1

    elif dataset == 'emnist':
        pattern_type='diffustion' 
        x = np.array(x.squeeze())
        if pattern_type == 'plus':
            start_idx = 0
            size = 3
            # vertical line  
            for i in range(start_idx, start_idx+size):
                x[i, start_idx+size//2] = 1
            
            # horizontal line
            for i in range(start_idx, start_idx+size):
                x[start_idx+size//2, i] = 1


        elif pattern_type == 'diffustion':
            x = np.array(x.squeeze())
            pattern_diffusion=0
            change_range = 4
            diffusion = int(random.random() * pattern_diffusion * change_range)
            height, width = x.shape
            x[height-(6+diffusion), width-(6+diffusion)] = 1
            diffusion = 0
            x[height-(5+diffusion), width-(5+diffusion)] = 1
            diffusion = int(random.random() * pattern_diffusion * change_range)
            x[height-(4-diffusion), width-(6+diffusion)] = 1
            diffusion = int(random.random() * pattern_diffusion * change_range)
            x[height-(6+diffusion), width-(4-diffusion)] = 1
            diffusion = int(random.random() * pattern_diffusion * change_range)
            x[height-(4-diffusion), width-(4-diffusion)] = 1

    elif dataset == 'fedemnist':
        if pattern_type == 'square':
            for i in range(21, 26):
                for j in range(21, 26):
                    x[i, j] = 0
    
        # elif pattern_type == 'copyright':
        #     trojan = cv2.imread('../watermark.png', cv2.IMREAD_GRAYSCALE)
        #     trojan = cv2.bitwise_not(trojan)
        #     trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)/255
        #     x = x - trojan
            
        # elif pattern_type == 'apple':
        #     trojan = cv2.imread('../apple.png', cv2.IMREAD_GRAYSCALE)
        #     trojan = cv2.bitwise_not(trojan)
        #     trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)/255
        #     x = x - trojan
            
        elif pattern_type == 'plus':
            start_idx = 8
            size = 7
            # vertical line  
            for i in range(start_idx, start_idx+size):
                x[i, start_idx] = 0
            
            # horizontal line
            for i in range(start_idx-size//2, start_idx+size//2 + 1):
                x[start_idx+size//2, i] = 0
            
    return x
