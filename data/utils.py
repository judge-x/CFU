import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import copy

from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd


def addBackooor(datasetName, dataset, datas, pixel, backrate):
    target_label = 9
    # backdoor = PoisoningAttackBackdoor(add_pattern_bd)
    # all_indices = np.arange(len(datas))
    # remove_indices = all_indices[np.all(datas.label == target_label, axis=1)]
    # backdoor.poison(datas[all_indices], y=target_label, broadcast=True)
    tmp = datas.indices
    count = 0
    backNum = int(len(datas) * backrate)
    for index in datas.indices:
        if count < backNum:
            if datasetName == 'cifar10' or datasetName == 'cifar100':
                data = dataset.data[index]
                label = dataset.targets[index]

                data = np.transpose(data, (2, 0, 1))

                data = addTrigger(datasetName, data, pixel).numpy()

                data = np.transpose(data, (1, 2, 0))
                # plt.figure(figsize=(8,8))
                # plt.imshow(data/255, interpolation='nearest')
                # plt.axis('off')
                # plt.savefig('backdoor_training.jpg')

                dataset.data[index] = data
                dataset.targets[index] = target_label
            elif datasetName =="mnist" or datasetName =="fmnist":
                # may need a litte repaire
                data, label = dataset[index]
                # data_ = addTrigger(datasetName, data, pixel).squeeze().numpy()
                data = addTrigger(datasetName, data, pixel).numpy()

                # plt.figure(figsize=(8,8))
                # plt.imshow(data_, cmap='gray')
                # plt.axis('off')
                # plt.savefig('backdoor_fmnist_trainf.jpg')

                dataset.data[index] = torch.tensor(data)
                dataset.targets[index] = target_label

        count += 1

    return datas


# def addTrigger(dataset, data, pixel):
        
#         # change origin data
#         new_data = np.copy(data)
#         pattern_diffusion=0
#         params=None
#         channels, height, width = new_data.shape
#         if pixel == 1:
#             for c in range(channels):
#                 if dataset == 'cifar10' or dataset == 'cifar100':
#                     new_data[c, height-3, width-3] = 1
#                     new_data[c, height-2, width-4] = 1
#                     new_data[c, height-4, width-2] = 1
#                     new_data[c, height-2, width-2] = 1
#                 elif dataset == 'fmnist':
#                     new_data[c, height-3, width-3] = 1
#                     new_data[c, height-2, width-4] = 1
#                     new_data[c, height-4, width-2] = 1
#                     new_data[c, height-2, width-2] = 1
        
#         elif pixel == 2:
#             change_range = 4
            
#             if dataset == 'cifar10' or dataset == 'cifar100':
#                 diffusion = int(random.random() * pattern_diffusion * change_range)
#                 new_data[0, height-(6+diffusion), width-(6+diffusion)] = 255
#                 new_data[1, height-(6+diffusion), width-(6+diffusion)] = 255
#                 new_data[2, height-(6+diffusion), width-(6+diffusion)] = 255

#                 diffusion = 0
#                 new_data[0, height-(5+diffusion), width-(5+diffusion)] = 255
#                 new_data[1, height-(5+diffusion), width-(5+diffusion)] = 255
#                 new_data[2, height-(5+diffusion), width-(5+diffusion)] = 255

#                 diffusion = int(random.random() * pattern_diffusion * change_range)
#                 new_data[0, height-(4-diffusion), width-(6+diffusion)] = 255
#                 new_data[1, height-(4-diffusion), width-(6+diffusion)] = 255
#                 new_data[2, height-(4-diffusion), width-(6+diffusion)] = 255

#                 diffusion = int(random.random() * pattern_diffusion * change_range)
#                 new_data[0, height-(6+diffusion), width-(4-diffusion)] = 255
#                 new_data[1, height-(6+diffusion), width-(4-diffusion)] = 255
#                 new_data[2, height-(6+diffusion), width-(4-diffusion)] = 255

#                 diffusion = int(random.random() * pattern_diffusion * change_range)
#                 new_data[0, height-(4-diffusion), width-(4-diffusion)] = 255
#                 new_data[1, height-(4-diffusion), width-(4-diffusion)] = 255
#                 new_data[2, height-(4-diffusion), width-(4-diffusion)] = 255
                
#             elif dataset=='mnist' or dataset == 'fmnist':
#                 diffusion = int(random.random() * pattern_diffusion * change_range) 
#                 new_data[0, height-(6+diffusion), width-(6+diffusion)] = 1

#                 diffusion = 0
#                 new_data[0, height-(5+diffusion), width-(5+diffusion)] = 1

#                 diffusion = int(random.random() * pattern_diffusion * change_range)
#                 new_data[0, height-(4-diffusion), width-(6+diffusion)] = 1

#                 diffusion = int(random.random() * pattern_diffusion * change_range)
#                 new_data[0, height-(6+diffusion), width-(4-diffusion)] = 1

#                 diffusion = int(random.random() * pattern_diffusion * change_range)
#                 new_data[0, height-(4-diffusion), width-(4-diffusion)] = 1


#         return torch.Tensor(new_data)


# def poison_dataset(name, dataset, datas=None, pixel=2, backrate=1):

#     data_idx = datas.indices
#     # all_idxs = (dataset.targets == args.base_class).nonzero().flatten().tolist()
#     # if data_idxs != None:
#     #     all_idxs = list(set(all_idxs).intersection(data_idxs))
#     target_class = 0
#     poison_frac = backrate
#     backNum = int(len(datas) * poison_frac)
#     poison_idxs = data_idx[len(datas)-backNum:len(datas)]
    
#     for idx in poison_idxs:
#         if name == 'fedemnist':
#             clean_img = dataset.inputs[idx]
#         else:
#             clean_img = dataset.data[idx]
#         bd_img = add_pattern_bd(clean_img, name, -1)

#         # plt.figure(figsize=(8,8))
#         # plt.imshow(bd_img/255, interpolation='nearest')
#         # plt.axis('off')
#         # plt.savefig('backdoor_training.jpg')

#         if name == 'fedemnist':
#              dataset.inputs[idx] = torch.tensor(bd_img)
#         else:
#             dataset.data[idx] = torch.tensor(bd_img)
#         dataset.targets[idx] = target_class    
#     return datas


# def add_pattern_bd(x, dataset='cifar10', agent_idx=-1):
#     """
#     adds a trojan pattern to the image
#     """
#     x = np.array(x.squeeze())
    
#     # if cifar is selected, we're doing a distributed backdoor attack (i.e., portions of trojan pattern is split between agents, only works for plus)
#     if dataset == 'cifar10':
#         pattern_type='plus'
#         if pattern_type == 'plus':
#             start_idx = 5
#             size = 6
#             if agent_idx == -1:
#                 # vertical line
#                 for d in range(0, 3):  
#                     for i in range(start_idx, start_idx+size+1):
#                         x[i, start_idx][d] = 0
#                 # horizontal line
#                 for d in range(0, 3):  
#                     for i in range(start_idx-size//2, start_idx+size//2 + 1):
#                         x[start_idx+size//2, i][d] = 0
#             else:# DBA attack
#                 #upper part of vertical 
                
#                 if agent_idx % 4 == 0:
#                     for d in range(0, 3):  
#                         for i in range(start_idx, start_idx+(size//2)+1):
#                             x[i, start_idx][d] = 0
                            
#                 #lower part of vertical
#                 elif agent_idx % 4 == 1:
#                     for d in range(0, 3):  
#                         for i in range(start_idx+(size//2)+1, start_idx+size+1):
#                             x[i, start_idx][d] = 0
                            
#                 #left-part of horizontal
#                 elif agent_idx % 4 == 2:
#                     for d in range(0, 3):  
#                         for i in range(start_idx-size//2, start_idx+size//4 + 1):
#                             x[start_idx+size//2, i][d] = 0
                            
#                 #right-part of horizontal
#                 elif agent_idx % 4 == 3:
#                     for d in range(0, 3):  
#                         for i in range(start_idx-size//4+1, start_idx+size//2 + 1):
#                             x[start_idx+size//2, i][d] = 0
#         else:
#             diffusion = int(random.random() * pattern_diffusion * change_range)
#             x[height-(6+diffusion), width-(6+diffusion)][0] = 255
#             x[height-(6+diffusion), width-(6+diffusion)][1] = 255
#             x[height-(6+diffusion), width-(6+diffusion)][2] = 255

#             diffusion = 0
#             x[height-(5+diffusion), width-(5+diffusion)][0] = 255
#             x[height-(5+diffusion), width-(5+diffusion)][1] = 255
#             x[height-(5+diffusion), width-(5+diffusion)][2] = 255

#             diffusion = int(random.random() * pattern_diffusion * change_range)
#             x[height-(4-diffusion), width-(6+diffusion)][0] = 255
#             x[height-(4-diffusion), width-(6+diffusion)][1] = 255
#             x[height-(4-diffusion), width-(6+diffusion)][2] = 255

#             diffusion = int(random.random() * pattern_diffusion * change_range)
#             x[height-(6+diffusion), width-(4-diffusion)][0] = 255
#             x[height-(6+diffusion), width-(4-diffusion)][1]= 255
#             x[height-(6+diffusion), width-(4-diffusion)][2] = 255

#             diffusion = int(random.random() * pattern_diffusion * change_range)
#             x[height-(4-diffusion), width-(4-diffusion)][0]= 255
#             x[height-(4-diffusion), width-(4-diffusion)][1] = 255
#             x[height-(4-diffusion), width-(4-diffusion)][2] = 255      
                              
#     elif dataset == 'fmnist':   
#         pattern_type='plus' 
#         if pattern_type == 'square':
#             for i in range(21, 26):
#                 for j in range(21, 26):
#                     x[i, j] = 255
        
#         # elif pattern_type == 'copyright':
#         #     trojan = cv2.imread('../watermark.png', cv2.IMREAD_GRAYSCALE)
#         #     trojan = cv2.bitwise_not(trojan)
#         #     trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
#         #     x = x + trojan
            
#         # elif pattern_type == 'apple':
#         #     trojan = cv2.imread('../apple.png', cv2.IMREAD_GRAYSCALE)
#         #     trojan = cv2.bitwise_not(trojan)
#         #     trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
#         #     x = x + trojan
            
#         elif pattern_type == 'plus':
#             start_idx = 1
#             size = 3
#             # vertical line  
#             for i in range(start_idx, start_idx+size):
#                 x[i, start_idx] = 255
            
#             # horizontal line
#             for i in range(start_idx-size//2, start_idx+size//2 + 1):
#                 x[start_idx+size//2, i] = 255

#     elif dataset == 'mnist':
#         pattern_diffusion=0
#         change_range = 4
#         diffusion = int(random.random() * pattern_diffusion * change_range)
#         height, width = x.shape
#         x[height-(6+diffusion), width-(6+diffusion)] = 1
#         diffusion = 0
#         x[height-(5+diffusion), width-(5+diffusion)] = 1
#         diffusion = int(random.random() * pattern_diffusion * change_range)
#         x[height-(4-diffusion), width-(6+diffusion)] = 1
#         diffusion = int(random.random() * pattern_diffusion * change_range)
#         x[height-(6+diffusion), width-(4-diffusion)] = 1
#         diffusion = int(random.random() * pattern_diffusion * change_range)
#         x[height-(4-diffusion), width-(4-diffusion)] = 1

#     elif dataset == 'fedemnist':
#         if pattern_type == 'square':
#             for i in range(21, 26):
#                 for j in range(21, 26):
#                     x[i, j] = 0
    
#         # elif pattern_type == 'copyright':
#         #     trojan = cv2.imread('../watermark.png', cv2.IMREAD_GRAYSCALE)
#         #     trojan = cv2.bitwise_not(trojan)
#         #     trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)/255
#         #     x = x - trojan
            
#         # elif pattern_type == 'apple':
#         #     trojan = cv2.imread('../apple.png', cv2.IMREAD_GRAYSCALE)
#         #     trojan = cv2.bitwise_not(trojan)
#         #     trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)/255
#         #     x = x - trojan
            
#         elif pattern_type == 'plus':
#             start_idx = 8
#             size = 5
#             # vertical line  
#             for i in range(start_idx, start_idx+size):
#                 x[i, start_idx] = 0
            
#             # horizontal line
#             for i in range(start_idx-size//2, start_idx+size//2 + 1):
#                 x[start_idx+size//2, i] = 0
            
#     return x

def plotbubble(datsetName, alpha, clientNumber, classNumber, sampleSizes, train):
    sizes = np.sqrt(sampleSizes)

    fig, ax  = plt.subplots()

    ax.set_xlim([0, clientNumber+1])
    ax.set_ylim([0, classNumber+1])
    ax.set_xticks(range(1, clientNumber+1))
    ax.set_yticks(range(1, classNumber+1))

    ax.set_xlabel("Client IDs")
    ax.set_ylabel("Class IDs")

    ax.grid(color='lightgrey', linestyle="-", linewidth=0.5)

    for x in range(clientNumber):
        for y in range(classNumber):
            ax.scatter(x+1, y+1, s=sizes[x*classNumber+y]*10, c='lightskyblue')

    plt.savefig("distribution_{}_{}_{}.pdf".format(datsetName, alpha, train))
    




