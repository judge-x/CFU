import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleNet(nn.Module):
    def __init__(self, name=None, created_time=None):
        super(SimpleNet, self).__init__()
        self.created_time = created_time
        self.name=name



    def visualize(self, vis, epoch, acc, loss=None, eid='main', is_poisoned=False, name=None):
        if name is None:
            name = self.name + '_poisoned' if is_poisoned else self.name
        vis.line(X=np.array([epoch]), Y=np.array([acc]), name=name, win='vacc_{0}'.format(self.created_time), env=eid,
                                update='append' if vis.win_exists('vacc_{0}'.format(self.created_time), env=eid) else None,
                                opts=dict(showlegend=True, title='Accuracy_{0}'.format(self.created_time),
                                          width=700, height=400))
        if loss is not None:
            vis.line(X=np.array([epoch]), Y=np.array([loss]), name=name, env=eid,
                                     win='vloss_{0}'.format(self.created_time),
                                     update='append' if vis.win_exists('vloss_{0}'.format(self.created_time), env=eid) else None,
                                     opts=dict(showlegend=True, title='Loss_{0}'.format(self.created_time), width=700, height=400))

        return


class CNN_EMNIST(nn.Module):
    def __init__(self, numberChannel=1) -> None:
        super(CNN_EMNIST, self).__init__()
        
        self.con_v1 = nn.Conv2d(numberChannel, out_channels=10, kernel_size=5)
        self.con_v2 = nn.Conv2d(10, 20, kernel_size=5)

        self.fc1 = nn.Linear(320, 50)
        self.fc1_drop = nn.Dropout2d()
        self.fc2 = nn.Linear(50, 37)

    def forward(self, x):
        x = F.relu(self.con_v1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.con_v2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 320)  #Flatten
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

class MLP_MNIST_SUP(nn.Module):
    def __init__(self, num_classes = 10, embeddingLen=128) -> None:
        super(MLP_MNIST_SUP, self).__init__()
        self.num_classes= num_classes
        
        self.fc1 = nn.Linear(28 * 28, embeddingLen)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(embeddingLen, num_classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        embedding = F.normalize(x)
        x = self.fc2(x)
        output = x
        return output, embedding
    
    def head(self, embedding):
        output = self.fc2(embedding)
        
        return output

# class CNN_MNIST_SUP(nn.Module):
#     def __init__(self, numberChannel=1) -> None:
#         super(CNN_MNIST_SUP, self).__init__()
        
#         self.con_v1 = nn.Conv2d(numberChannel, out_channels=10, kernel_size=5)
#         self.con_v2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.fc1 = nn.Linear(320, 50)
#         self.fc1_drop = nn.Dropout2d()
#         self.fc2 = nn.Linear(50, 10)

#     def forward(self, x):
#         x = F.relu(self.con_v1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.con_v2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 320)  #Flatten
#         x = F.relu(self.fc1(x))
#         embedding = F.normalize(x)
#         x = self.fc1_drop(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output, embedding
    
class CNN_FMNIST_SUP(nn.Module):
    def __init__(self, num_classes=10, embeddingLen=128):
        super(CNN_FMNIST_SUP, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        
        out = self.fc(out)
        embedding = F.normalize(out)
        out = self.fc2(out)
        return out, embedding
    
    def head(self, embedding):
        out = self.fc2(embedding)
        return out

class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    

class CNN_CIFAR10_SUP(nn.Module):
    def __init__(self,num_classes=10, embeddingLen=128):
        super(CNN_CIFAR10_SUP, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        embedding = F.normalize(x)
        x = self.fc3(x)
        return x, embedding
    


# class CNN_CIFAR10_SUP(SimpleNet):
#     def __init__(self, name=None, created_time=None):
#         super(CNN_CIFAR10_SUP, self).__init__(f'{name}_Simple', created_time)
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         embedding = F.normalize(x)
#         x = self.fc3(x)
#         return x, embedding
    

# class CNN_Classifier(nn.Module):
#     """Linear classifier"""
#     def __init__(self, num_classes=10,feat_dim=128):
#         super(CNN_Classifier, self).__init__()
#         self.fc = nn.Linear(feat_dim, num_classes)

#     def forward(self, features):
#         return self.fc(features)
    

class CNN_CIFAR10_SUP(nn.Module):
    def __init__(self):
        super(CNN_CIFAR10_SUP, self).__init__()
        # convolutional layer (sees 32x32x3 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # convolutional layer (sees 16x16x16 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # convolutional layer (sees 8x8x32 tensor)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (64 * 4 * 4 -> 500)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(512, 128)
        # dropout layer (p=0.25)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(-1, 64 * 4 * 4)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        embedding = F.normalize(x)
        x = self.fc3(x)

        return x, embedding
    
    def head(self, embedding):
        out = self.fc3(embedding)
        return out