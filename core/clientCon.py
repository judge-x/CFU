from torch.utils.data import DataLoader
import torch
import torch.optim as opti
import torch.nn.functional as F
import copy

from alive_progress import alive_bar
from .utils import *
from .lossFun.supcon import *

class ClientCon(object):
    def __init__(self, args, clientIndex, datasetTrain, datasetTest, model, classifier, device) -> None:
        self.args = args

        self.device = device
        self.clientIndex = clientIndex
        self.epochs = args.localEpochs
        self.model = copy.deepcopy(model)
        self.model = self.model.to(self.device)
        self.classifier = copy.deepcopy(classifier)
        self.classifier = self.classifier.to(self.device)

        self.lr = args.lr
        self.wdecay = args.wdecay
        self.momentum = args.momentum
        self.batchsize = args.batchsize

        self.trainNumber = len(datasetTrain)
        self.testNumber = len(datasetTest)

        self.dataloaderTrain = DataLoader(datasetTrain, batch_size=self.batchsize, shuffle=True, drop_last=True)
        self.dataloaderTest = DataLoader(datasetTest, batch_size=self.batchsize, shuffle=True, drop_last=True)

    def train(self, globalModel):
        optimizer = opti.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wdecay)
        # lossFun = torch.nn.CrossEntropyLoss()

        self.setModel(globalModel)
        self.model.train()

        lossFun = SupConLoss(temperature=0.1)
        self.model.train

        for e in range(self.epochs):
            lossSum = 0.0
            numTotal = 0
            lossAvg = 0.0
            accuracyCount = 0
            self.model.train()

            for _, (datas, labels) in enumerate(self.dataloaderTrain):

                datas = torch.cat([datas[0], datas[1]], dim=0)
                datas = datas.to(self.device)
                labels = labels.to(self.device)
                bsz = labels.shape[0]

                _, features = self.model(datas)
                f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1).to(self.device)

                loss = lossFun(features, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()    

                numTotal += bsz
                lossSum += loss.detach().item() * bsz
                lossAvg = lossSum/numTotal

                # _, preds = torch.max(outputs[:len(labels), :].detach().data, 1)
                # numCorrect += preds.eq(labels).sum().cpu().data.numpy()
                # features_ = self.model.encoder(datas[0])
                # output = self.classifier(features_.detach())  
                # accuracyCount += self.accuracy(output, labels)
                    

            print("Training Round {}: loss {}".format(e, lossAvg))

        return lossAvg

            # self.model.eval()
            # TestlossCount = 0
            # TestaccuracyCount = 0
            # lossEval = torch.nn.CrossEntropyLoss()
            # for _, (data, labels) in enumerate(self.dataloaderTest):

            #     data, labels = data.to(self.device), labels.to(self.device)
                
            #     _, predict = self.model(data)
            #     # loss = lossEval(predict, labels)

            #     # TestlossCount += loss.detach().item()
                 
            #     _, preds = torch.max(predict.detach().data, 1)
            #     # TestaccuracyCount += preds.eq(labels).sum().cpu().data.numpy()
            #     TestaccuracyCount = torch.sum(preds == labels).item()
            
            # print("Test Round {}: Accuracy{}".format(e, TestaccuracyCount/self.testNumber))

    def finetune(self):
        print("Conduct fine tuning. ")
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.classifier.train()
        lossFun = torch.nn.CrossEntropyLoss()
        optimizer = opti.SGD(self.model.parameters(), lr=0.01)
        finetuneEpochs = 5

        for e in range(finetuneEpochs):
            for _, (data, labels) in enumerate(self.dataloaderTest):

                data, labels = data[0].to(self.device), labels.to(self.device)
                
                features, _ = self.model(data)
                output = self.classifier(features.detach())

                loss = lossFun(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tmp = self.classifier.state_dict()
        
    def test(self):
        self.model.eval()
        # self.classifier.train()
        # # Fine-tuning
        lossFun = torch.nn.CrossEntropyLoss()
        # optimizer = opti.SGD(self.model.parameters(), lr=0.1)
        # for _, (data, labels) in enumerate(self.dataloaderTest):

        #     data, labels = data.to(self.device), labels.to(self.device)
            
        #     features, _ = self.model(data)
        #     output = self.classifier(features.detach())

        #     loss = lossFun(output, labels)

        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        # Validation
        self.classifier.eval()
        numTotal = 0
        lossCount = 0
        accuracyCount = 0

        lossAvg = 0
        accAvg = 0
        with torch.no_grad():
            for _, (data, labels) in enumerate(self.dataloaderTest):

                data, labels = data[0].to(self.device), labels.to(self.device)
                
                features, _ = self.model(data)
                output = self.classifier(features.detach())

                loss = lossFun(output, labels)

                numTotal += len(labels)
                lossCount += loss.detach().item() * len(labels)
                lossAvg = lossCount/numTotal
                
                # accuracyCount += self.accuracy(output, labels)

                _, preds = torch.max(output.detach().data, 1)
                accuracyCount += preds.eq(labels).sum().cpu().data.numpy()

            for param in self.model.parameters():
                param.requires_grad = True

        return lossAvg, accuracyCount/numTotal

    def accuracy(self, output, target):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = 1
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            
            correct_k = correct[:1].view(-1).float().sum(0, keepdim=True).detach().item()
            # accuracy = correct_k.mul_(100.0 / batch_size)
            return correct_k

    def setModel(self, model):

        self.model.load_state_dict(model)

    def getModel(self):

        return self.model.state_dict()