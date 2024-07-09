from torch.utils.data import DataLoader
import torch
import torch.optim as opti
import torch.nn.functional as F
import copy
from torch.optim.lr_scheduler import StepLR
from pytorch_metric_learning import losses
from .utils import *
from .lossFun.ewc import *
from .lossFun.ada_hessain import *
import math

class Client(object):
    def __init__(self, args, clientIndex, datasetTrain, datasetTest, model, device) -> None:
        self.args = args

        self.device = device
        self.clientIndex = clientIndex
        self.epochs = args.localEpochs
        self.model = copy.deepcopy(model)
        self.model = self.model.to(self.device)

        self.lr = args.lr
        self.wdecay = args.wdecay
        self.batchsize = args.batchsize

        self.trainNumber = len(datasetTrain)
        self.testNumber = len(datasetTest)

        self.previousModel = None
        self.deltamodel = None


        self.dataloaderTrain = DataLoader(datasetTrain, batch_size=self.batchsize, shuffle=False)
        self.dataloaderTest = DataLoader(datasetTest, batch_size=self.batchsize, shuffle=False)

        if self.args.dataset == "emnist":
            self.sample_per_class = torch.zeros(37)
        else:
            self.sample_per_class = torch.zeros(10)

        self.qualified_labels = []
        for x, y in self.dataloaderTrain:
            for yy in y:
                self.sample_per_class[yy.item()] += 1

        self.temperature = args.tau
        self.mu = args.mu


    def train(self, globalModel):
        optimizer = opti.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wdecay)
        lossFun = torch.nn.CrossEntropyLoss()

        self.setModel(globalModel)
        self.model.train()
        
        for e in range(self.epochs):
            lossCount = 0.0
            numCorrect =0
            numTotal = 0

            for _, (datas, labels) in enumerate(self.dataloaderTrain):
                datas = datas.to(self.device)
                labels = labels.to(self.device)              
                predict, _ = self.model(datas)

                loss = lossFun(predict, labels.long())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                lossCount += loss.detach().item() * len(labels)

                _, preds = torch.max(predict.detach().data, 1)
                numCorrect += preds.eq(labels).sum().cpu().data.numpy()
                numTotal += len(labels)
         
        return lossCount/numTotal, numCorrect/numTotal
    
    
    def trainCon(self, globalModel, teacherModel, pertrainedModel):
        optimizer = opti.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wdecay)
        lossFun = torch.nn.CrossEntropyLoss()
        # lossNTX = losses.NTXentLoss(0.01)

        self.setModel(globalModel)
        self.model.train()
        
        #init teacher model
        globalModel_ = copy.deepcopy(teacherModel).to(self.device)
        globalModel_.load_state_dict(globalModel)
        globalModel_.eval()
        teacherModel_ = copy.deepcopy(teacherModel).to(self.device)
        teacherModel_.eval()
        pertrainedModel_ = copy.deepcopy(pertrainedModel).to(self.device)
        pertrainedModel_.eval()

        for _ in range(self.epochs):
            lossCount = 0.0
            numCorrect = 0
            numTotal = 0

            for index, (datas, labels) in enumerate(self.dataloaderTrain):
                datas = datas.to(self.device)
                labels = labels.to(self.device)
                predict, embedding = self.model(datas)

                # unlearningRate = 0.9
                # unlearningLen = int(unlearningRate*len(labels))

                # unlearning
                _, embedding_glo = globalModel_(datas)
                _, embedding_init = teacherModel_(datas)
                logits = getconloss(embedding, embedding_init, embedding_glo, self.temperature)
                labelsZero = torch.zeros(datas.size(0)).long().to(self.device)
                lossUnlearn = lossFun(logits, labelsZero.long())
                loss = self.mu * lossUnlearn             

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                lossCount += loss.detach().item() * len(labels)

                _, preds = torch.max(predict.detach().data, 1)
                numCorrect += preds.eq(labels).sum().cpu().data.numpy()
                numTotal += len(labels)

            del datas, labels

        return lossCount/numTotal, numCorrect/numTotal
    

    def test(self, globalModel):
        model = copy.deepcopy(globalModel).to(self.device)
        model.eval()
        lossFun = torch.nn.CrossEntropyLoss()
        lossCount = 0
        accuracyCount = 0
        with torch.no_grad():
            for _, (data, labels) in enumerate(self.dataloaderTrain):

                data, labels = data.to(self.device), labels.to(self.device)
                
                predict, _ = model(data)
                loss = lossFun(predict, labels)

                lossCount += loss.detach().item() * len(labels)
                
                _, preds = torch.max(predict.detach().data, 1)
                accuracyCount += preds.eq(labels).sum().cpu().data.numpy()

        return lossCount/self.trainNumber, accuracyCount/self.trainNumber
    
    def setModel(self, model):

        self.model.load_state_dict(model)

    def setClassifier(self, model):

        self.classifier.load_state_dict(model)

    def getModel(self):

        return self.model.state_dict()
    
    def getClassifier(self):

        return self.classifier.state_dict()
    
    def scaleUp(self, globalParameter):
        print("Conduct Scale Up")
        gamme = 5
        localParameter = self.model.state_dict()
        # localParameter = {k: v.to(self.device) for k, v in localParameter.items()}

        for key, value in globalParameter.items():
            globalParameter[key] += (gamme * (localParameter[key] - globalParameter[key])).long()
        return globalParameter

    def getDeltaModel(self, globalParameter):

        self.deltamodel = copy.deepcopy(self.model.state_dict())

        # for key, value in local.items():
        #     local[key] -= globalModel[key]
 
        gamme = 5
        localParameter = self.getModel()

        for key, _ in self.deltamodel.items():
            self.deltamodel[key] =  (gamme * (localParameter[key] - globalParameter[key])).long() 
        

