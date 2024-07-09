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
    
    def trainFFMU(self, globalModel):
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

                for param in self.model.parameters():
                    if param.grad is not None:
                        noise = 0.01 * torch.randn_like(param.grad)
                        param.grad += noise
                        param.grad = torch.sign(param.grad)

                optimizer.step()
                optimizer.zero_grad()

                lossCount += loss.detach().item() * len(labels)

                _, preds = torch.max(predict.detach().data, 1)
                numCorrect += preds.eq(labels).sum().cpu().data.numpy()
                numTotal += len(labels)
        
        return lossCount/numTotal, numCorrect/numTotal

    def trainIFU(self, globalModel, globalRound):
        optimizer = opti.Adam(self.model.parameters(),lr=self.lr, weight_decay=self.wdecay)
        lossFun = torch.nn.CrossEntropyLoss()

        self.setModel(globalModel)
        globalModel_ = copy.deepcopy(self.model)
        self.model.train()

        for e in range(self.epochs):
            lossCount = 0.0
            numCorrect = 0
            numTotal = 0

            for _, (datas, labels) in enumerate(self.dataloaderTrain):

                datas = datas.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()

                predict, _ = self.model(datas)
                loss = lossFun(predict, labels.long())

                loss.backward()

                optimizer.step()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                lossCount += loss.detach().item() * len(labels)

                _, preds = torch.max(predict.detach().data, 1)
                numCorrect += preds.eq(labels).sum().cpu().data.numpy()
                numTotal += len(labels)
        
        psi = 0
        B = 1
        gamma = (self.args.globalEpochs-globalRound-1)*1
        deltaNorm = getDeltaNorm(globalModel_, self.model)
        psi = math.pow(B, gamma) * deltaNorm * (self.trainNumber/50000)

        return lossCount/numTotal, numCorrect/numTotal, psi
    
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

                # lossUnlearn = - torch.log(torch.exp(F.cosine_similarity(embedding[:unlearningLen], embedding_glo[:unlearningLen]) / self.temperature) / (torch.exp(F.cosine_similarity(embedding[:unlearningLen], embedding_glo[:unlearningLen]) / self.temperature) + torch.exp(F.cosine_similarity(embedding[:unlearningLen], embedding_init[:unlearningLen]) / self.temperature)))


                # enhance
                # lossRecovers = []
                # for _, (dataRecover, labelR) in enumerate(recoverDataloader):
                #     dataRecover = dataRecover.to(self.device)
                #     _, embeddingRecover = self.model(dataRecover)
                #     _, embeddingPretrain = pertrainedModel_(dataRecover)
                #     _, embeddingGlobal = globalModel_(dataRecover)
                #     logitsRecover = getconloss(embeddingRecover, embeddingPretrain, embeddingGlobal)
                #     labels = torch.zeros(dataRecover.size(0)).long().to(self.device)
                #     lossRecovers.append(lossFun(logitsRecover, labels.long()))
                # lossRecover = np.mean(lossRecovers)

                # loss = lossRecover + self.mu * lossUnlearn

                # lossClear = lossFun(predict[unlearningLen:], labels[unlearningLen:])
                # loss = lossClear + self.mu * torch.mean(lossUnlearn)
                # NTXLoss
                # cos=torch.nn.CosineSimilarity(dim=-1)
                # positive = cos(embedding, embedding_init).reshape(-1,1)
                # negetive = cos(embedding, embedding_glo).reshape(-1,1)
                # embeddings = torch.cat((negetive,positive))
                # indices = torch.arange(0, embedding.size(0), device=self.device)
                # labels_ = torch.cat((indices, indices))
                # loss = lossNTX(embeddings, labels_)

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
    

    def trainCan(self, globalModel, incompetentT, competentT):
        optimizer = opti.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wdecay)
        lossFun = torch.nn.CrossEntropyLoss()
        # lossNTX = losses.NTXentLoss(0.01)

        self.setModel(globalModel)
        self.model.train()
        
        #init teacher model
        incompetentT.to(self.device)
        competentT.to(self.device)
        incompetentT.eval()
        competentT.eval()

        for _ in range(self.epochs):
            lossCount = 0.0
            numCorrect = 0
            numTotal = 0

            for index, (datas, labels) in enumerate(self.dataloaderTrain):
                datas = datas.to(self.device)
                labels = labels.to(self.device)
                predict, _ = self.model(datas)

                predictCompenent, _ = competentT(datas)
                predictInCompenent, _ = incompetentT(datas)
                loss = 0 * F.kl_div(predict, predictCompenent, reduction='batchmean') + 1 * F.kl_div(predict, predictInCompenent, reduction='batchmean') 
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
    
    def trainRip(self, globalModel):
        optimizer = AdaHessian(self.model.parameters(), lr=self.lr, weight_decay=self.wdecay)
        lossFun = torch.nn.CrossEntropyLoss()

        self.setModel(globalModel)
        self.model.train()

        for _ in range(self.epochs):
            lossCount = 0.0
            numCorrect =0
            numTotal = 0
            for _, (datas, labels) in enumerate(self.dataloaderTrain):
                datas, labels = datas.to(self.device), labels.to(self.device)
                
                
                optimizer.zero_grad()
                predict, _ = self.model(datas)
                loss = lossFun(predict, labels)

                loss.backward(create_graph=True)
                optimizer.step()
                
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                lossCount += loss.detach().item() * len(labels)

                _, pred = torch.max(predict.detach().data, 1)
                numCorrect += pred.eq(labels).sum().cpu().data.numpy()
                numTotal += len(labels)

        del datas, labels
        return lossCount/numTotal, numCorrect/numTotal



    def trainAsc(self, globalModel):
        optimizer = opti.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wdecay)
        lossFun = torch.nn.CrossEntropyLoss()

        self.setModel(globalModel)
        self.model.train()
        
        for _ in range(self.epochs):
            lossCount = 0.0
            numCorrect =0
            numTotal = 0

            for _, (datas, labels) in enumerate(self.dataloaderTrain):


                datas, labels = datas.to(self.device), labels.to(self.device)


                predictClear, _ = self.model(datas)
                loss = -lossFun(predictClear, labels.long())

                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                lossCount += loss.detach().item() * len(labels)
                numCorrect += 0
                numTotal += len(labels)

            del datas, labels

        return lossCount/numTotal, numCorrect/numTotal

    def trainActive(self, globalModel):
        optimizer = opti.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wdecay)
        lossFun = torch.nn.CrossEntropyLoss()

        self.setModel(globalModel)
        self.model.train()
        ewc = EWC()
        for _ in range(self.epochs):
            lossCount = 0.0
            numCorrect =0
            numTotal = 0

            for _, (datas, labels) in enumerate(self.dataloaderTrain):

                for index in range(len(labels)):
                    labels[index] = random.randint(0,8)

                datas, labels = datas.to(self.device), labels.to(self.device)

                predictBack, _ = self.model(datas)
                lossback = lossFun(predictBack, labels)

                # #clear loss
                # predictClear, _ = self.model(datasClear)
                lossclear = ewc.regularize(self.model.named_parameters()).to(self.device)
                # lossclear = lossFun(predictClear, labelsClear)

                loss = lossback+lossclear

                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                lossCount += loss.detach().item() * len(labels)
                # _, preds = torch.max(predict.detach().data, 1)
                # numCorrect += preds.eq(labels).sum().cpu().data.numpy()
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
        

