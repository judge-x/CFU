from torch.utils.data import DataLoader
import torchvision
import torch
import torch.optim as opti
import torch.nn.functional as F
import torch.nn.utils as U
import copy
from .utils import *
import matplotlib.pyplot as plt
from .lossFun.ewc import *
from .lossFun.ada_hessain import *

class ClientBackdoor(object):
    def __init__(self, args, clientIndex, datasetTrain, datasetTest, model, device) -> None:
        self.args = args

        self.device = device
        self.clientIndex = clientIndex
        self.epochs = args.localEpochs
        self.model = copy.deepcopy(model)
        self.model = self.model.to(self.device)
        self.deltamodel = None

        self.lr = args.lr
        self.wdecay = args.wdecay
        self.momentum = args.momentum
        self.batchsize = args.batchsize

        self.trainNumber = len(datasetTrain)
        self.testNumber = len(datasetTest)
        
        #Backdoor
        self.backdoorTargetLabel = 9
        self.pixel = 2
        self.modelZero = None

        self.temperature = args.tau
        self.mu = args.mu

        # if self.args.unlearningChoice:
        #     shuffle = False
        # else:
        #     shuffle = True

        self.dataloaderTrain = DataLoader(datasetTrain, batch_size=self.batchsize, shuffle=False, drop_last=True)
        self.dataloaderTest = DataLoader(datasetTest, batch_size=self.batchsize, shuffle=False, drop_last=True)

    def train(self, globalModel, lr):
        # optimizer = opti.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wdecay)
        optimizer = opti.Adam(self.model.parameters(), lr=lr, weight_decay=self.wdecay)
        lossFun = torch.nn.CrossEntropyLoss()

        self.setModel(globalModel)
        self.model.train()
        
        for _ in range(self.epochs):
            lossCount = 0.0
            numCorrect =0
            numTotal = 0

            for _, (datas, labels) in enumerate(self.dataloaderTrain):

                # backdoor the former dataset
                if self.clientIndex == 0:
                    datasBackdoorLen = int(self.args.backRate * len(labels))
                    for index in range(datasBackdoorLen):
                        data = datas[index]
                        datas[index] = addTrigger(data, self.args.dataset)
                        labels[index] = self.backdoorTargetLabel

                datas = datas.to(self.device)
                labels = labels.to(self.device)

                # for index in range(len(datas)):
                #     plt.figure(figsize=(8,8))
                #     # #mnist/fmnist
                #     # plt.imshow(data.cpu().squeeze()/255, interpolation='nearest', cmap='gray')
                #     # cifar10
                #     image = np.transpose(datas[index].cpu().numpy(), (1, 2, 0))
                #     plt.imshow(image, interpolation='nearest')
                #     plt.axis('off')
                #     plt.savefig("./picture/backdoor_cifar10_trainf_{}.jpg".format(index))
                
                predict, _ = self.model(datas)
                loss = lossFun(predict, labels.long())

                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                lossCount += loss.detach().item() * len(labels)
                _, preds = torch.max(predict.detach().data, 1)
                numCorrect += preds.eq(labels).sum().cpu().data.numpy()
                numTotal += len(labels)

            del datas, labels

        # scale up
        # if self.clientIndex == 0:
        #     self.setModel(self.scaleUp(globalModel))

        return lossCount/numTotal, numCorrect/numTotal
    
    def trainCon(self, globalModel, teacherModel, pretrainModel):
        optimizer = opti.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wdecay)
        lossFun = torch.nn.CrossEntropyLoss()

        self.setModel(globalModel)
        self.model.train()
        
        #init teacher model
        globalModel_ = copy.deepcopy(teacherModel).to(self.device)
        globalModel_.load_state_dict(globalModel)
        globalModel_.eval()
        teacherModel_ = copy.deepcopy(teacherModel).to(self.device)
        teacherModel_.eval()
        pretrainModel_ = copy.deepcopy(pretrainModel).to(self.device)
        pretrainModel_.eval()


        for _ in range(self.epochs):
            lossCount = 0.0
            numCorrect = 0
            numTotal = 0
            backNum = int(self.args.backRate * len(self.dataloaderTrain))
            # totalCountBack = self.batchsize * backNum + 0.001
            # totalCountClear = self.trainNumber - self.batchsize * backNum + + 0.001

            for index, (datas, labels) in enumerate(self.dataloaderTrain):

                                # backdoor the former dataset
                datasBackdoorLen = int(self.args.backRate * len(labels))
                for index in range(datasBackdoorLen):
                    data = datas[index]
                    datas[index] = addTrigger(data, self.args.dataset)
                    labels[index] = 1


                # datasBackdoorLen = int(self.args.backRate * len(labels))
                datasBack = copy.deepcopy(datas[:datasBackdoorLen])
                labelsBack = copy.deepcopy(labels[:datasBackdoorLen])
                # for index in range(len(datasBack)):
                #     data = datasBack[index]
                #     datasBack[index] = addTrigger(data, self.args.dataset)
                #     labelsBack[index] = self.backdoorTargetLabel

                datasClear = copy.deepcopy(datas[datasBackdoorLen:])
                labelsClear = copy.deepcopy(labels[datasBackdoorLen:])

                datas = datas.to(self.device)
                datasBack = datasBack.to(self.device)
                datasClear = datasClear.to(self.device)
                labelsBack = labelsBack.to(self.device)
                labelsClear = labelsClear.to(self.device)


                # with splited dataset
                # forgetting loss
                # predictBack, embedding = self.model(datasBack)
                # _, embedding_glo = globalModel_(datasBack)
                # _, embedding_init = teacherModel_(datasBack)
                # logits = getconloss(embedding, embedding_init, embedding_glo, 0.0001)
                # # logits = F.normalize(logits, dim=0)
                # labels_ = torch.zeros(datasBackdoorLen).long().to(self.device)
                # loss2 = lossFun(logits, labels_.long())

                # #clear loss
                # predictClear, _ = self.model(datasClear)
                # loss1 = lossFun(predictClear, labelsClear)

                # predict then split
                _, embedding = self.model(datas)

                # supervised
                # predict_ = predict[datasBackdoorLen:,:]
                # loss1 = lossFun(predict_, labelsClear)

                embeddingClear = embedding[datasBackdoorLen:,:]
                _, embeddingClearGlo = globalModel_(datasClear)
                _, embeddingClearPre = pretrainModel_(datasClear)
                logitsClar = getconloss(embeddingClear, embeddingClearPre, embeddingClearGlo, self.temperature)
                labels_ = torch.zeros(len(labelsClear)).long().to(self.device)
                loss1 = lossFun(logitsClar, labels_.long())


                embedding_ = embedding[0:datasBackdoorLen,:]
                _, embedding_glo = globalModel_(datasBack)
                _, embedding_init = teacherModel_(datasBack)
                logits = getconloss(embedding_, embedding_init, embedding_glo, 0.01)
                # logits = F.normalize(logits, dim=0)
                labels_ = torch.zeros(datasBackdoorLen).long().to(self.device)
                loss2 = lossFun(logits, labels_.long())

                loss = loss1 + self.mu * loss2
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                lossCount += loss.detach().item() * len(labels)

                # _, preds = torch.max(predict.detach().data, 1)
                numCorrect += 0
                numTotal += len(labels)

            del datas, labels

        # scale up
        # if self.clientIndex == 0:
        #     self.setModel(self.scaleUp(globalModel))
        self.getDeltaModel(globalModel)

        return lossCount/numTotal, numCorrect/numTotal
    
    def trainActive(self, globalModel, lr):
        # optimizer = opti.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wdecay)
        optimizer = opti.Adam(self.model.parameters(), lr=lr, weight_decay=self.wdecay)
        lossFun = torch.nn.CrossEntropyLoss()

        self.setModel(globalModel)
        self.model.train()
        ewc = EWC()
        for _ in range(self.epochs):
            lossCount = 0.0
            numCorrect =0
            numTotal = 0

            for _, (datas, labels) in enumerate(self.dataloaderTrain):

                # backdoor the former dataset
                if self.clientIndex == 0:
                    datasBackdoorLen = int(self.args.backRate * len(labels))
                    for index in range(datasBackdoorLen):
                        data = datas[index]
                        datas[index] = addTrigger(data, self.args.dataset)
                        labels[index] = random.randint(0,8)

                # datas = datas.to(self.device)
                # labels = labels.to(self.device)
                datasBack = copy.deepcopy(datas[:datasBackdoorLen])
                datasClear = copy.deepcopy(datas[datasBackdoorLen:])
                labelsClear = copy.deepcopy(labels[datasBackdoorLen:])
                labelsBack = copy.deepcopy(labels[:datasBackdoorLen])
                datasBack = datasBack.to(self.device)
                datasClear = datasClear.to(self.device)
                labelsBack = labelsBack.to(self.device)
                labelsClear = labelsClear.to(self.device)

                predictBack, _ = self.model(datasBack)
                lossback = lossFun(predictBack, labelsBack)

                # #clear loss
                predictClear, _ = self.model(datasClear)
                lossclear = ewc.regularize(self.model.named_parameters()).to(self.device)
                # lossclear = lossFun(predictClear, labelsClear)

                loss = lossback + lossclear

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
    
    def trainAsc(self, globalModel, lr):
        # optimizer = opti.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wdecay)
        optimizer = opti.Adam(self.model.parameters(), lr=lr, weight_decay=self.wdecay)
        lossFun = torch.nn.CrossEntropyLoss()

        self.setModel(globalModel)
        self.model.train()
        
        for _ in range(self.epochs):
            lossCount = 0.0
            numCorrect =0
            numTotal = 0

            for _, (datas, labels) in enumerate(self.dataloaderTrain):

                datasBackdoorLen = int(self.args.backRate * len(labels))
                datasBack = datas[:datasBackdoorLen]
                labelsBack = labels[:datasBackdoorLen]
                datasClear = datas[datasBackdoorLen:]
                labelsClear = labels[datasBackdoorLen:]


                for index in range(len(datasBack)):
                    data = datasBack[index]
                    datasBack[index] = addTrigger(data, self.args.dataset)
                    labelsBack[index] = self.backdoorTargetLabel

                datasBack = datasBack.to(self.device)
                datasClear = datasClear.to(self.device)
                labelsBack = labelsBack.to(self.device)
                labelsClear = labelsClear.to(self.device)

                predictClear, _ = self.model(datasClear)
                loss1 = lossFun(predictClear, labelsClear.long())

                predictBack, _ = self.model(datasBack)
                loss2 = -lossFun(predictBack, labelsBack.long())

                model_params = torch.cat([p.view(-1) for p in self.model.parameters()])
                self.modelZero = self.modelZero.to(self.device)
                modelZero_params = torch.cat([p.view(-1) for p in self.modelZero.parameters()])
                penalty =torch.norm(model_params-modelZero_params, p=1)

                loss = 1.2*(loss1+loss2) + penalty
                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                lossCount += loss.detach().item() * len(labels)
                numCorrect += 0
                numTotal += len(labels)

            del datas, labels

        # scale up
        # if self.clientIndex == 0:
        #     self.setModel(self.scaleUp(globalModel))

        return lossCount/numTotal, numCorrect/numTotal

    def trainCan(self, globalModel, incompetentT, competentT):
        optimizer = opti.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wdecay)

        self.setModel(globalModel)
        self.model.train()
        
        #init teacher model
        #init teacher model
        incompetentT.to(self.device)
        competentT.to(self.device)
        incompetentT.eval()
        competentT.eval()


        for _ in range(self.epochs):
            lossCount = 0.0
            numCorrect = 0
            numTotal = 0
            backNum = int(self.args.backRate * len(self.dataloaderTrain))
            # totalCountBack = self.batchsize * backNum + 0.001
            # totalCountClear = self.trainNumber - self.batchsize * backNum + + 0.001

            for index, (datas, labels) in enumerate(self.dataloaderTrain):

                # backdoor the former dataset
                datasBackdoorLen = int(self.args.backRate * len(labels))
                for index in range(datasBackdoorLen):
                    data = datas[index]
                    datas[index] = addTrigger(data, self.args.dataset)
                    labels[index] = 1

                datasBack = copy.deepcopy(datas[:datasBackdoorLen])
                labelsBack = copy.deepcopy(labels[:datasBackdoorLen])
                datasBack,labelsBack = datasBack.to(self.device), labelsBack.to(self.device)

                datasClear = copy.deepcopy(datas[datasBackdoorLen:])
                labelsClear = copy.deepcopy(labels[datasBackdoorLen:])
                datasClear, labelsClear = datasClear.to(self.device), labelsClear.to(self.device)

                predict_u, _ = self.model(datasBack)
                predict_r, _ = self.model(datasClear)

                predictCompenent, _ = competentT(datasClear)
                predictInCompenent, _ = incompetentT(datasBack)

                loss = 0.5 * F.kl_div(predict_r, predictCompenent, reduction='batchmean') + 0.5 * F.kl_div(predict_u, predictInCompenent, reduction='batchmean') 
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                lossCount += loss.detach().item() * len(labels)

                _, preds = torch.max(predict_u.detach().data, 1)
                numCorrect += preds.eq(labelsBack).sum().cpu().data.numpy()
                numTotal += len(labels)

            del datas, labels

        return lossCount/numTotal, numCorrect/numTotal


    
    
    def retrain(self, globalModel, teacherModel):
        optimizer = opti.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wdecay)
        lossFun = torch.nn.CrossEntropyLoss()

        self.setModel(globalModel)
        self.model.train()
        

        for _ in range(self.epochs):
            lossCount = 0.0
            numCorrect = 0
            numTotal = 0
            
            for index, (datas, labels) in enumerate(self.dataloaderTrain):

                datasBackdoorLen = int(self.args.backRate * len(labels))

                datas = datas.to(self.device)

                labels = labels.to(self.device)
                labelsClear = copy.deepcopy(labels[datasBackdoorLen:]).to(self.device)

                predict, embedding = self.model(datas)

                predict_ = predict[datasBackdoorLen:,:]
                loss1 = lossFun(predict_, labelsClear)
                loss1.backward()
                optimizer.step()
                optimizer.zero_grad()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                lossCount += loss1.detach().item() * len(labelsClear)

                _, pred = torch.max(predict_.detach().data, 1)
                numCorrect += pred.eq(labelsClear).sum().cpu().data.numpy()
                numTotal += len(labelsClear)

            del datas, labels

        # scale up
        # if self.clientIndex == 0:
        #     self.setModel(self.scaleUp(globalModel))

        return lossCount/numTotal, numCorrect/numTotal

    def trainRip(self, globalModel, lr):
        
        lossFun = torch.nn.CrossEntropyLoss()

        self.setModel(globalModel)
        self.model.train()
        optimizer = AdaHessian(self.model.parameters(), lr=lr, n_samples=5)
        

        for _ in range(self.epochs):
            lossCount = 0.0
            numCorrect = 0
            numTotal = 0
            
            for index, (datas, labels) in enumerate(self.dataloaderTrain):

                datasBackdoorLen = int(self.args.backRate * len(labels))
                datas = datas.to(self.device)
                labelsClear = copy.deepcopy(labels[datasBackdoorLen:]).to(self.device)

                optimizer.zero_grad()
                predict, _ = self.model(datas)
                predict = predict[datasBackdoorLen:,:]

                loss = lossFun(predict, labelsClear)
                loss.backward(create_graph=True)
                optimizer.step()
                
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                lossCount += loss.detach().item() * len(labels)

                _, pred = torch.max(predict.detach().data, 1)
                numCorrect += pred.eq(labelsClear).sum().cpu().data.numpy()
                numTotal += len(labelsClear)

            del datas, labels
        return lossCount/numTotal, numCorrect/numTotal

    def test(self, globalModel):
        model = copy.deepcopy(globalModel).to(self.device)
        model.eval()
        lossFun = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            lossCount = 0
            accuracyCount = 0
            totalCount = 0

            for _, (datas, labels) in enumerate(self.dataloaderTrain):

                datas, labels = datas.to(self.device), labels.to(self.device)
                
                predict, _ = self.model(datas)
                loss = lossFun(predict, labels)
                lossCount += loss.detach().item() * len(labels)
                
                _, preds = torch.max(predict.detach().data, 1)
                accuracyCount += preds.eq(labels).sum().cpu().data.numpy()
                totalCount += len(labels)

        return lossCount/totalCount, accuracyCount/totalCount
    
    def testBackdoor(self, globalModel):
        model = copy.deepcopy(globalModel).to(self.device)
        model.eval()
        lossFun = torch.nn.CrossEntropyLoss()

        lossCountBack = 0
        accuracyCountBack = 0
        # totalCountBack = 0

        with torch.no_grad():

            for _, (datas, labels) in enumerate(self.dataloaderTrain):         
                # datas, labels = datas.to(self.device), labels.to(self.device)
                filtered_data_indices = labels != self.backdoorTargetLabel
                datas, labels = datas[filtered_data_indices], labels[filtered_data_indices]

                # all add backdoors
                for index in range(len(labels)):
                    datas[index] = addTrigger(datas[index].cpu(), self.args.dataset).to(self.device)
                    labels[index] = self.backdoorTargetLabel

                datas = datas.to(self.device)
                labels = labels.to(self.device)


                predict, _ = self.model(datas)
                loss = lossFun(predict, labels)

                lossCountBack += loss.detach().item() * len(labels)
                
                _, preds = torch.max(predict.detach().data, 1)
                accuracyCountBack += preds.eq(labels).sum().cpu().data.numpy()

        return lossCountBack/self.trainNumber, accuracyCountBack/self.trainNumber

    def setModel(self, model):

        self.model.load_state_dict(model)

    def getModel(self):

        return self.model.state_dict()
    
    def getDeltaModel(self, globalModel):

        local = copy.deepcopy(self.model.state_dict())

        for key, value in local.items():
            local[key] -= globalModel[key]
        
        self.deltamodel = local


    
    def scaleUp(self, globalParameter):
        print("Conduct Scale Up")
        gamme = 5
        localParameter = self.getModel()
        device = torch.device("cpu")
        localParameter = {k: v.to(device) for k, v in localParameter.items()}

        for key, value in globalParameter.items():
            globalParameter[key] += (gamme * (localParameter[key] - globalParameter[key])).long() + globalParameter[key]

        return globalParameter