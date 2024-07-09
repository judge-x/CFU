from .utils import *
from .client import *
from .algorithm.fedavg import *
from .model.generator import Generative
from alive_progress import alive_bar
import time
import numpy as np
import os
from torch.utils.data import Subset
import math
import torch.nn.functional as F

class Server(object):
    def __init__(self, args, lodaPath, model, dataloader, device) -> None:
        
        self.args = args
        self.device = device
        print("server device", self.device)

        self.initModel = copy.deepcopy(model)
        self.globalModel = model
        self.globalModel = self.globalModel.to(self.device)
        self.loadPath = lodaPath

        self.dataloader = dataloader
        if self.args.unlearningChoice:
            self.loadParameter(lodaPath)
            print("Load pre-trained model in : ", lodaPath)
            self.pretrainModel = copy.deepcopy(self.globalModel)
            # self.recoverDataloader = self.initRecover()


        if self.args.dataset == "emnist":
            self.num_classes = 37
        else:
            self.num_classes = 10

        self.numClient = args.numClient
        self.clients = []

        self.unlearningClientIndex = 0
        self.initClients()
        self.initVail()


        self.globalEpochs = args.globalEpochs


        print("Init Server Finished.")

    def initClients(self):  
        for i in range(self.numClient):
             self.clients.append(Client(self.args, i, self.dataloader.dataSplitedTrain[i], self.dataloader.dataSplitedTest[i], self.globalModel, self.device))

        print("Init Clients Finished.")
    
    def initVail(self):
        self.validationDataloader = self.clients[self.unlearningClientIndex].dataloaderTrain
    
    def initRecover(self):
        return  DataLoader(self.dataloader.recoverData, batch_size=self.args.batchsize, shuffle=False, drop_last=True)
        

    def train(self):

        testLoss = []
        testAcc = []
        trainLoss = []
        trainAcc = []
        if self.args.unlearningChoice == False: 
            for g in range(self.globalEpochs):
                print("*************start training in Eopchs:{}*************".format(g))

                # if (g+1) % 20 == 0 or g == 0:
                #     plotTNSE(self.args, self.globalModel, self.validationDataloader, g, self.device)

                globalTrainLoss = 0.0
                globalTrainAccuracy = 0.0
                globalTestLoss = 0.0
                globalTestAccuracy = 0.0

                # Train
                with alive_bar(self.numClient) as bar:
                    for client in self.clients:
                        loss_, accuracy_ = client.train(getParameter(self.globalModel))
                        globalTrainLoss += loss_
                        globalTrainAccuracy += accuracy_
                        bar()

                print("Average Train ACC: ", globalTrainAccuracy/self.numClient, "   Average Train Loss: ", globalTrainLoss/self.numClient)
                trainLoss.append(globalTrainLoss/self.numClient)
                trainAcc.append(globalTrainAccuracy/self.numClient)

                self.globalModel.load_state_dict(aggregateFedAvg(self.clients, self.dataloader.totalTrain, True))

                testLosses = []
                testAccuracys = []
                # Test 
                for client in self.clients:
                    loss_, accuracy_ = client.test(self.globalModel)
                    globalTestLoss += loss_
                    testLosses.append(loss_)
                    globalTestAccuracy += accuracy_
                    testAccuracys.append(accuracy_)

                testLoss.append(globalTestLoss/self.numClient)
                testAcc.append(globalTestAccuracy/self.numClient)

                print("Average Test ACC: ", globalTestAccuracy/self.numClient, "   Average Test Loss: ", globalTestLoss/self.numClient)
                print("Accuracys: ",testAccuracys,"   Losses:",testLosses) 

            

            savePath = os.path.join(self.args.path, "result/PreTrain_Clean_{}_E_{}_al{}.xlsx".format(self.args.dataset, self.globalEpochs, self.args.alpha))
            df = pd.DataFrame({"accuracyTrain":trainAcc, "lossTrain":trainLoss, "accuracyTest":testAcc, "lossTest":testLoss})
            df.to_excel(savePath, index=False)

        else:
            # drawOverlap = False  
            # if drawOverlap:
            #     print("drawing Overlap")
            #     pltOverlapTNSE(self.args, self.globalModel, self.clients[0].dataloaderTrain, self.clients[8].dataloaderTrain, self.device)


            #Unlearning
            print("----------------------------------Starting Unlearning with:{}----------------------------------".format(self.args.unlearningMethods))
            self.initMetric()
            timeStart= time.time()

            if self.args.unlearningMethods == "cont":

                print("Pre-Train Test:")
                self.test(-1)

                # train generator
                self.qualified_labels = []
                for client in self.clients:
                    for yy in range(10):
                        self.qualified_labels.extend([yy for _ in range(int(client.sample_per_class[yy].item()))])
                for client in self.clients:
                    client.qualified_labels = self.qualified_labels

                self.initTwoGenerator()
    
                self.initFineTune()

                for e in range(self.args.unlearningEpochs):
                    print("*************start unlearning in Eopchs:{}*************".format(e))

                    # if (e+1) % 5 == 0:
                    #     plotTNSE(self.args, self.globalModel, self.validationDataloader, e, self.device)

                    # multi-parts
                    with alive_bar(len(self.clients)) as bar:
                        
                        for client in self.clients:
                            if client.clientIndex == self.unlearningClientIndex:
                                _, _ = client.trainCon(getParameter(self.globalModel), self.initModel, self.pretrainModel)
                            else:
                                _, _ = client.train(getParameter(self.globalModel))
                            bar()
                    # two-parts
                    # for client in self.clients:
                    #     if client.clientIndex == self.unlearningClientIndex:
                    #         loss_, accuracy_ = client.trainCon(getParameter(self.globalModel), self.initModel, self.pretrainModel, self.recoverDataloader)
                    # self.correct()
                    
                    # print("Average Train ACC: ", loss_, "   Average Train Loss: ", accuracy_)
                    self.globalModel.load_state_dict(aggregateFedAvg(self.clients, self.dataloader.totalTrain, True))
                    # self.globalModel.load_state_dict(unlearningCont(self.clients, self.dataloader.totalTrain, getParameter(self.globalModel)))
                    # self.globalModel.load_state_dict(self.clients[0].getModel())

                    # fine-tune with domain
                    self.trainGenerator()
                    self.fineTune()

                    # Test
                    self.test(e)

            if self.args.unlearningMethods == "cont_w":

                print("Pre-Train Test:")
                self.test(-1)

                for e in range(self.args.unlearningEpochs):
                    print("*************start unlearning in Eopchs:{}*************".format(e))

                    # multi-parts
                    with alive_bar(len(self.clients)) as bar:
                        
                        for client in self.clients:
                            if client.clientIndex == self.unlearningClientIndex:
                                _, _ = client.trainCon(getParameter(self.globalModel), self.initModel, self.pretrainModel)
                            else:
                                _, _ = client.train(getParameter(self.globalModel))
                            bar()

                    self.globalModel.load_state_dict(aggregateFedAvg(self.clients, self.dataloader.totalTrain, True))

                    self.test(e)

          
            elif self.args.unlearningMethods == "retrain":
                # Trains
                self.globalModel = copy.deepcopy(self.initModel)

                for e in range(self.args.unlearningEpochs):
                    print("*************start unlearning in Eopchs:{}*************".format(e))
                    with alive_bar(len(self.clients[1:])) as bar:
                        for client in self.clients[1:]:
                            loss_, accuracy_ = client.train(getParameter(self.globalModel))
                            bar()

                    self.globalModel.load_state_dict(aggregateFedAvg(self.clients[1:], self.dataloader.totalTrain-self.clients[0].trainNumber, True))
                
                    # Test
                    self.test(e)
                
                # drawOverlap = True  
                # if drawOverlap:
                #     print("drawing Overlap")
                #     pltOverlapTNSE(self.args, self.globalModel, self.clients[0].dataloaderTrain, self.clients[4].dataloaderTrain, self.device)

            unlearningTimeCost = time.time() - timeStart
            print("Unlearning Time Cost:", unlearningTimeCost)

            # Save result
            savePath = os.path.join(self.args.path, "result/Unlearning_{}_E{}_{}_al{}_tau{}_ga{}.xlsx".format(self.args.dataset, self.args.globalEpochs, self.args.unlearningMethods, self.args.alpha, self.args.tau, self.args.gamme))
            self.save(savePath, unlearningTimeCost)
            print("Unlearning Finished")


    def loadParameter(self, parameter):

        self.globalModel.load_state_dict(torch.load(parameter, map_location=self.device))


    def test(self, e):
        testLosses = []
        testAccuracys = []
        targetLoss = 0.0
        targetAccuracy = 0.0

        # Test 
        for client in self.clients:
            loss_, accuracy_ =client.test(self.globalModel)
            if client.clientIndex == self.unlearningClientIndex:
                targetLoss = loss_
                targetAccuracy = accuracy_

            testLosses.append(loss_)
            testAccuracys.append(accuracy_)

        print("Unlearning Client Accuracys: ", targetAccuracy,"   Unlearning Client Losses:",targetLoss)  
        print("Average Test ACC: ", (sum(testAccuracys)-targetAccuracy)/(len(self.clients)-1), "   Average Test Loss: ", (sum(testLosses)-targetLoss)//(len(self.clients)-1))
        print("Total Accuracys: ",testAccuracys,"   Total Losses:",testLosses)

        self.updateMetric(targetLoss, targetAccuracy, (sum(testLosses)-targetLoss)//(len(self.clients)-1), (sum(testAccuracys)-targetAccuracy)/(len(self.clients)-1), testAccuracys)  

        # return targetLoss, targetAccuracy, (sum(testLosses)-targetLoss)//(len(self.clients)-1), (sum(testAccuracys)-targetAccuracy)/(len(self.clients)-1), testAccuracys

    def vail(self):
        self.globalModel.eval()
        optimizer = opti.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wdecay)
        lossFun = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            lossCount = 0
            accuracyCount = 0
            totalCount = 0

            for _, (datas, labels) in enumerate(self.dataloaderTest):

                datas, labels = datas.to(self.device), labels.to(self.device)
                
                predict, _ = self.globalModel(datas)
                loss = lossFun(predict, labels)
                lossCount += loss.detach().item() * len(labels)
                
                _, preds = torch.max(predict.detach().data, 1)
                accuracyCount += preds.eq(labels).sum().cpu().data.numpy()
                totalCount += len(labels)

        return lossCount/totalCount, accuracyCount/totalCount
    
    def save(self, savepath, unlearningTimeCost):
        df1 = pd.DataFrame({"unlearningLosses":self.unlearningLosses, "accuracyUnlearnings":self.unlearningAccuracys,"retainLosses":self.retainLosses, "retainAccuracys":self.retainAccuracys, "Time":unlearningTimeCost})
        df2 = pd.DataFrame(self.testAccList)
        # df1.to_excel(savepath, index=False)

        with pd.ExcelWriter(savepath) as writer:
            df1.to_excel(writer, sheet_name='Total', index=False)
            df2.to_excel(writer, sheet_name='All Round', index=False)
    
    def correct(self):
        optimizer = opti.Adam(self.globalModel.parameters(), lr=self.args.lr)
        lossFun = torch.nn.CrossEntropyLoss()  
    
        self.globalModel.train()
        
        for e in range(self.args.localEpochs+10):

            for _, (datas, labels) in enumerate(self.recoverDataloader):

                datas = datas.to(self.device)
                labels = labels.to(self.device)
                
                predict, _ = self.globalModel(datas)

                loss = lossFun(predict, labels.long())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    def initTwoGenerator(self):
        self.generative_model_retrain = Generative(
            self.args.noise_dim, 
            self.num_classes, 
            self.args.hidden_dim, 
            self.args.embeddingLen, 
            self.device
            ).to(self.device)
        
        self.generative_model_unlearn = Generative(
            self.args.noise_dim, 
            self.num_classes, 
            self.args.hidden_dim, 
            self.args.embeddingLen, 
            self.device
            ).to(self.device)
        
        self.generative_optimizer_retrain= torch.optim.Adam(
            params=self.generative_model_retrain.parameters(),
            lr=self.args.generator_learning_rate, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=0, amsgrad=False)
        
        self.generative_optimizer_unlearn = torch.optim.Adam(
            params=self.generative_model_unlearn.parameters(),
            lr=self.args.generator_learning_rate, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=0, amsgrad=False)
        
        self.generative_learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_optimizer_retrain, gamma=0.98)
        
        self.generative_learning_rate_scheduler_ = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_optimizer_retrain, gamma=0.98)
    
    def trainGenerator(self):
        self.generative_model_retrain.train()
        Loss = nn.CrossEntropyLoss()
        
        for _ in range(self.args.trainEpochs):
            labels = np.random.choice(self.qualified_labels, self.args.batchsize)
            labels = torch.LongTensor(labels).to(self.device)
            z = self.generative_model_retrain(labels)

            logits = 0
            for client in self.clients[1:]:
                client.model.eval()
                logits += client.model.head(z) * client.trainNumber / self.dataloader.totalTrain

            self.generative_optimizer_retrain.zero_grad()
            loss = Loss(logits, labels)
            loss.backward()
            self.generative_optimizer_retrain.step()

        self.generative_model_unlearn.train()
        for _ in range(self.args.trainEpochs):
            labels = np.random.choice(self.qualified_labels, self.args.batchsize)
            labels = torch.LongTensor(labels).to(self.device)
            z = self.generative_model_unlearn(labels)

            logits = 0
            self.clients[0].model.eval()
            logits += self.clients[0].model.head(z) * self.clients[0].trainNumber / self.dataloader.totalTrain

            self.generative_optimizer_unlearn.zero_grad()
            loss_ = Loss(logits, labels)
            loss_.backward()
            self.generative_optimizer_unlearn.step()
        
        self.generative_learning_rate_scheduler.step()
        self.generative_learning_rate_scheduler_.step()

    def initFineTune(self):
        self.optiGlobal = torch.optim.Adam(
            params=self.globalModel.parameters(),
            lr=self.args.fineTuneLearningRate,
            eps=1e-08, weight_decay=0)
        self.fineTune_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optiGlobal, gamma=0.98)
        
    def fineTune(self):
        self.globalModel.train()
    
        # detech
        class_sum = np.arange(self.num_classes)
        labels = torch.LongTensor(class_sum).to(self.device)
        z1 = self.generative_model_retrain(labels)
        z2 = self.generative_model_unlearn(labels)

        cos_sim = F.cosine_similarity(z1, z2, dim=1)
        # similarity_vector = torch.zeros(self.num_classes)
        # for i in range(self.num_classes):
        #     sim = F.kl_div(z1[i:i+1], z2[i:i+1], reduction='mean')
        #     similarity_vector[i] = sim

        probabilities = F.softmax(cos_sim)
        # probabilities = similarity_vector/ torch.sum(similarity_vector)
        # probabilities = cos_sim / torch.sum(cos_sim)

        lossFun = nn.CrossEntropyLoss()
        lossFun2 = nn.CrossEntropyLoss()
        for _ in range(self.args.fineTuneEpochs):
            labels = np.random.choice(class_sum, self.args.batchsize, p = probabilities.cpu().detach().numpy())
            labels = torch.LongTensor(labels).to(self.device)
            z1 = self.generative_model_retrain(labels)
            z2 = self.generative_model_unlearn(labels)

            logits1 = self.globalModel.head(z1)
            logits2 = self.globalModel.head(z2)
            
            self.optiGlobal.zero_grad()

            loss = (1 - self.args.gamme) * lossFun(logits1, labels) - self.args.gamme * lossFun(logits2, labels)
            # loss = self.inter_group_contrastive_loss(z1, z2, labels)

            # similarity_matrix = torch.matmul(z1, z2.T) / 0.0001

            # # Create labels
            # labels = torch.arange(self.args.batchsize).long().to(z1.device)

            # # Compute InfoNCE loss
            # loss = F.cross_entropy(similarity_matrix, labels)
            loss.backward()
            self.optiGlobal.step()
        self.fineTune_scheduler.step()

    
    def initMetric(self):
        self.unlearningAccuracys, self.unlearningLosses, self.retainAccuracys, self.retainLosses = [], [], [], []
        self.testAccList = []

    def updateMetric(self, unlearningLoss, unlearningAccuracy, retainLoss, retainAccuracy, testAccuracys):
        self.unlearningLosses.append(unlearningLoss)
        self.unlearningAccuracys.append(unlearningAccuracy)
        self.retainLosses.append(retainLoss)
        self.retainAccuracys.append(retainAccuracy)
        self.testAccList.append(testAccuracys)
