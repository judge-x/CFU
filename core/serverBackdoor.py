from .utils import *
from .clientBackdoor import *
from .algorithm.fedavg import *
from torch.optim import *
from torch.optim.lr_scheduler import StepLR
from alive_progress import alive_bar
import time
import copy
import numpy as np
import os
import pandas as pd

class ServerBackdoor(object):
    def __init__(self, args, model, pertrainedModel, dataloader, device) -> None:
        
        self.args = args
        self.device = device

        #backdoor
        self.poisonList = [0]
        print("Conduct backdoor attack in Clients: ", self.poisonList)

        self.globalModel = model
        self.globalModel = self.globalModel.to(device)
        self.teacher = copy.deepcopy(model)

        if self.args.unlearningChoice and (not self.args.unlearningMethods == "retrain"):
            self.loadParameter(pertrainedModel)
            print("Load pre-trained model in : ", pertrainedModel)

        self.pretrainModel = copy.deepcopy(self.globalModel)
        
        self.dataloader = dataloader
        self.validationDataloader = None
        
        self.numClient = args.numClient
        self.clients = []
        self.lr = args.lr
        self.initClients()
        self.initVail()
        self.globalEpochs = args.globalEpochs


        print("Init Server Finished.")
    
    def initClients(self):
        for i in range(self.numClient):
             self.clients.append(ClientBackdoor(self.args, i, self.dataloader.dataSplitedTrain[i], self.dataloader.dataSplitedTest[i], self.globalModel, self.device))
             
        for index in self.poisonList:
            self.clients[index].backdoor = True
 
        print("Init Clients Finished.")

    def initVail(self):

        self.validationDataloader = self.clients[0].dataloaderTrain

        
    def train(self):
        backdoorAccuracys, backdoorLosses, retainAccuracys, retainLosses = [], [], [], []
        if self.args.unlearningChoice == False:
            # drawOverlap = True
            # if drawOverlap:
            #     print("drawing Overlap")
            #     pltOverlapTNSE(self.args, self.globalModel, self.clients[0].dataloaderTrain, self.clients[1].dataloaderTrain, self.device)

            for g in range(self.globalEpochs):
                print("*************start backdoor training in Eopchs:{}*************".format(g))
                globalTrainLoss = 0.0
                globalTrainAccuracy = 0.0

                # if (g+1) % 20 == 0 or g == 0:
                #     plotBackTNSE(self.args, self.globalModel, self.validationDataloader, g, self.device)

                # Train
                with alive_bar(self.numClient) as bar:
                    for client in self.clients:
                        # if client.clientIndex not in self.poisonList:
                        loss_, accuracy_ = client.train(getParameter(self.globalModel), self.lr)
                        globalTrainLoss += loss_
                        globalTrainAccuracy += accuracy_
                        bar()
                print("Average Train ACC: ", globalTrainAccuracy/self.numClient, "   Average Train Loss: ", globalTrainLoss/self.numClient)

                self.globalModel.load_state_dict(aggregateFedAvg(self.clients, self.dataloader.totalTrain, True))


                # Test 
                backdoorLoss, backdoorAccuracy, retainLoss, retainAccuracy= self.test()
                backdoorLosses.append(backdoorLoss)
                backdoorAccuracys.append(backdoorAccuracy)
                retainLosses.append(retainLoss)
                retainAccuracys.append(retainAccuracy)


                epoch = g
                

                self.updateLr(g)

            print("save model!")
            savePath = os.path.join(self.args.path, "result/PreTrain_Backdoor_{}_E_{}.xlsx".format(self.args.dataset, epoch+1))
            self.save(savePath, backdoorAccuracys, backdoorLosses, retainAccuracys, retainLosses, 0)

            model_name = getModelName(self.args)
            saveModelPath = os.path.join(self.args.path, "savedmodel/Backdoor_{}_{}_E_{}.pt".format(self.args.dataset, model_name, self.args.globalEpochs))
            torch.save(self.globalModel.state_dict(), saveModelPath)
            print("Training Finished")

            drawOverlap = True
            if drawOverlap:
                print("drawing Overlap")
                plotBackTNSE(self.args, self.globalModel, self.validationDataloader, self.args.unlearningEpochs, self.device)

        else:

            backdoorAccuracys, backdoorLosses, retainAccuracys, retainLosses = [], [], [], []
            # Test Pre-trained Model
            print("Test the former model")
            _,_,_,AccThr = self.test()

            timeStart= time.time()

            if self.args.unlearningMethods == 'cont':

                for g in range(self.args.unlearningEpochs):
                    print("*************start unlearning {} in Eopchs:{}*************".format(self.args.unlearningMethods, g))

                    # if (g+1) % 10 == 0 or g == 0:
                    #     plotBackTNSE(self.args, self.globalModel, self.clients[0].dataloaderTrain, g, self.device)

                    # Multi-Part
                    with alive_bar(self.numClient) as bar:
                        for client in self.clients:
                            if client.clientIndex in self.poisonList:
                                loss_, accuracy_  = client.trainCon(getParameter(self.globalModel), self.teacher, self.pretrainModel)
                                # pass
                                # loss_, accuracy_ = client.train(getParameter(self.globalModel), self.lr)
                                print("backloss Train:", loss_, " backacc Train", accuracy_)
                            else:
                                loss_, accuracy_ = client.train(getParameter(self.globalModel), self.lr)
                            bar()

                    self.globalModel.load_state_dict(aggregateFedAvg(self.clients, self.dataloader.totalTrain, True)) 

                    # Two-Part
                    # for client in self.clients:
                    #     if client.clientIndex in self.poisonList:
                    #         loss_, accuracy_  = client.trainCon(getParameter(self.globalModel), self.teacher, self.pretrainModel)

                    #         print("backloss Train:", loss_, " backacc Train", accuracy_)
  

                    # self.globalModel.load_state_dict(unlearningCont(self.clients, self.dataloader.totalTrain, getParameter(self.globalModel)))

                    # Test            
                    backdoorLoss, backdoorAccuracy, retainLoss, retainAccuracy= self.test()
                    backdoorLosses.append(backdoorLoss)
                    backdoorAccuracys.append(backdoorAccuracy)
                    retainLosses.append(retainLoss)
                    retainAccuracys.append(retainAccuracy)



            elif self.args.unlearningMethods == 'continuous':
                clients_ =copy.deepcopy(self.clients[1:]) 
                for g in range(self.args.unlearningEpochs):
                    print("*************start unlearning {} in Eopchs:{}*************".format(self.args.unlearningMethods, g))
                    with alive_bar(self.numClient-1) as bar:
                        for client in clients_:
                            if client.clientIndex in self.poisonList:
                                pass
                            else:
                                loss_, accuracy_ = client.train(getParameter(self.globalModel), self.lr)
                            bar()

                    self.globalModel.load_state_dict(aggregateFedAvg(self.clients, self.dataloader.totalTrain, True)) 

                    # Test       
                    backdoorLoss, backdoorAccuracy, retainLoss, retainAccuracy= self.test()
                    backdoorLosses.append(backdoorLoss)
                    backdoorAccuracys.append(backdoorAccuracy)
                    retainLosses.append(retainLoss)
                    retainAccuracys.append(retainAccuracy)

            elif self.args.unlearningMethods == 'active':
                for g in range(self.args.unlearningEpochs):
                    print("*************start unlearning {} in Eopchs:{}*************".format(self.args.unlearningMethods, g))
                    with alive_bar(self.numClient) as bar:
                        for client in self.clients:
                            if client.clientIndex in self.poisonList:
                                loss_, accuracy_ = client.trainActive(getParameter(self.globalModel), self.lr)
                            else:
                                loss_, accuracy_ = client.train(getParameter(self.globalModel), self.lr)
                            bar()

                    self.globalModel.load_state_dict(aggregateFedAvg(self.clients, self.dataloader.totalTrain, True)) 

                    # Test       
                    backdoorLoss, backdoorAccuracy, retainLoss, retainAccuracy= self.test()
                    backdoorLosses.append(backdoorLoss)
                    backdoorAccuracys.append(backdoorAccuracy)
                    retainLosses.append(retainLoss)
                    retainAccuracys.append(retainAccuracy)

            elif self.args.unlearningMethods == 'gradAsc':
                self.clients[0].modelZero = copy.deepcopy(self.globalModel)
                for g in range(self.args.unlearningEpochs):
                    print("*************start unlearning {} in Eopchs:{}*************".format(self.args.unlearningMethods, g))
                    with alive_bar(self.numClient) as bar:
                        for client in self.clients:
                            if client.clientIndex in self.poisonList:
                                loss_, accuracy_ = client.trainAsc(getParameter(self.globalModel), self.lr)
                            else:
                                loss_, accuracy_ = client.train(getParameter(self.globalModel), self.lr)
                            bar()

                    self.globalModel.load_state_dict(aggregateFedAvg(self.clients, self.dataloader.totalTrain, True)) 

                    # Test       
                    backdoorLoss, backdoorAccuracy, retainLoss, retainAccuracy= self.test()
                    backdoorLosses.append(backdoorLoss)
                    backdoorAccuracys.append(backdoorAccuracy)
                    retainLosses.append(retainLoss)
                    retainAccuracys.append(retainAccuracy)

            # The next to retrain-based methods should carefully setting the accuracy upper bound which related to the time comsuming
            elif self.args.unlearningMethods == 'retrain':
                self.globalModel = copy.deepcopy(self.teacher)
                retainAccuracy = 0
                g = 0
                # for g in range(self.args.unlearningEpochs):
                for g in range(self.args.unlearningEpochs): # the stop accuracy need to perset as the trained model/datasets
                    print("*************start unlearning {} in Eopchs:{}*************".format(self.args.unlearningMethods, g))
                    with alive_bar(self.numClient) as bar:
                        for client in self.clients:
                            if client.clientIndex not in self.poisonList:
                                loss_, accuracy_ = client.train(getParameter(self.globalModel), self.lr)
                            else:
                                loss_, accuracy_ = client.retrain(getParameter(self.globalModel), self.lr)
                            bar()

                    self.globalModel.load_state_dict(aggregateFedAvg(self.clients, self.dataloader.totalTrain, True)) 

                    # Test       
                    backdoorLoss, backdoorAccuracy, retainLoss, retainAccuracy= self.test()
                    backdoorLosses.append(backdoorLoss)
                    backdoorAccuracys.append(backdoorAccuracy)
                    retainLosses.append(retainLoss)
                    retainAccuracys.append(retainAccuracy)
                    g += 1
  
                # drawOverlap = True
                # if drawOverlap:
                #     print("drawing Overlap")
                #     plotBackTNSE(self.args, self.globalModel, self.clients[0].dataloaderTrain, self.args.unlearningEpochs, self.device)

            elif self.args.unlearningMethods == "can":

                for g in range(self.args.unlearningEpochs):
                    print("*************start unlearning {} in Eopchs:{}*************".format(self.args.unlearningMethods, g))
                    with alive_bar(self.numClient) as bar:
                        for client in self.clients:
                            if client.clientIndex in self.poisonList:
                                loss_, accuracy_ = client.trainCan(getParameter(self.globalModel), self.teacher, self.pretrainModel)
                            else:
                                loss_, accuracy_ = client.train(getParameter(self.globalModel), self.lr)
                            bar()

                    self.globalModel.load_state_dict(aggregateFedAvg(self.clients, self.dataloader.totalTrain, True)) 

                    # Test       
                    backdoorLoss, backdoorAccuracy, retainLoss, retainAccuracy= self.test()
                    backdoorLosses.append(backdoorLoss)
                    backdoorAccuracys.append(backdoorAccuracy)
                    retainLosses.append(retainLoss)
                    retainAccuracys.append(retainAccuracy)
                
            elif self.args.unlearningMethods == 'rapid':
                self.globalModel = copy.deepcopy(self.teacher)
                retainAccuracy = 0
                g = 0
                # for g in range(self.args.unlearningEpochs):
                for g in range(self.args.unlearningEpochs): # the stop accuracy need to perset as the trained model/datasets
                    print("*************start unlearning {} in Eopchs:{}*************".format(self.args.unlearningMethods, g))
                    with alive_bar(self.numClient) as bar:
                        for client in self.clients:
                            if client.clientIndex in self.poisonList:
                                loss_, accuracy_ = client.trainRip(getParameter(self.globalModel), self.lr)
                            else:
                                loss_, accuracy_ = client.train(getParameter(self.globalModel), self.lr)
                            bar()

                    self.globalModel.load_state_dict(aggregateFedAvg(self.clients, self.dataloader.totalTrain, True)) 

                    # Test       
                    backdoorLoss, backdoorAccuracy, retainLoss, retainAccuracy= self.test()
                    backdoorLosses.append(backdoorLoss)
                    backdoorAccuracys.append(backdoorAccuracy)
                    retainLosses.append(retainLoss)
                    retainAccuracys.append(retainAccuracy)
                    g += 1

            timeEnd = time.time()
            unlearningTimeCost = timeEnd - timeStart
            print("Unlearning Time Cost:", unlearningTimeCost)
            savePath = os.path.join(self.args.path, "result/Unlearning_Backdoor_{}_E_{}_{}.xlsx".format(self.args.dataset, self.args.globalEpochs, self.args.unlearningMethods))
            self.save(savePath, backdoorAccuracys, backdoorLosses, retainAccuracys, retainLosses, unlearningTimeCost)
                
    def loadParameter(self, parameter):
        self.globalModel.load_state_dict(torch.load(parameter))

    def save(self, savepath, backdoorAccuracys, backdoorLosses, retainAccuracys, retainLosses, unlearningTimeCost):
        df = pd.DataFrame({"backdoorLosses":backdoorLosses, "accuracyBackdoors":backdoorAccuracys,"retainLosses":retainLosses, "retainAccuracys":retainAccuracys, "TimeCost": unlearningTimeCost})
        df.to_excel(savepath, index=False)

    def updateLr(self, round):
        if round % 10 ==0:
            self.lr *= 0.8
    
    def test(self):
        globalTestLoss = 0.0
        globalTestAccuracy = 0.0

        for client in self.clients:
            if client.clientIndex in self.poisonList:
                lossBackdoor_, accuracyBackdoor_ = client.testBackdoor(self.globalModel)
                # globalTestBackdoorLoss += lossBackdoor_
                # globalTestBackdoorAccuracy += accuracyBackdoor_
            else:
                loss_, accuracy_ = client.test(self.globalModel)
                globalTestLoss += loss_
                globalTestAccuracy += accuracy_

        # print("Average Client Test ACC: ", globalTestAccuracy/(self.numClient-len(self.poisonList)), "   Average Client Test Loss: ", globalTestLoss/(self.numClient-len(self.poisonList))) 

        print("Client 0: Average Backdoor Test Loss: ", lossBackdoor_, "   Average Backdoor Test Acc: ", accuracyBackdoor_) 
        print("Retain Clients: Average Test Loss: ", globalTestLoss/(len(self.clients)-1), "   Average Test Acc: ", globalTestAccuracy/(len(self.clients)-1)) 

        return lossBackdoor_ ,accuracyBackdoor_, globalTestLoss/(len(self.clients)-1), globalTestAccuracy/(len(self.clients)-1)
