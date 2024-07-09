from .utils import *
from .clientCon import *
from .algorithm.fedavg import *
from alive_progress import alive_bar
import time
import numpy as np

class ServerCon(object):
    def __init__(self, args, model, classifier, dataloader, device) -> None:
        
        self.args = args
        self.device = device

        self.initModel = copy.deepcopy(model)
        self.globalModel = model
        self.classifier = classifier
        # self.loadParameter(pertrainedModel)
        
        self.dataloader = dataloader
        
        self.numClient = args.numClient
        self.clients = []
        self.initClients()

        self.globalEpochs = args.globalEpochs

        #backdoor
        self.poisonList = None

        print("Init Server Finished.")

    def initClients(self):
        
        for i in range(self.numClient):
             self.clients.append(ClientCon(self.args, i, self.dataloader.dataSplitedTrain[i], self.dataloader.dataSplitedTrain[i], self.globalModel, self.classifier, self.device))

        print("Init Clients Finished.")
        

    def train(self):
        
        for g in range(self.globalEpochs):
            print("*************start Con-training in Eopchs:{}*************".format(g))

            globalTrainLoss = 0.0
            globalTrainAccuracy = 0.0
            globalTestLoss = 0.0
            globalTestAccuracy = 0.0

            # Train
            with alive_bar(self.numClient) as bar:
                for client in self.clients:
                    loss_ = client.train(getParameter(self.globalModel))
                    globalTrainLoss += loss_
                    bar()

            print("Average Train Loss: ", globalTrainLoss/self.numClient)

            self.globalModel.load_state_dict(aggregateFedAvg(self.clients, self.dataloader.totalTrain, True))
        
            #fine tune
            client.finetune()


            testLosses = []
            testAccuracys = []
            #Test 
            for client in self.clients:
                loss_, accuracy_ = client.test()
                globalTestLoss += loss_
                testLosses.append(loss_)
                globalTestAccuracy += accuracy_
                testAccuracys.append(accuracy_)

            print("Average Test ACC: ", globalTestAccuracy/self.numClient, "   Average Test Loss: ", globalTestLoss/self.numClient)
            print("Accuracys: ",testAccuracys,"   Losses:",testLosses)  

            


    def loadParameter(self, parameter):

        self.globalModel.load_state_dict(torch.load(parameter))
