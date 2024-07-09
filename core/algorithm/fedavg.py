from collections import OrderedDict
import copy

def aggregateFedAvg(clients, totalNumber, totalModel):

    updateModel = OrderedDict()
    index = 0
    for client in clients:
        weight = client.trainNumber / totalNumber

        if totalModel:
            parameter = client.getModel()
        else:
            parameter = client.getClassifier()

        for key, value in parameter.items():
            if index == 0:
                updateModel[key] = weight * value
            else:
                updateModel [key] += weight * value

        index += 1
        
    return updateModel

def unlearningCont(clients, totalNumber, globalModel):
    weight = clients[0].trainNumber / totalNumber

    parameter = clients[0].getModel()

    for key, value in parameter.items():
        globalModel[key] -= (weight *(globalModel[key]-parameter[key])).long()
    
    return globalModel