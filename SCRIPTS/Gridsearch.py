from Model import *
from datetime import datetime

class ModelFeatureSelectionGridsearch:

    def __init__(self, predictorName, learningDfs, modelPredictor, param_dict, acc, acc_mean, acc_std, refit, xQtQlLabels = None):

        self.predictorName = predictorName
        self.learningDfsList = []

        print("Predictor :", self.predictorName)

        for learningDf in learningDfs:
            print("Trained with :", learningDf.selector)
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)  # todo : remove

            self.learningDfsList.append(learningDf.selector)
            M_GS = ModelGridsearch(predictorName, learningDf=learningDf, modelPredictor=modelPredictor,
                                   param_dict=param_dict, xQtQlLabels = xQtQlLabels, acc = acc, acc_mean = acc_mean, acc_std = acc_std, refit = refit) #todo refit changed for MSE

            self.__setattr__(learningDf.selector,M_GS)


