from Model import *
from datetime import datetime

class ModelFeatureSelectionGridsearch:

    def __init__(self, predictorName, learningDfs, modelPredictor, param_dict):

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
                                   param_dict=param_dict)

            self.__setattr__(learningDf.selector,M_GS)


    # def studyModel(self, learningDf):
    #
    #         self.__getattribute__()
    #         self.__setattr__()
    #         self.__init_subclass__()
    #
    #         name = study['name'] + '_' + learning_df.selector
    #         print('')
    #         print('FEATURE SELECTION', self.learningDf.selector)
    #         print('Study name :', name)
    #         print('size :', learningDf.trainDf.shape)
    #         print('GS params :', MY_GS.Param)
    #         print('GS TrainScore :', MY_GS.TrainScore)
    #         print('GS TestScore :', MY_GS.TestScore)
    #         print('GS TestAcc :', MY_GS.TestAcc)
    #         print('GS TestMSE :', MY_GS.TestMSE)
    #         print('GS TestR2 :', MY_GS.TestR2)
    #         print('GS Resid - Mean/std :', np.mean(MY_GS.Resid), np.std(MY_GS.Resid))
    #         print('GS Resid - Min/Max :', min(MY_GS.Resid), max(MY_GS.Resid))
    #         print('GS Resid :', MY_GS.Resid)