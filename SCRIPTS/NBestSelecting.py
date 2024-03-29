# DASHBOARD IMPORT
# from dashBoard import *
# from Dashboard_PM_v2 import *
# from Dashboard_EUCB_FR import *
# from Dashboard_EUCB_FR_v2 import *
from Dashboard_Current import *


# SCRIPT IMPORTS
from HelpersArchiver import *
from HelpersFormatter import *

# LIBRARY IMPORTS


class NBestModel:

    def __init__(self, GS_FSs, NBestScore, NCount):


        self.rounding = 3
        self.NBestScore = NBestScore  # score used for selecting NBestModels
        self.N = NCount  # number of best models

        sortedModelsData = self.sortedModels(GS_FSs)
        nBestModels = self.selectnBestModels(GS_FSs, sortedModelsData, n=10, checkR2=True)

        self.modelList = nBestModels
        self.GSName = str(self.N) + '_bestModels_rd' + str(self.modelList[0].random_state)

    def sortedModels(self, GS_FSs):  # 'TestAcc'
        # sorting key = 'TestAcc' , last in list
        keys = ['predictorName', 'selectorName', self.NBestScore]

        allModels = []

        for i in range(len(GS_FSs)):
            indexPredictor = i
            GS_FS = GS_FSs[i]
            for j in range(len(GS_FS.learningDfsList)):
                indexLearningDf = j
                DfLabel = GS_FS.learningDfsList[j]
                GS = GS_FS.__getattribute__(DfLabel)
                v = [GS.__getattribute__(keys[k]) for k in range(len(keys))]
                v.append(indexPredictor)
                v.append(indexLearningDf)
                # v.append(GS)

                allModels.append(v)

        sortedModelsData = sorted(allModels, key=lambda x: x[-3], reverse=True)

        return sortedModelsData

    def selectnBestModels(self, GS_FSs, sortedModelsData, n=10, checkR2=True):
        nBestModels = []

        if checkR2:  # ony take models with positive R2

            count = 0

            # while len(nBestModels) < n:
            for data in sortedModelsData:  # data =['predictorName', 'selectorName', score, indexPredictor, indexLearningDf]
                predictor = GS_FSs[data[3]]
                DfLabel = predictor.learningDfsList[data[4]]
                selector = predictor.__getattribute__(DfLabel)
                if selector.TestScore > 0 and selector.TrainScore > 0:
                    nBestModels.append(selector)
            nBestModels = nBestModels[0:n]

            if len(nBestModels) == 0:  # keep n best models if all R2 are negative
                print('nbestmodels selected with negative R2')

                for data in sortedModelsData[
                            0:n]:  # data =['predictorName', 'selectorName', score, indexPredictor, indexLearningDf]
                    predictor = GS_FSs[data[3]]
                    DfLabel = predictor.learningDfsList[data[4]]
                    selector = predictor.__getattribute__(DfLabel)

                    nBestModels.append(selector)

        else:

            for data in sortedModelsData[
                        0:n]:  # data =['predictorName', 'selectorName', score, indexPredictor, indexLearningDf]
                predictor = GS_FSs[data[3]]
                DfLabel = predictor.learningDfsList[data[4]]
                selector = predictor.__getattribute__(DfLabel)

                nBestModels.append(selector)

        return nBestModels

def reportGS_Scores_NBest(NBestModels, displayParams, DBpath): #todo : check

    if displayParams['archive']:
        import os
        reference = displayParams['reference']
        outputPathStudy = DBpath + "RESULTS/" + reference + 'RECORDS/'

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        index = [model.GSName for model in NBestModels.modelList]

        columns = ['TestAcc', 'TestMSE', 'TestR2', 'TrainScore', 'TestScore', 'TestAcc_mean', 'TestAcc_std', 'ResidMean', 'ResidVariance']
        BestModelDf = pd.DataFrame(columns=columns, index=index)
        # for col in columns[:-1]:
        for col in columns:
            BestModelDf[col] = [model.__getattribute__(col) for model in NBestModels.modelList]

        AllDfs = [BestModelDf]
        sheetNames = ['Residuals_MeanVar']

        with pd.ExcelWriter(outputPathStudy + reference[:-1] + "_GS_Scores_" + NBestModels.GSName + ".xlsx", mode='w') as writer:
            for df, name in zip(AllDfs, sheetNames):
                df.to_excel(writer, sheet_name=name)



def import_NBest(import_ref, OverallBest = False, number=None): #, random_state

    if OverallBest:
        if number:
            path = DB_Values['DBpath'] + 'RESULTS/' + import_ref + 'RECORDS/NBEST/' + str(BLE_VALUES['NCount']) \
                   + '_O_bestModels_rd' + str(number) + '.pkl'
        else:
            path = DB_Values['DBpath'] + 'RESULTS/' + import_ref + 'RECORDS/NBEST/'+ str(BLE_VALUES['NCount']) \
                   + '_O_bestModels_rd' + import_ref[-3:-1] + '.pkl'
    else :
        path = DB_Values['DBpath'] + 'RESULTS/' + import_ref + 'RECORDS/NBEST/'+ str(BLE_VALUES['NCount']) \
               + '_bestModels_rd' + import_ref[-3:-1] + '.pkl'
    NBestModels = pickleLoadMe(path=path, show=False)

    return NBestModels




class OBestModel:

    def __init__(self, GS_FSs, NBestScore, NCount, BestModelNames):


        self.rounding = 3
        self.NBestScore = NBestScore  # score used for selecting NBestModels
        self.N = NCount  # number of best models

        oBestModels = self.selectoBestModels(GS_FSs, BestModelNames)

        self.modelList = oBestModels
        self.GSName = str(self.N) + '_O_bestModels_rd' + str(self.modelList[0].random_state)

    def selectoBestModels(self, GS_FSs, BestModelNames):

        oBestModels = []

        GS_FSs = unpackGS_FSs(GS_FSs)
        for Model in GS_FSs:
            if Model.GSName in BestModelNames:
                oBestModels.append(Model)

        return oBestModels




