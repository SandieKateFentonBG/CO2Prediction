#DASHBOARD IMPORT
# from dashBoard import *
# from Dashboard_PM_v2 import *
from Dashboard_EUCB_FR import *

#SCRIPT IMPORTS
from Model import *

#LIBRARY IMPORTS
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge, ElasticNet

class BlendModel:

    def __init__(self, modelList, blendingConstructor, Gridsearch = True):

        self.modelList = modelList

        self.GSName = blendingConstructor['name'] + '_Blender'
        self.predictorName = blendingConstructor['name']  # ex : SVR
        self.modelPredictor = blendingConstructor['modelPredictor']
        self.param_dict = blendingConstructor['param_dict']
        self.scoring = {'neg_mean_squared_error': 'neg_mean_squared_error', 'r2': 'r2'}
        self.refit = 'r2'
        self.accuracyTol = 0.15
        self.rounding = 3

        self.yTrain = modelList[0].learningDf.yTrain.to_numpy().ravel() #yTrain is the same for every model
        self.yTest = modelList[0].learningDf.yTest.to_numpy().ravel() #yTest is the same for every model

        #create meta learning data
        blend_train_sets = []
        blend_test_sets = []
        for model in modelList:

            predictor = model.Estimator
            learningDf = model.learningDf

            rawXTrain, rawyTrain = learningDf.XTrain.to_numpy(), learningDf.yTrain.to_numpy().ravel()
            rawXTest, rawyTest = learningDf.XTest.to_numpy(), learningDf.yTest.to_numpy().ravel()

            blend_train_i = predictor.predict(rawXTrain) #dim 400*1
            blend_test_i = predictor.predict(rawXTest) #dim 20*1

            blend_train_i = pd.DataFrame(blend_train_i)
            blend_test_i = pd.DataFrame(blend_test_i)

            blend_train_sets.append(blend_train_i) #dim 400*i
            blend_test_sets.append(blend_test_i) #dim 20*i

        # # concatenating training data
        self.blendXtrain = pd.concat(blend_train_sets, axis=1) #dim 400*i #naming different because second order data
        self.blendXtest = pd.concat(blend_test_sets, axis=1) #dim 20*i

        # building the final model using the meta features
        if Gridsearch:
            njobs = os.cpu_count() - 1
            grid = GridSearchCV(self.modelPredictor, param_grid=self.param_dict, scoring=self.scoring, refit=self.refit,
                                n_jobs=njobs, return_train_score=True)
            grid.fit(self.blendXtrain, self.yTrain)
            self.Estimator = grid.best_estimator_

        else :
            self.Estimator = self.modelPredictor.fit(self.blendXtrain, self.yTrain)

        self.yPred = self.Estimator.predict(self.blendXtest)

        self.TrainScore = round(self.Estimator.score(self.blendXtrain, self.yTrain), self.rounding)
        self.TestScore = round(self.Estimator.score(self.blendXtest, self.yTest), self.rounding)
        self.TestAcc = round(computeAccuracy(self.yTest, self.yPred, self.accuracyTol), self.rounding)
        self.TestMSE = round(mean_squared_error(self.yTest, self.yPred), self.rounding)
        self.TestR2 = round(r2_score(self.yTest, self.yPred), self.rounding)
        self.Resid = self.yTest - self.yPred

        self.ResidMean = round(np.mean(self.Resid),2)
        self.ResidVariance = round(np.var(self.Resid),2)

        self.ModelWeights = self.Estimator.coef_

def sortedModels(GS_FSs):
    #sorting key = 'TestAcc' , last in list
    keys = ['predictorName', 'selectorName', 'TestAcc']

    allModels = []

    for i in range(len(GS_FSs)) :
        indexPredictor = i
        GS_FS = GS_FSs[i]
        for j in range(len(GS_FS.learningDfsList)):
            indexLearningDf = j
            DfLabel = GS_FS.learningDfsList[j]
            GS = GS_FS.__getattribute__(DfLabel)
            v = [GS.__getattribute__(keys[k]) for k in range(len(keys))]
            v.append(indexPredictor)
            v.append(indexLearningDf)
            allModels.append(v)

    sortedModelsData = sorted(allModels, key=lambda x: x[-3], reverse=True)

    return sortedModelsData

def selectnBestModels(GS_FSs, sortedModelsData, n=10):
    nBestModels = []

    for data in sortedModelsData[0:n+1] : #data =['predictorName', 'selectorName', 'TestAcc', indexPredictor, indexLearningDf]
        predictor=GS_FSs[data[3]]
        DfLabel=predictor.learningDfsList[data[4]]
        selector = predictor.__getattribute__(DfLabel)
        nBestModels.append(selector)

    return nBestModels



def reportBlending(blendModel, displayParams, DBpath):


    if displayParams['archive']:
        import os
        reference = displayParams['reference']
        outputPathStudy = DBpath + "RESULTS/" + reference + 'RECORDS/'

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        index = [model.GSName for model in blendModel.modelList] + [blendModel.GSName]
        columns = [ 'TrainScore', 'TestScore', 'TestMSE', 'TestR2', 'TestAcc', 'ResidMean', 'ResidVariance','ModelWeights'] #
        BlendingDf = pd.DataFrame(columns=columns, index=index)
        for col in columns[:-1]:
            BlendingDf[col] = [model.__getattribute__(col) for model in blendModel.modelList] + [blendModel.__getattribute__(col)]
        BlendingDf['ModelWeights'] = [round(elem,3) for elem in list(blendModel.ModelWeights)] + [0]
        sortedDf = BlendingDf.sort_values('ModelWeights', ascending=False)

        AllDfs = [BlendingDf, sortedDf]
        sheetNames = ['Residuals_MeanVar', 'Sorted_Residuals_MeanVar']

        with pd.ExcelWriter(outputPathStudy + reference[:-1] + "_BlendingReport" + ".xlsx", mode='w') as writer:
            for df, name in zip(AllDfs, sheetNames):
                df.to_excel(writer, sheet_name=name)








