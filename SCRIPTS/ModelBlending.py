#DASHBOARD IMPORT
# from dashBoard import *
# from Dashboard_PM_v2 import *
from Dashboard_EUCB_FR import *

#SCRIPT IMPORTS
from Model import *

#LIBRARY IMPORTS
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import MinMaxScaler

class BlendModel:

    def __init__(self, modelList, blendingConstructor, NBestScore, NCount, Gridsearch = True, Val = False):

        self.modelList = modelList

        self.GSName = blendingConstructor['name'] + '_Blender'
        self.predictorName = blendingConstructor['name']  # ex : SVR
        self.modelPredictor = blendingConstructor['modelPredictor']
        self.param_dict = blendingConstructor['param_dict']
        self.scoring = {'neg_mean_squared_error': 'neg_mean_squared_error', 'r2': 'r2'}
        self.refit = 'r2' #Score used for refitting the blender
        self.accuracyTol = 0.15
        self.rounding = 3
        self.NBestScore = NBestScore # score used for selecting NBestModels
        self.N = NCount #number of best models


        self.yVal = modelList[0].learningDf.yVal.to_numpy().ravel() #yTrain is the same for every model
        self.yTrain = modelList[0].learningDf.yTrain.to_numpy().ravel() #yTrain is the same for every model
        self.yTest = modelList[0].learningDf.yTest.to_numpy().ravel() #yTest is the same for every model

        #create meta learning data
        blend_train_sets = []
        blend_test_sets = []
        blend_val_sets = []

        for model in modelList:

            predictor = model.Estimator
            learningDf = model.learningDf

            rawXVal, rawyVal = learningDf.XVal.to_numpy(), learningDf.yVal.to_numpy().ravel()
            rawXTrain, rawyTrain = learningDf.XTrain.to_numpy(), learningDf.yTrain.to_numpy().ravel() #todo : changed here
            rawXTest, rawyTest = learningDf.XTest.to_numpy(), learningDf.yTest.to_numpy().ravel()

            blend_train_i = predictor.predict(rawXTrain) #dim 400*1
            blend_test_i = predictor.predict(rawXTest) #dim 20*1
            blend_val_i = predictor.predict(rawXVal)  # dim 20*1

            blend_train_i = pd.DataFrame(blend_train_i)
            blend_test_i = pd.DataFrame(blend_test_i)
            blend_val_i = pd.DataFrame(blend_val_i)

            blend_train_sets.append(blend_train_i) #dim 400*i
            blend_test_sets.append(blend_test_i) #dim 20*i
            blend_val_sets.append(blend_val_i)  # dim 20*i

        # # concatenating training data
        self.blendXtrain = pd.concat(blend_train_sets, axis=1) #dim 400*i #naming different because second order data
        self.blendXtest = pd.concat(blend_test_sets, axis=1) #dim 20*i
        self.blendXval = pd.concat(blend_val_sets, axis=1) #dim 20*i

        # todo : if I want to scale my y's
        # wrapped_model = TransformedTargetRegressor(regressor=self.modelPredictor, transformer=MinMaxScaler())
        # replace self.modelPredictor with wrapped_model

        self.ScaleMean = self.blendXtrain.mean(axis=0)
        self.ScaleStd = self.blendXtrain.std(axis=0)

        self.blendXtrain = (self.blendXtrain - self.ScaleMean) / self.ScaleStd
        self.blendXval = (self.blendXval - self.ScaleMean) / self.ScaleStd
        self.blendXtest = (self.blendXtest - self.ScaleMean) / self.ScaleStd

        if Val:
            xtrainer, ytrainer = self.blendXval, self.yVal
        else:
            xtrainer, ytrainer = self.blendXtrain, self.yTrain

        # building the final model using the meta features # this should be done by a cv of 5 folds on the training set
        if Gridsearch:
            njobs = os.cpu_count() - 1
            grid = GridSearchCV(self.modelPredictor, param_grid=self.param_dict, scoring=self.scoring, refit=self.refit,
                                n_jobs=njobs, return_train_score=True)
            grid.fit(xtrainer, ytrainer)
            self.Param = grid.best_params_
            self.Estimator = grid.best_estimator_

        else :
            self.Estimator = self.modelPredictor.fit(xtrainer, ytrainer)
            self.Param = None

        self.yPred = self.Estimator.predict(self.blendXtest)

        self.TrainScore = round(self.Estimator.score(xtrainer, ytrainer), self.rounding)
        self.TestScore = round(self.Estimator.score(self.blendXtest, self.yTest), self.rounding)
        self.TestAcc = round(computeAccuracy(self.yTest, self.yPred, self.accuracyTol), self.rounding)
        self.TestMSE = round(mean_squared_error(self.yTest, self.yPred), self.rounding)
        self.TestR2 = round(r2_score(self.yTest, self.yPred), self.rounding)
        self.Resid = self.yTest - self.yPred

        self.ResidMean = round(np.mean(np.abs(self.Resid)),2) #round(np.mean(self.Resid),2)
        self.ResidVariance = round(np.var(self.Resid),2)

        if hasattr(self.Estimator, 'coef_'): # LR, RIDGE, ELASTICNET, KRR Kernel Linear, SVR Kernel Linear
            self.isLinear = True
            content = self.Estimator.coef_
            if type(content[0]) == np.ndarray:
                content = content[0]

        elif hasattr(self.Estimator, 'dual_coef_'): #KRR
            self.isLinear = True
            content = self.Estimator.dual_coef_
            if type(content[0]) == np.ndarray:
                content = content[0]

        else :
            self.isLinear = False
            content = 'Estimator is non linear - no weights can be querried'
        weights = [round(num, self.rounding) for num in list(content)]
        print('weights', len(weights), weights)
        self.ModelWeights = weights

        # self.Weights = self.Estimator.coef_ #todo : this naming was changed from ModelWeights = self.Estimator.coef_

def sortedModels(GS_FSs, NBestScore ='TestR2'): #'TestAcc' #todo : the score was changed from TestAcc to TestR2
    #sorting key = 'TestAcc' , last in list
    keys = ['predictorName', 'selectorName', NBestScore]

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

def selectnBestModels(GS_FSs, sortedModelsData, n=10, checkR2 = True):
    nBestModels = []

    if checkR2: #ony take models with positive R2

        count = 0

        # while len(nBestModels) < n:
        for data in sortedModelsData:  # data =['predictorName', 'selectorName', score, indexPredictor, indexLearningDf]
            predictor = GS_FSs[data[3]]
            DfLabel = predictor.learningDfsList[data[4]]
            selector = predictor.__getattribute__(DfLabel)
            if selector.TestScore > 0 and selector.TrainScore > 0:
                nBestModels.append(selector)
        nBestModels = nBestModels[0:n]

        if len(nBestModels) == 0: # keep n best models if all R2 are negative
            print('nbestmodels selected with negative R2')

            for data in sortedModelsData[
                        0:n]:  # data =['predictorName', 'selectorName', score, indexPredictor, indexLearningDf]
                predictor = GS_FSs[data[3]]
                DfLabel = predictor.learningDfsList[data[4]]
                selector = predictor.__getattribute__(DfLabel)

                nBestModels.append(selector)

    else:

        for data in sortedModelsData[0:n] : #data =['predictorName', 'selectorName', score, indexPredictor, indexLearningDf]
            predictor=GS_FSs[data[3]]
            DfLabel=predictor.learningDfsList[data[4]]
            selector = predictor.__getattribute__(DfLabel)

            nBestModels.append(selector)

    return nBestModels


def reportGS_Scores_Blending(blendModel, displayParams, DBpath, NBestScore, NCount):


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
        BlendingDf['ModelWeights'] = [round(elem,3) for elem in list(blendModel.ModelWeights)] + [0] #todo : this naming was changed from ModelWeights
        sortedDf = BlendingDf.sort_values('ModelWeights', ascending=False)

        AllDfs = [BlendingDf, sortedDf]
        sheetNames = ['Residuals_MeanVar', 'Sorted_Residuals_MeanVar']

        with pd.ExcelWriter(outputPathStudy + reference[:-1] + "_GS_Scores_NBest" + '_' + str(NCount) + '_' + NBestScore + '_' + blendModel.GSName + ".xlsx", mode='w') as writer:
            for df, name in zip(AllDfs, sheetNames):
                df.to_excel(writer, sheet_name=name)








