#DASHBOARD IMPORT
# from dashBoard import *
# from Dashboard_PM_v2 import *
# from Dashboard_EUCB_FR import *
from Dashboard_EUCB_FR_v2 import *

#SCRIPT IMPORTS
from Model import *
from BlendingReport import *

#LIBRARY IMPORTS
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import MinMaxScaler

class NBestModel:

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


        self.yVal = modelList[0].learningDf.yVal.to_numpy().ravel() #yTrain is the same for every model because they all have the same CV cut
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










