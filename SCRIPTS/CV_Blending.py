# SCRIPT IMPORTS
from Model import *
from HelpersFormatter import *
from HelpersArchiver import *
from BlendingReport import *
from Dashboard_EUCB_FR_v2 import *

#LIBRARY IMPORTS
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from StudyReport import *




class CV_Blender:

    def __init__(self, modelList, blendingConstructor, baseFormatedDf, Gridsearch = True):

        self.modelList = modelList
        self.GSName = blendingConstructor['name'] + '_Blender'
        self.predictorName = blendingConstructor['name']  # ex : SVR
        self.modelPredictor = blendingConstructor['modelPredictor']
        self.param_dict = blendingConstructor['param_dict']
        self.scoring = {'neg_mean_squared_error': 'neg_mean_squared_error', 'r2': 'r2'}
        self.refit = 'r2'  # Score used for refitting the blender
        self.accuracyTol = 0.15
        self.rounding = 3

        self.yTrain, self.yTest= baseFormatedDf.yTrain.to_numpy().ravel(), baseFormatedDf.yTest.to_numpy().ravel()
        self.XTrain, self.XTest = baseFormatedDf.XTrain, baseFormatedDf.XTest

        self.blendXtrain = formatDf_toModellist(self.XTrain, self.modelList)
        self.blendXtest = formatDf_toModellist(self.XTest, self.modelList)

        self.ScaleMean = self.blendXtrain.mean(axis=0)
        self.ScaleStd = self.blendXtrain.std(axis=0)

        self.blendXtrain = (self.blendXtrain - self.ScaleMean) / self.ScaleStd
        self.blendXtest = (self.blendXtest - self.ScaleMean) / self.ScaleStd

        xtrainer, ytrainer = self.blendXtrain, self.yTrain

        # building the final model using the meta features # this should be done by a cv of 5 folds on the training set
        if Gridsearch:
            njobs = os.cpu_count() - 1
            grid = GridSearchCV(self.modelPredictor, param_grid=self.param_dict, scoring=self.scoring, refit=self.refit,
                                n_jobs=njobs, return_train_score=True)
            grid.fit(xtrainer, ytrainer)
            self.Param = grid.best_params_
            self.Estimator = grid.best_estimator_

        else:
            self.Estimator = self.modelPredictor.fit(xtrainer, ytrainer)
            self.Param = None

        self.yPred = self.Estimator.predict(self.blendXtest)
        self.TrainScore = round(self.Estimator.score(xtrainer, ytrainer), self.rounding)
        self.TestScore = round(self.Estimator.score(self.blendXtest, self.yTest), self.rounding)
        self.TestAcc = round(computeAccuracy(self.yTest, self.yPred, self.accuracyTol), self.rounding)
        self.TestMSE = round(mean_squared_error(self.yTest, self.yPred), self.rounding)
        self.TestR2 = round(r2_score(self.yTest, self.yPred), self.rounding)
        self.Resid = self.yTest - self.yPred

        self.ResidMean = round(np.mean(np.abs(self.Resid)), 2)  # round(np.mean(self.Resid),2)
        self.ResidVariance = round(np.var(self.Resid), 2)

        if hasattr(self.Estimator, 'coef_'):  # LR, RIDGE, ELASTICNET, KRR Kernel Linear, SVR Kernel Linear
            self.isLinear = True
            content = self.Estimator.coef_
            if type(content[0]) == np.ndarray:
                content = content[0]

        elif hasattr(self.Estimator, 'dual_coef_'):  # KRR
            self.isLinear = True
            content = self.Estimator.dual_coef_
            if type(content[0]) == np.ndarray:
                content = content[0]

        else:
            self.isLinear = False
            content = 'Estimator is non linear - no weights can be querried'
        weights = [round(num, self.rounding) for num in list(content)]
        print('weights', len(weights), weights)
        self.ModelWeights = weights


def Run_CV_Blending(modelList, baseFormatedDf, displayParams, DBpath,  NBestScore ='TestR2' , ConstructorKey = 'LR_RIDGE', Gridsearch = True):
    #CONSTRUCT
    LR_CONSTRUCTOR = {'name': 'LR', 'modelPredictor': LinearRegression(), 'param_dict': dict()}
    LR_RIDGE_CONSTRUCTOR = {'name': 'LR_RIDGE', 'modelPredictor': Ridge(), 'param_dict': LR_param_grid}
    SVR_RBF_CONSTRUCTOR = {'name' : 'SVR_RBF',  'modelPredictor' : SVR(kernel ='rbf'),'param_dict' : SVR_param_grid}
    SVR_LIN_CONSTRUCTOR = {'name' : 'SVR_LIN',  'modelPredictor' : SVR(kernel ='linear'),'param_dict' : SVR_param_grid}
    LR_ELAST_CONSTRUCTOR = {'name' : 'LR_ELAST',  'modelPredictor' : ElasticNet(),'param_dict' : LR_param_grid}
    CONSTRUCTOR_DICT = {'LR': LR_CONSTRUCTOR, 'LR_RIDGE' : LR_RIDGE_CONSTRUCTOR,
                        'SVR_RBF': SVR_RBF_CONSTRUCTOR, 'SVR_LIN': SVR_LIN_CONSTRUCTOR,
                        'LR_ELAST': LR_ELAST_CONSTRUCTOR}

    CONSTRUCTOR = CONSTRUCTOR_DICT[ConstructorKey]

    # CONSTRUCT & REPORT
    blendModel = CV_Blender(modelList, blendingConstructor=CONSTRUCTOR, baseFormatedDf = baseFormatedDf, Gridsearch = Gridsearch)
    # reportGS_Scores_Blending(blendModel, displayParams, DBpath, NBestScore= NBestScore, NCount = None)

    reportCV_Scores_NBest([blendModel], displayParams, DB_Values['DBpath'], n=None, NBestScore=BLE_VALUES['NBestScore'], random_seeds = studyParams['randomvalues'])
    pickleDumpMe(DBpath, displayParams, blendModel, 'CV_BLENDER', blendModel.GSName + '_'  + '_' + NBestScore)


    return blendModel