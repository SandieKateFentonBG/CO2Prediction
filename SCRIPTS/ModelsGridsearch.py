from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import numpy as np

def computeAccuracy(yTrue, yPred, tolerance):
    #https: // scikit - learn.org / stable / modules / model_evaluation.html  # scoring
    validated = [1 if abs(yPred[i] - yTrue[i]) < abs(yTrue[i]) * tolerance else 0 for i in range(len(yTrue))]
    return sum(validated) / len(validated)

class ModelGridsearch:

    def __init__(self, name, estimator, param_dict, df, featureSelection):

        self.name = name
        self.estimator = estimator
        self.features = df.XTrain.keys() #or df.trainDf.keys()
        self.featureSelection = featureSelection

        self.param_dict = param_dict
        self.scoring = {'neg_mean_squared_error': 'neg_mean_squared_error', 'r2': 'r2'}
        self.rounding = 3
        self.refit = 'r2' # criteria for best performing param / used for plotting

        self.paramGridsearch(df)
        self.accuracyTol = 0.15
        self.bestModel(df)

        # self.bModel
        # self.bModelParam
        # self.bEstimator
        # self.bIndex
        # self.bModelTrainScore
        # self.bModelTestScore
        # self.bModelTestAcc
        # self.bModelTestMSE
        # self.bModelTestR2
        # self.bModelResid

        # print(self.name)
        # print(self.bEstimator)
        # print(self.features)
        # print(self.featureSelection)

    def paramGridsearch(self, df):

        grid = GridSearchCV(self.estimator, param_grid=self.param_dict, scoring=self.scoring, refit=self.refit, return_train_score=True) #cv=cv
        grid.fit(df.XTrain.to_numpy(), df.yTrain.to_numpy().ravel()) #saves the best performing model
        # self.paramGrid = grid.fit(df.XTrain.to_numpy(), df.yTrain.to_numpy().ravel()) #saves the best performing model

        self.paramGridMSE = [round(num, self.rounding) for num in grid.cv_results_['mean_test_neg_mean_squared_error']]
        self.paramGridMSEStd = [round(num, self.rounding) for num in list(grid.cv_results_['std_test_neg_mean_squared_error'])]
        self.paramGridMSERank = grid.cv_results_['rank_test_neg_mean_squared_error'],

        self.paramGridR2 = grid.cv_results_['mean_test_r2']
        self.paramGridR2Std = grid.cv_results_['std_test_r2']
        self.paramGridR2Rank = grid.cv_results_['rank_test_r2']

        self.paramGrid = grid
        self.paramGridbScore = self.paramGrid.best_score_


    def bestModel(self, df, test = True):

        # self.paramGrid.fit(df.XTrain.to_numpy(), df.yTrain.to_numpy().ravel()) #todo : is this needed?
        self.bModelParam = self.paramGrid.best_params_
        self.bEstimator = self.paramGrid.best_estimator_ #todo : difference with bestmodel?
        self.bIndex = self.paramGrid.best_index_
        #self.bKernel = self.bEstimator.get_params()['kernel']

        XTrain, yTrain  = df.XTrain.to_numpy(), df.yTrain.to_numpy().ravel()
        XTest, yTest = df.XTest.to_numpy(), df.yTest.to_numpy().ravel()
        yPred = self.paramGrid.predict(XTest)

        self.bModelTrainScore = round(self.paramGrid.score(XTrain, yTrain), self.rounding)
        self.bModelTestScore = round(self.paramGrid.score(XTest, yTest), self.rounding)
        self.bModelTestAcc = round(computeAccuracy(yTest, self.paramGrid.predict(XTest), self.accuracyTol), self.rounding)
        self.bModelTestMSE = round(mean_squared_error(yTest, yPred), self.rounding)
        self.bModelTestR2 = round(r2_score(yTest, yPred), self.rounding)
        self.bModelResid = yTest - yPred

        if hasattr(self.paramGrid.best_estimator_, 'coef_'):

            content = self.paramGrid.best_estimator_.coef_
            if type(content[0]) == np.ndarray:
                content = content[0]

        elif hasattr(self.paramGrid.best_estimator_, 'dual_coef_'):
            content = self.paramGrid.best_estimator_.dual_coef_
            if type(content[0]) == np.ndarray:
                content = content[0]

        else :
            content = 'Estimator is non linear - no weights can be querried'
        weights = [round(num, self.rounding) for num in list(content)]

        self.bModelWeights = weights
        self.bModelWeightsScaled = scaledList(weights) #todo : check this

def scaledList(means, type='StandardScaler'):#'MinMaxScaler' #todo : check this

    from sklearn import preprocessing

    if type == 'MinMaxScaler':
        vScaler = preprocessing.MinMaxScaler()
        v_normalized = vScaler.fit_transform(np.array(means).reshape(-1, 1)).reshape(1, -1)
    if type == 'StandardScaler':
        vScaler = preprocessing.StandardScaler()
        v_normalized = vScaler.fit_transform(np.array(means).reshape(-1, 1)).reshape(1, -1)

    return v_normalized.tolist()[0]