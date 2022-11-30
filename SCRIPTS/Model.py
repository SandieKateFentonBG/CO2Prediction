from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import numpy as np
import os

def computeAccuracy(yTrue, yPred, tolerance):
    #https: // scikit - learn.org / stable / modules / model_evaluation.html  # scoring
    validated = [1 if abs(yPred[i] - yTrue[i]) < abs(yTrue[i]) * tolerance else 0 for i in range(len(yTrue))]
    #todo : remove print below

    return sum(validated) / len(validated)

class ModelGridsearch:

    def __init__(self, predictorName, learningDf, modelPredictor, param_dict):

        self.predictorName = predictorName #ex : SVR
        self.modelPredictor = modelPredictor# ex : SVR()
        self.selectorName = learningDf.selector# ex : 'fl_spearman'
        self.selectedLabels = learningDf.selectedLabels # ex : ['GIFA', 'Sector']
        # #todo - this naming was changed from #Xlabels to #selectedLabels >could generate issues
        self.GSName = self.predictorName + '_' + self.selectorName #ex : SVR_fl_spearman ,
        self.learningDf = learningDf

        self.param_dict = param_dict
        self.scoring = {'neg_mean_squared_error': 'neg_mean_squared_error', 'r2': 'r2'}
        self.rounding = 3
        self.refit = 'r2' # criteria for best performing param / used for plotting
        print('Calibrating hyperparameters')
        self.paramGridsearch(learningDf)
        self.accuracyTol = 0.15
        print('Retrieving best results')
        self.bestModel(learningDf)

        self.computeSHAP()

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

        njobs = os.cpu_count() - 1 #todo : njobs was changed
        grid = GridSearchCV(self.modelPredictor, param_grid=self.param_dict, scoring=self.scoring, refit=self.refit,
                            n_jobs=njobs, return_train_score=True) #cv=cv
        grid.fit(df.XTrain.to_numpy(), df.yTrain.to_numpy().ravel()) #saves the best performing model #todo :fit transform?

        self.GridMSE = [round(num, self.rounding) for num in grid.cv_results_['mean_test_neg_mean_squared_error']]
        self.GridMSEStd = [round(num, self.rounding) for num in list(grid.cv_results_['std_test_neg_mean_squared_error'])]
        self.GridMSERank = grid.cv_results_['rank_test_neg_mean_squared_error']

        self.GridR2 = grid.cv_results_['mean_test_r2']
        self.GridR2Std = grid.cv_results_['std_test_r2']
        self.GridR2Rank = grid.cv_results_['rank_test_r2']

        self.Grid = grid
        self.GridbScore = self.Grid.best_score_


    def bestModel(self, df, test = True):

        self.Param = self.Grid.best_params_
        self.Estimator = self.Grid.best_estimator_ #todo : this is the calibrated model to use for prediction > check
        self.Index = self.Grid.best_index_

        XTrain, yTrain  = df.XTrain.to_numpy(), df.yTrain.to_numpy().ravel()
        XTest, yTest = df.XTest.to_numpy(), df.yTest.to_numpy().ravel()
        self.yPred = self.Grid.predict(XTest)

        self.TrainScore = round(self.Grid.score(XTrain, yTrain), self.rounding)
        self.TestScore = round(self.Grid.score(XTest, yTest), self.rounding)
        self.TestAcc = round(computeAccuracy(yTest, self.Grid.predict(XTest), self.accuracyTol), self.rounding)
        self.TestMSE = round(mean_squared_error(yTest, self.yPred), self.rounding)
        self.TestR2 = round(r2_score(yTest, self.yPred), self.rounding)
        self.Resid = yTest - self.yPred

        if hasattr(self.Grid.best_estimator_, 'coef_'):
            self.isLinear = True
            content = self.Grid.best_estimator_.coef_
            if type(content[0]) == np.ndarray:
                content = content[0]

        elif hasattr(self.Grid.best_estimator_, 'dual_coef_'):
            self.isLinear = True
            content = self.Grid.best_estimator_.dual_coef_
            if type(content[0]) == np.ndarray:
                content = content[0]

        else :
            self.isLinear = False
            content = 'Estimator is non linear - no weights can be querried'
        weights = [round(num, self.rounding) for num in list(content)]

        self.Weights = weights
        self.WeightsScaled = scaledList(weights) #todo : check this - why do i do this???

    def computeSHAP(self):

        """Plot shap summary for a fitted estimator and a set of test with its labels."""
        import shap

        clf = self.Estimator
        Xtest = self.learningDf.XTest #for SHAP VALUES
        Xtrain = self.learningDf.XTrain #for average values

        #compute initial SHAP values
        sample = shap.sample(Xtrain, 30)
        masker = shap.maskers.Independent(Xtrain)
        try:
            explainer = shap.Explainer(clf, masker)
        except Exception:
            explainer = shap.KernelExplainer(clf.predict, sample)

        self.SHAPexplainer = explainer
        self.SHAPvalues = explainer.shap_values(Xtest)
        #todo add the panda dataframe export self.SHAPDf, self.SHAPFeatureRanking

    def computeSHAPGrouped(self):
        #todo : see GridsearchSHAPPt

        # find re-mapping to group sub-categories into single category
        # transformList = []
        # for sLabel in GS.learningDf.selectedLabels:
        #     if sLabel in xQuantLabels:
        #         transformList.append(sLabel)
        #     else:
        #         for qLabel in xQualLabels:
        #             if qLabel in sLabel:
        #                 transformList.append(qLabel)
        # remap_dict = {i: transformList.count(i) for i in transformList}  # dictionary for remapping
        # keyList = list(remap_dict.keys())
        # lengthList = list(remap_dict.values())
        #
        # # compute new SHAP values
        # new_shap_values = []
        # for values in shap_values:
        #     # split shap values into a list for each feature
        #     values_split = np.split(values, np.cumsum(lengthList[:-1]))
        #     # sum values within each list
        #     values_sum = [sum(l) for l in values_split]
        #     new_shap_values.append(values_sum)
        # # replace SHAP values
        # shap_values = np.array(new_shap_values)

        pass



def scaledList(means, type='StandardScaler'):#'MinMaxScaler' #todo : check this

    from sklearn import preprocessing

    if type == 'MinMaxScaler':
        vScaler = preprocessing.MinMaxScaler()
        v_normalized = vScaler.fit_transform(np.array(means).reshape(-1, 1)).reshape(1, -1)
    if type == 'StandardScaler':
        vScaler = preprocessing.StandardScaler()
        v_normalized = vScaler.fit_transform(np.array(means).reshape(-1, 1)).reshape(1, -1)

    return v_normalized.tolist()[0]