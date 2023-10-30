from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import numpy as np
import os
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import MinMaxScaler

def computeAccuracy(yTrue, yPred, tolerance):
    #https: // scikit - learn.org / stable / modules / model_evaluation.html  # scoring
    validated = [1 if abs(yPred[i] - yTrue[i]) < abs(yTrue[i]) * tolerance else 0 for i in range(len(yTrue))]

    return sum(validated) / len(validated)

def computeAccuracy_std(yTrue, yPred, learningDf, tolerance):

    std = learningDf.ydf.std().values[0] # std = learningDf.MeanStdDf.loc[learningDf.yLabel, 'std'] # or  'std',learningDf.yLabel
    val = std * tolerance
    validated = [1 if abs(yPred[i] - yTrue[i]) < val else 0 for i in range(len(yTrue))]
    #todo check this > insert it > in exports > compare results

    return sum(validated) / len(validated)

def computeAccuracy_mean(yTrue, yPred, learningDf, tolerance):

    mea = learningDf.ydf.mean().values[0]
    val = mea * tolerance
    validated = [1 if abs(yPred[i] - yTrue[i]) < val else 0 for i in range(len(yTrue))]
    #todo check this

    return sum(validated) / len(validated)

def computeAccuracy_rev(yTrue, yPred, comparison, tolerance):

    val = comparison * tolerance
    validated = [1 if abs(yPred[i] - yTrue[i]) < val else 0 for i in range(len(yTrue))]
    #todo check this

    return sum(validated) / len(validated)

class ModelGridsearch:

    def __init__(self, predictorName, learningDf, modelPredictor, param_dict, acc, acc_mean, acc_std, refit, xQtQlLabels = None): #todo refit changed for MSE

        self.predictorName = predictorName #ex : SVR
        self.modelPredictor = modelPredictor# ex : SVR()
        self.selectorName = learningDf.selector# ex : 'fl_spearman'
        self.selectedLabels = learningDf.selectedLabels # ex : ['GIFA', 'Sector']
        self.GSName = self.predictorName + '_' + self.selectorName #ex : SVR_fl_spearman ,
        self.learningDf = learningDf
        self.random_state = learningDf.random_state


        self.param_dict = param_dict
        self.scoring = {'neg_mean_squared_error': 'neg_mean_squared_error', 'r2': 'r2'}
        self.rounding = 3
        self.refit = refit #'r2' # criteria for best performing param / used for plotting   #todo refit changed for MSE

        # refit = True means that after performing k - fold Cv (i.e., training on a subset of the data),
        # it refits the model using the best hyperparameters from the gridsearch, on the complete training set.

        print('Calibrating hyperparameters')
        self.paramGridsearch(learningDf)
        self.accuracyTol = acc
        self.accuracyTol_mean = acc_mean
        self.accuracyTol_std = acc_std

        print('Retrieving best results')
        self.computeBestModel(learningDf)

        self.computeSHAP(NbFtExtracted = 5)
        if xQtQlLabels :
            self.computeSHAPGrouped(xQtQlLabels, NbFtExtracted = 5)

    def paramGridsearch(self, df):

        njobs = os.cpu_count() - 1 #todo : njobs was changed

        grid = GridSearchCV(self.modelPredictor, param_grid=self.param_dict, scoring=self.scoring,
                            refit=self.refit,
                            n_jobs=njobs, return_train_score=True) #cv=cv
        grid.fit(df.XTrain.to_numpy(), df.yTrain.to_numpy().ravel())

        self.GridMSE = [round(num, self.rounding) for num in grid.cv_results_['mean_test_neg_mean_squared_error']]
        self.GridMSEStd = [round(num, self.rounding) for num in list(grid.cv_results_['std_test_neg_mean_squared_error'])]
        self.GridMSERank = grid.cv_results_['rank_test_neg_mean_squared_error']

        self.GridR2 = grid.cv_results_['mean_test_r2'] #this is R2 on testing from CV on training !
        self.GridR2Std = grid.cv_results_['std_test_r2'] #this is R2 on testing from CV on training !
        self.GridR2Rank = grid.cv_results_['rank_test_r2']

        self.Grid = grid
        self.GridbScore = self.Grid.best_score_


    def computeBestModel(self, df, test = True):
        # todo : naming was changed from bestModel
        self.Param = self.Grid.best_params_
        self.Estimator = self.Grid.best_estimator_ #todo : this is the calibrated model to use for prediction > check
        self.Index = self.Grid.best_index_

        XTrain, yTrain  = df.XTrain.to_numpy(), df.yTrain.to_numpy().ravel()
        XTest, yTest = df.XTest.to_numpy(), df.yTest.to_numpy().ravel()
        self.yPred = self.Grid.predict(XTest)

        self.TrainScore = round(self.Grid.score(XTrain, yTrain), self.rounding)
        self.TestScore = round(self.Grid.score(XTest, yTest), self.rounding)
        self.TestAcc = round(computeAccuracy(yTest, self.Grid.predict(XTest), self.accuracyTol), self.rounding)

        mea = self.learningDf.ydf.mean().values[0]
        std = self.learningDf.ydf.std().values[0]

        self.TestAcc_std = round(computeAccuracy_rev(yTest, self.Grid.predict(XTest), std, self.accuracyTol_std), self.rounding)
        self.TestAcc_mean = round(computeAccuracy_rev(yTest, self.Grid.predict(XTest), mea, self.accuracyTol_mean), self.rounding)

        self.TestMSE = round(mean_squared_error(yTest, self.yPred), self.rounding)
        self.TestR2 = round(r2_score(yTest, self.yPred), self.rounding)
        self.Resid = yTest - self.yPred
        self.ResidMean = round(np.mean(np.abs(self.Resid)),2) #
        self.ResidVariance = round(np.var(self.Resid),2)

        if hasattr(self.Grid.best_estimator_, 'coef_'): # LR, RIDGE, ELASTICNET, KRR Kernel Linear, SVR Kernel Linear
            self.isLinear = True
            content = self.Grid.best_estimator_.coef_
            if type(content[0]) == np.ndarray:
                content = content[0]
            self.Weights = [round(num, self.rounding) for num in list(content)]
            self.WeightsScaled = scaledList(self.Weights)

        elif hasattr(self.Grid.best_estimator_, 'dual_coef_'): #KRR
            self.isLinear = True
            content = self.Grid.best_estimator_.dual_coef_
            if type(content[0]) == np.ndarray:
                content = content[0]
            self.Weights = [round(num, self.rounding) for num in list(content)]
            self.WeightsScaled = scaledList(self.Weights)

        else :
            self.isLinear = False
            content = 'Estimator is non linear - no weights can be querried'
            self.Weights = 'Estimator is non linear - no weights can be querried'
            self.WeightsScaled = 'Estimator is non linear - no weights can be querried'


    def computeSHAP(self, NbFtExtracted):

        """Compute shap summary for a fitted estimator and a set of test with its labels."""
        import shap

        #compute initial SHAP values

        sample = shap.sample(self.learningDf.XTrain, nsamples = 30, random_state = 0)
        masker = shap.maskers.Independent(self.learningDf.XTrain)#for average values
        try:
            explainer = shap.Explainer(self.Estimator) #, masker
        except Exception:
            # explainer = shap.KernelExplainer(self.Estimator.predict, sample) #use thisif too slow
            explainer = shap.KernelExplainer(self.Estimator.predict, sample)
        self.SHAPexplainer = explainer
        self.SHAPvalues = explainer.shap_values(self.learningDf.XTest) #for SHAP VALUES

        # convert SHAP as dataframe
        df_shap_values = pd.DataFrame(data=self.SHAPvalues, columns=self.learningDf.XTrain.columns)
        SHAPdf = pd.DataFrame(columns=['feature', 'importance'])
        #todo : check this !
        for col in df_shap_values.columns:
            importance = df_shap_values[col].abs().mean()
            SHAPdf.loc[len(SHAPdf)] = [col, importance]
        self.SHAPdf = SHAPdf.sort_values('importance', ascending=False) # df (nX2*) col1 = featuresnames / col2 = meanSHApvalue  for all tested data

        # extract top n features and give score : lower is better (0 for highest)
        SHAPScoreDict = dict()
        topNFeatures = self.SHAPdf['feature'][:NbFtExtracted]
        for i in range(len(list(topNFeatures))):
            SHAPScoreDict[list(topNFeatures)[i]] = NbFtExtracted-i
        self.SHAPScoreDict = SHAPScoreDict

    def computeSHAPGrouped(self, xQtQlLabels, NbFtExtracted):

        """Compute shap summary for a fitted estimator and a set of test with its labels - categorical features will be grouped."""

        explainer = self.SHAPexplainer
        shap_values = self.SHAPvalues

        (xQuantLabels, xQualLabels) = xQtQlLabels

        # find re-mapping to group sub-categories into single category
        transformList = []
        # xQuantLabels = list(rdat.xQuanti.keys())
        # xQualLabels = list(rdat.xQuali.keys())
        for sLabel in self.learningDf.selectedLabels:
            if sLabel in xQuantLabels:
                transformList.append(sLabel)
            else:
                for qLabel in xQualLabels:
                    if qLabel in sLabel:
                        transformList.append(qLabel)
        self.SHAPGroup_RemapDict = {i: transformList.count(i) for i in transformList}  # dictionary for remapping
        SHAPGroupKeys = list(self.SHAPGroup_RemapDict.keys())
        lengthList = list(self.SHAPGroup_RemapDict.values())

        # compute new SHAP values
        new_shap_values = []
        for values in shap_values:
            # split shap values into a list for each feature
            values_split = np.split(values, np.cumsum(lengthList[:-1]))
            # sum values within each list
            values_sum = [sum(l) for l in values_split]
            new_shap_values.append(values_sum)
        # replace SHAP values
        self.SHAPGroupvalues = np.array(new_shap_values)

        # convert SHAP as dataframe
        df_shap_values = pd.DataFrame(data=self.SHAPGroupvalues, columns=SHAPGroupKeys)
        SHAPGroupDf = pd.DataFrame(columns=['feature', 'importance'])
        for col in df_shap_values.columns:
            importance = df_shap_values[col].abs().mean()
            SHAPGroupDf.loc[len(SHAPGroupDf)] = [col, importance]
        self.SHAPGroupDf = SHAPGroupDf.sort_values('importance', ascending=False)

        # extract top n features and give score : lower is better (0 for highest)
        SHAPGroupScoreDict = dict()
        topNFeatures = self.SHAPGroupDf['feature'][:NbFtExtracted]
        for i in range(len(list(topNFeatures))):
            SHAPGroupScoreDict[list(topNFeatures)[i]] = NbFtExtracted-i
        self.SHAPGroupScoreDict = SHAPGroupScoreDict






def scaledList(means, type='StandardScaler'):#'MinMaxScaler'

    from sklearn import preprocessing

    if type == 'MinMaxScaler':
        vScaler = preprocessing.MinMaxScaler()
        v_normalized = vScaler.fit_transform(np.array(means).reshape(-1, 1)).reshape(1, -1)
    if type == 'StandardScaler':
        vScaler = preprocessing.StandardScaler()
        v_normalized = vScaler.fit_transform(np.array(means).reshape(-1, 1)).reshape(1, -1)

    return v_normalized.tolist()[0]