from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
# from sklearn.model_selection import LeaveOneOut

#DEFAULT VALUES

rs = 42
n_features_to_select = 15
featureCount = [5, 10, 15, 20, 25]

"""
Questions
1. Trainscore is too high - What should i evaluate my rfe on? validation set? > what does the RFECV use as a score? 
2. Scaling, When should it be done? Look into pipeline
3. I inserted a random-state to have stable results - is this ok?
4. loo = LeaveOneOut() #alternative to cv - use this? 
"""

class WrapFeatures:

    def __init__(self, method, estimator, formatedDf, yLabel, n_features_to_select, featureCount, step= 1,
                 cv = KFold(n_splits=5, shuffle=True, random_state=42), scoring="r2"):

        # step - features removed at every iteration
        # KFold - cross-validation splitting - ensures same split for every function run

        self.method = method
        self.estimator = estimator #unfit estimator
        self.RFElimination(formatedDf, n_features_to_select, yLabel)
        self.RFEliminationCV(formatedDf, step, cv, scoring, yLabel)
        self.RFEHyperparameterSearch(formatedDf, featureCount)

        # self.n_features_to_select
        # self.rfe/rfeCV/rfehyp
            # self.rfe_droppedLabels
            # self.rfe_selectedLabels
            # self.rfe_trainScore
            # self.rfe_valScore

            # self.rfe_trainDf
            # self.rfe_valDf
            # self.rfe_testDf
            # self.rfe_XTrain
            # self.rfe_XVal
            # self.rfe_XTest
            # self.rfe_yTrain
            # self.rfe_yVal
            # self.rfe_yTest

            # self.rfeHyp_featureCount
            # self.rfeHyp_trainScore
            # self.rfeHyp_valScore

    def RFElimination(self, formatedDf, n_features_to_select, yLabel):

        """
        Look for n most important features
        (or, the best combination of n features, given their importance for the wrapped estimator)
        """
        self.n_features_to_select = n_features_to_select
        rfe = RFE(self.estimator, n_features_to_select=self.n_features_to_select)

        self.rfe = rfe.fit(formatedDf.XTrain.to_numpy(), formatedDf.yTrain.to_numpy().ravel())
        self.rfe_selectedLabels = formatedDf.XTrain.columns[rfe.support_]
        self.rfe_trainScore = self.rfe.score(formatedDf.XTrain.to_numpy(), formatedDf.yTrain.to_numpy().ravel())
        self.rfe_valScore = self.rfe.score(formatedDf.XVal.to_numpy(), formatedDf.yVal.to_numpy().ravel())

        self.rfe_droppedLabels = [label for label in formatedDf.XTrain.columns if label not in self.rfe_selectedLabels]

        self.rfe_trainDf = formatedDf.trainDf.drop(columns=self.rfe_droppedLabels)
        self.rfe_valDf = formatedDf.valDf.drop(columns=self.rfe_droppedLabels)
        self.rfe_testDf = formatedDf.testDf.drop(columns=self.rfe_droppedLabels)
        self.rfe_XTrain =self.rfe_trainDf.drop(columns=yLabel)
        self.rfe_XVal = self.rfe_valDf.drop(columns=yLabel)
        self.rfe_XTest = self.rfe_testDf.drop(columns=yLabel)
        self.rfe_yTrain = self.rfe_trainDf[yLabel]
        self.rfe_yVal = self.rfe_valDf[yLabel]
        self.rfe_yTest = self.rfe_testDf[yLabel]

    def RFEDisplay(self):

        print("RFE with:", self.method)
        print("Number of features fixed:", self.n_features_to_select)
        print("Score on training", self.rfe_trainScore)
        print('Selected feature labels', list(self.rfe_selectedLabels))
        print("Score on validation", self.rfe_valScore)
        print("")

    def RFEliminationCV(self, formatedDf, step, cv, scoring, yLabel):

        """    Look optimal number of features
        (or, the best combination of n features, given their importance for the wrapped estimator)"""

        # TODO : pipeline - insert scaling here?
        rfecv = RFECV(estimator=self.estimator, step=step, cv=cv, scoring=scoring)
        self.rfecv = rfecv.fit(formatedDf.XTrain.to_numpy(), formatedDf.yTrain.to_numpy().ravel())
        self.rfecv_trainScore = self.rfecv.score(formatedDf.XTrain.to_numpy(), formatedDf.yTrain.to_numpy().ravel())
        self.rfecv_valScore = self.rfecv.score(formatedDf.XVal.to_numpy(), formatedDf.yVal.to_numpy().ravel())
        self.rfecv_selectedLabels = formatedDf.XTrain.columns[self.rfecv.support_]

        self.rfecv_droppedLabels = [label for label in formatedDf.XTrain.columns if label not in self.rfecv_selectedLabels]

        self.rfecv_trainDf = formatedDf.trainDf.drop(columns=self.rfecv_droppedLabels)
        self.rfecv_valDf = formatedDf.valDf.drop(columns=self.rfecv_droppedLabels)
        self.rfecv_testDf = formatedDf.testDf.drop(columns=self.rfecv_droppedLabels)
        self.rfecv_XTrain =self.rfecv_trainDf.drop(columns=yLabel)
        self.rfecv_XVal = self.rfecv_valDf.drop(columns=yLabel)
        self.rfecv_XTest = self.rfecv_testDf.drop(columns=yLabel)
        self.rfecv_yTrain = self.rfecv_trainDf[yLabel]
        self.rfecv_yVal = self.rfecv_valDf[yLabel]
        self.rfecv_yTest = self.rfecv_testDf[yLabel]

    def RFEHyperparameterSearch(self, formatedDf, featureCount):

        """Look for number of features - by selecting the n most important features
        (or, the best combination of n features, given their importance for the wrapped estimator)"""

        self.rfeHyp_featureCount = featureCount
        trainScoreList = []
        testScoreList = []

        for f in featureCount:

            rfeHyp = RFE(self.estimator, n_features_to_select=f)
            rfeHyp = rfeHyp.fit(formatedDf.XTrain.to_numpy(), formatedDf.yTrain.to_numpy().ravel())

            trainScoreList.append(rfeHyp.score(formatedDf.XTrain.to_numpy(), formatedDf.yTrain.to_numpy().ravel()))
            testScoreList.append(rfeHyp.score(formatedDf.XVal.to_numpy(), formatedDf.yVal.to_numpy().ravel()))

        self.rfeHyp_trainScore = trainScoreList
        self.rfeHyp_valScore= testScoreList

    def RFECVDisplay(self):

        print("RFECV with:", self.method)
        print("Number of features from CV:", self.rfecv.n_features_)
        print("Score on training", self.rfecv_trainScore)
        print('Selected feature labels', list(self.rfecv_selectedLabels))
        print("Score on validation", self.rfecv_valScore)
        print("")

    def RFEHypSearchDisplay(self):

        print("RFE Param Search with:", self.method)
        print("Number of features compared", self.rfeHyp_featureCount)
        print("Score on training", self.rfeHyp_trainScore)
        print("Score on validation", self.rfeHyp_valScore)
        print("")

    def check(self):

        print((self.rfe_trainDf).shape)
        print((self.rfe_valDf).shape)
        print((self.rfe_testDf).shape)

        print((self.rfe_XTrain).shape)
        print((self.rfe_XVal).shape)
        print((self.rfe_XTest).shape)
        print((self.rfe_yTrain).shape)
        print((self.rfe_yVal).shape)
        print((self.rfe_yTest).shape)

        print((self.rfecv_trainDf).shape)
        print((self.rfecv_valDf).shape)
        print((self.rfecv_testDf).shape)

        print((self.rfecv_XTrain).shape)
        print((self.rfecv_XVal).shape)
        print((self.rfecv_XTest).shape)
        print((self.rfecv_yTrain).shape)
        print((self.rfecv_yVal).shape)
        print((self.rfecv_yTest).shape)

        print(self.rfeHyp_featureCount)
        print(self.rfeHyp_trainScore)
        print(self.rfeHyp_valScore)


# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
# https://blog.datadive.net/selecting-good-features-part-iv-stability-selection-rfe-and-everything-side-by-side/
#https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
#https://machinelearningmastery.com/rfe-feature-selection-in-python/
#https://www.section.io/engineering-education/recursive-feature-elimination/
#https://stackoverflow.com/questions/46062679/right-order-of-doing-feature-selection-pca-and-normalization