from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
from datetime import datetime
# from sklearn.model_selection import LeaveOneOut


class WrapFeatures:

    def __init__(self, method, estimator, formatedDf, rfe_hyp_feature_count, output_feature_count ='rfeCV', scoring="r2",
                 step= 1, cv = KFold(n_splits=5, shuffle=True, random_state=42), process ='long'):


        # step - features removed at every iteration
        # KFold - cross-validation splitting - ensures same split for every function run

        self.yLabel = formatedDf.yLabel
        self.random_state = formatedDf.random_state
        self.method = method # ex :'LR'
        self.estimator = estimator #unfit estimator # ex : LinearRegression()
        self.FtCountFrom = output_feature_count # ex : 'rfeHyp' > defines the number of feature for the output RFE

        # now = datetime.now()
        # current_time = now.strftime("%H:%M:%S")
        # print("Current Time =", current_time)  # todo : remove

        if process == 'long' :
            print('RFE - CV Calibration')
            self.RFEliminationCV(formatedDf, step, cv, scoring)
            print('RFE - Hyperparameter Calibration')
            self.RFEHyperparameterSearch(formatedDf, rfe_hyp_feature_count)
            print('RFE - Retrieving Resuts for ', output_feature_count)
            if output_feature_count == 'rfeHyp':
                self.RFElimination(formatedDf, self.rfeHyp_maxvalFtCount)
            if output_feature_count == 'rfeCV':
                self.RFElimination(formatedDf, self.rfecv.n_features_)
            if type(output_feature_count) == int:
                self.RFElimination(formatedDf, output_feature_count)

        if process == 'short':
            if self.FtCountFrom == 'rfeHyp' :
                print('RFE - Hyperparameter Calibration')
                self.RFEHyperparameterSearch(formatedDf, rfe_hyp_feature_count)
                self.RFElimination(formatedDf, self.rfeHyp_maxvalFtCount, pretrained = True)
            if self.FtCountFrom == 'rfeCV' :
                print('RFE - CV Calibration')
                self.RFEliminationCV(formatedDf, step, cv, scoring)
                self.RFElimination(formatedDf, self.rfecv.n_features_, pretrained = True)
            if type(output_feature_count) == int:
                print('RFE - Retrieving Resuts for ', output_feature_count)
                self.RFElimination(formatedDf, output_feature_count, pretrained = False)

        # self.trainDf
        # self.valDf
        # self.testDf
        # self.droppedLabels
        # self.selectedLabels
        self.selector = 'RFE_' + self.method

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
            # rfeHyp_maxvalScore
            # rfeHyp_maxvalidx
            # rfeHyp_maxvalFtCount


    def RFElimination(self, formatedDf, n_features_to_select, pretrained = False):

        """
        Look for n most important features
        (or, the best combination of n features, given their importance for the wrapped estimator)
        """

        self.n_features_to_select = n_features_to_select

        if pretrained :
            if self.FtCountFrom == 'rfeHyp':
                self.rfe = self.rfeHyp
                self.selectedLabels = self.rfeHyp_selectedLabels
                self.rfe_trainScore = self.rfeHyp_trainScore
                self.rfe_valScore = self.rfeHyp_valScore
                self.droppedLabels = self.rfeHyp_droppedLabels

            if self.FtCountFrom == 'rfeCV':
                self.rfe = self.rfecv
                self.rfe_trainScore = self.rfecv_trainScore
                self.rfe_valScore = self.rfecv_valScore
                self.selectedLabels = self.rfecv_selectedLabels
                self.droppedLabels = self.rfecv_droppedLabels

        else :
            rfe = RFE(self.estimator, n_features_to_select=self.n_features_to_select)
            self.rfe = rfe.fit(formatedDf.XTrain.to_numpy(), formatedDf.yTrain.to_numpy().ravel())
            self.selectedLabels = formatedDf.XTrain.columns[rfe.support_]   #this naming was changed from #Xlabels to #selectedLabels >could generate issues
            self.rfe_trainScore = self.rfe.score(formatedDf.XTrain.to_numpy(), formatedDf.yTrain.to_numpy().ravel())
            self.rfe_valScore = self.rfe.score(formatedDf.XVal.to_numpy(), formatedDf.yVal.to_numpy().ravel())
            self.droppedLabels = [label for label in formatedDf.XTrain.columns if label not in self.selectedLabels]



        self.trainDf = formatedDf.trainDf.drop(columns=self.droppedLabels)
        self.valDf = formatedDf.valDf.drop(columns=self.droppedLabels)
        self.testDf = formatedDf.testDf.drop(columns=self.droppedLabels)

        self.XTrain =self.trainDf.drop(columns=self.yLabel)
        self.XVal = self.valDf.drop(columns=self.yLabel)
        self.XTest = self.testDf.drop(columns=self.yLabel)
        self.yTrain = self.trainDf[self.yLabel]
        self.yVal = self.valDf[self.yLabel]
        self.yTest = self.testDf[self.yLabel]
        # self.yLabel = yLabel

    def RFEDisplay(self):

        print("RFE with:", self.method)
        print("Number of features fixed:", self.n_features_to_select)
        print("Score on training", self.rfe_trainScore)
        print('Selected feature labels', list(self.selectedLabels))
        print("Score on validation", self.rfe_valScore)
        print("")

    def RFEliminationCV(self, formatedDf, step, cv, scoring):

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
        self.rfecv_XTrain =self.rfecv_trainDf.drop(columns=self.yLabel)
        self.rfecv_XVal = self.rfecv_valDf.drop(columns=self.yLabel)
        self.rfecv_XTest = self.rfecv_testDf.drop(columns=self.yLabel)
        self.rfecv_yTrain = self.rfecv_trainDf[self.yLabel]
        self.rfecv_yVal = self.rfecv_valDf[self.yLabel]
        self.rfecv_yTest = self.rfecv_testDf[self.yLabel]

    def RFEHyperparameterSearch(self, formatedDf, featureCount):

        """Look for number of features - by selecting the n most important features
        (or, the best combination of n features, given their importance for the wrapped estimator)"""

        self.rfeHyp_featureCount = featureCount

        trainScoreList = []
        testScoreList = []
        selectedLabelsList = []
        rfeHypList = []

        for f in featureCount:

            rfeHyp = RFE(self.estimator, n_features_to_select=f)
            rfeHyp = rfeHyp.fit(formatedDf.XTrain.to_numpy(), formatedDf.yTrain.to_numpy().ravel())
            rfeHyp_selectedLabels = formatedDf.XTrain.columns[rfeHyp.support_]

            rfeHypList.append(rfeHyp)
            selectedLabelsList.append(rfeHyp_selectedLabels)
            trainScoreList.append(rfeHyp.score(formatedDf.XTrain.to_numpy(), formatedDf.yTrain.to_numpy().ravel()))
            testScoreList.append(rfeHyp.score(formatedDf.XVal.to_numpy(), formatedDf.yVal.to_numpy().ravel()))


        self.rfeHyp_trainScore = trainScoreList
        self.rfeHyp_valScore= testScoreList
        self.rfeHyp_maxvalScore = max(testScoreList)
        self.rfeHyp_maxvalidx = testScoreList.index(max(testScoreList))
        self.rfeHyp_maxvalFtCount = featureCount[self.rfeHyp_maxvalidx]
        self.rfeHyp_selectedLabels = selectedLabelsList[self.rfeHyp_maxvalidx]
        self.rfeHyp = rfeHypList[self.rfeHyp_maxvalidx]

        self.rfeHyp_droppedLabels = [label for label in formatedDf.XTrain.columns if label not in self.rfeHyp_selectedLabels]

        self.rfeHyp_trainDf = formatedDf.trainDf.drop(columns=self.rfeHyp_droppedLabels)
        self.rfeHyp_valDf = formatedDf.valDf.drop(columns=self.rfeHyp_droppedLabels)
        self.rfeHyp_testDf = formatedDf.testDf.drop(columns=self.rfeHyp_droppedLabels)
        self.rfeHyp_XTrain =self.rfeHyp_trainDf.drop(columns=self.yLabel)
        self.rfeHyp_XVal = self.rfeHyp_valDf.drop(columns=self.yLabel)
        self.rfeHyp_XTest = self.rfeHyp_testDf.drop(columns=self.yLabel)
        self.rfeHyp_yTrain = self.rfeHyp_trainDf[self.yLabel]
        self.rfeHyp_yVal = self.rfeHyp_valDf[self.yLabel]
        self.rfeHyp_yTest = self.rfeHyp_testDf[self.yLabel]

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




# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
# https://blog.datadive.net/selecting-good-features-part-iv-stability-selection-rfe-and-everything-side-by-side/
#https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
#https://machinelearningmastery.com/rfe-feature-selection-in-python/
#https://www.section.io/engineering-education/recursive-feature-elimination/
#https://stackoverflow.com/questions/46062679/right-order-of-doing-feature-selection-pca-and-normalization