# SCRIPT IMPORTS
from Model import *
from HelpersFormatter import *
from SCRIPTS.UNUSED.BlendingReport import *
# from Dashboard_EUCB_Structures import *
from Dashboard_Current import *

#LIBRARY IMPORTS
from sklearn.model_selection import KFold

def prepare_blending_cv(modelList):

    blend_elem_sets = []

    for model in modelList:
        XVal = model.learningDf.__getattribute__('XVal').to_numpy()
        XCheck = model.learningDf.__getattribute__('XCheck').to_numpy()
        yVal = model.learningDf.__getattribute__('yVal').to_numpy().ravel()
        yCheck = model.learningDf.__getattribute__('yCheck').to_numpy().ravel()
        XMeta = np.concatenate((XVal, XCheck), axis=0)
        yMeta = np.concatenate((yVal, yCheck), axis=0)

        blend_elem_i = model.Estimator.predict(XMeta)
        blend_elem_i = pd.DataFrame(blend_elem_i)
        blend_elem_sets.append(blend_elem_i)

    blendDf = pd.concat(blend_elem_sets, axis=1)

    return blendDf, yMeta

def split_blending_cv(X, y, k = 5):

    kf = KFold(n_splits=k, random_state=None)
    kfolds = []
    for train_index, test_index in kf.split(X):
        fold = []

        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        ScaleMean = X_train.mean(axis=0)
        ScaleStd = X_train.std(axis=0)

        X_train = (X_train - ScaleMean) / ScaleStd
        X_test = (X_test - ScaleMean) / ScaleStd

        fold.append(X_train)
        fold.append(X_test)
        fold.append(y_train)
        fold.append(y_test)
        fold.append(ScaleMean) #todo : added
        fold.append(ScaleStd) #todo : added

        kfolds.append(fold)
        #kfolds = [fold,..., fold, fold]
        # fold =  [X_train,X_test, y_train, y_test, ScaleMean, ScaleStd]

    return kfolds

def Blend_Learning_Data(modelList, type = 'XVal'):

    # create meta learning data
    blend_elem_sets = []

    for model in modelList:
        XVal = model.learningDf.__getattribute__(type).to_numpy()
        blend_elem_i = model.Estimator.predict(XVal)
        blend_elem_i = pd.DataFrame(blend_elem_i)
        blend_elem_sets.append(blend_elem_i)
        #data is already scaled

    blendDf = pd.concat(blend_elem_sets, axis=1)

    return blendDf


class Model_Blender:

    def __init__(self, modelList, blendingConstructor, acc, Gridsearch = True, Type ='NBest'):

        self.modelList = modelList
        self.GSName = blendingConstructor['name'] + '_Blender_' + Type
        self.Type = Type
        self.predictorName = blendingConstructor['name']  # ex : SVR
        self.modelPredictor = blendingConstructor['modelPredictor']
        self.param_dict = blendingConstructor['param_dict']
        self.scoring = {'neg_mean_squared_error': 'neg_mean_squared_error', 'r2': 'r2'}
        self.refit = 'r2'  # Score used for refitting the blender
        self.accuracyTol = acc
        self.rounding = 3

        #prepare data for outer loop - Cross-Validation
        blendDf, yMeta = prepare_blending_cv(modelList)
        kfolds = split_blending_cv(blendDf, yMeta, k = 5)

        #kfolds = [fold,..., fold, fold]
        # fold =  [X_train,X_test, y_train, y_test, ScaleMean, ScaleStd]

        self.Estimators = []
        self.Params = []
        self.yPreds = []
        self.TrainScores = []
        self.TestScores = []
        self.TestAccs = []
        self.TestMSEs = []
        self.TestR2s = []
        self.Resids = []
        self.ResidMeans = []
        self.ResidVariances = []
        self.blendXtrains, self.blendXtests, self.yTrains, self.yTests = [], [], [], []
        self.ScaleMeans = []
        self.ScaleStds = []

        for fold in kfolds:
            X_train, X_test, y_train, y_test, ScaleMean, ScaleStd = fold #todo : added 'ScaleMean, ScaleStd'
            xtrainer, ytrainer = X_train, y_train

            print('xtrainer', xtrainer)
            print('ytrainer', ytrainer)

            # building the final model using the meta features # this should be done by a cv of 5 folds on the training set
            if Gridsearch:
                njobs = os.cpu_count() - 1
                print("RUNNING GRIDSEARCH")
                grid = GridSearchCV(self.modelPredictor, param_grid=self.param_dict, scoring=self.scoring, refit=self.refit,
                                    n_jobs=njobs, return_train_score=True)
                grid.fit(xtrainer, ytrainer)
                Param = grid.best_params_
                Estimator = grid.best_estimator_

            else:
                Estimator = self.modelPredictor.fit(xtrainer, ytrainer)
                Param = None

            yPred = Estimator.predict(X_test)
            TrainScore = round(Estimator.score(xtrainer, ytrainer), self.rounding)
            TestScore = round(Estimator.score(X_test, y_test), self.rounding)
            TestAcc = round(computeAccuracy(y_test, yPred, self.accuracyTol), self.rounding)
            TestMSE = round(mean_squared_error(y_test, yPred), self.rounding)
            TestR2 = round(r2_score(y_test, yPred), self.rounding)
            Resid = y_test - yPred
            ResidMean = round(np.mean(np.abs(Resid)), 2)  # round(np.mean(self.Resid),2)
            ResidVariance = round(np.var(Resid), 2)

            self.Estimators.append(Estimator)
            self.Params.append(Param)
            self.yPreds.append(yPred)
            self.TrainScores.append(TrainScore)
            self.TestScores.append(TestScore)
            self.TestAccs.append(TestAcc)
            self.TestMSEs.append(TestMSE)
            self.TestR2s.append(TestR2)
            self.Resids.append(Resid)
            self.ResidMeans.append(ResidMean)
            self.ResidVariances.append(ResidVariance)
            self.blendXtrains.append(X_train)
            self.blendXtests.append(X_test)
            self.yTrains.append(y_train)
            self.yTests.append(y_test)
            self.ScaleMeans.append(ScaleMean) #todo : added 'ScaleMean, ScaleStd'
            self.ScaleStds.append(ScaleStd) #todo : added 'ScaleMean, ScaleStd'

        "Best Blender Model is selected as model with lowest variance residual "

        idx = get_minvalue(self.ResidVariances)

        self.Estimator = self.Estimators[idx]
        self.Param = self.Params[idx]
        self.yPred = self.yPreds[idx]
        self.TrainScore = self.TrainScores[idx]
        self.TestScore = self.TestScores[idx]
        self.TestAcc = self.TestAccs[idx]
        self.TestMSE = self.TestMSEs[idx]
        self.TestR2 = self.TestR2s[idx]
        self.Resid = self.Resids[idx]
        self.ResidMean = self.ResidMeans[idx]
        self.ResidVariance = self.ResidVariances[idx]
        self.blendXtrain = self.blendXtrains[idx]
        self.blendXtest = self.blendXtests[idx]
        self.yTrain = self.yTrains[idx]
        self.yTest = self.yTests[idx]
        self.ScaleMean = self.ScaleMeans[idx]
        self.ScaleStd = self.ScaleStds[idx]

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
        self.ModelWeights = weights

    def plot_Blender_CV_Residuals(self, displayParams, FORMAT_Values, DBpath):
        from StudyResiduals import ResidualPlot_Distri_Combined
        ResidualPlot_Distri_Combined([self], displayParams, FORMAT_Values, DBpath,
                                     studyFolder='GaussianPlot_' + self.GSName +'_BLENDER', Blender=True, CV = True)#BLE_VALUES['Regressor']

    def plotBlenderYellowResiduals(self, displayParams, DBpath, yLim=None, xLim=None,fontsize=None,studyFolder='BLENDER/'):

        if displayParams['showPlot'] or displayParams['archive']:
            import matplotlib.pyplot as plt
            from yellowbrick.regressor import ResidualsPlot

            title = 'Residuals for ' + str(self.GSName) + '- BEST PARAM (%s) ' % self.Param

            fig = plt.figure(figsize=(10, 5))  #
            if fontsize:
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.xlabel('Predicted Value ', fontsize=14)
                plt.ylabel('Residuals', fontsize=14)
            ax = plt.gca()
            if yLim:
                plt.ylim(yLim[0], yLim[1])
            if xLim:
                plt.xlim(xLim[0], xLim[1])
            visualizer = ResidualsPlot(self.Estimator, title=title, fig=fig,hist=True)
            visualizer.fit(self.blendXtrain, self.yTrain.ravel())  # Fit the training data to the visualizer
            visualizer.score(self.blendXtest, self.yTest.ravel())  # Evaluate the model on the test data

            reference, ref_prefix = displayParams['reference'], displayParams['ref_prefix']

            if displayParams['archive']:
                import os

                if self.Type == 'NBest': # store in seed folder
                    path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/' + studyFolder + 'Residuals'

                else : # save in combined
                    path, folder, subFolder = DBpath, "RESULTS/", ref_prefix + '_Combined/' + 'VISU/' + studyFolder + 'Residuals'

                outputFigPath = path + folder + subFolder

                if not os.path.isdir(outputFigPath):
                    os.makedirs(outputFigPath)

                visualizer.show(outpath=outputFigPath + '/' + str(self.GSName) + '.png')

            if displayParams['showPlot']:
                visualizer.show()

            plt.close()

def report_BL_NBest_CV(BL_NBest_All, displayParams, DBpath, random_seeds):

    import pandas as pd
    if displayParams['archive']:
        import os
        reference = displayParams['reference']
        path, folder, subFolder = DBpath, "RESULTS/", reference[:-6] + '_Combined/' + 'RECORDS/'
        outputPathStudy = path + folder + subFolder

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        AllDfs = []
        sheetNames = [str(elem) for elem in random_seeds]

        for blendModel in BL_NBest_All:
            BlendingDf = construct_Blending_Df(blendModel)
            AllDfs.append(BlendingDf)

        sheetNames += ['Avg']


        index = [model.GSName for model in BL_NBest_All[0].modelList] #+ ['NBest_Avg'] + [BLE_VALUES['Regressor'] + "_Blender_NBest"]
        columns = ['TrainScore', 'TestScore', 'TestMSE', 'TestAcc', 'ResidMean', 'ResidVariance', 'ModelWeights']
        Avgdf = pd.DataFrame(columns=columns, index=index, dtype=float)
        for id in index:
            for col in columns:
                lists = [elem.loc[id, col] for elem in AllDfs]
                Avgdf.loc[id, col] = np.mean(lists)
        NBest_Avg = Avgdf.mean(axis=0)
        NBest_Std = Avgdf.std(axis=0)
        Avgdf.loc['NBest_Avg', :] = NBest_Avg
        Avgdf.loc['NBest_Std', :] = NBest_Std

        # id_blender = BLE_VALUES['Regressor'] + "_Blender_NBest"
        id_blender = BL_NBest_All[0].GSName
        for col in columns:
            lists_bl = [elem.loc[id_blender, col] for elem in AllDfs]
            Avgdf.loc[id_blender + '_Avg', col] = np.mean(lists_bl)
            Avgdf.loc[id_blender + '_Std', col] = np.std(lists_bl)

        index = ['', 'BlenderAvg-BestModelAvg', 'BlenderAvg-NBestAvg']
        columns = ['TrainScore', 'TestScore', 'TestMSE', 'TestAcc', 'ResidMean', 'ResidVariance', 'ModelWeights']
        ExtraDf = pd.DataFrame(columns=columns, index=index, dtype=float)
        ExtraDf.loc['BlenderAvg-BestModelAvg', :] = (Avgdf.loc[id_blender + "_Avg", :] - Avgdf.iloc[0, :])
        ExtraDf.loc['BlenderAvg-NBestAvg', :] = (Avgdf.loc[id_blender + "_Avg", :] - Avgdf.loc['NBest_Avg', :])

        Combined_Df = pd.concat([Avgdf, ExtraDf])

        AllDfs.append(Combined_Df)

        with pd.ExcelWriter(
                outputPathStudy + reference[:-6] + '_' + BL_NBest_All[0].GSName + "_BL_Scores_NBest" + ".xlsx", mode='w') as writer:
            for df, name in zip(AllDfs, sheetNames):
                df.to_excel(writer, sheet_name=name)

def construct_NBest_Df(blendModel):

    index = [model.GSName for model in blendModel.modelList]
    columns = ['TrainScore', 'TestScore', 'TestMSE', 'TestAcc', 'ResidMean', 'ResidVariance',
               'ModelWeights']  #
    NBestDf = pd.DataFrame(columns=columns, index=index)
    for col in columns[:-1]:
        NBestDf[col] = [model.__getattribute__(col) for model in blendModel.modelList]
        if len(blendModel.ModelWeights) == len(blendModel.modelList):
            NBestDf['ModelWeights'] = [round(elem, 3) for elem in list(blendModel.ModelWeights)]
        else:
            NBestDf['ModelWeights'] = [0 for elem in list(blendModel.modelList)]
    # NBestDf.loc['NBest_Avg', :] = NBestDf.mean(axis=0)
    ExtraDf = pd.DataFrame(columns=columns)
    ExtraDf.loc['NBest_Avg', :] = NBestDf.mean(axis=0)
    Combined_Df = pd.concat([NBestDf, ExtraDf])

    return Combined_Df

def construct_Blender_Df(blendModel):

    index = [blendModel.GSName]
    columns = ['TrainScore', 'TestScore', 'TestMSE', 'TestAcc', 'ResidMean', 'ResidVariance','ModelWeights']  #
    BlendingDf = pd.DataFrame(columns=columns, index=index)
    for col in columns[:-1]:
        BlendingDf[col] = [blendModel.__getattribute__(col)]

    return BlendingDf

def construct_CVAnalysis_Blender_Df(blendModel):
    from statistics import mean, stdev

    index = ['Blender Fold' + str(elem) for elem in list(range(len(blendModel.Estimators)))] + ['Blender_Mean', 'Blender_Std']
    columns = ['TrainScore', 'TestScore', 'TestMSE', 'TestAcc', 'ResidMean', 'ResidVariance','ModelWeights']  #
    AnalysisDf = pd.DataFrame(columns=columns, index=index)
    for col, name in zip(['TrainScores', 'TestScores', 'TestMSEs', 'TestAccs', 'ResidMeans', 'ResidVariances'],
                         ['TrainScore', 'TestScore', 'TestMSE', 'TestAcc', 'ResidMean', 'ResidVariance']):
        AnalysisDf[name] = blendModel.__getattribute__(col) + [mean(blendModel.__getattribute__(col))] + [stdev(blendModel.__getattribute__(col))]  #[i] for i in range(len())] + [self.__getattribute__(col)]

    return AnalysisDf

def construct_Blending_Df(blendModel):

    NBest_Df = construct_NBest_Df(blendModel)
    Blender_Df = construct_Blender_Df(blendModel)
    CVAnalysis_Blender_Df = construct_CVAnalysis_Blender_Df(blendModel)
    index = ['', 'BestBlender-BestModel', 'BestBlender-NBestAvg', 'AvgBlender-BestModel', 'AvgBlender-NBestAvg']
    columns = ['TrainScore', 'TestScore', 'TestMSE', 'TestAcc', 'ResidMean', 'ResidVariance','ModelWeights']
    ExtraDf = pd.DataFrame(columns=columns, index=index)
    ExtraDf.loc['BestBlender-BestModel', :] = (Blender_Df.loc[blendModel.GSName, :] - NBest_Df.iloc[0, :])
    ExtraDf.loc['BestBlender-NBestAvg', :] = (Blender_Df.loc[blendModel.GSName, :] - NBest_Df.iloc[-1, :])
    ExtraDf.loc['AvgBlender-BestModel', :] = (CVAnalysis_Blender_Df.loc['Blender_Mean', :] - NBest_Df.iloc[0, :])
    ExtraDf.loc['AvgBlender-NBestAvg', :] = (CVAnalysis_Blender_Df.loc['Blender_Mean', :] - NBest_Df.loc['NBest_Avg', :])

    Blending_Df = pd.concat([NBest_Df, Blender_Df, CVAnalysis_Blender_Df, ExtraDf])

    return Blending_Df

def report_Blending_NBest(blendModel, displayParams, DBpath):

    if displayParams['archive']:

        reference = displayParams['reference']
        BlendingDf = construct_Blending_Df(blendModel)

        import os
        outputPathStudy = DBpath + "RESULTS/" + reference + 'RECORDS/'
        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        sortedDf = BlendingDf.sort_values('ModelWeights', ascending=False)

        AllDfs = [BlendingDf, sortedDf]
        sheetNames = ['Residuals_MeanVar', 'Sorted_Residuals_MeanVar']

        with pd.ExcelWriter(outputPathStudy + reference[:-1] + '_' + blendModel.GSName + "_Scores" + ".xlsx", mode='w') as writer:
            for df, name in zip(AllDfs, sheetNames):
                df.to_excel(writer, sheet_name=name) #todo :BLE_VALUES['Regressor']










