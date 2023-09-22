# SCRIPT IMPORTS
from Model import *
from HelpersFormatter import *
from Dashboard_Current import *

#LIBRARY IMPORTS
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

    def __init__(self, modelList, blendingConstructor, acc, refit, grid_select, Gridsearch = True, Type ='NBest'):

        self.modelList = modelList
        self.GSName = blendingConstructor['name'] + '_Blender_' + Type
        self.Type = Type
        self.predictorName = blendingConstructor['name']  # ex : SVR
        self.modelPredictor = blendingConstructor['modelPredictor']
        self.param_dict = blendingConstructor['param_dict']
        self.scoring = {'neg_mean_squared_error': 'neg_mean_squared_error', 'r2': 'r2'}
        self.refit = refit  # 'r2'Score used for refitting the blender  #todo : changed to fit BLE_Values input
        self.grid_select = grid_select #[metric, minimize] #todo : changed to fit BLE_Values input
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
            X_train, X_test, y_train, y_test, ScaleMean, ScaleStd = fold
            xtrainer, ytrainer = X_train, y_train

            # print('xtrainer', xtrainer)
            # print('ytrainer', ytrainer)

            # building the final model using the meta features # this should be done by a cv of 5 folds on the training set
            if Gridsearch:
                njobs = os.cpu_count() - 1
                print("RUNNING GRIDSEARCH")
                grid = GridSearchCV(self.modelPredictor, param_grid=self.param_dict, scoring=self.scoring,
                                    refit=self.refit,
                                    n_jobs=njobs, return_train_score=True)
                grid.fit(xtrainer, ytrainer)
                self.Grid = grid
                Param = grid.best_params_
                Estimator = grid.best_estimator_

            else:
                Estimator = self.modelPredictor.fit(xtrainer, ytrainer)
                Param = None

            yPred = Estimator.predict(X_test)
            TrainScore = round(self.Grid.best_score_, self.rounding)
            # cross-validated score using the specified scoring function
            # how well the model generalizes to different subsets of the training data (cross-validation folds)
            TestScore = round(self.Grid.score(X_test, y_test), self.rounding)
            # score on the test data using the same scoring function
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
            self.ScaleMeans.append(ScaleMean)
            self.ScaleStds.append(ScaleStd)

        "Best Blender Model is selected as model with lowest variance residual "

        metric = self.grid_select[0]
        minimize = self.grid_select[1]
        if minimize:
            idx = get_minvalue(self.__getattribute__(metric))
        else :
            idx = get_maxvalue(self.__getattribute__(metric))

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
        reference = displayParams['ref_prefix'] + '_Combined/'
        path, folder, subFolder = DBpath, "RESULTS/", reference + 'RECORDS/'
        outputPathStudy = path + folder + subFolder

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        AllDfs = []
        sheetNames = [str(elem) for elem in random_seeds]

        for blendModel in BL_NBest_All:
            BlendingDf = construct_Blending_Df(blendModel)
            AllDfs.append(BlendingDf)
            # print(BlendingDf) #todo

        sheetNames += ['Avg']


        index = [model.GSName for model in BL_NBest_All[0].modelList] #+ ['NBest_Avg'] + [BLE_VALUES['Regressor'] + "_Blender_NBest"]
        columns = [ 'TestAcc', 'TestMSE', 'TestR2','TrainScore', 'TestScore','ResidMean', 'ResidVariance', 'ModelWeights']

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
        columns = [ 'TestAcc', 'TestMSE', 'TestR2','TrainScore', 'TestScore','ResidMean', 'ResidVariance', 'ModelWeights']
        ExtraDf = pd.DataFrame(columns=columns, index=index, dtype=float)
        ExtraDf.loc['BlenderAvg-BestModelAvg', :] = (Avgdf.loc[id_blender + "_Avg", :] - Avgdf.iloc[0, :])
        ExtraDf.loc['BlenderAvg-NBestAvg', :] = (Avgdf.loc[id_blender + "_Avg", :] - Avgdf.loc['NBest_Avg', :])

        Combined_Df = pd.concat([Avgdf, ExtraDf])

        AllDfs.append(Combined_Df)

        with pd.ExcelWriter(
                outputPathStudy + displayParams['ref_prefix'] + '_' + BL_NBest_All[0].GSName + "_BL_Scores_NBest" + ".xlsx", mode='w') as writer:
            for df, name in zip(AllDfs, sheetNames):
                df.to_excel(writer, sheet_name=name)

def construct_NBest_Df(blendModel):

    index = [model.GSName for model in blendModel.modelList]
    columns = ['TestAcc', 'TestMSE', 'TestR2', 'TrainScore', 'TestScore', 'ResidMean', 'ResidVariance','ModelWeights']

    NBestDf = pd.DataFrame(columns=columns, index=index)
    for col in columns[:-1]:
        NBestDf[col] = [model.__getattribute__(col) for model in blendModel.modelList]
        if len(blendModel.ModelWeights) == len(blendModel.modelList):
            NBestDf['ModelWeights'] = [round(elem, 3) for elem in list(blendModel.ModelWeights)]
        else:
            NBestDf['ModelWeights'] = [0 for elem in list(blendModel.modelList)]
    ExtraDf = pd.DataFrame(columns=columns)
    ExtraDf.loc['NBest_Avg', :] = NBestDf.mean(axis=0)
    Combined_Df = pd.concat([NBestDf, ExtraDf])

    return Combined_Df

def construct_Blender_Df(blendModel):

    index = [blendModel.GSName]
    columns = ['TestAcc', 'TestMSE', 'TestR2', 'TrainScore', 'TestScore', 'ResidMean', 'ResidVariance','ModelWeights']
    BlendingDf = pd.DataFrame(columns=columns, index=index)
    for col in columns[:-1]:
        BlendingDf[col] = [blendModel.__getattribute__(col)]

    return BlendingDf

def construct_CVAnalysis_Blender_Df(blendModel):
    from statistics import mean, stdev

    index = ['Blender Fold' + str(elem) for elem in list(range(len(blendModel.Estimators)))] + ['Blender_Mean', 'Blender_Std']
    columns = [ 'TestAcc', 'TestMSE', 'TestR2', 'TrainScore', 'TestScore', 'ResidMean', 'ResidVariance','ModelWeights']

    AnalysisDf = pd.DataFrame(columns=columns, index=index)

    for col, name in zip([ 'TestAccs', 'TestMSEs', 'TestR2s', 'TrainScores', 'TestScores','ResidMeans', 'ResidVariances'],
                         ['TestAcc', 'TestMSE', 'TestR2', 'TrainScore', 'TestScore', 'ResidMean', 'ResidVariance']):
        AnalysisDf[name] = blendModel.__getattribute__(col) + [mean(blendModel.__getattribute__(col))] + [stdev(blendModel.__getattribute__(col))]  #[i] for i in range(len())] + [self.__getattribute__(col)]

    return AnalysisDf

def construct_Blending_Df(blendModel):

    NBest_Df = construct_NBest_Df(blendModel)
    Blender_Df = construct_Blender_Df(blendModel)

    CVAnalysis_Blender_Df = construct_CVAnalysis_Blender_Df(blendModel)


    index = ['', 'BestBlender-BestModel', 'BestBlender-NBestAvg', 'AvgBlender-BestModel', 'AvgBlender-NBestAvg']
    columns = ['TestAcc', 'TestMSE', 'TestR2', 'TrainScore', 'TestScore', 'ResidMean', 'ResidVariance','ModelWeights']

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
                df.to_excel(writer, sheet_name=name)


def plot_distri_blender(studies_Blender, DBpath,displayParams, focus = 'TestAcc', unit = '[%]', setxLim = [0.5, 1],
                        fontsize = 12, adaptXLim = True, binwidth = 0.05):

    from scipy.stats import norm


    listResVal = []
    for blendModel in studies_Blender:
        CVAnalysis_Blender_Df = construct_CVAnalysis_Blender_Df(blendModel)
        listResVal += CVAnalysis_Blender_Df.loc['Blender Fold0':'Blender Fold4', focus].tolist()
    listResVal = [elem*100 for elem in listResVal]

    extra = studies_Blender[0].GSName
    title = focus + ' distribution for ' + extra
    arr = np.array(listResVal)
    mean = np.mean(arr)
    variance = np.var(arr)
    sigma = np.sqrt(variance)

    if adaptXLim :
        resmin, resmax = min(listResVal), max(listResVal)
        if resmax > setxLim[1]:
            import math
            setxLim[1] = math.ceil(resmax / 100) * 100
            print("residuals out of binrange  :", resmax)
            print("bin max changed to :", setxLim[1])
        if resmin < setxLim[0]:
            import math
            setxLim[0] = math.floor(resmin / 100) * 100
            print("residuals out of binrange  :", resmin)
            print("bin min changed to :", setxLim[0])

    x = focus + unit
    fig, ax = plt.subplots()
    # plot the histplot and the kde
    try:
        ax = sns.histplot(listResVal, kde=True, legend=False) #binwidth=binwidth

    except np.core._exceptions._ArrayMemoryError:
        ax = sns.histplot(listResVal, kde=True, legend=False, bins='sturges', label="Residuals kde curve") #

    for k in ['right', 'top']:
        ax.spines[k].set_visible(False)

    plt.setp(ax.patches, linewidth=0)
    plt.title(title, fontsize=fontsize)
    plt.xlabel(x, fontsize=fontsize)
    plt.ylabel("Count (" + extra + ")", fontsize=fontsize)

    plt.figure(1)
    if setxLim:
        xLim = (setxLim[0], setxLim[1])
    else:
        xLim = (min(arr), max(arr))
    plt.xlim(xLim)

    ref_prefix = displayParams["ref_prefix"]
    reference = displayParams['reference']

    if displayParams['archive']:
        path, folder, subFolder = DBpath, "RESULTS/",  ref_prefix + '_Combined/' + 'VISU/' + focus
        import os
        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)
        plt.savefig(outputFigPath + '/' + 'Distri_Combined' + '-' + extra + '.png')

    if displayParams['showPlot']:
        plt.show()
    plt.close()

    return listResVal

def plot_distri_blenders(Blenders_NBest_CV, DBpath, displayParams, focus = 'TestAcc', unit = '[%]', setxLim = [0.5, 1],
                        fontsize = 12, adaptXLim = True, binwidth = 0.05):
    blender_results = []
    detail = []
    for blender_type in Blenders_NBest_CV:

        listResVal = plot_distri_blender(blender_type, DBpath = DBpath, displayParams =displayParams, focus=focus,
                            unit=unit, setxLim=setxLim,
                            fontsize=fontsize, adaptXLim=adaptXLim, binwidth=binwidth)
        blender_results.append(listResVal)
        arr = np.array(listResVal)
        mean = round(np.mean(arr), 2)
        variance = np.var(arr)
        sigma = round(np.sqrt(variance), 2)
        detail.append([mean, sigma])


    columns = [studies_Blender[0].GSName for studies_Blender in Blenders_NBest_CV]
    df = pd.DataFrame(blender_results, index = columns)

    title = focus + ' distribution'
    x = 'Blended Models'
    # fig = plt.figure(figsize=(10, 5))figsize=(5, 5)

    fig, ax = plt.subplots()

    ax = sns.boxplot(data=df.T, showmeans=True) #todo

    ax_2 = ax.axes
    lines = ax_2.get_lines()
    categories = ax_2.get_xticks()

    for cat, det in zip(categories,detail) :
        # every 4th line at the interval of 6 is median line
        # 0 -> p25 1 -> p75 2 -> lower whisker 3 -> upper whisker 4 -> p50 5 -> upper extreme value
        # y = round(lines[4 + cat * 6].get_ydata()[0], 1)
        y = det[0]

        ax_2.text(
            cat,
            y,
            f'{y}',
            ha='center',
            va='center',
            fontweight='bold',
            size=10,
            color='white',
            bbox=dict(facecolor='#445A64'))

    for k in ['right', 'top']:
        ax.spines[k].set_visible(False)

    plt.setp(ax.patches, linewidth=0)
    plt.title(title, fontsize=fontsize)
    plt.xlabel(x, fontsize=fontsize)
    plt.ylabel(focus + ' ' + unit, fontsize=fontsize)

    ax.figure.tight_layout()

    ref_prefix = displayParams["ref_prefix"]
    reference = displayParams['reference']

    if displayParams['archive']:
        path, folder, subFolder = DBpath, "RESULTS/",  ref_prefix + '_Combined/' + 'VISU/' + focus
        import os
        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)
        plt.savefig(outputFigPath + '/' + 'BoxPlot' + focus + '.png')

    if displayParams['showPlot']:
        plt.show()
    plt.close()

def RUN_Blender_Combined_NBest(CV_BlenderNBest, displayParams, DBpath, randomvalues, focus = 'TestAcc', unit = '[%]'):

    for blender_type in CV_BlenderNBest:
        report_BL_NBest_CV(blender_type, displayParams, DBpath, randomvalues)

    plot_distri_blenders(CV_BlenderNBest, DBpath= DBpath, displayParams =displayParams,
                         focus = focus, unit = unit, setxLim = [50,100],
                        fontsize = 12, adaptXLim = False, binwidth = 0.05)











