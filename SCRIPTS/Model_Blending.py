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

    def __init__(self, modelList, blendingConstructor, Gridsearch = True, Type ='NBest'):

        self.modelList = modelList
        self.GSName = blendingConstructor['name'] + '_Blender_' + Type
        self.Type = Type
        self.predictorName = blendingConstructor['name']  # ex : SVR
        self.modelPredictor = blendingConstructor['modelPredictor']
        self.param_dict = blendingConstructor['param_dict']
        self.scoring = {'neg_mean_squared_error': 'neg_mean_squared_error', 'r2': 'r2'}
        self.refit = 'r2'  # Score used for refitting the blender
        self.accuracyTol = 0.15
        self.rounding = 3

        # xVal and xCheck : same samples for every seeds and models, but different features depending on learningDf
        self.blendXtrain = Blend_Learning_Data(modelList, type='XVal')
        self.blendXtest = Blend_Learning_Data(modelList, type='XCheck')

        # yVal, yCheck are identical for all modls and seed > fixed seed
        self.yTrain = modelList[0].learningDf.__getattribute__('yVal').to_numpy().ravel()
        self.yTest = modelList[0].learningDf.__getattribute__('yCheck').to_numpy().ravel()

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
        self.ModelWeights = weights


    def construct_Blending_Df(self):

        index = [model.GSName for model in self.modelList] + [self.GSName]
        columns = ['TrainScore', 'TestScore', 'TestMSE', 'TestR2', 'TestAcc', 'ResidMean', 'ResidVariance',
                   'ModelWeights']  #
        BlendingDf = pd.DataFrame(columns=columns, index=index)
        for col in columns[:-1]:
            BlendingDf[col] = [model.__getattribute__(col) for model in self.modelList] + [self.__getattribute__(col)]
        BlendingDf['ModelWeights'] = [round(elem, 3) for elem in list(self.ModelWeights)] + [0]

        return BlendingDf

    def report_Blending_NBest(self, displayParams, DBpath):

        if displayParams['archive']:

            reference = displayParams['reference']
            BlendingDf = self.construct_Blending_Df()

            if self.Type == 'NBest':

                import os
                outputPathStudy = DBpath + "RESULTS/" + reference + 'RECORDS/'
                if not os.path.isdir(outputPathStudy):
                    os.makedirs(outputPathStudy)

                sortedDf = BlendingDf.sort_values('ModelWeights', ascending=False)

                AllDfs = [BlendingDf, sortedDf]
                sheetNames = ['Residuals_MeanVar', 'Sorted_Residuals_MeanVar']

                with pd.ExcelWriter(outputPathStudy + reference[:-1] + "_BL_Scores_NBest" + '_' + str(
                        len(self.modelList)) + '_' + self.GSName + ".xlsx", mode='w') as writer:
                    for df, name in zip(AllDfs, sheetNames):
                        df.to_excel(writer, sheet_name=name)


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

            reference = displayParams['reference']

            if displayParams['archive']:
                import os

                if self.Type == 'NBest': # store in seed folder
                    path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/' + studyFolder + 'Residuals'

                else : # save in combined
                    path, folder, subFolder = DBpath, "RESULTS/", reference[:-6] + '_Combined/' + 'VISU/' + studyFolder + 'Residuals'

                outputFigPath = path + folder + subFolder

                if not os.path.isdir(outputFigPath):
                    os.makedirs(outputFigPath)

                visualizer.show(outpath=outputFigPath + '/' + str(self.GSName) + '.png')

            if displayParams['showPlot']:
                visualizer.show()

            plt.close()

def report_BL_NBest_All(BL_NBest_All, displayParams, DBpath, random_seeds):

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
            BlendingDf = blendModel.construct_Blending_Df()
            AllDfs.append(BlendingDf)

        dflist = []
        for df, blendModel in zip(AllDfs, BL_NBest_All):
            slice = df.iloc[0:len(df)-1, :]
            index = ['NBest_Avg', 'Blender_Increase']
            columns = ['TrainScore', 'TestScore', 'TestMSE',  'TestAcc', 'ResidMean', 'ResidVariance','ModelWeights'] #'TestR2',
            IncDf = pd.DataFrame(columns=columns, index=index)
            IncDf.loc['NBest_Avg', :] = df.iloc[0:len(df)-1, :].mean(axis=0)
            IncDf.loc['Blender_Increase', :] = ((df.loc[blendModel.GSName, :] / df.iloc[0:len(df)-1, :].mean(axis=0)) - 1)
            nwdf = pd.concat([df, IncDf])
            dflist.append(nwdf)

        with pd.ExcelWriter(
                outputPathStudy + reference[:-6] + "_BL_Scores_NBest" + ".xlsx", mode='w') as writer:
            for df, name in zip(dflist, sheetNames):
                df.to_excel(writer, sheet_name=name)




def report_Blending_CV(self, displayParams, DBpath):

    # THIS MOVES TO COMBINED
    if self.Type == 'CV':
        sheetNames = [model.random_state for model in self.modelList]

        pass

















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

    #SINGLE
    blendModel = Model_Blender(modelList, blendingConstructor=CONSTRUCTOR, baseFormatedDf = baseFormatedDf, Gridsearch = Gridsearch)
    reportCV_Scores_NBest([blendModel], displayParams, DB_Values['DBpath'], n=None, NBestScore=BLE_VALUES['NBestScore'], random_seeds = None)
    pickleDumpMe(DBpath, displayParams, blendModel, 'CV_BLENDER', blendModel.GSName + "_" + blendModel.modelList[0].GSName + '_' + NBestScore, combined = True)
    return blendModel

    #MULTIPLE
    blendModel_ls =[]

    for blist in modelList:
        blendModel = Model_Blender(blist, blendingConstructor=CONSTRUCTOR, baseFormatedDf=baseFormatedDf,
                                   Gridsearch=Gridsearch)
        blendModel_ls.append(blendModel)
        pickleDumpMe(DBpath, displayParams, blendModel, 'CV_BLENDER',
                     blendModel.GSName + "_" + blendModel.modelList[0].GSName + '_' + NBestScore, combined=True)

    #
    # GS_FS_List_Labels = ['LR', 'LR_RIDGE', 'KRR_RBF', 'KRR_LIN', 'KRR_POL', 'SVR_LIN', 'SVR_RBF']
    #
    # blendModel_ls = []
    # for GSName in GS_FS_List_Labels:
    #     n = GSName
    #     for FS in ['_RFE_GBR', '_NoSelector', '_RFE_DTR', '_RFE_RFR', '_fl_pearson', '_fl_spearman']:
    #         n += FS
    #
    #     path = DB_Values['DBpath'] + 'RESULTS/CSTB_res_nf_EC_Combined/RECORDS/CV_BLENDER/LR_RIDGE_Blender_' \
    #            + n + '_' + NBestScore + '.pkl'
    #
    #     Blender = pickleLoadMe(path=path, show=False)
    #     blendModel_ls.append(Blender)

    reportCV_Scores_NBest(blendModel_ls, displayParams, DB_Values['DBpath'], n=None,
                          NBestScore=BLE_VALUES['NBestScore'], random_seeds=None)

    return blendModel_ls
