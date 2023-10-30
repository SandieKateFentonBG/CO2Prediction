from Model import *
from HelpersFormatter import *

def AccuracyCheck(Blender_List, displayParams, DBpath, tolerance):

    import pandas as pd

    blenderDf = []
    for blender_fold in Blender_List:
        ModelDfs = []
        for Model in blender_fold.modelList:
            yPred, PredictionDict = trackPrediction(Model)
            yTrue, yPred = PredictionDict['yTest'], PredictionDict['yPred']
            SampleAcc = abs((yPred - yTrue) / yTrue)
            SampleAccBool = [1 if SampleAcc[i] < tolerance else 0 for i in range(len(yTrue))]

            AccuracyDf_Model = pd.DataFrame(columns=[i for i in range(len(PredictionDict['yTest']))],
                                       index=[Model.GSName + '_' + l for l in ['yTest', 'yPred', 'Resid', 'SampleAcc','SampleAccBool']])

            AccuracyDf_Model.loc[Model.GSName + '_' +'yTest', :] = PredictionDict['yTest']
            AccuracyDf_Model.loc[Model.GSName + '_' +'yPred', :] = PredictionDict['yPred']
            AccuracyDf_Model.loc[Model.GSName + '_' +'Resid', :] = PredictionDict['Resid']
            AccuracyDf_Model.loc[Model.GSName + '_' +'SampleAcc', :] = SampleAcc
            AccuracyDf_Model.loc[Model.GSName + '_' +'SampleAccBool', :] = SampleAccBool
            AccuracyDf_Model.loc[:, 'Mean'] = AccuracyDf_Model.abs().mean(axis=1) #todo : check if abs makes sense
            AccuracyDf_Model.loc[:, 'Stdv'] = AccuracyDf_Model.abs().std(axis=1)

            ModelDfs.append(AccuracyDf_Model) #list of 10

        FoldDf = pd.concat(ModelDfs, axis=0)#1 Df merging 10
        blenderDf.append(FoldDf) #list of 5

    reference, ref_prefix = displayParams['reference'], displayParams['ref_prefix']

    if displayParams['archive']:
        import os

        path, folder, subFolder = DBpath, "RESULTS/", ref_prefix + '_Combined/' + 'RECORDS'+ '/'
        # path, folder, subFolder = DBpath, "RESULTS/", name
        outputPathStudy = path + folder + subFolder

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        sheetNames = ["PA_" + str(i) for i in range(len(blenderDf))]

        with pd.ExcelWriter(outputPathStudy + ref_prefix + "_AccuracyCheck" +  ".xlsx", mode='w') as writer:
            for df, sh in zip(blenderDf, sheetNames):
                df.to_excel(writer, sheet_name=sh, freeze_panes=(0, 1))

def SampleAccuracy(yTrue, yPred, metric, tolerance, mea, std):
    # mea = Model.learningDf.ydf.mean().values[0]
    # std = Model.learningDf.ydf.std().values[0]  # std = learningDf.MeanStdDf.loc[learningDf.yLabel, 'std'] # or  'std',learningDf.yLabel
    if metric == 'TestAcc':
        threshhold = yTrue
    elif metric == 'TestAcc_mean':
        threshhold = mea
    elif metric == 'TestAcc_std':
        threshhold = std

    SampleAcc = abs((yPred - yTrue)/ threshhold)
    SampleAccBool = [1 if SampleAcc[i] < tolerance else 0 for i in range(len(yTrue))]
    val = threshhold*tolerance

    return SampleAcc, SampleAccBool, val

def modelAccuracy(Model, metric, tolerance, mea, std, baseModel = True):

    yPred, PredictionDict = trackPrediction(Model, baseModel=baseModel)
    yTrue, yPred = PredictionDict['yTest'], PredictionDict['yPred']
    SampleAcc, SampleAccBool, threshhold = SampleAccuracy(yTrue, yPred, metric, tolerance, mea, std)
    AccuracyDf_Model = pd.DataFrame(columns=[i for i in range(len(PredictionDict['yTest']))],
                                    index=[Model.GSName + '_' + l for l in
                                           ['yTest', 'yPred', 'Resid', 'Treshhold', 'SampleAccBool']])

    AccuracyDf_Model.loc[Model.GSName + '_' + 'yTest', :] = PredictionDict['yTest']
    AccuracyDf_Model.loc[Model.GSName + '_' + 'yPred', :] = PredictionDict['yPred']
    AccuracyDf_Model.loc[Model.GSName + '_' + 'Resid', :] = PredictionDict['Resid']
    AccuracyDf_Model.loc[Model.GSName + '_' + 'Treshhold', :] = threshhold
    AccuracyDf_Model.loc[Model.GSName + '_' + 'SampleAccBool', :] = SampleAccBool
    AccuracyDf_Model.loc[:, 'Mean'] = AccuracyDf_Model.abs().mean(axis=1)  # todo : check if abs makes sense
    AccuracyDf_Model.loc[:, 'Stdv'] = AccuracyDf_Model.abs().std(axis=1)

    return AccuracyDf_Model


def constructBaseAccuracyDf(Blender, metric, tolerance, mea, std):

    baseModelDf = []
    for blender_fold in Blender: #5
        ModelDfs = []
        for Model in blender_fold.modelList: #10
            AccuracyDf_Model = modelAccuracy(Model, metric, tolerance, mea, std, baseModel = True)
            ModelDfs.append(AccuracyDf_Model)
        FoldDf = pd.concat(ModelDfs, axis=0)#1 Df merging 10
        baseModelDf.append(FoldDf) #list of 5

    return baseModelDf #1 list of 5

def constructBlenderAccuracyDf(Blender_List, metric, tolerance, mea, std):

    blenderDf_List = []
    #base models
    for blender in Blender_List:
        bl_df = []
        for blender_fold in blender: #5
            AccuracyDf_Blender = modelAccuracy(blender_fold, metric, tolerance, mea, std, baseModel = False)
            bl_df.append(AccuracyDf_Blender)
        blenderDf_List.append(bl_df)

    return blenderDf_List # 2 lists of 5

def constructAccuracyDf(Blender_List, metric, tolerance):
    mea = Blender_List[0][0].modelList[0].learningDf.ydf.mean().values[0]
    std = Blender_List[0][0].modelList[0].learningDf.ydf.std().values[0]

    baseModelDf_List = constructBaseAccuracyDf(Blender_List[0], metric, tolerance, mea, std)
    blenderDf_List = constructBlenderAccuracyDf(Blender_List, metric, tolerance, mea, std)
    GroupedDf = []

    for i in range(len(baseModelDf_List)):
        elem = [baseModelDf_List[i]]
        for bl in blenderDf_List:
            elem += [bl[i]]
        df = pd.concat(elem, axis=0)
        GroupedDf.append(df)

    return GroupedDf

def AccuracyCheck_Comparison(Blender_List, displayParams, DBpath, metrics, tolerances):

    import pandas as pd

    AllDfs = []
    sheetNames = []

    for metric, tolerance in zip(metrics, tolerances):

        GroupedDf = constructAccuracyDf(Blender_List, metric, tolerance) #list of 5

        AllDfs.extend(GroupedDf)
        sheetNames.extend(metric + "_" + str(i) for i in range(1, len(GroupedDf)+1))

    reference, ref_prefix = displayParams['reference'], displayParams['ref_prefix']

    if displayParams['archive']:
        import os

        path, folder, subFolder = DBpath, "RESULTS/", ref_prefix + '_Combined/' + 'RECORDS'+ '/'
        outputPathStudy = path + folder + subFolder

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        with pd.ExcelWriter(outputPathStudy + ref_prefix +"_AccuracyComparison" +  ".xlsx", mode='w') as writer:
            for df, sh in zip(AllDfs, sheetNames):
                df.to_excel(writer, sheet_name=sh, freeze_panes=(0, 1))

def trackPrediction(GS, baseModel = True):

    predictor = GS.Estimator

    rounding = 3
    if baseModel:
        learningDf = GS.learningDf
        XTrain, yTrain = learningDf.XTrain.to_numpy(), learningDf.yTrain.to_numpy().ravel()
        XTest, yTest = learningDf.XTest.to_numpy(), learningDf.yTest.to_numpy().ravel()

    else:
        learningDf = GS.modelList[0].learningDf
        XTrain, yTrain = learningDf.XTrain.to_numpy(), learningDf.yTrain.to_numpy().ravel()
        XTest, yTest = learningDf.XTest.to_numpy(), learningDf.yTest.to_numpy().ravel()

        XTrain = formatDf_toBlender(XTrain, GS, Scale=True).values
        XTest = formatDf_toBlender(XTest, GS, Scale=True).values


    yPred = predictor.predict(XTest)

    TrainScore = round(predictor.score(XTrain, yTrain), rounding)
    TestScore = round(predictor.score(XTest, yTest), rounding)
    TestAcc = round(computeAccuracy(yTest, predictor.predict(XTest), GS.accuracyTol), rounding)
    TestAcc_std = round(computeAccuracy_std(yTest, predictor.predict(XTest), learningDf, GS.accuracyTol_std), rounding)
    TestAcc_mean = round(computeAccuracy_mean(yTest, predictor.predict(XTest), learningDf, GS.accuracyTol_mean), rounding)

    TestMSE = round(mean_squared_error(yTest, yPred), rounding)
    TestR2 = round(r2_score(yTest, yPred), rounding)
    Resid = yTest - yPred

    PredictionDict = dict()
    PredictionDict['GS.GSName'] = GS.GSName
    PredictionDict['GS.XTrain.shape'] = XTrain.shape
    PredictionDict['GS.XTest.shape'] = XTest.shape
    PredictionDict['yPred'] = yPred
    PredictionDict['yTest'] = yTest
    PredictionDict['Resid'] = Resid

    PredictionDict['TrainScore'] = TrainScore
    PredictionDict['TestScore'] = TestScore
    PredictionDict['TestMSE'] = TestMSE
    PredictionDict['TestAcc'] = TestAcc
    PredictionDict['TestR2'] = TestR2
    PredictionDict['TestAcc_std'] = TestAcc_std
    PredictionDict['TestAcc_mean'] = TestAcc_mean

    return yPred, PredictionDict

def computePrediction_NBest(CV_BlenderNBest):

    for BlenderNBest in CV_BlenderNBest:
        for Model in BlenderNBest.modelList:
            yPred, PredictionDict = trackPrediction(Model)