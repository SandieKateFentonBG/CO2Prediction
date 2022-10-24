import numpy as np

def pickleDumpMe( DBpath, displayParams, obj, objFolder, objName):
    #objFolder = DATA; FILTER; WRAPPER; GS
    if displayParams['archive']:
        reference = displayParams['reference']
        import os
        import pickle
        outputFigPath = DBpath + "RESULTS/" + reference + 'RECORDS/' + objFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)
        outputFigPath = f'{outputFigPath}/{objName}.pkl'

        with open(outputFigPath, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('FILE has been saved here :', outputFigPath)


def pickleLoadMe(path, show = False):
    import pickle
    with open(path, 'rb') as handle:
        obj = pickle.load(handle)
    if show:
        print(obj)
    return obj


def saveStudy(DBpath, displayParams, obj, objFolder = 'report'):

    import inspect
    if displayParams['archive']:
        import os
        reference = displayParams['reference']
        outputPathStudy = DBpath + "RESULTS/" + reference + 'RECORDS/' + objFolder + '/'

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        with open(outputPathStudy  + "Archive.txt", 'a') as f:
            print('', file=f)
            test = inspect.getmembers(obj)
            for r in test:
                print(r, file=f)

        f.close()

def reportFeatureSelection(DBpath, displayParams, df, learningDf, baseFormatedDf, spearmanFilter, RFEs, objFolder ='REPORT'):

    if displayParams['archive']:
        import os
        reference = displayParams['reference']
        outputPathStudy = DBpath + "RESULTS/" + reference + 'RECORDS/' + objFolder + '/'

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        with open(outputPathStudy + "FeatureSelection.txt", 'w', encoding='UTF8', newline='') as e:
            import csv
            writer = csv.writer(e, delimiter=";")

            writer.writerow(['DATA'])

            writer.writerow(["Full df", df.shape])
            writer.writerow(["Outliers removed ", learningDf.shape])
            writer.writerow(['FORMAT'])
            writer.writerow(["train", type(baseFormatedDf.trainDf), baseFormatedDf.trainDf.shape])
            writer.writerow(["validate", type(baseFormatedDf.valDf), baseFormatedDf.valDf.shape])
            writer.writerow(["test", type(baseFormatedDf.testDf), baseFormatedDf.testDf.shape])
            writer.writerow(['FILTER'])
            writer.writerow(['FILTER - SPEARMAN CORRELATION'])
            writer.writerow(['LABELS : ', spearmanFilter.trainDf.shape, type(spearmanFilter.trainDf)])
            writer.writerow([spearmanFilter.selectedLabels])
            writer.writerow(['RFE'])
            for RFE in RFEs:
                writer.writerow(["RFE with:", RFE.method])
                writer.writerow(["Number of features fixed:", RFE.n_features_to_select])
                writer.writerow(["Score on training", RFE.rfe_trainScore])
                writer.writerow(['Selected feature labels', list(RFE.selectedLabels)])
                writer.writerow(["Score on validation", RFE.rfe_valScore])
                writer.writerow([""])


        e.close()


def reportModels(DBpath, displayParams, models, learningDf, objFolder ='REPORT', display = True):

    if displayParams['archive']:
        import os
        reference = displayParams['reference']
        outputPathStudy = DBpath + "RESULTS/" + reference + 'RECORDS/' + objFolder + '/'

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        with open(outputPathStudy + "Models.txt", 'w', encoding='UTF8', newline='') as e:
            import csv
            writer = csv.writer(e, delimiter=";")

            writer.writerow(['MODEL GRIDSEARCH'])
            writer.writerow([''])
            for m in models:

                writer.writerow(['Model :', m.predictorName])
                writer.writerow(['calibrated with data :', learningDf.trainDf.shape])
                writer.writerow(['from feature selection :', learningDf.selector])
                writer.writerow(['GS params :', m.Param])
                writer.writerow(['GS TrainScore :', m.TrainScore])
                writer.writerow(['GS TestScore :', m.TestScore])
                writer.writerow(['GS TestAcc :', m.TestAcc])
                writer.writerow(['GS TestMSE :', m.TestMSE])
                writer.writerow(['GS TestR2 :', m.TestR2])
                writer.writerow(['GS Resid - Mean/std :', np.mean(m.Resid), np.std(m.Resid)])
                writer.writerow(['GS Resid - Min/Max :', min(m.Resid), max(m.Resid)])
                writer.writerow(['GS Resid :', m.Resid])
                writer.writerow([''])

        e.close()

        if display :

            print('MODEL GRIDSEARCH')
            print('')
            for m in models:

                print('Model :', m.predictorName)
                print('calibrated with data :', learningDf.trainDf.shape)
                print('from feature selection :', learningDf.selector)
                print('GS params :', m.Param)
                print('GS TrainScore :', m.TrainScore)
                print('GS TestScore :', m.TestScore)
                print('GS TestAcc :', m.TestAcc)
                print('GS TestMSE :', m.TestMSE)
                print('GS TestR2 :', m.TestR2)
                print('GS Resid - Mean/std :', np.mean(m.Resid), np.std(m.Resid))
                print('GS Resid - Min/Max :', min(m.Resid), max(m.Resid))
                print('GS Resid :', m.Resid)
                print('')

def reportGridsearch (DBpath, displayParams, modelGS, objFolder ='REPORT', display = True):

    if displayParams['archive']:
        import os
        reference = displayParams['reference']
        outputPathStudy = DBpath + "RESULTS/" + reference + 'RECORDS/' + objFolder + '/'

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        with open(outputPathStudy + modelGS.predictorName + ".txt", 'w', encoding='UTF8', newline='') as e:
            import csv
            writer = csv.writer(e, delimiter=";")

            writer.writerow(['MODEL x FEATURE SELECTION GRIDSEARCH'])
            writer.writerow(['>>', modelGS.predictorName])
            writer.writerow([''])

            for DfLabel in modelGS.learningDfsList:
                writer.writerow(['Model :', modelGS.predictorName])
                GS = getattr(modelGS, DfLabel)
                writer.writerow(['calibrated with data :', GS.learningDf.trainDf.shape])
                writer.writerow(['from feature selection  :', DfLabel])
                writer.writerow(['GS params :', GS.Param])
                writer.writerow(['GS TrainScore :', GS.TrainScore])
                writer.writerow(['GS TestScore :', GS.TestScore])
                writer.writerow(['GS TestAcc :', GS.TestAcc])
                writer.writerow(['GS TestMSE :', GS.TestMSE])
                writer.writerow(['GS TestR2 :', GS.TestR2])
                writer.writerow(['GS Resid - Mean/std :', np.mean(GS.Resid), np.std(GS.Resid)])
                writer.writerow(['GS Resid - Min/Max :', min(GS.Resid), max(GS.Resid)])
                writer.writerow(['GS Resid :', GS.Resid])
                writer.writerow([''])

        e.close()

        if display:

            print('MODEL x FEATURE SELECTION GRIDSEARCH')
            print('>>Model :', modelGS.predictorName)
            print('')

            for DfLabel in modelGS.learningDfsList:
                print('Model :', modelGS.predictorName)
                GS = getattr(modelGS, DfLabel)
                print('calibrated with data :', GS.learningDf.trainDf.shape)
                print('from feature selection  :', DfLabel)
                print('GS params :', GS.Param)
                print('GS TrainScore :', GS.TrainScore)
                print('GS TestScore :', GS.TestScore)
                print('GS TestAcc :', GS.TestAcc)
                print('GS TestMSE :', GS.TestMSE)
                print('GS TestR2 :', GS.TestR2)
                print('GS Resid - Mean/std :', np.mean(GS.Resid), np.std(GS.Resid))
                print('GS Resid - Min/Max :', min(GS.Resid), max(GS.Resid))
                print('GS Resid :', GS.Resid)
                print('')
