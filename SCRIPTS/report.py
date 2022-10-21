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

def reportStudy(DBpath, displayParams, df, learningDf, baseFormatedDf, spearmanFilter, RFEs, GSs, KRR, objFolder = 'report'):

    if displayParams['archive']:
        import os
        reference = displayParams['reference']
        outputPathStudy = DBpath + "RESULTS/" + reference + 'RECORDS/' + objFolder + '/'

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        with open(outputPathStudy + "Archive.txt", 'w', encoding='UTF8', newline='') as e:
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

            writer.writerow(['GRIDSEARCH'])
            #todo : continue saving study info

        e.close()
