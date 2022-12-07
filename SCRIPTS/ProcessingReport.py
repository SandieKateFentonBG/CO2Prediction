def reportProcessing(DBpath, displayParams, df, learningDf, baseFormatedDf, FiltersLs, RFEs, objFolder ='FS', display = True):

    if displayParams['archive']:
        import os
        reference = displayParams['reference']
        outputPathStudy = DBpath + "RESULTS/" + reference + 'RECORDS/' + objFolder + '/'

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        with open(outputPathStudy + "FeatureSelection.txt", 'w', encoding='UTF8', newline='') as e:
            import csv
            writer = csv.writer(e, delimiter=":")

            writer.writerow(['DATA'])

            writer.writerow(["Full df ", df.shape])
            writer.writerow(["Outliers removed ", learningDf.shape])
            writer.writerow('')
            writer.writerow(['FORMAT'])
            writer.writerow(["train", type(baseFormatedDf.trainDf), baseFormatedDf.trainDf.shape])
            writer.writerow(["validate", type(baseFormatedDf.valDf), baseFormatedDf.valDf.shape])
            writer.writerow(["test", type(baseFormatedDf.testDf), baseFormatedDf.testDf.shape])
            writer.writerow([''])
            writer.writerow(['FILTER'])
            for filter in FiltersLs :
                writer.writerow(['FILTER ', filter.method])
                writer.writerow(['LABELS ', filter.trainDf.shape, type(filter.trainDf)])
                writer.writerow([filter.selectedLabels])
                writer.writerow('')
            writer.writerow(['RFE'])
            for RFE in RFEs:
                writer.writerow(["RFE with  ", RFE.method])
                writer.writerow(["Number of features fixed ", RFE.n_features_to_select])
                writer.writerow(["Score on training ", RFE.rfe_trainScore])
                writer.writerow(['Selected feature labels ', list(RFE.selectedLabels)])
                writer.writerow(["Score on validation ", RFE.rfe_valScore])
                writer.writerow('')

        e.close()

        if display :

            print('DATA')

            print("Full df :", df.shape)
            print("Outliers removed :",learningDf.shape)
            print('')
            print('FORMAT')
            print("train", type(baseFormatedDf.trainDf), baseFormatedDf.trainDf.shape)
            print("validate", type(baseFormatedDf.valDf), baseFormatedDf.valDf.shape)
            print("test", type(baseFormatedDf.testDf), baseFormatedDf.testDf.shape)
            print('')
            print('FILTER')
            for filter in FiltersLs :
                print('FILTER :', filter.method)
                print('LABELS :', filter.trainDf.shape, type(filter.trainDf))
                print(filter.selectedLabels)
                print('')
            print('RFE')
            for RFE in RFEs:
                print("RFE with :" , RFE.method)
                print("Number of features fixed :", RFE.n_features_to_select)
                print("Score on training :", RFE.rfe_trainScore)
                print('Selected feature labels :', list(RFE.selectedLabels))
                print("Score on validation :", RFE.rfe_valScore)
                print('')

def dfAsTable (DBpath, displayParams, df, objFolder ='DATA', name = "DF"):

    import pandas as pd

    if displayParams['archive']:
        import os
        reference = displayParams['reference']
        outputPathStudy = DBpath + "RESULTS/" + reference + 'RECORDS/' + objFolder + '/'

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        with pd.ExcelWriter(outputPathStudy + name + ".xlsx", mode='w') as writer:

                df.to_excel(writer)