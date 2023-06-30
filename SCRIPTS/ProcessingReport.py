def reportProcessing(DBpath, displayParams, df, learningDf, baseFormatedDf, FiltersLs, RFEs, objFolder ='FS', display = True, combined=False, number=None):

    if displayParams['archive']:
        import os
        if combined:
            reference = displayParams['ref_prefix'] + '_Combined/'
        elif number :
            reference = displayParams['ref_prefix'] + '_rd' + str(number) + '/'
        else:
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
            writer.writerow(["test", type(baseFormatedDf.testDf), baseFormatedDf.testDf.shape])
            writer.writerow(["validate", type(baseFormatedDf.valDf), baseFormatedDf.valDf.shape])
            writer.writerow(["check", type(baseFormatedDf.checkDf), baseFormatedDf.checkDf.shape])
            writer.writerow([''])
            writer.writerow(['FILTER'])
            if len(FiltersLs)>0:
                for filter in FiltersLs :
                    writer.writerow(['FILTER ', filter.method])
                    writer.writerow(['LABELS ', filter.trainDf.shape, type(filter.trainDf)])
                    writer.writerow([filter.selectedDict])
                    writer.writerow('')
                writer.writerow(['RFE'])
            if len(RFEs)>0 :
                for RFE in RFEs:
                    writer.writerow(["RFE with  ", RFE.method])
                    writer.writerow(["Number of features fixed ", RFE.n_features_to_select])
                    writer.writerow(["Score on training ", RFE.rfe_valScore])
                    writer.writerow(['Selected feature labels ', list(RFE.selectedDict)])
                    writer.writerow(["Score on validation ", RFE.rfe_checkScore])
                    writer.writerow('')

        e.close()

        if display :

            print('DATA')

            print("Full df :", df.shape)
            print("Outliers removed :",learningDf.shape)
            print('')
            print('FORMAT')
            print("train", type(baseFormatedDf.trainDf), baseFormatedDf.trainDf.shape)
            print("test", type(baseFormatedDf.testDf), baseFormatedDf.testDf.shape)
            print("validate", type(baseFormatedDf.valDf), baseFormatedDf.valDf.shape)
            print("validate", type(baseFormatedDf.checkDf), baseFormatedDf.checkDf.shape)
            print('')
            print('FILTER')
            for filter in FiltersLs :
                print('FILTER :', filter.method)
                print('LABELS :', filter.trainDf.shape, type(filter.trainDf))
                print(filter.selectedDict)
                print('')
            print('RFE')
            for RFE in RFEs:
                print("RFE with :" , RFE.method)
                print("Number of features fixed :", RFE.n_features_to_select)
                print("Score on training :", RFE.rfe_valScore)
                print('Selected feature labels :', list(RFE.selectedDict))
                print("Score on validation :", RFE.rfe_checkScore)
                print('')

def dfAsTable (DBpath, displayParams, df, objFolder ='DATA', name = "DF", combined = False, number=None):

    import pandas as pd

    if displayParams['archive']:
        import os
        if combined :
            reference = displayParams['ref_prefix'] + '_Combined/'
        elif number :
            reference = displayParams['ref_prefix'] + '_rd' + str(number) + '/'
        else:
            reference = displayParams['reference']

        # reference = displayParams['reference']
        outputPathStudy = DBpath + "RESULTS/" + reference + 'RECORDS/' + objFolder + '/'

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        with pd.ExcelWriter(outputPathStudy + name + ".xlsx", mode='w') as writer:

                df.to_excel(writer)