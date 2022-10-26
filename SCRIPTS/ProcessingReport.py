def reportProcessing(DBpath, displayParams, df, learningDf, baseFormatedDf, spearmanFilter, RFEs, objFolder ='FS', display = True):

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
            writer.writerow(['FILTER - SPEARMAN CORRELATION'])
            writer.writerow(['LABELS ', spearmanFilter.trainDf.shape, type(spearmanFilter.trainDf)])
            writer.writerow([spearmanFilter.selectedLabels])
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
            print('FILTER - SPEARMAN CORRELATION')
            print('LABELS :' , spearmanFilter.trainDf.shape, type(spearmanFilter.trainDf))
            print(spearmanFilter.selectedLabels)
            print('')
            print('RFE')
            for RFE in RFEs:
                print("RFE with :" , RFE.method)
                print("Number of features fixed :", RFE.n_features_to_select)
                print("Score on training :", RFE.rfe_trainScore)
                print('Selected feature labels :', list(RFE.selectedLabels))
                print("Score on validation :", RFE.rfe_valScore)
                print('')