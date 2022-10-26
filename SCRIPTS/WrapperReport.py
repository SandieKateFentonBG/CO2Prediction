def reportRFE(DBpath, displayParams, RFEs, objFolder ='FS', display = True):

    if displayParams['archive']:
        import os
        reference = displayParams['reference']
        outputPathStudy = DBpath + "RESULTS/" + reference + 'RECORDS/' + objFolder + '/'

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        with open(outputPathStudy + "RFE.txt", 'w', encoding='UTF8', newline='') as e:
            import csv
            writer = csv.writer(e, delimiter=";")
            writer.writerow('')
            writer.writerow(['RECURSIVE FEATURE ELIMINATION'])

            for RFE in RFEs:
                writer.writerow(["RFECV with %s :" % RFE.method])
                writer.writerow(["Number of features from CV %s :" % RFE.rfecv.n_features_])
                writer.writerow(["Score on training %s :" % RFE.rfecv_trainScore])
                writer.writerow(["Selected feature labels %s :" % list(RFE.rfecv_selectedLabels)])
                writer.writerow(["Score on validation %s :" % RFE.rfecv_valScore])
                writer.writerow('')

                writer.writerow(["RFE Param Search with %s :" % RFE.method])
                writer.writerow(["Number of features compared %s :" % RFE.rfeHyp_featureCount])
                writer.writerow(["Score on training %s :" % RFE.rfeHyp_trainScore])
                writer.writerow(["Score on validation %s :" % RFE.rfeHyp_valScore])
                writer.writerow('')

                writer.writerow(["RFE with %s :" % RFE.method])
                writer.writerow(["Number of features fixed %s :" % RFE.n_features_to_select])
                writer.writerow(["Score on training %s :" % RFE.rfe_trainScore])
                writer.writerow(["Selected feature labels %s :" % list(RFE.selectedLabels)])
                writer.writerow(["Score on validation %s :" % RFE.rfe_valScore])
                writer.writerow('')

        e.close()

        if display :

            for RFE in RFEs:
                print("RFECV with:", RFE.method)
                print("Number of features from CV:", RFE.rfecv.n_features_)
                print("Score on training", RFE.rfecv_trainScore)
                print('Selected feature labels', list(RFE.rfecv_selectedLabels))
                print("Score on validation", RFE.rfecv_valScore)
                print('')

                print("RFE Param Search with:", RFE.method)
                print("Number of features compared", RFE.rfeHyp_featureCount)
                print("Score on training", RFE.rfeHyp_trainScore)
                print("Score on validation", RFE.rfeHyp_valScore)
                print('')

                print("RFE with:", RFE.method)
                print("Number of features fixed:", RFE.n_features_to_select)
                print("Score on training", RFE.rfe_trainScore)
                print('Selected feature labels', list(RFE.selectedLabels))
                print("Score on validation", RFE.rfe_valScore)
                print('')