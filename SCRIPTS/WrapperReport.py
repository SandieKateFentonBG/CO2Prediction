def reportRFE(DBpath, displayParams, RFEs, objFolder ='FS', display = True, process = 'short'):

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
                if process == 'long':

                    writer.writerow(["RFECV with %s :" % RFE.method])
                    writer.writerow(["Number of features from CV %s :" % RFE.rfecv.n_features_])
                    writer.writerow(["Score on validation %s :" % RFE.rfecv_valScore])
                    writer.writerow(["Selected feature labels %s :" % list(RFE.rfecv_selectedLabels)])
                    writer.writerow(["Score on check %s :" % RFE.rfecv_checkScore])
                    writer.writerow('')

                    writer.writerow(["RFE Param Search with %s :" % RFE.method])
                    writer.writerow(["Number of features compared %s :" % RFE.rfeHyp_featureCount])
                    writer.writerow(["Score on validation %s :" % RFE.rfeHyp_valScore])
                    writer.writerow(["Score on check %s :" % RFE.rfeHyp_checkScore])
                    writer.writerow('')

                    writer.writerow(["RFE with %s :" % RFE.method])
                    writer.writerow(["Number of features fixed %s :" % RFE.n_features_to_select])
                    writer.writerow(["Score on validation %s :" % RFE.rfe_valScore])
                    writer.writerow(["Selected feature labels %s :" % list(RFE.selectedLabels)])
                    writer.writerow(["Score on check %s :" % RFE.rfe_checkScore])
                    writer.writerow('')

                else :
                    writer.writerow(["RFE with %s :" % RFE.method])
                    writer.writerow(["Number of features based on %s :" % RFE.FtCountFrom])
                    writer.writerow(["Number of features  %s :" % RFE.n_features_to_select])
                    writer.writerow(["Score on validation %s :" % RFE.rfe_valScore])
                    writer.writerow(["Selected feature labels %s :" % list(RFE.selectedLabels)])
                    writer.writerow(["Score on check %s :" % RFE.rfe_checkScore])
                    writer.writerow('')

        e.close()

        if display :

            for RFE in RFEs:
                if process == 'long':
                    print("RFECV with:", RFE.method)
                    print("Number of features from CV:", RFE.rfecv.n_features_)
                    print("Score on validation", RFE.rfecv_valScore)
                    print('Selected feature labels', list(RFE.rfecv_selectedLabels))
                    print("Score on check", RFE.rfecv_checkScore)
                    print('')

                    print("RFE Param Search with:", RFE.method)
                    print("Number of features compared", RFE.rfeHyp_featureCount)
                    print("Score on validation", RFE.rfeHyp_valScore)
                    print("Score on check", RFE.rfeHyp_checkScore)
                    print('')

                    print("RFE with:", RFE.method)
                    print("Number of features fixed:", RFE.n_features_to_select)
                    print("Score on validation", RFE.rfe_valScore)
                    print('Selected feature labels', list(RFE.selectedLabels))
                    print("Score on check", RFE.rfe_checkScore)
                    print('')

                else :
                    print("RFE with:", RFE.method)
                    print("Number of features based on :", RFE.FtCountFrom)
                    print("Number of features :", RFE.n_features_to_select)
                    print("Score on validation", RFE.rfe_valScore)
                    print('Selected feature labels', list(RFE.selectedLabels))
                    print("Score on check", RFE.rfe_checkScore)
                    print('')