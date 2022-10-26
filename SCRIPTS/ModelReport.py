import numpy as np

def reportModels(DBpath, displayParams, models, learningDf, objFolder ='GS', display = True):

    if displayParams['archive']:
        import os
        reference = displayParams['reference']
        outputPathStudy = DBpath + "RESULTS/" + reference + 'RECORDS/' + objFolder + '/'

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        with open(outputPathStudy + "MODELS_GS.txt", 'w', encoding='UTF8', newline='') as e:
            import csv
            writer = csv.writer(e, delimiter=":")

            writer.writerow(['MODEL GRIDSEARCH'])
            writer.writerow('')
            for m in models:

                writer.writerow(['Model ', m.predictorName])
                writer.writerow(['calibrated with data ', learningDf.trainDf.shape])
                writer.writerow(['from feature selection ', learningDf.selector])
                writer.writerow(['GS params ', m.Param])
                writer.writerow(['GS TrainScore ', m.TrainScore])
                writer.writerow(['GS TestScore ', m.TestScore])
                writer.writerow(['GS TestAcc ', m.TestAcc])
                writer.writerow(['GS TestMSE ', m.TestMSE])
                writer.writerow(['GS TestR2 ', m.TestR2])
                writer.writerow(['GS Resid - Mean/std ', np.mean(m.Resid), np.std(m.Resid)])
                writer.writerow(['GS Resid - Min/Max ', min(m.Resid), max(m.Resid)])
                writer.writerow(['GS Resid ', m.Resid])
                writer.writerow('')

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