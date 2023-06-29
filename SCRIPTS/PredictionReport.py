from HelpersVisualizer import *

def reportGS_TxtScores_All (DBpath, displayParams, modelGS, objFolder ='GS_FS', display = True):

    if displayParams['archive']:
        import os

        reference = displayParams['reference']
        outputPathStudy = DBpath + "RESULTS/" + reference + 'RECORDS/' + objFolder + '/'

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        with open(outputPathStudy + modelGS.predictorName + "_FS.txt", 'w', encoding='UTF8', newline='') as e:
            import csv
            writer = csv.writer(e, delimiter=";")

            writer.writerow(['MODEL x FEATURE SELECTION GRIDSEARCH'])
            writer.writerow(['>>', modelGS.predictorName])
            writer.writerow('')

            for DfLabel in modelGS.learningDfsList:
                writer.writerow(['Model ', modelGS.predictorName])
                GS = getattr(modelGS, DfLabel)
                writer.writerow(['calibrated with data ', GS.learningDf.trainDf.shape])
                writer.writerow(['from feature selection  ', DfLabel])
                writer.writerow(['GS params ', GS.Param])
                writer.writerow(['GS TrainScore ', GS.TrainScore])
                writer.writerow(['GS TestScore ', GS.TestScore])
                writer.writerow(['GS TestAcc ', GS.TestAcc])
                writer.writerow(['GS TestMSE ', GS.TestMSE])
                writer.writerow(['GS TestR2 ', GS.TestR2])
                writer.writerow(['GS Resid - Mean/std ', np.mean(GS.Resid), np.std(GS.Resid)])
                writer.writerow(['GS Resid - Min/Max ', min(GS.Resid), max(GS.Resid)])
                writer.writerow(['GS Resid ', GS.Resid])
                writer.writerow('')

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


def reportGS_Scores_All (DBpath, displayParams, GS_FSs, scoreList, display = False):

    if displayParams['archive']:
        import os
        reference = displayParams['reference']
        outputPathStudy = DBpath + "RESULTS/" + reference + 'RECORDS/'

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        scoreDfs = []
        for score in scoreList :
            scoreDf = GS_ConstructDataframe(GS_FSs, score)
            scoreDfs.append(scoreDf)
            if display :
                print('')
                print('Results for :', score)
                print(scoreDf)

        with pd.ExcelWriter(outputPathStudy + reference[:-1] + "_GS_Scores_All" + ".xlsx", mode='w') as writer:
            for scoreDf, score in zip(scoreDfs, scoreList):
                scoreDf.to_excel(writer, sheet_name=score)

