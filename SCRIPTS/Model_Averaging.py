

from GridsearchParamPt import GS_ParameterPlot2D




def avgModel(example, AvgDict):

    new = example
    for learningDflabel in example.learningDfsList: #6
        GS = new.__getattribute__(learningDflabel) #54

        columns = ['TestAcc-Mean', 'TestAcc-Std', 'TestMSE-Mean', 'TestMSE-Std', 'Resid-Mean', 'Resid-Std',
                   'ResidVariance-Mean', 'ResidVariance-Std', 'TrainScore-Mean', 'TrainScore-Std', 'TestR2-Mean',
                   'TestR2-Std']

        TestAcc, TestAccStd, TestMSE, TestMSEStd, Resid, ResidStd, ResidVariance, \
        ResidVarianceStd, TrainScore, TrainScoreStd, TestR2, TestR2Std = AvgDict.loc[GS.GSName, :]


        setattr(GS, 'TrainScore', TrainScore)
        setattr(GS,'TestScore', TestR2)
        setattr(GS,'TestAcc', TestAcc)
        setattr(GS,'TestMSE', TestMSE)
        setattr(GS,'TestR2', TestR2)
        setattr(GS,'ResidMean', Resid)
        setattr(GS,'ResidVariance', ResidVariance)

        #todo : add > pickle dump new GSFS

    return new


# todo : run as in MainGS FS steps > for all keys
GS_ParameterPlot2D(GS_FSs, displayParams, DBpath, content='GS_FS', yLim=None, score='TestAcc', studyFolder='GS_FS/', combined = True)










