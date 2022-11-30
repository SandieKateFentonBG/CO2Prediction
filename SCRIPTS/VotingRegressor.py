ModelLs = []
ModelLsName = []
for GS_FS in GS_FSs:
    for learningDflabel in GS_FS.learningDfsList:
        GS = GS_FS.__getattribute__(learningDflabel)
        ModelLs.append(GS.bModel)
        ModelLsName.append(GS.GSName)

ereg = VotingRegressor([(name, model) for (name,model) in zip(ModelLs, ModelLsName)])

ereg.fit(X, y)