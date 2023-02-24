import pandas as pd

def formatDf(data, model):
    XDf = data
    if model.learningDf.droppedLabels != '':
        droppedLabels = model.learningDf.droppedLabels
        XDf = data.drop(columns=droppedLabels)
    return XDf

def formatDf_toBlender(data, blender, Scale = True):

        #create meta learning data
        blend_sample_sets = []

        for model in blender.modelList:

            XDf = formatDf(data, model)
            blend_train_i = model.Estimator.predict(XDf)
            blend_train_i = pd.DataFrame(blend_train_i)
            blend_sample_sets.append(blend_train_i)

        blendXDf = pd.concat(blend_sample_sets, axis=1)

        if Scale:
            blendXDf = (blendXDf - blender.ScaleMean) / blender.ScaleStd

        return blendXDf


def formatDf_toModellist(data, modelList):
    # create meta learning data
    blend_sample_sets = []

    for model in modelList:

        XDf = formatDf(data, model).to_numpy()
        blend_train_i = model.Estimator.predict(XDf)
        blend_train_i = pd.DataFrame(blend_train_i)
        blend_sample_sets.append(blend_train_i)

    blendXDf = pd.concat(blend_sample_sets, axis=1)

    return blendXDf

def unpackGS_FSs(GS_FSs, remove = ''):
    Model_List = []
    for GS_FS in GS_FSs: #9
        for learningDflabel in GS_FS.learningDfsList: #6
            GS = GS_FS.__getattribute__(learningDflabel) #54
            if GS.predictorName != remove:
                Model_List.append(GS)
    return Model_List


def repackGS_FSs(Model_List): #10 list de 54

    models_ls = [] #54 list de 10
    for i in range(len(Model_List[0])):  # 54
        seeds_ls = []
        for j in range(len(Model_List)):  # 10
             #54
            seeds_ls.append(Model_List[j][i])
        models_ls.append(seeds_ls)
    print("should be 54", len(models_ls))
    print("should be 10", len(models_ls[0]))

    return models_ls
