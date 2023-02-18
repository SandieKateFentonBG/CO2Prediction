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
