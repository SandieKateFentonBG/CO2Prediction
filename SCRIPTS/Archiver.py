def saveStudy(displayParams, Content):

    if displayParams['archive']:
        import os
        outputPathStudy = displayParams["outputPath"] + displayParams["reference"]
        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        with open(outputPathStudy + '/Records' + displayParams["reference"] + ".txt", 'a') as f:
            print('', file=f)
            if type(Content) == dict:
                for k,v in Content.items():
                    print(k, ":", v, file=f)
            else:
                for r in Content:
                    print(r, file=f)

        f.close()

def printStudy(displayParams, Content):

    if displayParams['showResults']:
        for m in Content:
            for k, v in m.items():
                print(k, ':', v)

def exportStudy(displayParams, inputData, prepData, modelsData, sortedModels):

    if displayParams['archive']:

        import os
        outputPathStudy = displayParams["outputPath"] + displayParams["reference"]
        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        import csv
        with open(outputPathStudy + '/Records_' + displayParams["reference"] + ".csv", 'w', encoding='UTF8', newline='') as e:
            writer = csv.writer(e, delimiter = ";")

            writer.writerow(['INPUT DATA'])
            for inputk, inputv in inputData.items():
                writer.writerow([inputk, inputv])
            writer.writerow('')

            writer.writerow(['PREPROCESSED DATA'])
            for inputk, inputv in prepData.items():
                writer.writerow([inputk, inputv])
            writer.writerow('')

            writer.writerow(['MODELS DATA'])
            k = modelsData[0].keys()
            writer.writerow(k)
            for i in range(len(modelsData)):
                v = modelsData[i].values()
                writer.writerow(v)

            writer.writerow(['SORTED MODELS DATA'])
            k = sortedModels[0].keys()
            writer.writerow(k)
            for i in range(len(sortedModels)):
                v = sortedModels[i].values()
                writer.writerow(v)

        e.close()

def saveInput(csvPath, outputPath, displayParams, xQualLabels, xQuantLabels, yLabels, processingParams, modelingParams, powers, mixVariables   ):

    Content = dict()
    Content['csvPath'] = csvPath
    Content['outputPath'] = outputPath
    Content['displayParams'] = displayParams
    Content['xQualLabels'] = xQualLabels
    Content['xQuantLabels'] = xQuantLabels
    Content['yLabels'] = yLabels
    Content['processingParams'] = processingParams
    Content['modelingParams'] = modelingParams
    Content['powers'] = powers
    Content['mixVariables'] = mixVariables

    saveStudy(displayParams, Content)

    return Content

def pickleDumpMe(displayParams, dic):

    if displayParams['archive']:
        import pickle
        import os
        outputPathStudy = displayParams["outputPath"] + displayParams["reference"]
        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        with open(outputPathStudy + '/Records', 'wb') as handle:
        # with open(set_up_dict['training_settings'].output_path + set_up_dict['reference'] + name, 'wb') as handle:
            pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('FILE has been saved here :', outputPathStudy + '/Records')


def pickleLoadMe(path, name = '/Records', show = False):
    import pickle
    with open(path + name, 'rb') as handle:
        mydict = pickle.load(handle)
    if show:
        print(name)
        for k, v in mydict.items():
                print(' ', k, ' : ', v)
    return mydict