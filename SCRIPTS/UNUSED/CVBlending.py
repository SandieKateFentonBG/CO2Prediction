
# todo : find how to split training and testing without any data leakage?

class CVBlend:

    def __init__(self, CV_AllModels): #,, blendingConstructor NBestScore, NCount, Gridsearch = True, Val = False

        self.CV_AllModels = CV_AllModels
        self.repackGS_FS(CV_AllModels)

        for name in self.Model_GSNames:
            print(name)
            for elem in self.__getattribute__(name):
                print(elem.GSName)

        # self.GSName = blendingConstructor['name'] + '_Blender'
        # self.predictorName = blendingConstructor['name']  # ex : SVR
        # self.modelPredictor = blendingConstructor['modelPredictor']
        # self.param_dict = blendingConstructor['param_dict']
        # self.scoring = {'neg_mean_squared_error': 'neg_mean_squared_error', 'r2': 'r2'}
        # self.refit = 'r2' #Score used for refitting the blender

        # self.Train_GS_FS = CV_AllModels[:-2] #" train on 80%"
        # self.Test_GS_FS = CV_AllModels[-2:] #" test on 20 %"
        #
        # self.yTrain = modelList[0].learningDf.yTrain.to_numpy().ravel() #yTrain is the same for every model
        # self.yTest = modelList[0].learningDf.yTest.to_numpy().ravel() #yTest is the same for every model

        # #create meta learning data
        # blend_train_sets = []
        # blend_test_sets = []
        # blend_val_sets = []
        #
        # for model in modelList:
        #
        #     predictor = model.Estimator
        #     learningDf = model.learningDf
        #
        #     rawXVal, rawyVal = learningDf.XVal.to_numpy(), learningDf.yVal.to_numpy().ravel()
        #     rawXTrain, rawyTrain = learningDf.XTrain.to_numpy(), learningDf.yTrain.to_numpy().ravel() #todo : changed here
        #     rawXTest, rawyTest = learningDf.XTest.to_numpy(), learningDf.yTest.to_numpy().ravel()
        #
        #     blend_train_i = predictor.predict(rawXTrain) #dim 400*1
        #     blend_test_i = predictor.predict(rawXTest) #dim 20*1
        #     blend_val_i = predictor.predict(rawXVal)  # dim 20*1
        #
        #     blend_train_i = pd.DataFrame(blend_train_i)
        #     blend_test_i = pd.DataFrame(blend_test_i)
        #     blend_val_i = pd.DataFrame(blend_val_i)
        #
        #     blend_train_sets.append(blend_train_i) #dim 400*i
        #     blend_test_sets.append(blend_test_i) #dim 20*i
        #     blend_val_sets.append(blend_val_i)  # dim 20*i



    def repackGS_FS(self, CV_AllModels):
            CV_Unpacked = []
            for run in CV_AllModels: #10
                Model_List,Model_GSNames = unpackModels(run, remove='') #54 models
                CV_Unpacked.append(Model_List) #10

            self.Model_GSNames = Model_GSNames

            for name in self.Model_GSNames:
                setattr(self, name, [])

            for Model_List in CV_Unpacked:  # 10
                for i in range(len(self.Model_GSNames)):  # 54
                    self.__getattribute__(self.Model_GSNames[i]).append(Model_List[i])


            for name in Model_GSNames:
                for elem in self.__getattribute__(name):
                    print("all", elem.GSName)


def unpackModels(GS_FSs, remove = ''):

    Model_List, Model_GSNames = [], []
    for GS_FS in GS_FSs:
        for learningDflabel in GS_FS.learningDfsList:
            GS = GS_FS.__getattribute__(learningDflabel)
            name = GS.GSName
            if GS.predictorName != remove:
                Model_List.append(GS)
                Model_GSNames.append(name)

    return Model_List,Model_GSNames