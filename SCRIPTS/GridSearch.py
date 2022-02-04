from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

"""
Docum

Linear Models : https://scikit-learn.org/stable/modules/linear_model.html#elastic-net

examples : https://www.programcreek.com/python/example/91151/sklearn.model_selection.GridSearchCV

gridsearch doc : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV

kernel vs svr : https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_ridge_regression.html
"""


def computeAccuracy(yTrue, yPred):
    #https: // scikit - learn.org / stable / modules / model_evaluation.html  # scoring
    tolerance = 0.05
    validated = [1 if abs(yPred[i] - yTrue[i]) < abs(yTrue[i]) * tolerance else 0 for i in range(len(yTrue))]
    return sum(validated) / len(validated)

def plot(yTest, yPred):
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = [18, 18]
    l1, = plt.plot(yTest, 'g')
    l2, = plt.plot(yPred, 'r', alpha=0.7)
    plt.legend(['Ground truth', 'Predicted'])
    plt.show()

class GridSearch:

    """
    Goal - for each model
    - build it
    - run it with different parameters
    - store the best parameter and score
    """

    def __init__(self):
        self.linearReg = LinearRegression()
        self.lassoReg = Lasso() #for overfitting
        self.ridgeReg = Ridge()
        self.elasticNetReg = ElasticNet()
        self.supportVector = SVR()
        self.kernelRidgeReg = KernelRidge() #for underfitting
        #self.normalModel = buildNormalModel()

    def searchCV(self, model, paramkey, paramValues, xTrain, yTrain, xTest, yTest, custom = False):

        parameters = dict()
        parameters[paramkey] = paramValues
        if custom:
            score = make_scorer(computeAccuracy(), greater_is_better=True)
            grid = GridSearchCV(model, scoring=score, param_grid=parameters)

        else:
            grid = GridSearchCV(model, param_grid=parameters)
        grid.fit(xTrain, yTrain.ravel())
        # train = grid.score(xTrain, yTrain.ravel())
        # test = grid.score(xTest, yTest.ravel()) #why are these scores different to mean test score? what do they rpz?
        scores = grid.cv_results_['mean_test_score']
        # returns R2 for each model/trained parameter, evaluated on the test samples on training data
        # print(scores, train, test)
        print(model,  ": best parameters %s with a mean_test_score of %0.2f" % (grid.best_params_, grid.best_score_))
        return grid

    def run(self, modelWithParam, xTrain, yTrain, xTest, yTest, display):

        clf = modelWithParam
        clf.fit(xTrain, yTrain.ravel())
        scores = clf.score(xTest, yTest.ravel())
        yPred = clf.predict(xTest)
        # yPred = clf.predict(xTest).tolist()
        # rTrue = yTest.values.tolist()
        # yTrue = [item for sublist in rTrue for item in sublist]
        accuracy = computeAccuracy(yTest, clf.predict(xTest))
        mse = mean_squared_error(yTest, clf.predict(xTest))
        if display:
            plot(yTest, yPred)

        return clf, accuracy, mse

    def update(self, modelName, model, bestParameters):
        update = model(bestParameters)
        self.modelName = update
        # if modelName == 'linearReg':
        #     self.linearReg = update
        # if modelName == 'lassoReg':
        #     self.lassoReg = update
        # if modelName == 'ridgeReg':
        #     self.ridgeReg = update
        # if modelName == 'elasticNetReg':
        #     self.elasticNetReg = update
        # if modelName == 'supportVector':
        #     self.supportVector = update
        # if modelName == 'kernelRidgeReg':
        #     self.kernelRidgeReg = update
          #todo : how to do this in a generic way?
        pass

    def save(self):
        pass