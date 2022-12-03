
RFEEstimators = {'LinearRegression': LinearRegression(),
                 'DecisionTreeRegressor': DecisionTreeRegressor(random_state = rs),
                  'RandomForestRegressor' : RandomForestRegressor(random_state = rs),
                  'GradientBoostingRegressor' : GradientBoostingRegressor(random_state = rs),
                  'DecisionTreeClassifier' : DecisionTreeClassifier(random_state = rs)}


class FilterSelection:

    def __init__(self, name, estimator, selectedFeatures, trainscore, testscore):

        self.name = name
        self.estimator = estimator
        self.selectedFeatures = selectedFeatures
        self.trainScore = trainscore
        self.testScore = testscore

class RFESelection:

    def __init__(self, name, estimator, selectedFeatures, trainscore, testscore):

        self.name = name
        self.estimator = estimator
        self.selectedFeatures = selectedFeatures
        self.trainScore = trainscore
        self.testScore = testscore


example = Selection(name, estimator, selectedFeatures, trainscore, testscore)

LinearRegression = Selection(name, estimator, selectedFeatures, trainscore, testscore)

allSolutions = [Selection(name, estimator, selectedFeatures, trainscore, testscore) for name in RFEEstimators]
