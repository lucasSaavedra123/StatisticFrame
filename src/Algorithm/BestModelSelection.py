from statisticframe.Algorithm.Algorithm import Algorithm
from statisticframe.Model.LinearRegressionModel import LinearRegressionModel
import statisticframe.Utils.Utils as Utils


class BestModelSelection(Algorithm):

    def result(self):
        return self.__result

    def description(self):
        return 'Best Model Selection'

    def run(self, debug=False):

        models = []
        inputVariablesNames = list(self.inputDataSet.columns)
        combinationsOfVariables = Utils.powerset(inputVariablesNames)
        combinationsOfVariables.remove([])

        for combinationOfVariables in combinationsOfVariables:
            model = LinearRegressionModel(self.inputDataSet[combinationOfVariables], self.outputDataSet)
            models.append(model)

        self.__result = models
