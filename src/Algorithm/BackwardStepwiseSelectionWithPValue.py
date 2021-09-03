import pathmagic
from Algorithm.Algorithm import Algorithm
from Model.LinearRegressionModel import LinearRegressionModel
import Utils


class BackwardStepwiseSelectionWithPValue(Algorithm):

    def result(self):
        return self.__result

    def run(self, debug=False):

        bestModelsForEachIteration = []
        selectedVariables =  list(self.inputDataSet.columns)
        quantityOfInputVariables = len(selectedVariables)

        for iteration in range(quantityOfInputVariables):
            model = LinearRegressionModel(self.inputDataSet[selectedVariables], self.outputDataSet)
            bestModelsForEachIteration.append(model)
            highestPValueVariableName = model.highestPValueVariableName()
            selectedVariables.remove(highestPValueVariableName)
            
        self.__result = bestModelsForEachIteration
