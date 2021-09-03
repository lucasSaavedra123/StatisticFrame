import pathmagic
from Algorithm.Algorithm import Algorithm
from Model.LinearRegressionModel import LinearRegressionModel
import Utils
import matplotlib.pyplot as plt
import numpy as np


class BackwardStepwiseSelectionWithPValue(Algorithm):

    def result(self):
        return self.__result

    def description(self):
        return "Backward Stepwise Selection with P Value"

    def run(self, debug=False):

        bestModelsForEachIteration = []
        selectedVariables =  list(self.inputDataSet.columns)
        quantityOfInputVariables = len(selectedVariables)

        for iteration in range(quantityOfInputVariables-1):
            model = LinearRegressionModel(self.inputDataSet[selectedVariables], self.outputDataSet)
            bestModelsForEachIteration.append(model)
            highestPValueVariableName = model.highestPValueVariableName()
            selectedVariables.remove(highestPValueVariableName)         

        model = LinearRegressionModel(self.inputDataSet[selectedVariables], self.outputDataSet)
        bestModelsForEachIteration.append(model)

        self.__result = bestModelsForEachIteration
