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

    def plot(self):
        self.run()
        plt.clf()

        plt.xlabel("Iterations")
        plt.ylabel("R2")
        plt.title("Backward Stepwise Selection With P Value")

        iterations = np.arange(len(self.result()))

        R2Values = []

        for model in self.result():
            R2Values.append(model.adjustedR2())

        R2Values = np.array(R2Values)

        plt.scatter(iterations, R2Values)
        plt.plot(iterations, R2Values)
        plt.show()