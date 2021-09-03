from typing import Tuple
import pathmagic
from Algorithm.Algorithm import Algorithm
from Model.LinearRegressionModel import LinearRegressionModel
import Utils
import matplotlib.pyplot as plt
import numpy as np

class BackwardStepwiseSelection(Algorithm):

    def result(self):
        return self.__result

    def description(self):
        return "Backward Stepwise Selection"

    def run(self, debug=False):

        bestModelsForEachIteration = []
        inputVariablesNames = list(self.inputDataSet.columns)
        quantityOfInputVariables = len(inputVariablesNames)
        selectedVariables = inputVariablesNames.copy()

        bestModelsForEachIteration.append(LinearRegressionModel(self.inputDataSet, self.outputDataSet))

        for iteration in range(quantityOfInputVariables-1):

            currentIterationModels = []
            selectedVariables = inputVariablesNames.copy()

            for inputVariableName in inputVariablesNames:
                selectedVariables.remove(inputVariableName)
                model = LinearRegressionModel(self.inputDataSet[selectedVariables], self.outputDataSet)
                selectedVariables.append(inputVariableName)
                currentIterationModels.append(model)
        
            bestCurrentModel = Utils.pickModelWithHighestAdjustedR2(currentIterationModels)
            bestModelsForEachIteration.append(bestCurrentModel)
            newVariableNameToDelete = list(set(selectedVariables) - set(model.inputVariablesNames()))[0]           
            inputVariablesNames.remove(newVariableNameToDelete)

            if debug:
                print("Iteration: ", iteration, "Removed Variable: ", newVariableNameToDelete, "Model Saved:", bestCurrentModel)

        bestModelsForEachIteration.append(LinearRegressionModel(self.inputDataSet[selectedVariables], self.outputDataSet))
        self.__result = bestModelsForEachIteration

    def plot(self):
        self.run()
        plt.clf()

        plt.xlabel("Iterations")
        plt.ylabel("R2")
        plt.title("Backward Stepwise Selection")

        iterations = np.arange(len(self.result()))

        R2Values = []

        for model in self.result():
            R2Values.append(model.adjustedR2())

        R2Values = np.array(R2Values)

        plt.scatter(iterations, R2Values)
        plt.plot(iterations, R2Values)
        plt.show()