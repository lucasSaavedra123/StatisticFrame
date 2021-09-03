from numpy.core.arrayprint import printoptions
import pathmagic
from Algorithm.Algorithm import Algorithm
from Model.LinearRegressionModel import LinearRegressionModel
import Utils
import matplotlib.pyplot as plt
import numpy as np


class ForwardStepwiseSelection(Algorithm):

    def result(self):
        return self.__result

    def run(self, debug=False):

        bestModelsForEachIteration = []
        inputVariablesNames = list(self.inputDataSet.columns)
        quantityOfInputVariables = len(inputVariablesNames)
        selectedVariables = []

        for iteration in range(quantityOfInputVariables):

            currentIterationModels = []

            for inputVariableName in inputVariablesNames:
                model = LinearRegressionModel(self.inputDataSet[selectedVariables+[inputVariableName]], self.outputDataSet)
                currentIterationModels.append(model)
            
            
            bestCurrentModel = Utils.pickModelWithHighestAdjustedR2(currentIterationModels)
            bestModelsForEachIteration.append(bestCurrentModel)
            newVariableNameToSelect = list(set(bestCurrentModel.inputVariablesNames()).difference(set(selectedVariables)))[0]
            selectedVariables.append(newVariableNameToSelect)
            inputVariablesNames.remove(newVariableNameToSelect)
            
            if debug:
                print("Iteration: ", iteration, "Selected Variables: ", selectedVariables)

        self.__result = bestModelsForEachIteration


    def plot(self):
        self.run()
        plt.clf()

        plt.xlabel("Iterations")
        plt.ylabel("R2")
        plt.title("Forward Stepwise Selection")

        iterations = np.arange(len(self.result()))

        R2Values = []

        for model in self.result():
            R2Values.append(model.adjustedR2())

        R2Values = np.array(R2Values)

        plt.scatter(iterations, R2Values)
        plt.plot(iterations, R2Values)
        plt.show()