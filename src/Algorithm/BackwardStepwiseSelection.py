import pathmagic
from Algorithm.Algorithm import Algorithm
from Model.LinearRegressionModel import LinearRegressionModel
import Utils


class BackwardStepwiseSelection(Algorithm):

    def result(self):
        return self.__result

    def run(self, debug=False):

        bestModelsForEachIteration = []
        inputVariablesNames = list(self.inputDataSet.columns)
        quantityOfInputVariables = len(inputVariablesNames)
        selectedVariables = inputVariablesNames.copy()

        for iteration in range(quantityOfInputVariables):

            currentIterationModels = []

            for inputVariableName in inputVariablesNames:
                if len(selectedVariables) == 1:
                    model = LinearRegressionModel(self.inputDataSet[selectedVariables], self.outputDataSet)
                    currentIterationModels.append(model)
                else:
                    selectedVariables.remove(inputVariableName)
                    model = LinearRegressionModel(self.inputDataSet[selectedVariables], self.outputDataSet)
                    selectedVariables.append(inputVariableName)
                    currentIterationModels.append(model)
            
            bestCurrentModel = Utils.pickModelWithHighestAdjustedR2(currentIterationModels)
            bestModelsForEachIteration.append(bestCurrentModel)

            if len(selectedVariables) != 1:
                newVariableNameToDelete = list(set(inputVariablesNames).difference(set(model.inputVariablesNames())))[0]            
                selectedVariables.remove(newVariableNameToDelete)
                inputVariablesNames.remove(newVariableNameToDelete)

            if debug:
                print("Iteration: ", iteration, "Selected Variables: ", selectedVariables)

        self.__result = bestModelsForEachIteration
