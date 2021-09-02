import Model
import statsmodels.api as sm
import Utils

class Algorithm():

    def __init__(self, inputDataset, outputDataset):
        self.__result = None
        self.inputDataSet = inputDataset
        self.outputDataSet = outputDataset

    def result(self):
        return self.__result

    def run(self):
        pass

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
                model = Model.LinearRegressionModel(self.inputDataSet[selectedVariables+[inputVariableName]], self.outputDataSet)
                currentIterationModels.append(model)
            
            bestCurrentModel = Utils.pickModelWithHighestAdjustedR2(currentIterationModels)
            bestModelsForEachIteration.append(bestCurrentModel)
            newVariableNameToSelect = list(set(model.inputVariablesNames()).difference(set(selectedVariables)))[0]
            selectedVariables.append(newVariableNameToSelect)
            inputVariablesNames.remove(newVariableNameToSelect)
            
            if debug:
                print("Iteration: ", iteration, "Selected Variables: ", selectedVariables)

        self.__result = bestModelsForEachIteration
