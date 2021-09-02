import Model
import statsmodels.api as sm

class Algorithm():

    def __init__(self, inputDataset, outputDataset):
        self.__result = None
        self.inputDataSet = sm.add_constant(inputDataset)
        self.outputDataSet = outputDataset

    def result(self):
        return self.__result

    def run(self):
        pass

class ForwardStepwiseSelection(Algorithm):

    def result(self):
        return self.__result

    def __pickbestModel(self, models):
        
        best_model = None
        best_value = 0

        for model in models:
            if model.adjustedR2() > best_value:
                best_model = model

        return best_model

    def run(self):

        bestModelsForEachIteration = []
        inputVariablesNames = list(self.inputDataSet.columns.drop('const'))
        quantityOfInputVariables = len(inputVariablesNames)
        selectedVariables = ['const']

        for iteration in range(quantityOfInputVariables):

            current_iteration_models = []

            for inputVariableName in inputVariablesNames:
                model = Model.LinearRegressionModel(self.inputDataSet[selectedVariables+[inputVariableName]], self.outputDataSet)
                current_iteration_models.append(model)
            
            bestCurrentModel = self.__pickbestModel(current_iteration_models)
            bestModelsForEachIteration.append(bestCurrentModel)
            newVariableNameToSelect = list(set(model.inputVariablesNames()).difference(set(selectedVariables)))[0]
            selectedVariables.append(newVariableNameToSelect)
            inputVariablesNames.remove(newVariableNameToSelect)


        self.__result = bestModelsForEachIteration
