from statisticframe.Model.Model import Model
import pandas as pd
import statsmodels.api as sm
import statisticframe.Utils.Utils as Utils
import operator


class LinearRegressionModel(Model):
    @classmethod
    def fromDataSetFileToPredictVariable(self, filename, predictorsVariablesNames, variableToPredictName):
        originalDataSet = pd.read_csv(filename)
        inputDataSet = Utils.addDummyVariablesToDataSet(originalDataSet[predictorsVariablesNames])
        outputDataSet = originalDataSet[[variableToPredictName]]
        return LinearRegressionModel(inputDataSet, outputDataSet)

    def __str__(self):
        return "Linear Regression Model: (%s)->(%s)" % (self.inputVariablesNames(), self.outputVariableName())

    def __init__(self, inputDataSet, outputDataSet):
        self.inputDataSet = Utils.addDummyVariablesToDataSet(inputDataSet)
        self.outputDataSet = outputDataSet

        if 'const' not in list(self.inputDataSet):
            self.model = sm.OLS(self.outputDataSet, sm.add_constant(self.inputDataSet)).fit()
        else:
            self.model = sm.OLS(self.outputDataSet, self.inputDataSet).fit()

    def predict(self, input):
        realInput = {'const': 1}
        for variableName in self.inputVariablesNames():
            value = input.get(variableName)
            if value is not None:
                realInput[variableName] = value
        return self.model.predict(pd.DataFrame(realInput))

    def inputVariablesNames(self):
        return list(self.inputDataSet.columns)

    def highestPValueVariableName(self):
        p_values = dict(self.model.pvalues)
        p_values.pop('const')
        return max(p_values.items(), key=operator.itemgetter(1))[0]

    def outputVariableName(self):
        return list(self.outputDataSet.columns)[0]
