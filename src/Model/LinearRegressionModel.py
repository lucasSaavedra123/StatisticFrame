import pathmagic
from Model.Model import Model
import pandas as pd
import statsmodels.api as sm
import Utils
import operator


class LinearRegressionModel(Model):
    @classmethod
    def fromDataSetFileToPredictVariable(self, filename, predictorsVariablesNames, variableToPredictName):
        originalDataSet = pd.read_csv(filename)
        inputDataSet = Utils.addDummyVariablesToDataSet(originalDataSet[predictorsVariablesNames])
        outputDataSet = originalDataSet[[variableToPredictName]]
        return LinearRegressionModel(inputDataSet, outputDataSet)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Input: (%s), R2: %s" % (self.inputVariablesNames(), self.adjustedR2())

    def __init__(self, inputDataSet, outputDataSet):
        self.inputDataSet = Utils.addDummyVariablesToDataSet(inputDataSet)

        if 'const' not in list(self.inputDataSet):
            self.inputDataSet = sm.add_constant(self.inputDataSet)

        self.outputDataSet = outputDataSet
        self.model = sm.OLS(self.outputDataSet, self.inputDataSet).fit()

    def predict(self, input):
        realInput = {'const': 1}
        for variableName in self.inputVariablesNames():
            value = input.get(variableName)
            if value is not None:
                realInput[variableName] = value
        return self.model.predict(pd.DataFrame(realInput))[0]

    def adjustedR2(self):
        return self.model.rsquared_adj

    def inputVariablesNames(self):
        return list(self.inputDataSet.columns.drop('const'))

    def highestPValueVariableName(self):
        p_values = dict(self.model.pvalues)
        p_values.pop('const')
        return max(p_values.items(), key=operator.itemgetter(1))[0]
