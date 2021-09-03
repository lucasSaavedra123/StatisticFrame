import pathmagic
from Model.Model import Model
import pandas as pd
import statsmodels.api as sm
import numpy as np
import copy
import operator

class LinearRegressionModel(Model):
    @classmethod
    def fromDataSetFileToPredictVariable(self, filename, predictorsVariablesNames, variableToPredictName):
        originalDataSet = pd.read_csv(filename)
        inputDataSet = originalDataSet[predictorsVariablesNames]
        outputDataSet = originalDataSet[[variableToPredictName]]
        return LinearRegressionModel(inputDataSet, outputDataSet)


    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Input: (%s), R2: %s" % (self.inputVariablesNames(), self.adjustedR2())

    def __init__(self, inputDataSet, outputDataSet):

        if 'const' not in list(inputDataSet.columns):
            self.inputDataSet = sm.add_constant(inputDataSet)
        else:
            self.inputDataSet = inputDataSet

        self.outputDataSet = outputDataSet
        self.model = sm.OLS(self.outputDataSet, self.inputDataSet).fit()

    def predict(self, input):
        return self.model.predict(pd.DataFrame(input))[0]

    def adjustedR2(self):
        return self.model.rsquared_adj

    def inputVariablesNames(self):
        return list(self.inputDataSet.columns.drop('const'))

    def highestPValueVariableName(self):
        p_values = dict(self.model.pvalues)
        p_values.pop('const')
        return max(p_values.items(), key=operator.itemgetter(1))[0]
