import pandas as pd
import statsmodels.api as sm
import numpy as np


class Model():
    pass


class LinearRegressionModel(Model):
    @classmethod
    def fromDataSetFileToPredictVariable(self, filename, predictorsVariablesNames, variableToPredictName):
        originalDataSet = pd.read_csv(filename)
        inputDataSet = originalDataSet[predictorsVariablesNames]
        outputDataSet = originalDataSet[[variableToPredictName]]
        return LinearRegressionModel(inputDataSet, outputDataSet)

    def __init__(self, inputDataSet, outputDataSet):
        self.inputDataSet = sm.add_constant(inputDataSet)
        self.outputDataSet = outputDataSet
        self.model = sm.OLS(outputDataSet, inputDataSet).fit()

    def predict(self, input):
        return self.model.predict(pd.DataFrame(input))[0]

    def adjustedR2(self):
        return self.model.rsquared_adj

    def inputVariablesNames(self):
        return list(self.inputDataSet.columns)