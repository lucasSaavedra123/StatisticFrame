from Model.Model import Model
import pandas as pd
import statsmodels.api as sm
import Utils
import numpy as np
import operator


class LogarithmicModel(Model):
    @classmethod
    def fromDataSetFileToPredictVariable(self, filename, predictorsVariablesNames, variableToPredictName):
        originalDataSet = pd.read_csv(filename)
        inputDataSet = Utils.addDummyVariablesToDataSet(originalDataSet[predictorsVariablesNames])
        outputDataSet = originalDataSet[[variableToPredictName]]
        return LogarithmicModel(inputDataSet, outputDataSet)

    def __str__(self):
        return "Logarithmic Model: (%s)->(%s)" % (self.inputVariablesNames(), self.outputVariableName())

    def __init__(self, inputDataSet, outputDataSet):
        self.inputDataSet = Utils.addDummyVariablesToDataSet(inputDataSet)
        self.outputDataSet = outputDataSet

        inputDataSetTransformed = np.log10(self.inputDataSet)
        outputDataSetTransformed = np.log10(self.outputDataSet)
        self.model = sm.OLS(outputDataSetTransformed, sm.add_constant(inputDataSetTransformed)).fit()

    def predict(self, input):
        multiplicator = np.array(self.coefficients()['b_multiplicator'])
        exponent = np.array(self.coefficients()['b_exponent'])
        X = np.array(input[self.inputVariablesNames()[0]])
        return multiplicator * np.power(X, exponent)

    def inputVariablesNames(self):
        return list(self.inputDataSet.columns)

    def highestPValueVariableName(self):
        p_values = dict(self.model.pvalues)
        p_values.pop('const')
        return max(p_values.items(), key=operator.itemgetter(1))[0]

    def outputVariableName(self):
        return list(self.outputDataSet.columns)[0]

    def coefficients(self):
        rawCoefficients = super().coefficients()
        return {
            'b_multiplicator': pow(10, rawCoefficients['b0']),
            'b_exponent': rawCoefficients['b_'+self.inputVariablesNames()[0]]
        }
