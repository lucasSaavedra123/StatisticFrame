import pathmagic
from Model.Model import Model
import pandas as pd
import statsmodels.api as sm
import Utils
import operator
from sklearn.preprocessing import PolynomialFeatures


class PolynomialRegressionModel(Model):
    @classmethod
    def fromDataSetFileToPredictVariable(self, filename, predictorsVariablesNames, variableToPredictName, grade):
        originalDataSet = pd.read_csv(filename)
        inputDataSet = Utils.addDummyVariablesToDataSet(originalDataSet[predictorsVariablesNames])
        outputDataSet = originalDataSet[[variableToPredictName]]
        return PolynomialRegressionModel(inputDataSet, outputDataSet, grade)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Linear Regression Model: (%s)->(%s)" % (self.inputVariablesNames(), self.outputVariableName())

    def __init__(self, inputDataSet, outputDataSet, grade):
        self.grade = grade
        self.__inputVariableName = list(inputDataSet.columns)[0]
        self.__outputVariableName = list(outputDataSet.columns)[0]
        self.inputDataSet = inputDataSet
        self.outputDataSet = outputDataSet
        self.model = sm.OLS(self.outputDataSet, PolynomialFeatures(degree = grade).fit_transform(inputDataSet)).fit()

    def predict(self, input):
        realInput = pd.DataFrame(input)
        realInput = PolynomialFeatures(degree = self.grade).fit_transform(realInput)
        return self.model.predict(realInput)

    def adjustedR2(self):
        return self.model.rsquared_adj

    def inputVariablesNames(self):
        return [self.__inputVariableName]

    def highestPValueVariableName(self):
        p_values = dict(self.model.pvalues)
        p_values.pop('const')
        return max(p_values.items(), key=operator.itemgetter(1))[0]

    def outputVariableName(self):
        return self.__outputVariableName

    def quantityOfPredictors(self):
        return 1