from Model.Model import Model
import pandas as pd
import statsmodels.api as sm
import Utils
import operator
from sklearn.preprocessing import PolynomialFeatures


class PolynomialRegressionModel(Model):
    @classmethod
    def fromDataSetFileToPredictVariableWithGrade(self, filename, predictorsVariablesNames, variableToPredictName, grade):
        originalDataSet = pd.read_csv(filename)
        inputDataSet = Utils.addDummyVariablesToDataSet(originalDataSet[predictorsVariablesNames])
        outputDataSet = originalDataSet[[variableToPredictName]]
        return PolynomialRegressionModel(inputDataSet, outputDataSet, grade)

    def __str__(self):
        return "Polynomial Regression Model of grade %s: (%s)->(%s)" % (self.grade, self.inputVariablesNames(), self.outputVariableName())

    def __init__(self, inputDataSet, outputDataSet, grade):
        self.grade = grade
        self.__inputVariableName = list(inputDataSet.columns)[0]
        self.__outputVariableName = list(outputDataSet.columns)[0]
        self.inputDataSet = inputDataSet
        self.outputDataSet = outputDataSet
        self.model = sm.OLS(self.outputDataSet, PolynomialFeatures(degree=grade).fit_transform(inputDataSet)).fit()

    def predict(self, input):
        realInput = pd.DataFrame(input)
        realInput = PolynomialFeatures(degree=self.grade).fit_transform(realInput)
        return self.model.predict(realInput)

    def inputVariablesNames(self):
        return [self.__inputVariableName]

    def outputVariableName(self):
        return self.__outputVariableName

    def quantityOfPredictors(self):
        return 1

    def coefficients(self):

        coefficientsDictionary = {}

        for index in range(self.grade+1):
            if index == 0:
                coefficientsDictionary['b0'] = self.model.params[index]
            else:
                coefficientsDictionary['b_'+self.inputVariablesNames()[0]+'_'+str(index)] = self.model.params[index]

        return coefficientsDictionary
