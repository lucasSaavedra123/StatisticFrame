import matplotlib.pyplot as plt
import numpy as np
import operator


class Model():

    @staticmethod
    def notValidPlottingMessage():
        return "To plot, you should have 1 predictors"

    def __repr__(self):
        return self.__str__()

    def plot(self):

        if self.quantityOfPredictors() > 1:
            raise Exception(Model.notValidPlottingMessage())
        else:
            plt.xlabel(self.inputVariablesNames()[0])
            plt.ylabel(self.outputVariableName()[0])
            plt.title(str(self))
            plt.scatter(self.inputDataSet[self.inputVariablesNames()], self.outputDataSet)

            exampleInput = np.arange(self.inputDataSet.min()[0], self.inputDataSet.max()[0], 0.1)
            examplePrediction = self.predict({self.inputVariablesNames()[0]: list(exampleInput)})
            plt.plot(exampleInput, examplePrediction, color='red')

            plt.show()

    def adjustedR2(self):
        return self.model.rsquared_adj

    def highestPValueVariableName(self):
        p_values = dict(self.model.pvalues)
        p_values.pop('const')
        return max(p_values.items(), key=operator.itemgetter(1))[0]

    def adjustedR2(self):
        return self.model.rsquared_adj

    def quantityOfPredictors(self):
        return len(self.inputVariablesNames())

    def coefficients(self):

        coefficientsDictionary = {}
        listOfCoefficientsSuffixes = ['0'] + self.inputVariablesNames()

        for index in range(len(listOfCoefficientsSuffixes)):
            if index == 0:
                coefficientsDictionary['b0'] = self.model.params[index]
            else:
                coefficientsDictionary['b_'+listOfCoefficientsSuffixes[index]] = self.model.params[index]

        return coefficientsDictionary
