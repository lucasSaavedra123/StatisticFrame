import matplotlib.pyplot as plt
import numpy as np


class Model():

    def plot(self):

        if self.quantityOfPredictors() > 1:
            raise Exception("To plot, you should have 1 predictors")
        else:
            plt.xlabel(self.inputVariablesNames()[0])
            plt.ylabel(self.outputVariableName()[0])
            plt.title(str(self))
            plt.scatter(self.inputDataSet[self.inputVariablesNames()], self.outputDataSet)

            exampleInput = np.arange(self.inputDataSet.min()[0],self.inputDataSet.max()[0],1)
            examplePrediction = self.predict({self.inputVariablesNames()[0]: list(exampleInput)})
            plt.plot(exampleInput, examplePrediction)

            plt.show()
