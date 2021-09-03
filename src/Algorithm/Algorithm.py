import pathmagic
import Utils
import matplotlib.pyplot as plt
import numpy as np


class Algorithm():

    def __init__(self, inputDataset, outputDataset):
        self.__result = None
        self.inputDataSet = inputDataset
        self.outputDataSet = outputDataset

    def result(self):
        return self.__result

    def run(self):
        pass

    def plot(self):
        pass

    def __str__(self):
        return self.description()

    def plot(self):
        self.run()
        plt.clf()

        plt.xlabel("Iterations")
        plt.ylabel("R2")
        plt.title(self.description())

        iterations = np.arange(len(self.result()))

        R2Values = []

        for model in self.result():
            R2Values.append(model.adjustedR2())

        R2Values = np.array(R2Values)

        plt.scatter(iterations, R2Values)
        plt.plot(iterations, R2Values)
        plt.show()
