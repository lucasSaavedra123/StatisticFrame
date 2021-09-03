from Algorithm.ForwardStepwiseSelection import ForwardStepwiseSelection
from Algorithm.BackwardStepwiseSelection import BackwardStepwiseSelection
from Algorithm.BackwardStepwiseSelectionWithPValue import BackwardStepwiseSelectionWithPValue
import Utils
import pandas as pd
import matplotlib.pyplot as plt

originalDataSet = pd.read_csv('insurance-ready.csv')
outputDataSet = originalDataSet[['charges']]
inputDataSet = originalDataSet.drop('charges', axis=1)


algorithm = ForwardStepwiseSelection(inputDataSet, outputDataSet).plot()
algorithm = BackwardStepwiseSelection(inputDataSet, outputDataSet).plot()
algorithm = BackwardStepwiseSelectionWithPValue(inputDataSet, outputDataSet).plot()
plt.show()
