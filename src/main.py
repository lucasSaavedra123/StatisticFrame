from Algorithm import ForwardStepwiseSelection
import Utils
import pandas as pd

originalDataSet = pd.read_csv('insurance-ready.csv')
outputDataSet = originalDataSet[['charges']]
inputDataSet = originalDataSet.drop('charges', 1)

algorithm = ForwardStepwiseSelection(inputDataSet, outputDataSet)
algorithm.run(debug=True)
model = Utils.pickModelWithHighestAdjustedR2(algorithm.result())

print(model.inputVariablesNames())
