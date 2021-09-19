from statisticframe.Algorithm.ForwardStepwiseSelection import ForwardStepwiseSelection
from statisticframe.Algorithm.BackwardStepwiseSelection import BackwardStepwiseSelection
from statisticframe.Algorithm.BackwardStepwiseSelectionWithPValue import BackwardStepwiseSelectionWithPValue
from statisticframe.Algorithm.BestModelSelection import BestModelSelection
import statisticframe.Utils.Utils as Utils
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# Some Algorithms plots
originalDataSet = pd.read_csv("tests/insurance.csv")
outputDataSet = originalDataSet[["charges"]]
inputDataSet = originalDataSet.drop("charges", axis=1)
inputDataSet = Utils.addDummyVariablesToDataSet(inputDataSet)

input = {
    "age": [25],
    "sex_female": [0],
    "sex_male": [1],
    "bmi": [26.3],
    "children": [0],
    "smoker_no": [0],
    "smoker_yes": [1],
    "region_northeast": [1],
    "region_northwest": [0],
    "region_southeast": [0],
    "region_southwest": [0],
}

algorithms = [
    ForwardStepwiseSelection(inputDataSet, outputDataSet),
    BackwardStepwiseSelection(inputDataSet, outputDataSet),
    BackwardStepwiseSelectionWithPValue(inputDataSet, outputDataSet),
    BestModelSelection(inputDataSet, outputDataSet)
]

for algorithm in algorithms:
    algorithm.plot()
    model = Utils.pickModelWithHighestAdjustedR2(algorithm.result())
    print(algorithm, "->", model.predict(input), "predicted with:", model)
