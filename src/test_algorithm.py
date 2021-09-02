import unittest
from Algorithm import ForwardStepwiseSelection
import Utils
import pandas as pd

class TestAlgorithm(unittest.TestCase):

    def test_forwards_stepwise_selection(self):

        originalDataSet = pd.read_csv('insurance-ready.csv')
        outputDataSet = originalDataSet[['charges']]
        inputDataSet = originalDataSet.drop('charges', 1)

        algorithm = ForwardStepwiseSelection(inputDataSet, outputDataSet)
        algorithm.run()
        model = Utils.pickModelWithHighestAdjustedR2(algorithm.result())

        result = model.inputVariablesNames().sort()
        expectedResult = ['smoker_no', 'smoker_yes', 'age', 'bmi', 'children', 'region_northeast', 'region_northwest'].sort()
        self.assertEqual(expectedResult, result)


if __name__ == '__main__':
    unittest.main()
