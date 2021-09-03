import pathmagic
import unittest
from Algorithm.ForwardStepwiseSelection import ForwardStepwiseSelection
from Algorithm.BackwardStepwiseSelection import BackwardStepwiseSelection
from Algorithm.BackwardStepwiseSelectionWithPValue import BackwardStepwiseSelectionWithPValue
import Utils
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class TestAlgorithm(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        originalDataSet = pd.read_csv('insurance-ready.csv')
        self.outputDataSet = originalDataSet[['charges']]
        self.inputDataSet = originalDataSet.drop('charges', 1)

    def test_forwards_stepwise_selection(self):
        algorithm = ForwardStepwiseSelection(self.inputDataSet, self.outputDataSet)
        algorithm.run()
        model = Utils.pickModelWithHighestAdjustedR2(algorithm.result())

        result = model.inputVariablesNames().sort()
        expectedResult = ['smoker_no', 'smoker_yes', 'age', 'bmi', 'children', 'region_northeast', 'region_northwest'].sort()
        self.assertEqual(expectedResult, result)

    def test_backwards_stepwise_selection(self):
        algorithm = BackwardStepwiseSelection(self.inputDataSet, self.outputDataSet)
        algorithm.run()
        model = Utils.pickModelWithHighestAdjustedR2(algorithm.result())

        result = model.inputVariablesNames().sort()
        expectedResult = ['region_southwest', 'age', 'children', 'region_southeast', 'bmi', 'smoker_yes', 'const'].sort()
        self.assertEqual(expectedResult, result)

    def test_backwards_stepwise_selection_with_p_value(self):
        algorithm = BackwardStepwiseSelectionWithPValue(self.inputDataSet, self.outputDataSet)
        algorithm.run()
        model = Utils.pickModelWithHighestAdjustedR2(algorithm.result())

        result = model.inputVariablesNames().sort()
        expectedResult = ['age', 'bmi', 'children', 'smoker_no', 'smoker_yes', 'region_southeast', 'region_southwest'].sort()
        self.assertEqual(expectedResult, result)
  

if __name__ == '__main__':
    unittest.main()
