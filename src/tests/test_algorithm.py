import pathmagic
import unittest
from Algorithm.ForwardStepwiseSelection import ForwardStepwiseSelection
from Algorithm.BackwardStepwiseSelection import BackwardStepwiseSelection
from Algorithm.BackwardStepwiseSelectionWithPValue import BackwardStepwiseSelectionWithPValue
import Utils
import pandas as pd


class TestAlgorithm(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        originalDataSet = pd.read_csv('insurance.csv')
        self.outputDataSet = originalDataSet[['charges']]
        self.inputDataSet = originalDataSet.drop('charges', 1)
        self.inputDataSet = Utils.addDummyVariablesToDataSet(self.inputDataSet)

    def assertAlgortihmThrowsExpectedOutput(self, algorithm, expectedOutput):
        algorithm.run()
        model = Utils.pickModelWithHighestAdjustedR2(algorithm.result())
        result = model.inputVariablesNames().sort()
        expectedResult = expectedOutput.sort()
        self.assertEqual(expectedResult, result)

    def assertAlgortihmHasExpectedNumberOfMOdelsCreated(self, algorithm, expectedNumberOfMOdelsCreated):
        algorithm.run()
        models = algorithm.result()
        self.assertEqual(len(models), expectedNumberOfMOdelsCreated)

    def test_forwards_stepwise_selection(self):
        self.assertAlgortihmThrowsExpectedOutput(ForwardStepwiseSelection(self.inputDataSet, self.outputDataSet), ['smoker_no', 'age', 'bmi', 'children', 'region_northeast', 'region_northwest'])

    def test_forwards_stepwise_selection_has_ten_models_created(self):
        self.assertAlgortihmHasExpectedNumberOfMOdelsCreated(ForwardStepwiseSelection(self.inputDataSet, self.outputDataSet), 8)

    def test_backwards_stepwise_selection(self):
        self.assertAlgortihmThrowsExpectedOutput(BackwardStepwiseSelection(self.inputDataSet, self.outputDataSet), ['bmi', 'children', 'smoker_no', 'region_northeast', 'region_northwest', 'age'])

    def test_backwards_stepwise_selection_has_ten_models_created(self):
        self.assertAlgortihmHasExpectedNumberOfMOdelsCreated(BackwardStepwiseSelection(self.inputDataSet, self.outputDataSet), 8)

    def test_backwards_stepwise_selection_with_p_value(self):
        self.assertAlgortihmThrowsExpectedOutput(BackwardStepwiseSelectionWithPValue(self.inputDataSet, self.outputDataSet), ['age', 'bmi', 'children', 'smoker_no', 'region_northeast', 'region_northwest'])

    def test_backwards_stepwise_selection_with_p_value_has_ten_models_created(self):
        self.assertAlgortihmHasExpectedNumberOfMOdelsCreated(BackwardStepwiseSelectionWithPValue(self.inputDataSet, self.outputDataSet), 8)


if __name__ == '__main__':
    unittest.main()
