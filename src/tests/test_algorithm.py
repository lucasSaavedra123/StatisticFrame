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

    def assertAlgortihmThrowsExpectedOutput(self, algorithm, expectedOutput):
        algorithm.run()
        model = Utils.pickModelWithHighestAdjustedR2(algorithm.result())
        result = model.inputVariablesNames().sort()
        expectedResult = expectedOutput.sort()
        self.assertEqual(expectedResult, result)

    def assertAlgortihmHasExpectedNumberOfMOdelsCreated(self, algorithm, expectedNumberOfMOdelsCreated):
        algorithm.run()
        models = algorithm.result()

        for model in models:
            print(model)

        self.assertEqual(len(models), expectedNumberOfMOdelsCreated)

    def test_forwards_stepwise_selection(self):
        self.assertAlgortihmThrowsExpectedOutput(ForwardStepwiseSelection(self.inputDataSet, self.outputDataSet), ['smoker_no', 'smoker_yes', 'age', 'bmi', 'children', 'region_northeast', 'region_northwest'])

    def test_forwards_stepwise_selection_has_ten_models_created(self):
        self.assertAlgortihmHasExpectedNumberOfMOdelsCreated(ForwardStepwiseSelection(self.inputDataSet, self.outputDataSet), 11)

    def test_backwards_stepwise_selection(self):
        self.assertAlgortihmThrowsExpectedOutput(BackwardStepwiseSelection(self.inputDataSet, self.outputDataSet), ['region_southwest', 'age', 'children', 'region_southeast', 'bmi', 'smoker_yes', 'const'])

    def test_backwards_stepwise_selection_has_ten_models_created(self):
        self.assertAlgortihmHasExpectedNumberOfMOdelsCreated(BackwardStepwiseSelection(self.inputDataSet, self.outputDataSet), 11)

    def test_backwards_stepwise_selection_with_p_value(self):
        self.assertAlgortihmThrowsExpectedOutput(BackwardStepwiseSelectionWithPValue(self.inputDataSet, self.outputDataSet), ['age', 'bmi', 'children', 'smoker_no', 'smoker_yes', 'region_southeast', 'region_southwest'])

    def test_backwards_stepwise_selection_with_p_value_has_ten_models_created(self):
            self.assertAlgortihmHasExpectedNumberOfMOdelsCreated(BackwardStepwiseSelectionWithPValue(self.inputDataSet, self.outputDataSet), 11)


if __name__ == '__main__':
    unittest.main()
