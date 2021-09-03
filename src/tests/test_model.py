import pathmagic
import unittest
from Model.LinearRegressionModel import LinearRegressionModel


class TestModels(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.exampleOneLinearModel = LinearRegressionModel.fromDataSetFileToPredictVariable('exampleOne.csv', ['X'], 'Y')
        self.exampleTwoLinearModel = LinearRegressionModel.fromDataSetFileToPredictVariable('exampleTwo.csv', ['X'], 'Y')

    def test_a_simple_linear_regression_model_predicts_two(self):
        prediction = self.exampleOneLinearModel.predict({"X":[2]})
        self.assertEqual(round(prediction), 2)

    def test_a_simple_linear_regression_model_predicts_fifty(self):
        prediction = self.exampleOneLinearModel.predict({"X":[50]})
        self.assertEqual(round(prediction), 50)

    def test_a_little_more_complex_linear_regression_model_predicts_zero_seventeen(self):
        prediction = self.exampleTwoLinearModel.predict({"X":[8]})
        self.assertEqual(round(prediction, 1), 1.7)


if __name__ == '__main__':
    unittest.main()
