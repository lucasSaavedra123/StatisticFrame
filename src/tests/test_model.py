import pathmagic
import unittest
from Model.LinearRegressionModel import LinearRegressionModel


class TestModels(unittest.TestCase):

    def test_a_simple_linear_regression_model_predicts_two(self):
        aLinearModel = LinearRegressionModel.fromDataSetFileToPredictVariable('exampleOne.csv', ['X'], 'Y')
        prediction = aLinearModel.predict({"X":[2]})
        self.assertEqual(prediction, 2)

    def test_a_simple_linear_regression_model_predicts_fifty(self):
        aLinearModel = LinearRegressionModel.fromDataSetFileToPredictVariable('exampleOne.csv',['X'],'Y')
        prediction = aLinearModel.predict({"X":[50]})
        self.assertEqual(prediction, 50)

    def test_a_little_more_complex_linear_regression_model_predicts_zero_seventeen(self):
        aLinearModel = LinearRegressionModel.fromDataSetFileToPredictVariable('exampleTwo.csv',['X'],'Y')
        prediction = aLinearModel.predict({"X":[8]})
        self.assertEqual(round(prediction, 1), 1.7)


if __name__ == '__main__':
    unittest.main()
