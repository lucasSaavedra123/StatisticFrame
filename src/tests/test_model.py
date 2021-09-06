import pathmagic
import unittest
import Utils
import pandas as pd
from Model.LinearRegressionModel import LinearRegressionModel


class TestModels(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.exampleOneLinearModel = LinearRegressionModel.fromDataSetFileToPredictVariable('exampleOne.csv', ['X'], 'Y')
        self.exampleTwoLinearModel = LinearRegressionModel.fromDataSetFileToPredictVariable('exampleTwo.csv', ['X'], 'Y')

        originalDataSet = pd.read_csv("insurance.csv")
        outputDataSet = originalDataSet[["charges"]]
        inputDataSet = originalDataSet.drop("charges", axis=1)
        inputDataSet = Utils.addDummyVariablesToDataSet(inputDataSet)
        inputDataSet = inputDataSet[['smoker_no', 'age', 'bmi', 'children', 'region_northeast', 'region_northwest']]

        self.insuranceModel = LinearRegressionModel(inputDataSet, outputDataSet)

    def test_a_simple_linear_regression_model_predicts_two(self):
        prediction = self.exampleOneLinearModel.predict({'X': [2]})[0]
        self.assertEqual(round(prediction), 2)

    def test_a_simple_linear_regression_model_predicts_fifty(self):
        prediction = self.exampleOneLinearModel.predict({'X': [50]})[0]
        self.assertEqual(round(prediction), 50)

    def test_a_little_more_complex_linear_regression_model_predicts_zero_seventeen(self):

        prediction = self.exampleTwoLinearModel.predict({'X': [8]})[0]
        self.assertEqual(round(prediction, 1), 1.7)

    def test_predict_charges_for_northeast_person(self):

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
            "region_southwest": [0]
        }

        prediction = self.insuranceModel.predict(input)[0]
        self.assertEqual(round(prediction, 0), 27175)

    def test_model_with_more_than_one_predictors_fails(self):

        with self.assertRaises(Exception) as context:
            self.insuranceModel.plot()

        self.assertTrue("To plot, you should have 1 predictors" in context.exception)


if __name__ == '__main__':
    unittest.main()
