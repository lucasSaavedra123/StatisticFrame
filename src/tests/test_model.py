import pathmagic
import unittest
import pandas as pd
from statisticframe.Model.LinearRegressionModel import LinearRegressionModel
from statisticframe.Model.PolynomialRegressionModel import PolynomialRegressionModel
from statisticframe.Model.LogarithmicModel import LogarithmicModel
from statisticframe.Model.Model import Model
import statisticframe.Utils as Utils

class TestModels(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.exampleOneLinearModel = LinearRegressionModel.fromDataSetFileToPredictVariable('exampleOne.csv', ['X'], 'Y')
        self.exampleTwoLinearModel = LinearRegressionModel.fromDataSetFileToPredictVariable('exampleTwo.csv', ['X'], 'Y')
        self.exampleOnePolynomialModel = PolynomialRegressionModel.fromDataSetFileToPredictVariableWithGrade('exampleOne.csv', ['X'], 'Y', grade=1)
        self.exampleTwoPolynomialModel = PolynomialRegressionModel.fromDataSetFileToPredictVariableWithGrade('exampleThree.csv', ['X'], 'Y', grade=2)

        originalDataSet = pd.read_csv("insurance.csv")
        outputDataSet = originalDataSet[["charges"]]
        inputDataSet = originalDataSet.drop("charges", axis=1)
        inputDataSet = Utils.addDummyVariablesToDataSet(inputDataSet)
        inputDataSet = inputDataSet[['smoker_no', 'age', 'bmi', 'children', 'region_northeast', 'region_northwest']]

        self.insuranceModel = LinearRegressionModel(inputDataSet, outputDataSet)

        originalDataSet = pd.read_csv("Chwirut1.csv")
        outputDataSet = originalDataSet[["ultrasonic_response"]]
        inputDataSet = originalDataSet[["metal_distance"]]

        self.deviceMeasuresModel = LogarithmicModel(inputDataSet, outputDataSet)

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

        self.assertTrue(Model().notValidPlottingMessage() in str(context.exception))

    def test_a_simple_polynomial_regression_of_grade_one_predict_equaly_to_linear_one(self):
        linearModelPrediction = self.exampleOneLinearModel.predict({'X': [2]})[0]
        oneGradePolynomialModelPrediction = self.exampleOnePolynomialModel.predict({'X': [2]})[0]
        self.assertEqual(round(linearModelPrediction), round(oneGradePolynomialModelPrediction))

    def test_a_simple_polynomial_regression_of_grade_two_predicts_eighty_one(self):
        prediction = self.exampleTwoPolynomialModel.predict({'X': [9]})[0]
        self.assertEqual(round(prediction), 81)

    def test_a_simple_polynomial_regression_of_grade_two_predicts_with_negative_input_forty_nine(self):
        prediction = self.exampleTwoPolynomialModel.predict({'X': [-7]})[0]
        self.assertEqual(round(prediction), 49)

    def test_a_logarithmic_regression_predicts_nearly_twenty_one(self):
        prediction = self.deviceMeasuresModel.predict({'metal_distance': [2]})[0]
        self.assertEqual(round(prediction), 22)

    def test_a_logarithmic_regression_predicts_forty_six(self):
        prediction = self.deviceMeasuresModel.predict({'metal_distance': [1]})[0]
        self.assertEqual(round(prediction), 46)

    def test_a_simple_linear_regression_has_b0_equal_zero(self):
        coefficientB0 = self.exampleOneLinearModel.coefficients().get('b0')
        self.assertEqual(round(coefficientB0), 0)

    def test_a_simple_linear_regression_has_bX_equal_one(self):
        coefficientBX = self.exampleOneLinearModel.coefficients().get('b_X')
        self.assertEqual(round(coefficientBX), 1)

    def test_a_linear_regression_has_b0_equal_to_zero_dot_one(self):
        coefficientB0 = self.exampleTwoLinearModel.coefficients().get('b0')
        self.assertEqual(round(coefficientB0, 2), 0.1)

    def test_a_linear_regression_has_bX_equal_to_zero_dot_two(self):
        coefficientBX = self.exampleTwoLinearModel.coefficients().get('b_X')
        self.assertEqual(round(coefficientBX, 2), 0.2)

    def test_a_polynomial_regression_has_b0_equal_to(self):
        coefficientB0 = self.exampleTwoPolynomialModel.coefficients().get('b0')
        self.assertEqual(round(coefficientB0), 0)

    def test_a_polynomial_regression_has_bX1_equal_to_zero_dot_two(self):
        coefficientBX1 = self.exampleTwoPolynomialModel.coefficients().get('b_X_1')
        self.assertEqual(round(coefficientBX1), 0)

    def test_a_polynomial_regression_has_bX2_equal_to_zero_dot_two(self):
        coefficientBX2 = self.exampleTwoPolynomialModel.coefficients().get('b_X_2')
        self.assertEqual(round(coefficientBX2), 1)

    def test_coefficients_return_all_coefficients_values_model(self):
        coefficients = self.insuranceModel.coefficients()
        self.assertTrue(coefficients.get('b0') is not None)
        self.assertTrue(coefficients.get('b_smoker_no') is not None)
        self.assertTrue(coefficients.get('b_age') is not None)
        self.assertTrue(coefficients.get('b_bmi') is not None)
        self.assertTrue(coefficients.get('b_children') is not None)
        self.assertTrue(coefficients.get('b_region_northeast') is not None)
        self.assertTrue(coefficients.get('b_region_northwest') is not None)

    def test_coefficients_return_all_coefficients_values_model(self):
        coefficients = self.deviceMeasuresModel.coefficients()
        self.assertEquals(round(coefficients.get('b_multiplicator')), 46)
        self.assertEquals(round(coefficients.get('b_exponent'), 2), -1.08)


if __name__ == '__main__':
    unittest.main()
