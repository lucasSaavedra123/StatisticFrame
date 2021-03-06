from statisticframe.Model.LogarithmicModel import LogarithmicModel
from statisticframe.Model.LinearRegressionModel import LinearRegressionModel
from statisticframe.Model.PolynomialRegressionModel import PolynomialRegressionModel
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


originalDataSet = pd.read_csv("tests/50_Startups.csv")
outputDataSet = originalDataSet[["Profit"]]
inputDataSet = originalDataSet[["R&D Spend"]]

# By now, we plot models of one predictor. And works very well...
aModel = LinearRegressionModel(inputDataSet, outputDataSet)
aModel.plot()

originalDataSet = pd.read_csv("tests/boston.csv")
outputDataSet = originalDataSet[["MEDV"]]
inputDataSet = originalDataSet[["LSTAT"]]

# We can do more...
aModel = PolynomialRegressionModel(inputDataSet, outputDataSet, grade=2)
aModel.plot()

originalDataSet = pd.read_csv("tests/Chwirut1.csv")
outputDataSet = originalDataSet[["ultrasonic_response"]]
inputDataSet = originalDataSet[["metal_distance"]]

# Logarithmic models too? Yes...
aModel = LogarithmicModel(inputDataSet, outputDataSet)
aModel.plot()
