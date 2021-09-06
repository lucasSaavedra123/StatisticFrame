from Model.LinearRegressionModel import LinearRegressionModel
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


originalDataSet = pd.read_csv("tests/50_Startups.csv")
outputDataSet = originalDataSet[["Profit"]]
inputDataSet = originalDataSet[["R&D Spend"]]


#By now, we plot models of one predictor. And works very well...
aModel = LinearRegressionModel(inputDataSet, outputDataSet)
aModel.plot()
