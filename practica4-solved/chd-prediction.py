import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from data_analysis import *


def train(predictors, target):
    """
    Given a datframe and a target variable
    return a logistic regression model
    """
    return LogisticRegression().fit(predictors, target)


def predict(x, model, threshold=0.5):
    """
    Using the proability that our model return for each
    possibility (0 or 1), create the proper predictions according
    the given threshold. Return the array with the results.
    """

    predictions = model.predict_proba(x)
    cleanPrediction = (predictions[:, 0] >= threshold).astype(int)
    x["prediction"] = cleanPrediction

    return x


def report(result):
    result.to_csv("resultado.csv", index=False)


def main():
    df = create_dataframe('framingham.csv')
    testPatients = pd.DataFrame({
        "male": [1, 0],
        "age": [45, 43],
        "education": [4, 4],
        "currentSmoker": [0, 1],
        "cigsPerDay": [0, 0],
        "BPMeds": [0, 0],
        "prevalentStroke": [1, 0],
        "prevalentHyp": [1, 1],
        "diabetes": [0, 1],
        "totChol": [205, 250],
        "sysBP": [125, 155],
        "diaBP": [110, 130],
        "BMI": [26, 28],
        "heartRate": [85, 90],
        "glucose": [85, 88]
    })

    predictors = ['age', 'BMI', 'cigsPerDay', 'diaBP', 'glucose', 'heartRate', 'sysBP', 'totChol']
    x = df[predictors]
    x_patients = testPatients[predictors]
    aModel = train(x, df[['TenYearCHD']])

    report(predict(x_patients, aModel, 0.75))


if __name__ == '__main__':
    main()