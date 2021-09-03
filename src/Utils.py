import pandas as pd


def pickModelWithHighestAdjustedR2(models):
    modelWithHighestAdjustedR2 = None
    highestAdjustedR2 = 0

    for model in models:
        adjustedR2OfCurrentModel = model.adjustedR2()

        if adjustedR2OfCurrentModel >= highestAdjustedR2:
            modelWithHighestAdjustedR2 = model
            highestAdjustedR2 = adjustedR2OfCurrentModel

    return modelWithHighestAdjustedR2


def categoricalVariablesNamesFromDataset(dataset):
    cols = dataset.columns
    num_cols = dataset._get_numeric_data().columns
    return list(set(cols) - set(num_cols))


def addDummyVariablesToDataSet(dataframe):

    categoricalVariablesNames = categoricalVariablesNamesFromDataset(dataframe)

    for categoricalVariableName in categoricalVariablesNames:
        new_columns = pd.get_dummies(dataframe[[categoricalVariableName]], prefix=categoricalVariableName)
        column_location = dataframe.columns.get_loc(categoricalVariableName)
        dataframe = dataframe.drop([categoricalVariableName], axis=1)
        for index in range(len(new_columns.columns)-1):
            dataframe.insert(column_location + index, new_columns.columns[index], new_columns[[new_columns.columns[index]]])

    return dataframe


def powerset(s):
    x = len(s)
    set = []
    for i in range(1 << x):
        set.append([s[j] for j in range(x) if (i & (1 << j))])

    return set
