from data_analysis import *
from data_preparation import *
from data_analysis import *
import statsmodels.api as sm
import operator
import matplotlib.pyplot as plt
import numpy as np


def forward_stepwise_selection(dataframe, target):
    """
    Given a dataframe and a target variable
    implement the forward stepwise selection algorithm and return
    and array with all the r2 values
    """
    r2_values = []
    target_dataframe = dataframe[[target]]
    dataframe = dataframe.drop(target, axis=1)
    dataframe = sm.add_constant(dataframe)
    columns_to_iterate = dataframe.columns
    columns_to_iterate = columns_to_iterate.drop('const', 1)
    columns_selected = ['const']
    number_of_variables = len(columns_to_iterate)

    for modelIndex in range(number_of_variables):
        r2 = {}

        for column in columns_to_iterate:
            predictor_variables = dataframe[columns_selected + [column]]
            model = sm.OLS(target_dataframe, predictor_variables)
            regression = model.fit()
            r2[column] = regression.rsquared_adj

        column_added_with_max_r2 = max(r2.items(), key=operator.itemgetter(1))[0]
        print(column_added_with_max_r2)
        max_r2_of_columns = r2[column_added_with_max_r2]
        r2_values.append(([column_added_with_max_r2]+columns_selected, max_r2_of_columns))
        columns_to_iterate = columns_to_iterate.drop(column_added_with_max_r2)
        columns_selected.append(column_added_with_max_r2)
        print(columns_selected)
    return r2_values


def backward_stepwise_selection(dataframe, target):
    """
    Given a dataframe and a target variable
    implement the backward stepwise selection algorithm and return
    and array with all the r2 values
    """
    r2_values = []
    target_dataframe = dataframe[[target]]
    dataframe = dataframe.drop(target, axis=1)
    dataframe = sm.add_constant(dataframe)
    columns_to_iterate = dataframe.columns
    columns_to_iterate = columns_to_iterate.drop('const', 1)
    columns_selected =  list(columns_to_iterate)
    number_of_variables = len(columns_to_iterate)

    for modelIndex in range(number_of_variables):
        r2 = {}

        for column in columns_to_iterate:
            new_columns = ['const'] + list(set(columns_selected).difference(set([column])))
            predictor_variables = dataframe[new_columns]
            model = sm.OLS(target_dataframe, predictor_variables)
            regression = model.fit()
            r2[column] = regression.rsquared_adj

        column_added_with_max_r2 = max(r2.items(), key=operator.itemgetter(1))[0]
        print(column_added_with_max_r2)
        max_r2_of_columns = r2[column_added_with_max_r2]
        r2_values.append((list(set(columns_selected) - set([column_added_with_max_r2]))+['const'], max_r2_of_columns))
        columns_to_iterate = columns_to_iterate.drop(column_added_with_max_r2)
        columns_selected.remove(column_added_with_max_r2)

    return r2_values


def backward_stepwise_selection_pvalues(dataframe, target):
    """
    Given a dataframe and a target variable
    implement the forward stepwise selection algorithm with p values
    and return and array with all the r2 values
    """
    target_dataframe = dataframe[[target]]
    dataframe = dataframe.drop(target, axis=1)
    dataframe = sm.add_constant(dataframe)
    columns_to_iterate = dataframe.columns
    columns_to_iterate = columns_to_iterate.drop('const', 1)
    number_of_variables = len(columns_to_iterate)
    columns_to_iterate = list(columns_to_iterate)
    r2_models = []

    for iterator in range(number_of_variables):
        predictor_variables = dataframe[['const']+columns_to_iterate]
        model = sm.OLS(target_dataframe, predictor_variables)
        regression = model.fit()
        r2_models.append((list(predictor_variables.columns), regression.rsquared_adj))
        p_values = regression.pvalues.drop('const')
        pvalue_index = p_values.argmax()
        pvalue_var = p_values.keys()[pvalue_index]
        columns_to_iterate.remove(pvalue_var)

    return r2_models

def best_variables_of_model(dictionary):
    best_model = []
    best_last_r2 = 0

    for item in dictionary:
        if item[1] >= best_last_r2:
            best_model = item[0]
            best_last_r2 = item[1]

    return best_model


def r2_variation(r2_adj, title, x_label, y_label, show=True):
    """
    Plot the scatter and the curve that shows how R2 value vary over
    every iteration. A title and labels for the graph must be included.
    Also, highlight the point where there is a maximum R2 value. Add information
    about that point. The amount of independent variables is given.
    """
    x = np.arange(len(r2_adj))
    y = []
    for i in r2_adj:
        y.append(i[1])

    y = np.array(y)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.scatter(x, y)
    plt.plot(x, y)

    if show:
        plt.show()


def create_model(r2_adj, var_model, dataframe, target, mode):
    """
    Creates a linear regression model with the variable/s that has
    the highest r2 value given by a selection stepwise algorithm
    """
    best_variables = best_variables_of_model(r2_adj)
    best_variables.remove('const')
    inputDataFrame = dataframe[best_variables]
    inputDataFrame = sm.add_constant(inputDataFrame)
    outputDataFrame = dataframe[[target]]

    return sm.OLS(outputDataFrame, inputDataFrame).fit()


def predict_on_model(model, input, ind_vars):
    input_values = {'const': [1]}

    for independent_variable in list(dict(model.pvalues).keys()):
        value = input.get(independent_variable)
        if value is not None:
            input_values[independent_variable] = value

    return model.predict(pd.DataFrame(input_values))[0]


def main():

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

    df = create_dataframe('insurance.csv')
    df = add_dummies(df, ['sex', 'smoker', 'region'])
    results = [forward_stepwise_selection(df, 'charges'), backward_stepwise_selection(df, 'charges'), backward_stepwise_selection_pvalues(df, 'charges')]

    for result in results:
        print(best_variables_of_model(result))

    r2_variation(results[0], "Forward Stepwise Selection", "Iterations", "R2", False)
    r2_variation(results[1], "Backward Stepwise Selection",  "Iterations", "R2", False)
    r2_variation(results[2], "Backward Stepwise Selection with p values", "Iterations", "R2", False)
    plt.show()

    for result in results:
        best_variables_to_use = best_variables_of_model(result)
        model = create_model(result, best_variables_to_use, df, 'charges', '')
        print(predict_on_model(model, input, best_variables_to_use))


if __name__ == '__main__':
    main()