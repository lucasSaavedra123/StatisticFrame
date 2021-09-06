import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from data_analysis import create_dataframe, plot_scatter, save_plots

PARAMS = ['R&D Spend', 'Administration', 'Marketing Spend', 'State', 'Profit']
# Complete with the Independent variables that you will use
IND_VAR = ['R&D Spend']
DEP_VAR = ['Profit']

# Client variables
INPUT = {
    'R&D Spend': 40000,
    'Administration': 12000,
    'Marketing Spend': 129300,
    'Profit': 180000
}


def train_model(dataframe):
    """
    Train a linear regression model with
    its dependent and indepent variables
    from a specific dataframe
    """
    explicative_variables_data = dataframe[IND_VAR]
    target_variable_data = dataframe[DEP_VAR]
    return linear_model.LinearRegression().fit(explicative_variables_data, target_variable_data)


def betas(regr_model):
    """
    Return beta values given a
    specific model
    """

    coefficients_dictionary = {
        "b0": regr_model.intercept_[0]
    }

    for coefficient_index in range(len(list(regr_model.coef_[0]))):
        coefficients_dictionary["b"+str(IND_VAR[coefficient_index])] = list(regr_model.coef_[0])[coefficient_index]

    return coefficients_dictionary


def mse(regr_model, dataframe):
    """
    Calculate the Sum of Square Errors
    and then divide them by the amount of instances
    Do not use mean_squared_errors.
    """
    mse_result = 0
    n = dataframe.shape[0]
    explicative_variables_data = dataframe[IND_VAR]
    target_variable_data = np.transpose(dataframe[DEP_VAR])
    predictions = regr_model.predict(explicative_variables_data)

    for index in range(n):
        difference_between_prediction_and_real_result = predictions[index] - target_variable_data[index]
        mse_result += np.power(difference_between_prediction_and_real_result, 2) / n

    return mse_result


def expected_output(filename):
    """"
    Predict the profit of a certain startup given the
    variables that you use in your model
    """
    df = create_dataframe(filename)
    model = train_model(df)
    input_values = []

    for independent_variable in IND_VAR:
        value = INPUT.get(independent_variable)
        if value is not None:
            input_values.append(value)

    return model.predict(np.array(input_values))[0][0]


def save_model_plot_if_possible(model, df):
    plt.clf()
    if len(IND_VAR) == 1:
        plt.plot(df[IND_VAR], model.predict(df[IND_VAR]))
        plt.scatter(df[IND_VAR], df[DEP_VAR])
        plt.xlabel(IND_VAR[0])
        plt.ylabel(DEP_VAR[0])
        plt.savefig('%s_%s.png' % (IND_VAR[0], DEP_VAR[0]))


def print_report_on(filename):
    # This will be useful to test your model
    df = create_dataframe(filename)
    model = train_model(df)
    betas_dictionary = betas(model)
    print(df.describe())
    print("Betas: ", betas(model))
    print("MSE: ", mse(model, df)[0])
    print("r2 Score: ", r2_score(df[DEP_VAR], model.predict(df[IND_VAR])))
    if len(IND_VAR) == 1:
        print("Confidence Interval (95%):")
  
        standards_deviations = df[IND_VAR+DEP_VAR].std(numeric_only=None)

        sd_dep_var = list(standards_deviations[DEP_VAR])[0]
        sd_ind_var = list(standards_deviations[IND_VAR])[0]
        mean_ind_var = list(df[IND_VAR].mean())[0]

        n = len(df.index)

        se_0 = math.sqrt(pow(sd_dep_var, 2) * ((1/n)+(pow(mean_ind_var, 2)/(pow(sd_ind_var, 2)*n))))
        se_1 = math.sqrt(pow(sd_dep_var, 2)/(pow(sd_ind_var, 2)*n))

        confidence_interval_b0 = [betas_dictionary['b0']-2*se_0, betas_dictionary['b0']+2*se_0]
        confidence_interval_bx = [betas_dictionary['b'+IND_VAR[0]]-2*se_1, betas_dictionary['b'+IND_VAR[0]]+2*se_1]
        print("IC_95_b0: %s" % confidence_interval_b0)
        print("IC_95_b%s: %s" % (IND_VAR[0], confidence_interval_bx))

    save_model_plot_if_possible(model, df)


def main():
    filename = '50_Startups.csv'
    print_report_on(filename)
    print("Expected Output: ", expected_output(filename))


if __name__ == '__main__':
    main()
