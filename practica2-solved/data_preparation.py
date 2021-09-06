from data_analysis import *
import pandas as pd


def set_dummy_variable(dataframe, column, label):
    """
    Given a dataframe create dummy variables for a specific column
    Replace the old column and add the new ones
    """
    new_columns = pd.get_dummies(column, prefix=label)
    column_location = dataframe.columns.get_loc(label)
    dataframe = dataframe.drop([label], axis=1)
    for index in range(len(new_columns.columns)):
        dataframe.insert(column_location + index, new_columns.columns[index], new_columns[[new_columns.columns[index]]])
    return dataframe


def add_dummies(dataframe, columns):
    """
    Add dummy variables for every column given
    return the dataframe
    """
    for column_label in columns:
        dataframe = set_dummy_variable(dataframe, dataframe[[column_label]], column_label)

    return dataframe


def main():
    # Program steps
    df = create_dataframe('insurance.csv')
    df = add_dummies(df, ['sex', 'smoker', 'region'])
    df.to_csv('insurance-ready.csv', index=False)
    # visualize data
    # plot_scatter(df[['bmi']], df[['charges']], 'bmi', 'charges')
    # plt.show()


if __name__ == '__main__':
    main()
