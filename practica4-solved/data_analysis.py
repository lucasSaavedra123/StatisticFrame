import pandas as pd
import matplotlib.pyplot as plt


def create_dataframe(filename):
    """
    Given a filename in csv format
    create a panda dataframe and return it
    with no empty rows
    """
    df = pd.read_csv(filename)
    df = df.dropna()
    return df


def plot_histogram(dataframe, ind_variable, target, bins):
    """
    Plot a histogram of a given dependent variable
    of the whole data available and the ones that
    have target variable equal to 1
    """
    dataframe = dataframe[dataframe[target]==1]
    plt.hist(dataframe[[ind_variable]], bins=bins)
    plt.show()


def main():
    # For testing purposes, to analyze the data
    df = create_dataframe('framingham.csv')
    print(df.head())
    plot_histogram(df, 'age', 'TenYearCHD', 10)


if __name__ == '__main__':
    main()