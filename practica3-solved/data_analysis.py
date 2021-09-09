import pandas as pd
import matplotlib.pyplot as plt
import os


def create_dataframe(filename):
    """
    Given a filename in csv format
    create a panda dataframe and return it
    """
    return pd.read_csv(filename)


def plot_scatter(x, y, x_label, y_label):
    """
    Plot a scatter of (x, y) points
    given as a parameter
    """
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def save_plots(x, y, x_label, y_label, directory):
    """
    Plot a scatter of (x, y) points
    given as a parameter
    """
    plt.clf()
    try:
        plt.scatter(x, y)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig(directory + chr(92) + ('%s_%s.png' % (x_label, y_label)))
    except TypeError:
        print("(%s, %s) is not plottable" % (x_label, y_label))


def save_plots_against_target_variable(filename, target_variable_name):
    df = create_dataframe(filename)
    new_directory_name = os.path.abspath(os.getcwd()) + chr(92) + filename.split('.')[0]

    if not os.path.exists(new_directory_name):
        os.mkdir(new_directory_name)

    variable_names = list(df.columns.values)
    try:
        variable_names.remove(target_variable_name)
    except ValueError:
        raise Exception("Target variable should be in dataframe. Possibles target variables: %s" % variable_names)


    for x_value in variable_names:
        save_plots(df[[x_value]], df[[target_variable_name]], x_value, target_variable_name, new_directory_name)


if __name__ == '__main__':
    save_plots_against_target_variable('Chwirut1.csv', 'ultrasonic_response')
