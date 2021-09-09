from data_analysis import *
from Model import Model
from PolynomialRegressionModel import PolynomialRegressionModel
from LogarithmicModel import LogarithmicModel


"""
Reference:	Chwirut, D., NIST (1979). 
Ultrasonic Reference Block Study.
"""


def train_model(dep_variables, target, mode, degree=1):
    """
    Return a non linear regression model either polynomial
    or logarithmic according to mode parameter.
    """
    if mode == 'p':
        return PolynomialRegressionModel(dep_variables, target, degree)
    elif mode == 'l':
        return LogarithmicModel(dep_variables, target)
    else:
        raise Exception("mode should be 'l' or 'p', not %s" % mode )


def plot_solution(model, dataframe):
    """
    Generate x values and with your model predict y
    values. Plot them as a curve and with a given dataframe
    plot every (x, y) value
    """
    model.plot()


def percentage_difference(initial,final):
    """
    >>> percentage_difference(1,2)
    1.0
    >>> percentage_difference(2,1)
    -0.5
    >>> percentage_difference(5.98,5.98)
    0.0
    """
    return ( (final-initial) / initial )


def is_calibrated(x, y, tolerance=0.05):
    """
    From the values obtained from the calibration
    determine if the ultrasonic device is ready for use or not.
    """
    df = create_dataframe('Chwirut1.csv')
    model = train_model(df[['metal_distance']], df[['ultrasonic_response']], 'l')

    result = model.predict({'metal_distance': [x]})
    if abs(percentage_difference(result, x)) > tolerance:
        return False
    else:
        return True

def main():
    # this should work
    df = create_dataframe('Chwirut1.csv')
    # polinomial of degree 2
    train_model(df[['metal_distance']], df[['ultrasonic_response']], 'p', 2).plot()
    # polinomial of degree 3
    train_model(df[['metal_distance']], df[['ultrasonic_response']], 'p', 3).plot()
    # logarithmic
    train_model(df[['metal_distance']], df[['ultrasonic_response']], 'l').plot()
    # the program
    print('press -1 to stop, 1 to continue')
    i = 1
    while i:
        print('Metal distance')
        x = int(input())
        print('Ultrasonic Value')
        y = int(input())
        print("Result: %s" % is_calibrated(x, y))
        print('Continue?')
        i = int(input())


if __name__ == '__main__':
    main()