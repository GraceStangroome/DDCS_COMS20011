import random
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# no longer using
# def straight_line(x, y):
#     least_square = least_squares(x, y)
#     least_squares_x, least_squares_y = least_square[0], least_square[1]
#     x_endpoint_min = x.min()
#     x_endpoint_max = x.max()
#     y_endpoint_min = np.sin(least_squares_x + least_squares_y * x_endpoint_min)
#     y_endpoint_max = np.sin(least_squares_x + least_squares_y * x_endpoint_max)
#     return [x_endpoint_min, x_endpoint_max], [y_endpoint_min, y_endpoint_max]


def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()


def least_squares_formula(x, y):
    return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)


def least_squares(x, y):
    ones = np.ones(x.shape)
    x_with_ones = np.column_stack((ones, x))
    return least_squares_formula(x_with_ones, y)


def quadratic_resizer(x):
    return np.column_stack((np.ones(x.shape), x, x ** 2, x ** 3))


def unknown_resizer(x):
    return np.column_stack((x, np.ones(x.shape)))


def polynomial_line(xs, ys):
    resized_x = quadratic_resizer(xs)
    matrix = least_squares_formula(resized_x, ys)
    resulting_x = np.linspace(X[0], X[19], 20)  # uses X over 20 so that it covers all the data
    resulting_y = quadratic_resizer(resulting_x) @ matrix  # @ is matrix multiplication
    plt.plot(resulting_x, resulting_y, 'b-', lw=1)
    return resulting_x, resulting_y, matrix


def unknown_line(xs, ys):
    calculated_xs = np.sin(xs)
    coefficients = least_squares_formula(unknown_resizer(calculated_xs), ys)
    bias = np.column_stack((calculated_xs, np.ones(calculated_xs.shape)))
    calculated_ys = bias @ coefficients
    plt.plot(xs, calculated_ys, 'y-', lw=1)
    return xs, calculated_ys, coefficients


def calculate_pdf(data):
    fraction = 1 / (np.sqrt(np.var(data)) * np.sqrt(2 * np.pi))
    exponent = - ((np.power((data - np.mean(data)), 2)) / (2 * np.var(data)))
    result = fraction * np.power(np.e, exponent)
    return result


def squared_error(estimatedY, testY):
    error = 0
    for i in range(4):
        difference = estimatedY[i] - testY[i]
        error += difference * difference
    return error


# estimatedY = []
# for x in testX:
#  estimatedY.append( unknownCoefficient[0] * sin(x) + unknownCoefficient[1])
# need to find the difference at each position
def find_error(coeffs, x_test, y_test):
    for i in range(len(x_test)):
    deviation_squared = (coeffs * x_test - y_test) ** 2
    this_error = np.sum(deviation_squared)
    return this_error


def run_calculations():
    # Using 80% of the data for training
    x_train, y_train = np.empty(16), np.empty(16)
    x_test, y_test = np.empty(4), np.empty(4)
    options = [i for i in range(0, 20)]  # using X for no reason, could be Y, but both X and Y should be the same length
    # randomly choosing the 16 elements of X and Y for training
    for i in range(16):
        for_training = random.choice(options)
        x_train[i] = X[for_training]
        y_train[i] = Y[for_training]
        options.remove(for_training)
    # the remaining options are for testing
    for i in range(4):
        x_test[i] = X[options[i]]
        y_test[i] = Y[options[i]]

    # calculating the two possible lines
    polynomial_xs, polynomial_ys, poly_coefs = polynomial_line(x_train, y_train)
    unknown_func_xs, unknown_func_ys, unknown_coefs = unknown_line(x_train, y_train)
    # pdf = calculate_pdf(Y)
    # print("Pdf:", pdf)

    # calculating errors
    polynomial_error = find_error(poly_coefs, polynomial_xs, y_test)
    unknown_error = find_error(unknown_coefs, unknown_func_xs, y_test)

    # deciding which is better, recalculating for the whole data set and returning it
    this_error = min(polynomial_error, unknown_error)
    print("Polynomial error: ", polynomial_error)
    print("Unknown error: ", unknown_error)
    if this_error == polynomial_error:
        print("Chose poly")
        true_xs, true_ys, true_coefs = polynomial_line(X, Y)
    else:
        print("Chose unknown")
        true_xs, true_ys, true_coefs = unknown_line(X, Y)
    true_error = find_error(true_coefs, true_xs, Y)
    return true_xs, true_ys, true_error


datafile = sys.argv[1]  # sys.argv contains the arguments passed to the program
totalX, totalY = load_points_from_file(datafile)
start = 0
end = 20
error = 0
while end <= len(totalX):  # data is in chunks of 20
    X = totalX[start:end]
    Y = totalY[start:end]
    line_xs, line_ys, calculated_error = run_calculations()
    error += calculated_error
    plt.plot(line_xs, line_ys, 'r-', lw=3)
    start += 20
    end += 20
view_data_segments(totalX, totalY)
print("Total Error: ", error)
print("Finished")
