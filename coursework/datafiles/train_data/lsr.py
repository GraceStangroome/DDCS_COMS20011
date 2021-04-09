import random
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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


def linear_line(xs, ys):
    least_square = least_squares(xs, ys)
    least_squares_x, least_squares_y = least_square[0], least_square[1]
    x_endpoint_min = xs.min()
    x_endpoint_max = xs.max()
    y_endpoint_min = least_squares_x + least_squares_y * x_endpoint_min
    y_endpoint_max = least_squares_x + least_squares_y * x_endpoint_max
    plt.plot([x_endpoint_min, x_endpoint_max], [y_endpoint_min, y_endpoint_max], 'g-', lw=1)
    return [x_endpoint_min, x_endpoint_max], [y_endpoint_min, y_endpoint_max], least_squares_y, least_squares_x


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


def squared_error(estimated_y, test_y):
    this_error = 0
    for i in range(len(estimated_y)):
        squared_difference = (estimated_y[i] - test_y[i]) ** 2
        this_error += squared_difference
    return this_error


def find_linear_error(gradient, y_intercept, x_test, y_test):
    y_estimates = []
    for x in x_test:
        y_estimates.append(gradient * x + y_intercept)
    return squared_error(y_estimates, y_test)


def find_poly_error(coeffs, x_test, y_test):
    y_estimates = []
    for x in x_test:
        y_estimates.append(coeffs[0] * x ** 3 + coeffs[1] * x ** 2 + coeffs[2] * x + coeffs[3])
    return squared_error(y_estimates, y_test)


def find_unknown_error(coeffs, x_test, y_test):
    y_estimates = []
    for x in x_test:
        y_estimates.append(coeffs[0] * np.sin(x) + coeffs[1])
    return squared_error(y_estimates, y_test)


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
    linear_xs, linear_ys, gradient, y_intercept = linear_line(x_train, y_train)
    polynomial_xs, polynomial_ys, poly_coefs = polynomial_line(x_train, y_train)
    unknown_func_xs, unknown_func_ys, unknown_coefs = unknown_line(x_train, y_train)
    # pdf = calculate_pdf(Y)
    # print("Pdf:", pdf)

    # calculating errors
    linear_error = find_linear_error(gradient, y_intercept, x_test, y_test)
    polynomial_error = find_poly_error(poly_coefs, x_test, y_test)
    unknown_error = find_unknown_error(unknown_coefs, x_test, y_test)

    # deciding which is better, recalculating for the whole data set and returning it
    this_error = min(linear_error, polynomial_error, unknown_error)
    print("Linear error: ", linear_error)
    print("Polynomial error: ", polynomial_error)
    print("Unknown error: ", unknown_error)
    if this_error == linear_error:
        print("Chose linear")
        true_xs, true_ys, true_gradient, true_y_intercept = linear_line(X, Y)
        true_error = find_linear_error(true_gradient, true_y_intercept, true_xs, Y)
    elif this_error == polynomial_error:
        print("Chose poly")
        true_xs, true_ys, true_coefs = polynomial_line(X, Y)
        true_error = find_poly_error(true_coefs, true_xs, Y)
    else:
        print("Chose unknown")
        true_xs, true_ys, true_coefs = unknown_line(X, Y)
        true_error = find_unknown_error(true_coefs, true_xs, Y)
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
