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
    coefficients = least_squares(xs, ys)
    resulting_x = np.linspace(X[0], X[19], 20)  # uses X over 20 so that it covers all the data
    y_intercept, gradient = coefficients[0], coefficients[1]
    calculated_ys = y_intercept + gradient * resulting_x
    return resulting_x, calculated_ys, gradient, y_intercept


def polynomial_line(xs, ys):
    resized_x = quadratic_resizer(xs)
    matrix = least_squares_formula(resized_x, ys)
    resulting_x = np.linspace(X[0], X[19], 20)  # uses X over 20 so that it covers all the data
    resulting_y = quadratic_resizer(resulting_x) @ matrix  # @ is matrix multiplication
    return resulting_x, resulting_y, matrix


def unknown_line(xs, ys):
    calculated_xs = np.sin(xs)
    bias = unknown_resizer(calculated_xs)
    coefficients = least_squares_formula(bias, ys)
    calculated_ys = bias @ coefficients
    return xs, calculated_ys, coefficients


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
        y_estimates.append((coeffs[3] * (x ** 3)) + (coeffs[2] * (x ** 2)) + (coeffs[1] * x) + coeffs[0])
    return squared_error(y_estimates, y_test)


def find_unknown_error(coeffs, x_test, y_test):
    y_estimates = []
    for x in x_test:
        y_estimates.append(coeffs[0] * np.sin(x) + coeffs[1])
    return squared_error(y_estimates, y_test)


def find_best(all_results):
    i = 0
    # I tried using sorted but it didn't seem to work
    smallest_error_so_far = 99999999999  # set to really high, so that it is overwritten
    best_result = []
    for result in all_results:
        if result[1] < smallest_error_so_far:
            smallest_error_so_far = result[1]
            best_result = result
        i += 1
    return best_result


def cross_validation():
    j = 0
    all_results = []
    # Using 80% of the data for training
    while j in range(100):  # repeating 100 times to get good results
        x_train, y_train = np.empty(16), np.empty(16)
        x_test, y_test = np.empty(4), np.empty(4)
        options = [i for i in range(0, 20)]  # using X for no reason, could be Y as len(X) == len(Y)
        for i in range(4):  # it won't necessarily add a new one every time
            # randomly choosing the 4 elements of X and Y for testing
            for_testing = random.choice(options)
            x_test[i] = X[for_testing]
            y_test[i] = Y[for_testing]
            options.remove(for_testing)
        # the remaining options are for training
        for i in range(16):
            x_train[i] = X[options[i]]
            y_train[i] = Y[options[i]]

        linear_xs, linear_ys, gradient, y_intercept = linear_line(x_train, y_train)
        polynomial_xs, polynomial_ys, poly_coefs = polynomial_line(x_train, y_train)
        unknown_func_xs, unknown_func_ys, unknown_coefs = unknown_line(x_train, y_train)

        # calculating errors
        linear_error = find_linear_error(gradient, y_intercept, x_test, y_test)
        polynomial_error = find_poly_error(poly_coefs, x_test, y_test)
        unknown_error = find_unknown_error(unknown_coefs, x_test, y_test)

        # deciding which is better, and storing it
        this_error = min(linear_error, polynomial_error, unknown_error)
        if this_error == linear_error:
            all_results.append(["linear", linear_error])
        elif this_error == polynomial_error:
            all_results.append(["polynomial", polynomial_error])
        else:
            all_results.append(["unknown", unknown_error])
        j += 1
    return all_results


def run_calculations():
    chosen_one = find_best(cross_validation())
    # recalculating the lines and errors for the whole data set and returning it
    # print("Chose ", chosen_one[0], " with error of ", chosen_one[1])
    if chosen_one[0] == "linear":
        true_xs, true_ys, true_gradient, true_y_intercept = linear_line(X, Y)
        true_error = find_linear_error(true_gradient, true_y_intercept, X, Y)
    elif chosen_one[0] == "polynomial":
        true_xs, true_ys, true_coefficient = polynomial_line(X, Y)
        true_error = find_poly_error(true_coefficient, X, Y)
    else:
        true_xs, true_ys, true_coefficient = unknown_line(X, Y)
        true_error = find_unknown_error(true_coefficient, X, Y)
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
if len(sys.argv) >= 3:
    if sys.argv[2] == "--plot":
        view_data_segments(totalX, totalY)  # show each section
print(error)
