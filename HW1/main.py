import numpy as np
import matplotlib.pyplot as plt

# function to fit and predict the model
def fit_predict(X, Y, degree):
    z = np.polyfit(X, Y, degree)
    y_hat = np.polyval(z, X)
    return z, y_hat

# function to predict the model
def predict(model, X):
    y_hat = np.polyval(model, X)
    return y_hat

# function to plot the model
def plot(y_hat, X, train_x, Y, ax, degree):
    ax.plot(X, y_hat, color="r", label=str(degree) + " degree")
    ax.scatter(train_x, Y, color="b", label="data points")
    ax.set_title("Fitting with " + str(degree) + " degree")
    ax.set_xlabel("X")
    ax.set_ylabel("R")


# function to create a figure
def create_fig():
    fig = plt.figure()
    ax = fig.subplots()
    return ax


# function to plot errors on each degree
def error_plot(errors, degrees, ax, dataset):
    if dataset == "Train":
        ax.plot(degrees, errors, color="b", label=dataset + " error")
    else:
        ax.plot(degrees, errors, color="r", label=dataset + " error")

    ax.set_title("Error plot")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Error")

# function to calculate SSE
def calculate_SSE(y_hat, y):
    errors = []
    for i in range(len(y_hat)):
        errors.append(np.sum((y_hat[i] - y) ** 2))
    return errors


def main():
    # load the data
    train_data = np.loadtxt(
        "data/train.csv", delimiter=",", skiprows=1, dtype=np.float32
    )
    test_data = np.loadtxt("data/test.csv", delimiter=",",
                           skiprows=1, dtype=np.float32)
    # sort the data
    train_data = train_data[train_data[:, 0].argsort()]
    test_data = test_data[test_data[:, 0].argsort()]

    train_x, train_y = train_data[:, 0], train_data[:, 1]
    test_x, test_y = test_data[:, 0], test_data[:, 1]

    # list of degrees
    degrees = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    # list of figures
    figures = []

    # create the figures for each degree
    for i in range(len(degrees)):
        figures.append(create_fig())

    # error figure
    error_fig = create_fig()

    # list of results
    train_y_hats = []
    test_y_hats = []
    plts = []

    # Fitting
    for degree in degrees:
        # Fitting and predicting
        model, train_y_hat = fit_predict(train_x, train_y, degree)
        train_y_hats.append(train_y_hat)

        # predicting on test data
        test_y_hat = predict(model, test_x)
        test_y_hats.append(test_y_hat)
        
        # predicting for the plot
        lin = np.linspace(min(train_x), max(train_x), 100)
        y_h = predict(model, lin)
        plts.append((lin, y_h))

    # Calculating and plotting train errors for each degree
    train_errors = calculate_SSE(train_y_hats, train_y)
    error_plot(train_errors, degrees, error_fig, "Train")
    
    # Calculating and plotting test errors for each degree
    test_errors = calculate_SSE(test_y_hats, test_y)
    error_plot(test_errors, degrees, error_fig, "Test")
    
    # Plotting the models separately
    for i in range(len(degrees)):
        plot(plts[i][1], plts[i][0], train_x,
             train_y, figures[i], degree=i)
        figures[i].legend()

    error_fig.legend()
    plt.show()


if __name__ == "__main__":
    main()
