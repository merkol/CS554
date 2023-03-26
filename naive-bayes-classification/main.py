import numpy as np
import matplotlib.pyplot as plt
from utils import gaussian_dist, plot_line, calcualate_prior, create_fig, classify_and_visualize, accuracy, predict


if __name__ == "__main__":
    plt1 = create_fig()
    train_post = create_fig()
    test_post = create_fig()

    # load data
    train_data = np.loadtxt("data/train.csv", delimiter=",")
    test_data = np.loadtxt("data/test.csv", delimiter=",")

    # Splitting data into Polo and Golf
    polo_x, golf_x = (
        train_data[train_data[:, 1] == 0, 0],
        train_data[train_data[:, 1] == 1, 0],
    )
    
    # Calculating mean and variance of Polo and Golf
    polo_mean , polo_var = np.mean(polo_x) , np.var(polo_x)
    golf_mean , golf_var = np.mean(golf_x) , np.var(golf_x)
    
    # Calculating P(Age | Polo) and P(Age | Golf)
    polo_dist = gaussian_dist(polo_x, polo_mean, polo_var)
    golf_dist = gaussian_dist(golf_x, golf_mean, golf_var)

    # Plotting P(Age | Polo) and P(Age | Golf)
    plot_line(polo_x, polo_dist,"Gaussian Distribution","Age","probabilities","P(Age | Polo)","orange",plt1)
    plot_line(golf_x, golf_dist,"Gaussian Distribution", "Age","probabilities","P(Age | Golf)","red", plt1)
    
    # Calculate P(Polo) and P(Golf)
    p_polo , p_golf = calcualate_prior(train_data)
    
    # Calculate P(Polo | Age) and P(Golf | Age) on both training and test data
    train_results = [predict(train_data[i, 0], p_golf, p_polo, golf_mean, golf_var, polo_mean, polo_var) for i in range(len(train_data))]
    test_results = [predict(test_data[i, 0], p_golf, p_polo, golf_mean, golf_var, polo_mean, polo_var) for i in range(len(test_data))]
    
    # Visualize the results
    classify_and_visualize(train_data, train_results, train_post)
    classify_and_visualize(test_data, test_results, test_post)
    
    getattr(train_post, 'set_title')('Posteriors on training')
    getattr(test_post, 'set_title')('Posteriors on test')
    
    # Calculate accuracy
    print(f"Accuracy of Train : {accuracy(train_data[:, 1], [1 if proba[1] > proba[0] else 0 for proba in train_results])}")
    print(f"Accuracy of Test : {accuracy(test_data[:, 1], [1 if proba[1] > proba[0] else 0 for proba in test_results])}")
    
    
    plt.show()
