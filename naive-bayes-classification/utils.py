import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score

# Gaussian distribution


def gaussian_dist(x, mu, var):
    return np.exp(-(x - mu)**2 / (2 * var)) / (np.sqrt(2 * np.pi * var))

# Generate data
def generate_data(mu, sigma, n, label):
    # Generate n integer data points with label
    return np.append(np.random.normal(mu, sigma, (n, 1)), label * np.ones((n, 1), dtype=int), axis=1)

# Merge data
def merge_data(data1, data2):
    merged = np.concatenate((data1, data2), axis=0)
    # Sort data by age for better visualization
    return merged[merged[: , 0].argsort()]


def calcualate_prior(data):
    # Calculate P(Golf) and P(Polo)
    polo = data[data[:, 1] == 0, 0]
    golf = data[data[:, 1] == 1, 0]
    return len(polo) / len(data), len(golf) / len(data)


def predict(age, p_golf, p_polo, golf_mean, golf_var, polo_mean, polo_var):
    # Calculate P(Age) = P(Age | Polo) * P(Polo) + P(Age | Golf) * P(Golf)
    p_age_polo = gaussian_dist(age, polo_mean, polo_var)
    p_age_golf = gaussian_dist(age, golf_mean, golf_var)
    
    p_age = (p_polo * p_age_polo) + (p_golf * p_age_golf)
    
    # Calculate P(Class | Age) = P(Age | Class) * P(Class) 
    #                                   / P(Age)
    
    p_polo_age = (p_polo * p_age_polo) / p_age
    p_golf_age = (p_golf * p_age_golf) / p_age
    return p_polo_age, p_golf_age


# Calculate accuracy
def accuracy(true, pred):
    return (true == pred).sum() / len(true)

def precision(true, pred ,labels):
    return precision_score(true, pred, labels =labels, zero_division=0, average = None )

def recall(true, pred, labels):
    return recall_score(true, pred, labels = labels, zero_division=0, average= None)



# Classify and visualize 
def classify_and_visualize(data, results, reject = False):
    cm = confusion_matrix(data[:, 1], results) if not reject else confusion_matrix(data[:, 1], results, labels=[0, 1, -1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Polo", "Golf"]) if not reject else ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Polo", "Golf", "Reject"])
    disp.plot()


def create_fig():
    fig = plt.figure()
    ax = fig.subplots()
    return ax

def calculate_thresholds(higher_cost, lower_cost, train_probas, test_probas, higher_class):
    select_threshold = (higher_cost - 1) / higher_cost
    not_select_threshold = 1 / lower_cost
            
    train_results = []
    for proba in train_probas:
        if proba[higher_class] > select_threshold:
            train_results.append(higher_class)
        elif proba[higher_class] < not_select_threshold:
            train_results.append(not higher_class)
        else:
            train_results.append(-1)
            
    test_results = []
    for proba in test_probas:
        if proba[higher_class] > select_threshold:
            test_results.append(higher_class)
        elif proba[higher_class] < not_select_threshold:
            test_results.append(not higher_class)
        else:
            test_results.append(-1)
            
    return train_results, test_results, select_threshold, not_select_threshold
# Plotting data
def plot_line(X, Y, title, xlabel, ylabel, label, color, ax):
    ax.plot(X, Y, label=label, color = color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

def plot_scatter(X, Y, title, xlabel, ylabel, label, color, ax):
    ax.scatter(X, Y, label=label, color = color, s = 10)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()



