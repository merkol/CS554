import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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


# Classify and visualize 
def classify_and_visualize(data, results):
    cm = confusion_matrix(data[:, 1], results)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Polo", "Golf"])
    disp.plot()


def create_fig():
    fig = plt.figure()
    ax = fig.subplots()
    return ax
    
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



