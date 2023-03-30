import numpy as np
import os
from utils import generate_data, merge_data

if __name__ == "__main__":
    # Generating data for training
    polo_train = generate_data(mu=29, sigma=6, n = 100, label=0)
    golf_train = generate_data(mu=44, sigma=4,n = 100, label=1)

    # Generating data for testing
    polo_test = generate_data(mu=29, sigma=6, n=30, label=0)
    golf_test = generate_data(mu=44, sigma=4, n=30, label=1)

    # Merging data
    train = merge_data(polo_train, golf_train)
    test = merge_data(polo_test, golf_test)

    # Creating data directory
    os.makedirs("data") if not os.path.exists("data") else None

    # Saving data to csv files
    np.savetxt("data/train.csv", train, delimiter=",", fmt="%u")
    np.savetxt("data/test.csv", test, delimiter=",", fmt="%u")
