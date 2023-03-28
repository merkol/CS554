import numpy as np
import matplotlib.pyplot as plt
from utils import *

def run():
    # Setting parameters
    equal_loss = True
    
    # Creating figures
    likelihoods = create_fig()
    train_post = create_fig()
    test_post = create_fig()
    

    # load data
    train_data = np.loadtxt("data/train.csv", delimiter=",")
    test_data = np.loadtxt("data/test.csv", delimiter=",")
    
    # Creating a list of data to loop over
    data_loop = [train_data, test_data]

    # Splitting data into Polo and Golf
    polo_x, golf_x, polo_y, golf_y = (
        train_data[train_data[:, 1] == 0, 0],
        train_data[train_data[:, 1] == 1, 0],
        train_data[train_data[:, 1] == 0, 1],
        train_data[train_data[:, 1] == 1, 1]
    )
    
    # Calculating mean and variance of Polo and Golf
    polo_mean , polo_var = np.mean(polo_x) , np.var(polo_x)
    golf_mean , golf_var = np.mean(golf_x) , np.var(golf_x)
    
    # Calculating P(Age | Polo) and P(Age | Golf) likelihoods
    polo_dist = gaussian_dist(polo_x, polo_mean, polo_var)
    golf_dist = gaussian_dist(golf_x, golf_mean, golf_var)
    
    
    # Plotting P(Age | Polo) and P(Age | Golf) likelihoods
    plot_line(polo_x, polo_dist,"Gaussian Distribution","Age","probabilities","P(Age | Polo)","orange",likelihoods)
    plot_line(golf_x, golf_dist,"Gaussian Distribution", "Age","probabilities","P(Age | Golf)","red", likelihoods)
    
    # Scattering Polo and Golf Values 
    plot_scatter(polo_x, polo_y,"Gaussian Distribution", "Age","probabilities","Polo","orange",likelihoods)
    plot_scatter(golf_x, polo_y,"Gaussian Distribution", "Age","probabilities","Golf","red",likelihoods)
    
    # Calculate P(Polo) and P(Golf)
    p_polo , p_golf = calcualate_prior(train_data)
    
    
    # Calculate P(Polo | Age) and P(Golf | Age) on both training and test data
    train_probas = [predict(train_data[i, 0], p_golf, p_polo, golf_mean, golf_var, polo_mean, polo_var) for i in range(len(train_data))]
    test_probas = [predict(test_data[i, 0], p_golf, p_polo, golf_mean, golf_var, polo_mean, polo_var) for i in range(len(test_data))]
    

    # Plotting P(Polo | Age) and P(Golf | Age) on both training and test data
    for data in data_loop:
        results = train_probas if data is train_data else test_probas
        axes = train_post if data is train_data else test_post
        title = "Posteriors on training" if data is train_data else "Posteriors on test"
        plot_line(data[:, 0], [proba[0] for proba in results],title,"Age","probabilities","P(Polo | Age)","orange", axes)
        plot_line(data[:, 0], [proba[1] for proba in results],title,"Age","probabilities","P(Golf | Age)","red", axes)
    
    # Calculating the results and define reject threshold
    if equal_loss:
        train_results = [1 if proba[1] > proba[0] else 0 for proba in train_probas]
        test_results = [1 if proba[1] > proba[0] else 0 for proba in test_probas]
    else:
        polo_cost, golf_cost = 10, 5
        
        if polo_cost > golf_cost:
            thresholds_and_results = calculate_thresholds(polo_cost, golf_cost, train_probas, test_probas,0)
            
            train_results = thresholds_and_results[0]
            test_results = thresholds_and_results[1]
            
            train_post.axhline(y=thresholds_and_results[2], color='blue', label="Reject Threshold Polo")
            train_post.axhline(y=thresholds_and_results[3], color='blue')
            train_post.legend()
            
        elif polo_cost < golf_cost:
            
            thresholds_and_results = calculate_thresholds(golf_cost, polo_cost, train_probas, test_probas,1)
            
            train_results = thresholds_and_results[0]
            test_results = thresholds_and_results[1]
            
            train_post.axhline(y=thresholds_and_results[2], color='blue', label="Reject Threshold Golf")
            train_post.axhline(y=thresholds_and_results[3], color='blue')
            train_post.legend()
            
        
    # Visualize the results
    classify_and_visualize(train_data, train_results, not equal_loss)
    classify_and_visualize(test_data, test_results, not equal_loss)
    
    # Calculate accuracy
    print(f"Accuracy of Train : {accuracy(train_data[:, 1], train_results).round(2)}")
    print(f"Accuracy of Test : {accuracy(test_data[:, 1], test_results).round(2)}")
    print()
    print(f"Precision of Train : {precision(train_data[:, 1], train_results, labels = [0,1] if equal_loss else [0,1,-1]).round(2)}")
    print(f"Recall of Train : {recall(train_data[:, 1], train_results, labels = [0, 1] if equal_loss else [0,1,-1]).round(2)}")
    print()
    print(f"Precision of Test : {precision(test_data[:, 1], test_results, labels = [0, 1] if equal_loss else [0,1,-1]).round(2)}")
    print(f"Recall of Test : {recall(test_data[:, 1], test_results, labels = [0, 1] if equal_loss else [0,1,-1]).round(2)}")
    
    # Comparing with sklearn
    # print(predict(35, p_golf, p_polo, golf_mean, golf_var, polo_mean, polo_var))
    
    plt.show()


if __name__ == "__main__":
    run()