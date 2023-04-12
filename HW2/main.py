import numpy as np
import math

def accuracy(y_true, y_pred):
    sum = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            sum += 1
    return sum / len(y_true)

def confusion_matrix(y_true, y_pred):
    matrix = np.zeros((len(set(y_true)), len(set(y_true))))
    for i in range(len(y_true)):
        matrix[int(y_true[i])][int(y_pred[i])] += 1
    return matrix
                

class NearestMeanClassfier:
    def __init__(self):
        self.means = None
        self.classes = None
        
    # Calculating the mean of each feature for each class    
    def mean(self, X):
        sums = []
        for column in X.T:
            sum = 0
            for i in column:
                sum += i
            sums.append(sum)
        return [round(x / len(column),2) for x in sums]
    
    # Calculating the euclidean distance between an instance and the mean of each class
    def distance(self, x, mean):
        sums = []
        dst = (x-mean)**2
        for row in dst:
            sum = 0
            for element in row:
                sum += element
            sums.append(sum)
        return [math.sqrt(x) for x in sums]
            
    # Calculating the mean of each feature for each class         
    def fit(self, X, y):
        self.classes = list(set(y))
        self.means = [self.mean(X[y==c]) for c in self.classes]
        
    # Predicting the class of each instance in the dataset    
    def predict(self, X):
        return np.array([self.distance(X, m) for m in self.means]).argmin(axis=0)

class NearestNeighbourClassifier:
    def __init__(self,k=1):
        self.k = k

    def fit(self, X, Y):
        self.train_x = X
        self.train_y = Y
    
    # Predict instances on given set by calculatuon euclidean distance
    def predict(self, X):
        total_distances = []
        for i in range(len(X)):
            distances = []
            for j in range(len(self.train_x)):
                distances.append((self.euclidean_distance(X[i], self.train_x[j]), self.train_y[j]))
            distances.sort(key = lambda x: x[0])
            total_distances.append(distances)
       
       # Choosing the closest but if the distance is zero (that datapoint exists in the training set) choose the next closest one
        return [total_distances[k][0][1] if total_distances[k][0][0] != 0 else total_distances[k][1][1] for k in range(len(total_distances))]
            
    
    # Calculating the euclidean distance between two instances without using vectorization
    def euclidean_distance(self, x1, x2):
        return math.sqrt(sum((x1 - x2)**2))
        
        

if __name__ == "__main__":
    train = np.loadtxt("data/train_iris.csv", delimiter=",", skiprows=1)
    test = np.loadtxt("data/test_iris.csv", delimiter=",", skiprows=1)
    
    train_x , train_y, test_x, test_y = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
    
    print("----------------------Nearest Mean Classifier----------------------\n")
    
    NMC = NearestMeanClassfier()
    
    NMC.fit(train_x, train_y)
    
    predictions_train = NMC.predict(train_x)
    predictions_test = NMC.predict(test_x)

    accuracy_train = accuracy(train_y, predictions_train)
    accuracy_test = accuracy(test_y, predictions_test)
    
    print(f"Means of each class: {NMC.means} \n")
    
    print(f"Accuracy train: {accuracy_train:0.2f} , Error train: {1 - accuracy_train:0.2f}")
    print(f"Accuracy test: {accuracy_test:0.2f} , Error test: {1 - accuracy_test:0.2f} \n")
    
    print("Confusion matrix on train:")
    print(confusion_matrix(train_y, predictions_train), "\n")
    print("Confusion matrix on test:")
    print(confusion_matrix(test_y, predictions_test))

    print("----------------------Nearest Neighbour Classifier----------------------\n")
    

    NNC = NearestNeighbourClassifier()
    
    NNC.fit(train_x, train_y)

    train_predictions = NNC.predict(train_x)
    test_predictions = NNC.predict(test_x)

    train_accuracy = accuracy(train_y, train_predictions)
    test_accuracy = accuracy(test_y, test_predictions)

    print(f"Accuracy train: {train_accuracy:0.2f} , Error train: {1 - train_accuracy:0.2f}")
    print(f"Acuracy test: {test_accuracy:0.2f} , Error test: {1 - test_accuracy:0.2f} \n")
    
    print("Confusion matrix on train:")
    print(confusion_matrix(train_y, train_predictions), "\n")
    print("Confusion matrix on test:")
    print(confusion_matrix(test_y, test_predictions))
    
    