import numpy as np
import matplotlib.pyplot as plt


def plot_centroids(centroids):
    if len(centroids) == 10:
        x, y = 2, 5
    elif len(centroids) == 20:
        x, y = 4, 5
    elif len(centroids) == 30:
        x, y = 6, 5
    fig, ax = plt.subplots(x, y, figsize=(8, 8))
    fig.suptitle("Cluster centroids for MNIST for k = " + str(len(centroids)), fontsize=12)
    for i, ax in enumerate(ax.flat):
        ax.imshow(centroids[i].reshape(28, 28), cmap='gray')
        ax.axis('off')

                
class KMeansClustering:
    def __init__(self,k):
        self.k = k
        self.centers = None
    
    def random_centroids(self, X):
        self.centers = X[np.random.choice(X.shape[0], self.k)]
        self.labels = np.zeros(X.shape[0])
        
    def update_centers(self, X):
        distances = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            distances[:, i] = self.euclidean_distance(X, self.centers[i])
        closest = np.argmin(distances, axis=1)
        self.error = 0
        for j in range(self.k):
            if np.sum(closest == j) > 0: 
                self.centers[j] = np.mean(X[closest == j], axis=0)
                self.labels[closest == j] = j   
                self.error += np.sum((X[closest == j] - self.centers[j]) ** 2)
            else:
                # Reinitialize the centroid randomly if the cluster has no points besides itself
                self.centers[j] = X[np.random.choice(X.shape[0])]
        
    def fit(self, X):
        while True: 
            old_centers = self.centers.copy()
            self.update_centers(X)
            if np.all(old_centers == self.centers):
                break
    
    def euclidean_distance(self, x, center):
        return np.sqrt(np.sum((x - center)**2, axis=1))
    
        
        

if __name__ == "__main__":
    train = np.loadtxt("data.csv", delimiter=",", skiprows=1)
    
    k = [1, 2, 3, 4, 5, 6]
    models, mean_errors, best_models = [], [], []
    
    for i in k:
        models.clear()
        errors = []
        for j in range(10):
            model = KMeansClustering(k=i)
            model.random_centroids(train)
            model.fit(train)
            errors.append(model.error)
            models.append(model)
        mean_errors.append(np.mean(errors))

        min_error = np.argmin(errors)
        min_model = models[min_error]
        best_models.append(min_model)
        
    plt.figure()
    plt.plot(k, mean_errors)
    plt.xticks(k)
    plt.ylabel("Error")
    plt.xlabel("k")
    plt.title("Mean Reconstruction Loss vs k")



    fig, ax = plt.subplots(2, 3, figsize=(10, 6))
    fig.suptitle("K-Means Cluster Assignments", fontsize=12)
    for i, ax in enumerate(ax.flat):
        ax.scatter(train[:,0], train[:,1], c=best_models[i].labels)
        ax.scatter(best_models[i].centers[:,0], best_models[i].centers[:,1], c='gray', s=100)
        ax.set_title(f"K-Means Solution K={i+1}", fontsize=10)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
    plt.subplots_adjust(hspace=0.6)
    
        
    mnist = np.loadtxt("mnist.csv", delimiter=",", skiprows=1)
    mnist = mnist[:,1:]
    
    
    k = [10, 20, 30]
    
    errors.clear()
    for i in k:
        model = KMeansClustering(k=i)
        model.random_centroids(mnist)
        model.fit(mnist)
        errors.append(model.error)
        plot_centroids(model.centers)
        
    plt.figure()
    plt.plot(k, errors)
    plt.xticks(k)
    plt.yticks(errors)
    plt.ylabel("Error")
    plt.xlabel("k")
    plt.title("Error vs k for MNIST")
    plt.show()
    
