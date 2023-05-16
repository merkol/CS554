import numpy as np
import matplotlib.pyplot as plt

## Utils

# function to plot the model
def plot(y_hat, X, Y, ax, network):
    ax.plot(X, y_hat, color="r", label=network.__class__.__name__ + " with hidden units " + str(network.hidden) )
    ax.scatter(X, Y ,color="b", label="data points")
    ax.set_title("Fitting with " + str(network.__class__.__name__) + " with hidden units " + str(network.hidden))
    ax.set_xlabel("X")
    ax.set_ylabel("R")

# function to create a figure
def create_fig():
    fig = plt.figure()
    ax = fig.subplots()
    return ax

class SLP():
    def __init__(self, lr, iters) -> None:
        self.w0 =  np.random.randn(1, 1) * 0.01
        self.b0 = np.zeros(1)
        self.hidden = 0
        self.lr = lr
        self.iters = iters
        
    def forward(self, x):
        self.a = np.dot(x, self.w0) + self.b0
        return self.a
    
    def backward(self, x, y):
        error = (self.a - y)
        
        dw = (1/y.size) * np.dot(x.T, error)
        db = (1/y.size) * np.sum(error)
        
        self.w0 -= self.lr * dw
        self.b0 -= self.lr * db
    
    def train(self, X, Y):
        self.minimum_loss = np.inf
        for iter in range(self.iters):
            
            self.forward(X)
            self.backward(X, Y)
            epoch_loss = self.loss(Y)
            if epoch_loss < self.minimum_loss:
                self.minimum_loss = epoch_loss
            if (iter+1) % 100 == 0:
                print(f"Iteration {iter+1}/{self.iters}, Loss: {epoch_loss:.4f}")
        print()
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def loss(self, y):
        return 0.5 * np.mean((y - self.a) ** 2)
    
class MLP(SLP):
    def __init__(self,hidden, lr, iters) -> None:
        super().__init__(lr, iters)
        self.hidden = hidden
        self.w0 = np.random.randn(1, self.hidden) * 0.01
        self.b0 = np.zeros(self.hidden)
        self.w1 = np.random.randn(self.hidden, 1)  * 0.01
        self.b1 = np.zeros(1)
    
    def forward(self, x):
        self.z0 = np.dot(x, self.w0) + self.b0
        self.a0 = self.sigmoid(self.z0)
        self.a1 = np.dot(self.a0, self.w1) + self.b1
        return self.a1
        
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def backward(self, x, y):
        
        error = (self.a1 - y)
        
        # calculate the gradients
        dw1 = (1/y.size) * np.dot(self.a0.T, error)
        db1 = (1/y.size) * np.sum(error)
    
        hidden_error = np.dot(error, self.w1.T) * self.sigmoid_derivative(self.z0)
        dw0 = (1/y.size) * np.dot(x.T, hidden_error)
        db0 = (1/y.size) * np.sum(hidden_error)
        
        # Update the weights and biases
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1
        self.w0 -= self.lr * dw0
        self.b0 -= self.lr * db0
        

    def train(self, X, Y):
        print(f"--------Training MLP with {self.hidden} hidden layers--------")
        return super().train(X, Y)
    
    def loss(self, y):
        return 0.5 * np.mean((y - self.a1) ** 2)
    

def run():
    # load the data
    train_data = np.loadtxt("data/train.csv", delimiter=",", skiprows=1)
    test_data = np.loadtxt("data/test.csv", delimiter=",", skiprows=1)
    
    # sort the data
    train_data = train_data[train_data[:, 0].argsort()]
    test_data = test_data[test_data[:, 0].argsort()]
    
    # split the data into x and y
    x_train, y_train = train_data[:, 0].reshape(-1,1), train_data[:, 1].reshape(-1,1)
    x_test, y_test = test_data[:, 0].reshape(-1,1), test_data[:, 1].reshape(-1,1)

    # create the networks and train them
    slp = SLP(lr = 1e-1, iters = 30)
    print("-------Training Single Layer Perceptron------")
    slp.train(x_train, y_train)
    
    mlp_10 = MLP(hidden = 10, lr = 0.1, iters = 5000)
    mlp_10.train(x_train, y_train)
    
    mlp_20 = MLP(hidden = 20, lr = 0.1, iters = 5000)
    mlp_20.train(x_train, y_train)
    
    mlp_50 = MLP(hidden = 50, lr = 0.1, iters = 5000)
    mlp_50.train(x_train, y_train)
    
    networks = [slp, mlp_10, mlp_20, mlp_50]
    
    # list of figures
    figures = []
    loss_fig = create_fig()

    # create the figures for each network
    for i in range(len(networks)):
        figures.append(create_fig())
        
    # linearly spaced x values for smooth plotting
    lin = np.linspace(-2.5, 2.5, 20).reshape(-1, 1)

    # Train predictions
    train_predictions = [network.forward(lin) for network in networks]
    # train_predictions = [preds[preds[:, 0].argsort()] for preds in train_predictions]

    # Train losses
    train_losses = [slp.minimum_loss, mlp_10.minimum_loss, mlp_20.minimum_loss, mlp_50.minimum_loss]

    # Plotting the models separately
    for i in range(len(networks)):
        plot(train_predictions[i], lin,
             y_train, figures[i], network=networks[i])
        figures[i].legend()
    
    # Test predictions
    test_predictions = [network.forward(x_test) for network in networks]
    test_predictions = [preds[preds[:, 0].argsort()] for preds in test_predictions]
    
    # Test losses
    test_losses = [network.loss(y_test) for network in networks]

    # Plotting the train and test losses
    loss_fig.plot([network.hidden for network in networks], train_losses, label="Train Loss", color="blue")
    loss_fig.plot([network.hidden for network in networks], test_losses, label="Test Loss", color="red")
    loss_fig.set_title("Loss vs Hidden Layers")
    loss_fig.set_xlabel("Hidden Layers")
    loss_fig.set_ylabel("Loss")
    loss_fig.set_xticks([0, 10, 20, 50])
    loss_fig.legend()
    plt.show()
    

if __name__ == "__main__":
    run()
    