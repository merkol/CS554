import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
## Utils

# function to plot the model
def plot(y_hat, X, x_train, Y, ax, network):
    ax.plot(X, y_hat, color="r", label=network.__class__.__name__ + " with hidden units " + str(network.hidden) )
    ax.scatter(x_train, Y ,color="b", label="data points")
    ax.set_title("Fitting with " + str(network.__class__.__name__) + " with hidden units " + str(network.hidden))
    ax.set_xlabel("X")
    ax.set_ylabel("R")

def plot_loss_vs_iters(networks):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Iterations vs Loss")
    for i, ax in enumerate(ax.flat):
        ax.plot(range(networks[i].iters), networks[i].losses, label=networks[i].__class__.__name__ + " with hidden units " + str(networks[i].hidden), color="red")
        ax.plot(range(networks[i].iters), networks[i].val_losses, label=networks[i].__class__.__name__ + " with hidden units " + str(networks[i].hidden) + " test", color="blue")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss")
        ax.legend()

def plot_networks(networks, train_predictions, x_train, y_train, lin):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for i, ax in enumerate(ax.flat):
        plot(train_predictions[i], lin, x_train,
             y_train, ax, network=networks[i])
        ax.legend()

# function to create a figure
def create_fig():
    fig = plt.figure()
    ax = fig.subplots()
    return ax

class SLP():
    def __init__(self, lr, iters) -> None:
        self.w0 = self.xavier_init(1, 1)
        self.b0 = np.zeros(1)
        self.hidden = 0
        self.lr = lr
        self.iters = iters
        self.minimum_loss = np.inf
        self.losses = []
        self.val_losses = []

    def xavier_init(self, n_input, n_output):
        return np.random.randn(n_input, n_output) * np.sqrt(1.0/(n_input + n_output))

    def forward(self, x):
        self.a = np.dot(x, self.w0) + self.b0
        return self.a

    def backward(self, x, y):
        error = (self.a - y)

        dw = (1/y.size) * np.dot(x.T, error)
        db = (1/y.size) * np.sum(error)

        self.w0 -= self.lr * dw
        self.b0 -= self.lr * db

    def train(self, X, Y, val, val_y):
        for iter in range(self.iters):

            self.forward(X)
            self.backward(X, Y)
            iter_loss = self.loss(Y)
            self.losses.append(iter_loss)
            if iter_loss < self.minimum_loss:
                self.minimum_loss = iter_loss
            if (iter+1) % 100 == 0:
                print(f"Iteration {iter+1}/{self.iters}, Training Loss: {iter_loss:.4f}")

            self.forward(val)
            val_loss = self.loss(val_y)
            self.val_losses.append(val_loss)

        print()

    def loss(self, y):
        return np.mean((y - self.a) ** 2)

class MLP(SLP):
    def __init__(self,hidden, lr, iters) -> None:
        self.hidden = hidden
        self.w0 = self.xavier_init(1, self.hidden)
        self.b0 = np.zeros(self.hidden)
        self.w1 = self.xavier_init(self.hidden, 1)
        self.b1 = np.zeros(1)
        self.lr = lr
        self.iters = iters
        self.minimum_loss = np.inf
        self.losses = []
        self.val_losses = []

    def forward(self, x):
        self.z0 = np.dot(x, self.w0) + self.b0
        self.a0 = self.sigmoid(self.z0)
        self.a1 = np.dot(self.a0, self.w1) + self.b1
        return self.a1

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

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

        # Update the weights and biasesq
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1
        self.w0 -= self.lr * dw0
        self.b0 -= self.lr * db0


    def train(self, X, Y, val, val_y):
        print(f"--------Training MLP with {self.hidden} hidden layers--------")
        return super().train(X, Y, val, val_y)

    def loss(self, y):
        return np.mean((y - self.a1) ** 2)


def run():
    # load the data
    ## print current np random seed
    train_data = np.loadtxt("data/train.csv", delimiter=",", skiprows=1)
    test_data = np.loadtxt("data/test.csv", delimiter=",", skiprows=1)

    # sort the data
    train_data = train_data[train_data[:, 0].argsort()]
    test_data = test_data[test_data[:, 0].argsort()]

    # split the data into x and y
    x_train, y_train = train_data[:, 0].reshape(-1,1), train_data[:, 1].reshape(-1,1)
    x_test, y_test = test_data[:, 0].reshape(-1,1), test_data[:, 1].reshape(-1,1)



    # create the networks and train them
    slp = SLP(lr = 0.01, iters = 50)
    print("-------Training Single Layer Perceptron------")
    slp.train(x_train, y_train, x_test, y_test)

    mlp_2 = MLP(hidden = 2, lr = 0.1, iters = 10000)
    mlp_2.train(x_train, y_train, x_test, y_test)

    mlp_4 = MLP(hidden = 4, lr = 0.1, iters = 10000)
    mlp_4.train(x_train, y_train, x_test, y_test)

    mlp_8 = MLP(hidden = 8, lr = 0.1, iters = 10000)
    mlp_8.train(x_train, y_train, x_test, y_test)


    networks = [slp, mlp_2, mlp_4, mlp_8]

    # list of figures
    loss_fig = create_fig()

    # linearly spaced x values for smooth plotting
    lin = np.linspace(x_train.min(), x_train.max(), 1000).reshape(-1, 1)

    # Train predictions
    train_predictions = [network.forward(lin) for network in networks]
    # train_predictions = [preds[preds[:, 0].argsort()] for preds in train_predictions]

    # Train losses
    train_losses = [slp.minimum_loss, mlp_2.minimum_loss, mlp_4.minimum_loss, mlp_8.minimum_loss]

    plot_networks(networks, train_predictions, x_train, y_train, lin)
    plot_loss_vs_iters(networks)

    # Test predictions
    test_predictions = [network.forward(x_test) for network in networks]
    test_predictions = [preds[preds[:, 0].argsort()] for preds in test_predictions]

    # Test losses
    test_losses = [network.loss(y_test) for network in networks]

    # Plotting the train and test losses
    loss_fig.plot([network.hidden for network in networks], train_losses, label="Train Loss", color="blue")
    loss_fig.plot([network.hidden for network in networks], test_losses, label="Test Loss", color="red")
    loss_fig.set_title("Network Complexity vs Loss")
    loss_fig.set_xlabel("Network Complexity")
    loss_fig.set_ylabel("Loss")
    loss_fig.set_xticks([0, 2, 4, 8])
    loss_fig.legend()
    plt.show()


if __name__ == "__main__":
    run()
