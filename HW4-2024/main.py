import torch
import torch.nn as nn
import torch.utils
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F
    
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_loss(train_loss_list, val_loss_list, epoch, title):
    plt.figure(figsize=(12, 8))
    plt.plot(epoch, train_loss_list, label="Train Loss", color="red")
    plt.plot(epoch, val_loss_list, label="Validation Loss", color="blue")
    plt.text(epoch[-1], train_loss_list[-1], f'{train_loss_list[-1]:.4f}', ha='right', va='bottom', color='red')
    plt.text(epoch[-1], val_loss_list[-1], f'{val_loss_list[-1]:.4f}', ha='right', va='top', color='blue')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.savefig(f"{title}.png")
    plt.show()


def plot_ae_outputs(model, title, classes=10, samples_per_class=1, version="autoencoder"):
    plt.figure(figsize=(16, 2 * 4.5 * samples_per_class))
    targets = test_dataset.targets.numpy()
    t_idx = {i: np.where(targets == i)[0][:samples_per_class] for i in range(classes)}
    for i in range(classes):
        for j in range(samples_per_class):
            # plot original image
            ax = plt.subplot(2 * samples_per_class, classes, i + 1 + j * 2 * classes)
            img = test_dataset[t_idx[i][j]][0].unsqueeze(0).to(device)
            model.eval()
            with torch.no_grad():
                if version == "fully-connected":
                    img = img.view(-1, 784)
                rec_img = model(img)
            if version == "fully-connected":
                img = img.view(28, 28)
                rec_img = rec_img.view(28, 28)
            plt.imshow(img.cpu().squeeze().numpy(), cmap="gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == classes // 2:
                ax.set_title("Original images")

            # plot reconstructed image
            ax = plt.subplot(2 * samples_per_class, classes, i + 1 + (j * 2 + 1) * classes)
            plt.imshow(rec_img.cpu().squeeze().numpy(), cmap="gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == classes // 2:
                ax.set_title("Reconstructed images")

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.savefig(f"{title}.png", bbox_inches="tight", pad_inches=0)
    plt.show()



class CNNAE_v1(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(True),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_linear = nn.Sequential(
            nn.Linear(128 * 4 * 4 , hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 128 * 4 * 4),
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 4, 4))
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=(3, 3), stride=2, padding=0, output_padding=1),
        )
    
    def encode(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_linear(x)
        return x
    
    def decode(self, x):
        x = self.decoder_linear(x)
        x = self.unflatten(x)
        x = self.decoder_cnn(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        x = torch.sigmoid(x)
        return x
    
class CNNAE_v2(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(True),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_linear = nn.Sequential(
            nn.Linear(784 * 4  , latent_dim),
        )
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, 784 * 4),
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 7, 7))
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=(3, 3), stride=2, padding=0, output_padding=1),
        )
    
    def encode(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_linear(x)
        return x
    
    def decode(self, x):
        x = self.decoder_linear(x)
        x = self.unflatten(x)
        x = self.decoder_cnn(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        x = torch.sigmoid(x)
        return x

class FCAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(True),
            nn.Linear(256, latent_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 784),
        )
    
    def encode(self, x):
        x = self.encoder(x)
        return x
    
    def decode(self, x):
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        x = torch.sigmoid(x)
        return x
    
# Train the model
def train(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs ,version="autoencoder"):
    mean_train_loss, validation_loss = [], []
    model.train()
    for epoch in range(epochs):
        train_loss = []
        for step, (batch, label) in enumerate(train_loader):
            model.train()
            if version == "fully-connected":
                batch = batch.view(-1, 784)              
            batch = batch.to(device)
            label = label.to(device)
            output = model(batch)  
            optimizer.zero_grad()
            if version == "autoencoder" or version == "fully-connected":
                loss = criterion(output, batch)
                loss.backward()
            else:
                loss = criterion(output, label)
                loss.backward(retain_graph=True)
            train_loss.append(loss.item())
            optimizer.step()

        mean_train_loss.append(np.mean(train_loss))

        # Validation
        model.eval()
        with torch.no_grad():
            preds, ys = [], []
            for step, (batch, label) in enumerate(test_loader):
                batch = batch.to(device)
                label = label.to(device)
                if version == "fully-connected":
                    batch = batch.view(-1, 784)
                output = model(batch)  
                preds.append(output.cpu())
                if version == "autoencoder" or version == "fully-connected":
                    ys.append(batch.cpu())
                else:
                    ys.append(label.cpu())
            preds, ys = torch.cat(preds) , torch.cat(ys)
            val_loss = criterion(preds, ys)  
            validation_loss.append(val_loss.data)
            scheduler.step(val_loss)
            
        print(
            f"{version.capitalize()} Model, Epoch: {epoch}, Train Loss: {mean_train_loss[epoch]:.4f}, Val Loss: {validation_loss[epoch]:.4f}"
        )
        
    return mean_train_loss, validation_loss


def create_dataset_from_hidden_representation(model, loader, device, version="autoencoder"):
    hidden_representations = torch.tensor([], dtype=torch.float32, device=device)
    labels = torch.tensor([], dtype=torch.int64, device=device)
    for batch, label in loader:
        batch = batch.to(device)
        label = label.to(device)
        if version == "fully-connected":
            batch = batch.view(-1, 784)
        hidden_representation = model.encode(batch)
        hidden_representations = torch.cat((hidden_representations, hidden_representation), dim=0)
        labels = torch.cat((labels, label), dim=0)
    return hidden_representations, labels  

def calculate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch, label in loader:
            batch = batch.to(device)
            label = label.to(device)
            output = model(batch)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    return correct / total

if __name__ == "__main__":

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root="./mnist", train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./mnist", train=False, transform=transforms.ToTensor(), download=True
    )

    # Data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=512, shuffle=True)

    # Hyper Parameters
    EPOCH = 20
    LR = 0.001

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    latent_dim, hidden_dim = 16, 256
    version = "autoencoder"# "autoencoder" or "fully-connected"
    model_name = "CNNAE_v1" # "CNNAE_v1", "CNNAE_v2", "FCAE"
    
    if model_name == "CNNAE_v1":
        model = CNNAE_v1(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    elif model_name == "CNNAE_v2":
        model = CNNAE_v2(latent_dim=latent_dim).to(device)
    else:
        model = FCAE(latent_dim=latent_dim).to(device)
    print(f"Number of parameters: {count_parameters(model)}")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", patience=1, factor=0.5
    )
    mean_train_loss, validation_loss = train(model, train_loader, test_loader, criterion, optimizer, scheduler, EPOCH, version=version) # "autoencoder" or "fully-connected"
    
    plot_ae_outputs(model, title=model.__class__.__name__ ,version=version)
    plot_loss(mean_train_loss, validation_loss, np.linspace(0, EPOCH, EPOCH), model.__class__.__name__ + " autoencoder loss")
    
    train_hidden, train_labels = create_dataset_from_hidden_representation(model, train_loader, device, version=version)
    test_hidden, test_labels = create_dataset_from_hidden_representation(model, test_loader, device, version=version)
    
    # create dataloader from hidden representations
    train_loader = DataLoader(torch.utils.data.TensorDataset(train_hidden, train_labels), batch_size=512, shuffle=True)
    test_loader = DataLoader(torch.utils.data.TensorDataset(test_hidden, test_labels), batch_size=512, shuffle=True)

    # Train a classifier on the hidden representations
    classifier = nn.Sequential(
        nn.Linear(latent_dim, 10),
    ).to(device)
    
    EPOCH = 40
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", patience=1, factor=0.5
    )
    mean_train_loss, validation_loss = train(classifier, train_loader, test_loader, criterion, optimizer, scheduler, EPOCH, version="classifier") # keep it="classifier"
    
    plot_loss(mean_train_loss, validation_loss, np.linspace(0, EPOCH, EPOCH), model.__class__.__name__ + " classifier loss")
    print(f"Classifier Accuracy: {calculate_accuracy(classifier, test_loader, device):.4f}")
    

