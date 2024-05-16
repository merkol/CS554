import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


def plot_loss(train_loss_list, val_loss_list, epoch):
    plt.figure(figsize=(12, 8))
    plt.plot(epoch, train_loss_list, label="Train Loss", color="red")
    plt.plot(epoch, val_loss_list, label="Validation Loss", color="blue")
    plt.title("Loss")
    plt.legend()
    plt.savefig("loss.png")
    plt.show()


def plot_ae_outputs(model, classes=10, samples_per_class=1):
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
                rec_img = model(img)
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
    plt.savefig("ae_outputs.png", bbox_inches="tight", pad_inches=0)
    plt.show()


transform = transforms.Compose([
    transforms.ToTensor(),
])


# FashionMNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root="./mnist", train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root="./mnist", train=False, transform=transform, download=True
)

# Data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=512, shuffle=True)

# Hyper Parameters
EPOCH = 20
LR = 0.001


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1),
            nn.LeakyReLU(True),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_linear = nn.Sequential(
            nn.Linear(128 * 4 * 4 , hidden_dim),
            nn.LeakyReLU(True),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(True),
            nn.Linear(hidden_dim, 128 * 4 * 4),
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 4, 4))
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=(3, 3), stride=2, padding=0, output_padding=1),
        )

    def forward(self, x):
        # Encoder
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_linear(x)

        # Decoder
        x = self.decoder_linear(x)
        x = self.unflatten(x)
        x = self.decoder_cnn(x)
        x = torch.sigmoid(x)
        return x


# Train the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
print(f"Using {device}")
model = AutoEncoder(latent_dim=64, hidden_dim=256).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, mode="min", patience=1, factor=0.5
)

mean_train_loss, validation_loss = [], []
for epoch in range(EPOCH):
    train_loss = []
    for step, (batch, _) in enumerate(train_loader):
        model.train()
        batch = batch.to(device)
        output = model(batch)  
        loss = criterion(output, batch)  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    mean_train_loss.append(np.mean(train_loss))

    # Validation
    model.eval()
    with torch.no_grad():
        preds, ys = [], []
        for step, (batch, _) in enumerate(test_loader):
            batch = batch.to(device)
            output = model(batch)  
            preds.append(output.cpu())
            ys.append(batch.cpu())
        preds, ys = torch.cat(preds) , torch.cat(ys)
        val_loss = criterion(preds, ys)  
        validation_loss.append(val_loss.data)
        scheduler.step(val_loss)
    print(
        f"Epoch: {epoch}, Train Loss: {mean_train_loss[epoch]:.4f}, Val Loss: {validation_loss[epoch]:.4f}"
    )

plot_ae_outputs(model)
plot_loss(mean_train_loss, validation_loss, np.linspace(0, EPOCH, EPOCH))
