import os
import time
import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import random_split
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

data_dir = "./cifar10"

train_dataset = torchvision.datasets.CIFAR10(data_dir, train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10(data_dir, train=False, download=True)

train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

train_dataset.transform = train_transform
test_dataset.transform = test_transform

m = len(train_dataset)

train_data, val_data = random_split(train_dataset, [int(m - m * 0.2), int(m * 0.2)])
batch_size = 256

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


class Encoder(nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        return x


class Decoder(nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.ConvTranspose2d(16, 6, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 3, kernel_size=5),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        return x


loss_fn = torch.nn.MSELoss()

lr = 0.001
torch.manual_seed(0)

d = 4
print(train_dataset[0][0].shape)
encoder = Encoder(encoded_space_dim=d, fc2_input_dim=128)
decoder = Decoder(encoded_space_dim=d, fc2_input_dim=128)
params_to_optimize = [
    {"params": encoder.parameters()},
    {"params": decoder.parameters()},
]

optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
print(f"Selected device: {device}")

encoder.to(device)
decoder.to(device)


def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    encoder.train()
    decoder.train()
    train_loss = []
    for image_batch, _ in dataloader:
        image_batch = image_batch.to(device)
        encoded_data = encoder(image_batch)
        decoded_data = decoder(encoded_data)
        loss = loss_fn(decoded_data, image_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print("\t partial train loss (single batch): %f" % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            image_batch = image_batch.to(device)
            encoded_data = encoder(image_batch)
            decoded_data = decoder(encoded_data)
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data


def calculate_metrics(original, reconstructed):
    psnr = peak_signal_noise_ratio(original, reconstructed)  # Determine the appropriate win_size
    ssim = structural_similarity(
        original,
        reconstructed,
        win_size=3,
        multichannel=True,
        data_range=(original.max() - original.min()),
    )
    return psnr, ssim


def plot_ae_outputs(encoder, decoder, test_dataset, save_dir, n=10):
    plt.figure(figsize=(16, 4.5))
    targets = np.array(test_dataset.targets)
    t_idx = {i: np.where(targets == i)[0][0] for i in range(n)}

    psnr_list = []
    ssim_list = []

    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            rec_img = decoder(encoder(img))

        img = img.cpu().squeeze().numpy().transpose(1, 2, 0)
        rec_img = rec_img.cpu().squeeze().numpy().transpose(1, 2, 0)

        plt.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if i == n // 2:
            ax.set_title("Original images")

        ax = plt.subplot(2, n, i + 1 + n)
        rec_img = np.clip(rec_img, 0, 1)
        rec_img = np.clip(rec_img, 0, 255)
        plt.imshow(rec_img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if i == n // 2:
            ax.set_title("Reconstructed images")

        # Save reconstructed images
        save_path = os.path.join(save_dir, f"reconstructed_{i}.png")
        plt.imsave(save_path, rec_img)

        # Calculate metrics
        psnr, ssim = calculate_metrics(img, rec_img)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

    plt.savefig(os.path.join(save_dir, "plot.png"))
    plt.close()

    print(f"Average PSNR: {np.mean(psnr_list):.2f}")
    print(f"Average SSIM: {np.mean(ssim_list):.4f}")

    return np.mean(psnr_list), np.mean(ssim_list)


num_epochs = 100
psnr_list, ssim_list = [], []
diz_loss = {"train_loss": [], "val_loss": []}
st = time.time()
for epoch in range(num_epochs):
    train_loss = train_epoch(encoder, decoder, device, train_loader, loss_fn, optimizer)
    val_loss = test_epoch(encoder, decoder, device, test_loader, loss_fn)
    print(
        "\n EPOCH {}/{} \t train loss {} \t val loss {}".format(
            epoch + 1, num_epochs, train_loss, val_loss
        )
    )
    diz_loss["train_loss"].append(train_loss)
    diz_loss["val_loss"].append(val_loss)
    psnr, ssim = plot_ae_outputs(
        encoder, decoder, test_dataset=test_dataset, save_dir="./saved", n=10
    )

    psnr_list.append(psnr)
    ssim_list.append(ssim)
    # Plot psnr and ssim
et = time.time()
print(f"ELAPSED TIME IN MINUTES: {(et-st)/60:.2f}")
plt.figure(figsize=(6, 6))
plt.plot(list(range(num_epochs)), psnr_list, "-r", label="PSNR")
plt.title("PSNR")
plt.show()
plt.savefig(os.path.join("./saved", "psnr.png"))

plt.figure(figsize=(6, 6))
plt.plot(list(range(num_epochs)), ssim_list, label="SSIM")
plt.title("SSIM")
plt.show()
plt.savefig(os.path.join("./saved", "ssim.png"))
