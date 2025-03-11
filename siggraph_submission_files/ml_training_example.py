import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from ml_pipeline import ACESModel, BrightnessDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}")
model = ACESModel().to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()

train_set = BrightnessDataset("images", "brightnesses", downsample_ratio=4)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)

max_epochs = 10
e = 0
while e < max_epochs:
    train_losses = []
    for sample, brightness in tqdm(train_loader, desc=f"Epoch {e}"):
        sample = sample.to(device)
        brightness = brightness.float().to(device)
        prediction = model(sample)
        loss = loss_fn(prediction, brightness)
        train_losses.append(loss.detach().cpu().item())
        optim.zero_grad()
        loss.backward()
        optim.step()
    print(f"Average training loss: {np.mean(train_losses)}")
    print(f"Current learned params: {model.weights.detach().cpu().numpy()}")
