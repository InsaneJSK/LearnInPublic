# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19"
# !pip install opendatasets --quiet
import opendatasets as od
od.download("https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification")

# %%
import torch
from torch import nn
from torch.optim import Adam
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# %%
train_df = pd.read_csv("/kaggle/working/bean-leaf-lesions-classification/train.csv")
val_df = pd.read_csv("/kaggle/working/bean-leaf-lesions-classification/val.csv")

train_df["image:FILE"] = "/kaggle/working/bean-leaf-lesions-classification/" + train_df["image:FILE"]
val_df["image:FILE"] = "/kaggle/working/bean-leaf-lesions-classification/" + val_df["image:FILE"]

# %%
train_df.head()

# %%
train_df.shape, val_df.shape

# %%
train_df['category'].value_counts()

# %%
train_df.info()

# %%
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])


# %%
class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe
        self.transform = transform
        self.labels = torch.tensor(dataframe["category"]).to(device)

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.labels[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)/255.0

        return image, label


# %%
train_dataset = CustomImageDataset(dataframe = train_df, transform = transform)
val_dataset = CustomImageDataset(dataframe = val_df, transform = transform)

# %%
n_rows = 3
n_cols = 3

f, axarr = plt.subplots(n_rows, n_cols)

for row in range(n_rows):
    for col in range(n_cols):
        image = train_dataset[np.random.randint(0, train_dataset.__len__())][0].cpu()
        axarr[row, col].imshow((image*255.0).squeeze().permute(1, 2, 0))
        axarr[row, col].axis("off")

plt.tight_layout()
plt.show()

# %%
LR = 1e-3
BATCH_SIZE = 4
EPOCHS = 15

# %%
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True)

# %%
googlenet_model = models.googlenet(weights = "DEFAULT")

# %%
for param in googlenet_model.parameters():
    param.required_grad = True

# %%
googlenet_model.fc

# %%
num_classes = len(train_df["category"].unique())
num_classes

# %%
googlenet_model.fc = torch.nn.Linear(googlenet_model.fc.in_features, num_classes)
googlenet_model.fc

# %% _kg_hide-output=false
googlenet_model.to(device)

# %%
loss_fun = nn.CrossEntropyLoss()
optimizer = Adam(googlenet_model.parameters(), lr = LR)

total_loss_train_plot = []
total_acc_train_plot = []

for epoch in range(EPOCHS):
    total_acc_train = 0
    total_loss_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = googlenet_model(inputs)
        train_loss = loss_fun(outputs, labels)
        total_loss_train += train_loss.item()

        train_loss.backward()

        train_acc = (torch.argmax(outputs, axis = 1) == labels).sum().item()
        total_acc_train += train_acc
        optimizer.step()

    total_loss_train_plot.append(round(total_loss_train/1000, 4))
    total_acc_train_plot.append(round(total_acc_train/train_dataset.__len__()*100, 4))
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {round(total_loss_train/1000, 4)}, Train Accuracy: {round(total_acc_train/train_dataset.__len__()*100, 4)} %")

# %%
with torch.no_grad():
    total_acc_test = 0

    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        prediction = googlenet_model(inputs)
        acc = (torch.argmax(prediction, axis = 1) == labels).sum().item()
        total_acc_test += acc

# %%
print(round(total_acc_test/val_dataset.__len__()*100, 2))

# %%
for param in googlenet_model.parameters():
    param.requires_grad = False

googlenet_model.fc = torch.nn.Linear(googlenet_model.fc.in_features, num_classes)
googlenet_model.fc
googlenet_model.to(device)

# %%
loss_fun = nn.CrossEntropyLoss()
optimizer = Adam(googlenet_model.parameters(), lr = LR)

total_loss_train_plot = []
total_acc_train_plot = []

for epoch in range(EPOCHS):
    total_acc_train = 0
    total_loss_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = googlenet_model(inputs)
        train_loss = loss_fun(outputs, labels)
        total_loss_train += train_loss.item()

        train_loss.backward()

        train_acc = (torch.argmax(outputs, axis = 1) == labels).sum().item()
        total_acc_train += train_acc
        optimizer.step()

    total_loss_train_plot.append(round(total_loss_train/1000, 4))
    total_acc_train_plot.append(round(total_acc_train/train_dataset.__len__()*100, 4))
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {round(total_loss_train/1000, 4)}, Train Accuracy: {round(total_acc_train/train_dataset.__len__()*100, 4)} %")

# %%
with torch.no_grad():
    total_acc_test = 0

    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        prediction = googlenet_model(inputs)
        acc = (torch.argmax(prediction, axis = 1) == labels).sum().item()
        total_acc_test += acc

# %%
print(round(total_acc_test/val_dataset.__len__()*100, 2))

# %%
