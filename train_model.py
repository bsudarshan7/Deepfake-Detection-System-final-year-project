import os

import torch

import torch.nn as nn

import torch.optim as optim

from torchvision import datasets, transforms, models

from torch.utils.data import DataLoader



# =========================

# DEVICE

# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)



# =========================

# TRANSFORM

# =========================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

])



# =========================

# DATASET

# =========================

train_data = datasets.ImageFolder("dataset/train", transform=transform)

test_data  = datasets.ImageFolder("dataset/test", transform=transform)



train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

test_loader  = DataLoader(test_data, batch_size=16)



# IMPORTANT: class order used by ImageFolder

class_names = train_data.classes

print("Classes:", class_names)



# =========================

# MODEL

# =========================

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)



# Freeze backbone

for p in model.features.parameters():

    p.requires_grad = False



# Replace classifier for 2 classes

model.classifier[1] = nn.Linear(model.last_channel, 2)

model = model.to(device)



# =========================

# LOSS / OPTIMIZER

# =========================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)



# =========================

# TRAIN

# =========================

epochs = 10

for epoch in range(epochs):

    model.train()

    total_loss = 0.0



    for images, labels in train_loader:

        images, labels = images.to(device), labels.to(device)



        outputs = model(images)

        loss = criterion(outputs, labels)



        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



        total_loss += loss.item()



    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")



# =========================

# TEST

# =========================

model.eval()

correct = 0

total = 0



with torch.no_grad():

    for images, labels in test_loader:

        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        _, pred = torch.max(outputs, 1)



        total += labels.size(0)

        correct += (pred == labels).sum().item()



acc = 100 * correct / total

print(f"\nTest Accuracy: {acc:.2f}%")



# =========================

# SAVE (MODEL + CLASS ORDER)

# =========================

save_obj = {

    "state_dict": model.state_dict(),

    "class_names": class_names  # e.g. ['fake','real'] OR ['real','fake']

}

torch.save({

    "state_dict": model.state_dict(),

    "class_names": train_data.classes

}, "deepfake_model.pth")

print("Model + class_names saved ✅")