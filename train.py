from model import DroneClassifier
import torch.optim as optim
import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# https://www.mdpi.com/2313-433X/8/8/218 drone classification

# Hyperparameters
learning_rate = 0.001
batch_size = 32
epochs = 10

# Dataset and data augmentation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Replace 'drone_dataset' with the actual dataset
train_dataset = datasets.ImageFolder('data/drone/train', transform=transform)
test_dataset = datasets.ImageFolder('pdata/drone/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, Loss, Optimizer
model = DroneClassifier()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        labels = labels.float().view(-1, 1)  # Convert labels to float
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')

torch.save(model, 'drone_classifier_model.pth')

# Testing loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        labels = labels.float().view(-1, 1)
        outputs = model(images)
        predicted = (outputs >= 0.5).float()  # Apply threshold
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')