import torch
import torch.nn as nn

# Define the CNN model
class DroneClassifier(nn.Module):
    def __init__(self):
        super(DroneClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)  # Assuming input image size is 64x64
        self.fc2 = nn.Linear(128, 1)  # Binary classification: drone (1) or not drone (0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)  # Flatten the image
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)  # Binary classification
        return x

class PeopleClassifier(DroneClassifier):
    # Inherit from DroneClassifier (same architecture)
    pass

class GroundRobotClassifier(DroneClassifier):
    # Inherit from DroneClassifier (same architecture)
    pass