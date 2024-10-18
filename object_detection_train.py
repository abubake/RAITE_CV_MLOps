# conda env: test

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from dataset import RAITEDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import matplotlib.pyplot as plt

# Trying drone or not drone to start

# Custom dataset loader for your own data
transform = transforms.Compose([transforms.ToTensor()])
#train_dataset = CocoDetection(root="path_to_images", annFile="path_to_annotations", transform=transform)
classes = 2
dir_path = "data/archive/ugv_dataset_backpacks_v5/train/images"
ann_path = "data/archive/ugv_dataset_backpacks_v5/train/labels" # contains type and contains other info

width = 400
height = 400
train_dataset = RAITEDataset(dir_path, ann_path, width, height, classes, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True) # TODO: right now only batch size 1 is workin- need to improve that

# Load pre-trained Faster R-CNN
model = fasterrcnn_resnet50_fpn(pretrained=True)

# for param in model.backbone.parameters():
#     param.requires_grad = False

# Replace the head of the network for your number of classes (e.g., 3: drones, people, ground robots)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

model.train()

num_epochs = 40
loss_per_epoch = []

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    
    for images, targets, _, _ in train_loader:
        targets['boxes'] = targets['boxes'].squeeze(0)
        targets['labels'] = targets['labels'].squeeze(0)

        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in targets.items()}]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass and optimization step
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    # Calculate average loss for the epoch
    average_loss = running_loss / len(train_loader)
    loss_per_epoch.append(average_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss}")

# Save the model after training
torch.save(model, 'models/ugvs/fasterrcnn_resnet50_fpn_ugv_backpacks_v5.pth')
#torch.save(model, 'models/fasterrcnn_resnet50_fpn_ugv_v1.pth')

# Plotting the loss per epoch
plt.figure()
plt.plot(range(1, num_epochs + 1), loss_per_epoch, marker='o')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('loss_per_epoch.png')
plt.close()