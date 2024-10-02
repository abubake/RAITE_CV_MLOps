# from dataset import RAITEDataset
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
# from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import os
import cv2
import numpy as np

device = 'cuda'
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((200, 200)),
    transforms.ToTensor()
])

dir_path = "data/archive/drone_dataset/test/lab"
all_images = [f for f in os.listdir(dir_path) if f.endswith(".jpg") or f.endswith(".png")]

test_images = []
original_images = []  # To store original images for visualization

for image_name in all_images:
    image_path = os.path.join(dir_path, image_name)
    
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    original_images.append(image.copy())  # Store a copy of the original image for visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image_rgb)  # The tensor will be [C, H, W] now
    test_images.append(image_tensor)

# Move images to the same device as the model
test_images = [image.to(device) for image in test_images]

# Load the weights
model = torch.load('models/fasterrcnn_resnet50_fpn_labData.pth')
model.to(device)
model.eval()

# Function to enable dropout during inference
def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

# Enable dropout
enable_dropout(model)

# Perform multiple stochastic forward passes
num_samples = 10  # Number of forward passes for uncertainty estimation
all_predictions = []

for _ in range(num_samples):
    with torch.no_grad():
        predictions = model(test_images[:12])
        all_predictions.append(predictions)

# Process predictions: Extract bounding boxes and compute mean and variance
for i, image_predictions in enumerate(zip(*all_predictions)):
    boxes_list = []
    scores_list = []
    
    # Extract predictions for each forward pass
    for prediction in image_predictions:
        boxes = prediction['boxes'].cpu().numpy()  # Move to CPU and convert to NumPy
        scores = prediction['scores'].cpu().numpy()  # Move to CPU and convert to NumPy
        boxes_list.append(boxes)
        scores_list.append(scores)

    # Compute the mean and variance of bounding boxes and scores
    boxes_mean = np.mean(boxes_list, axis=0)
    boxes_var = np.var(boxes_list, axis=0)
    scores_mean = np.mean(scores_list, axis=0)
    scores_var = np.var(scores_list, axis=0)

    original_height, original_width = original_images[i].shape[:2]

    print(boxes_var)
    print(scores_var)

#     # Draw bounding boxes on the original image
#     for j, box in enumerate(boxes_mean):
#         if scores_mean[j] > 0.1:  # Use mean score to filter predictions
#             x1, y1, x2, y2 = map(int, box)
            
#             # Rescale the bounding box coordinates to match the original image dimensions
#             x1 = int(x1 * (original_width / 200))
#             y1 = int(y1 * (original_height / 200))
#             x2 = int(x2 * (original_width / 200))
#             y2 = int(y2 * (original_height / 200))

#             # Draw the rectangle on the original image
#             cv2.rectangle(original_images[i], (x1, y1), (x2, y2), (0, 255, 0), 2)
            
#             # Add variance info for the bounding box as text
#             label_text = f"Score: {scores_mean[j]:.2f}, Var: {scores_var[j]:.2f}"
#             cv2.putText(original_images[i], label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 5)

# # Display the images with bounding boxes
# for i, img in enumerate(original_images[:12]):
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#     plt.figure(figsize=(10, 10))
#     plt.imshow(img_rgb)
#     plt.title(f"Image {i+1} with Predicted Bounding Boxes")
#     plt.axis("off")
#     plt.show()
