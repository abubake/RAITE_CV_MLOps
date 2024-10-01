#from dataset import RAITEDataset
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
#from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import os
import cv2

device = 'cuda'
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((200, 200)),
    transforms.ToTensor()
])
#classes = 2 # not sure if I acttualy need classes here
dir_path = "data/archive/drone_dataset/test/lab"
#ann_path = "data/archive/drone_dataset/valid/labels" # not actually needed here
#width = 200
#height = 200
#test_dataset = RAITEDataset(dir_path, ann_path, width, height, classes, transform=transform)
#train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True) # TODO: right now only batch size 1 is workin- need to improve that
all_images = [f for f in os.listdir(dir_path) if f.endswith(".jpg") or f.endswith(".png")]

test_images = []
original_images = []  # To store original images for visualization
transformed_images = []

for image_name in all_images:
    image_path = os.path.join(dir_path, image_name)
    
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    original_images.append(image.copy())  # Store a copy of the original image for visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image_rgb)  # The tensor will be [C, H, W] now
    test_images.append(image_tensor)
    transformed_images.append(image_tensor)  # Store the transformed image

# Move images to the same device as the model
test_images = [image.to(device) for image in test_images]

# Load the weights
model = torch.load('fasterrcnn_resnet50_fpn_labData.pth')
model.to(device)
model.eval()

# for i, image_tensor in enumerate(transformed_images[:2]):  # Change the range if you want to visualize more images
#     # Convert tensor back to NumPy for visualization
#     image_np = image_tensor.permute(1, 2, 0).cpu().numpy()  # Change shape to [H, W, C]
    
#     # Plotting
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image_np)
#     plt.title(f"Transformed Image {i+1} Before Prediction")
#     plt.axis("off")
#     plt.show()


with torch.no_grad():
    predictions = model(test_images[:12])

for prediction in predictions:
    print(prediction)

# Visualize bounding boxes on original images
for i, prediction in enumerate(predictions):
    # Extract bounding boxes, labels, and scores
    boxes = prediction['boxes'].cpu().numpy()  # Move to CPU and convert to NumPy
    labels = prediction['labels'].cpu().numpy()  # Move to CPU and convert to NumPy
    scores = prediction['scores'].cpu().numpy()  # Move to CPU and convert to NumPy

    original_height, original_width = original_images[i].shape[:2]
    
    # Loop through each predicted box and draw it
    for box, score in zip(boxes, scores):
        if score > 0.1:  # Only consider predictions with confidence score > 0.5
            # Extract box coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box)

            # Rescale the bounding box coordinates to match the original image dimensions
            x1 = int(x1 * (original_width / 200))
            y1 = int(y1 * (original_height / 200))
            x2 = int(x2 * (original_width / 200))
            y2 = int(y2 * (original_height / 200))
            
            # Draw the rectangle on the original image
            cv2.rectangle(original_images[i], (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Put a label with the score
            label_text = f"Score: {score:.2f}"
            cv2.putText(original_images[i], label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 5)

# Display the images with bounding boxes
for i, img in enumerate(original_images[:12]):
    # Convert BGR to RGB for visualization with matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.title(f"Image {i+1} with Predicted Bounding Boxes")
    plt.axis("off")
    plt.show()