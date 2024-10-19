#from dataset import RAITEDataset
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
#from torch.utils.data import DataLoader
from centerpoint_tracker import CentroidTracker
from vision_attack_detection import attackDetector
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

# dir_path = "data/archive/test_sets/drone/t2_autonomyPark150/images"
# dir_path = "/home/eherrin@ad.ufl.edu/code/gitlab_dev/raiteclassify/data/archive/test_sets/drone/t2_autonomyPark150/images"
# dir_path = "/home/eherrin@ad.ufl.edu/code/gitlab_dev/raiteclassify/data/archive/test_sets/special_cases/cars/images"
dir_path = "/home/eherrin@ad.ufl.edu/code/gitlab_dev/raiteclassify/data/archive/test_sets/special_cases/backpacks/images"

all_images = sorted(
            [f for f in os.listdir(dir_path) if f.endswith(".jpg") or f.endswith(".png")]
        )

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

centerTracker = CentroidTracker(maxDisappeared=2)
anomalyDetector = attackDetector(brightness_threshold=50)

# Load the weights
model = torch.load('models/ugvs/fasterrcnn_resnet50_fpn_ugv_v3.pth')
model.to(device)
model.eval()

start_index = 60 # change based off of range we are looking at
end_index = 80
with torch.no_grad():
    predictions = model(test_images[start_index:end_index]) # or w/o list, test_images[:N]

for prediction in predictions:
    print(prediction)

# Visualize bounding boxes on original images
score_list = []
for i, prediction in enumerate(predictions):
    # Extract bounding boxes, labels, and scores
    boxes = prediction['boxes'].cpu().numpy()  # Move to CPU and convert to NumPy
    labels = prediction['labels'].cpu().numpy()  # Move to CPU and convert to NumPy
    scores = prediction['scores'].cpu().numpy()  # Move to CPU and convert to NumPy


    original_height, original_width = original_images[i+start_index].shape[:2]
    
    # Loop through each predicted box and draw it
    rescaled_boxes = []
    for box, score in zip(boxes, scores):

        score_list.append(score)
        if score > 0:  # Only consider predictions with confidence score > 0.5
            # Extract box coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box)

            # Rescale the bounding box coordinates to match the original image dimensions
            x1 = int(x1 * (original_width / 200))
            y1 = int(y1 * (original_height / 200))
            x2 = int(x2 * (original_width / 200))
            y2 = int(y2 * (original_height / 200))

            rescaled_boxes.append((x1, y1, x2, y2))
            # Draw the rectangle on the original image
            cv2.rectangle(original_images[i+start_index], (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Put a label with the score
            label_text = f"Score: {score:.2f}"
            cv2.putText(original_images[i+start_index], label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 5)
    
    # For the bounding boxes, find the centerpoints:
    centerTracker.update(boxes=rescaled_boxes)
    
    # Draw centerpoints and IDs on the original image
    centroids = []
    for objectID, centroid in centerTracker.objects.items():
        # Draw the centroid
        cx, cy = centroid  # Assuming centroid is in the form (cx, cy)
        centroids.append(centroid)
        cv2.circle(original_images[i+start_index], (int(cx), int(cy)), 7, (0, 0, 255), -1)  # Draw the centerpoint in blue
        
        # Put the ID text next to the centroid
        cv2.putText(original_images[i+start_index], f"ID: {objectID}", (int(cx), int(cy) - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
    
    # tests detection of attacks on the given set of frames
    anomalyDetector.detect_attack(frame_index=i+start_index, frame=original_images[i+start_index], detections=rescaled_boxes, centroids=centroids)


# Display the images with bounding boxes
for i, img in enumerate(original_images[start_index:end_index]):
    # Convert BGR to RGB for visualization with matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.title(f"Image {i+1} with Predicted Bounding Boxes")
    plt.axis("off")
    plt.show()

print(np.mean(score_list))
print(anomalyDetector.attacks)
