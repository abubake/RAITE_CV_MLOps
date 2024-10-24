import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

class RAITEDataset(Dataset):
    '''
    Dataset object for the raite data.

    Expects bbox annotations in format [class - center_x - center_y - width - height]
    '''
    def __init__(self, dir_path, ann_path, width, height, classes: dict[int, int], transform=None):
        self.transform = transform
        self.dir_path = dir_path
        self.ann_path = ann_path
        self.height = height
        self.width = width
        self.classes = classes
        #self.all_images = [f for f in os.listdir(dir_path) if f.endswith(".jpg") or f.endswith(".png")]
        self.all_images = sorted(
            [f for f in os.listdir(dir_path) if f.endswith(".jpg") or f.endswith(".png")]
        )

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)
        if image_name.endswith(".jpg"):
            ann_path = os.path.join(self.ann_path, image_name.replace(".jpg", ".txt"))
        elif image_name.endswith(".png"):
            ann_path = os.path.join(self.ann_path, image_name.replace(".png", ".txt"))
        else:
            print("Sorry, only png and jpg file types accepted.")

        image_orig = cv2.imread(image_path)
        image = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB).astype(np.float32)
        original_height, original_width = image.shape[:2]

        scale_x = self.width / original_width
        scale_y = self.height / original_height
        
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        boxes = []
        labels = []
        with open(ann_path) as f:
            for line in f:
                label, x_center, y_center, width, height = map(float, line.split())
                label = float(self.classes[int(label)])

                # Convert normalized bbox (center, width, height) to (xmin, ymin, xmax, ymax)
                xmin = (x_center - width / 2) * original_width
                ymin = (y_center - height / 2) * original_height
                xmax = (x_center + width / 2) * original_width
                ymax = (y_center + height / 2) * original_height

                xmin = xmin * scale_x
                ymin = ymin * scale_y
                xmax = xmax * scale_x
                ymax = ymax * scale_y

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(label))

        boxes = torch.as_tensor(boxes, dtype=torch.float32) #.squeeze(0) current [1,4] which is number of boxes, by dims
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        if self.transform:
            image_resized = self.transform(image_resized)

        return image_resized, target, image, boxes # image and boxes aren't readlly needed.
    
    @staticmethod
    def display_bounding_boxes_gt(image, boxes, title="Bounding Boxes"):
        """Draw bounding boxes on the image and display it."""
        thickness = 7
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), thickness)  # Draw red box
        plt.imshow(image.astype(np.uint8))
        plt.title(title)
        plt.axis('off')  # Hide axis
        plt.show()

    @staticmethod
    def display_bounding_boxes(image, boxes, title="Bounding Boxes"):
        """Draw bounding boxes on the image and display it."""
        thickness = 2
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), thickness)  # Draw red box
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')  # Hide axis
        plt.show()



if __name__ == '__main__':
    dataset = RAITEDataset('data/archive/drone_dataset/train/images', 
                           'data/archive/drone_dataset/train/labels', 
                           200, 200, 2)

    image_resized, target, original_image, boxes = dataset[49]  
    
    # Display bounding boxes on the original image (simply get rid of scale for this to be right)
    RAITEDataset.display_bounding_boxes_gt(original_image.copy(), boxes, title="Original Image with Bounding Boxes")

    # Display bounding boxes on the resized image
    RAITEDataset.display_bounding_boxes(image_resized, target['boxes'], title="Resized Image with Bounding Boxes")
    