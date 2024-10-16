import os
import json
from pycocotools.coco import COCO
import cv2

# Set the paths
annFile = 'path/to/annotations/instances_train2017.json'  # Adjust path accordingly
images_dir = 'path/to/train2017/'  # Directory containing images
output_images_dir = 'output/images/'  # Directory to save filtered images
output_annotations_dir = 'output/annotations/'  # Directory to save YOLO format annotations

# Create output directories if they don't exist
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_annotations_dir, exist_ok=True)

# Load COCO annotations
coco = COCO(annFile)

# Get the category ID for 'person'
person_category_id = coco.getCatIds(catNms=['person'])[0]

# Get all image IDs that contain people
image_ids = coco.getImgIds(catIds=[person_category_id])

# Process each image and its annotations
for img_id in image_ids:
    # Load image data
    img_data = coco.loadImgs(img_id)[0]
    img_filename = img_data['file_name']
    img_path = os.path.join(images_dir, img_filename)
    
    # Read and save the image
    image = cv2.imread(img_path)
    output_img_path = os.path.join(output_images_dir, img_filename)
    cv2.imwrite(output_img_path, image)

    # Get annotations for this image
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[person_category_id])
    anns = coco.loadAnns(ann_ids)

    # Prepare YOLO format annotations
    yolo_annotations = []
    img_width, img_height = img_data['width'], img_data['height']

    for ann in anns:
        # Get bounding box and convert to YOLO format
        x, y, width, height = ann['bbox']
        
        # Normalize the bounding box coordinates
        x_center = (x + width / 2) / img_width
        y_center = (y + height / 2) / img_height
        width_normalized = width / img_width
        height_normalized = height / img_height
        
        # Append class id (1 for person) and normalized coordinates
        yolo_annotations.append(f"1 {x_center} {y_center} {width_normalized} {height_normalized}")

    # Save the YOLO format annotations to a text file
    output_anno_path = os.path.join(output_annotations_dir, f"{os.path.splitext(img_filename)[0]}.txt")
    with open(output_anno_path, 'w') as f:
        f.write("\n".join(yolo_annotations))

print(f"Processed {len(image_ids)} images and saved corresponding YOLO annotations.")
