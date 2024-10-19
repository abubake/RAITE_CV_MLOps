# Script for modifying frames in a dataset
import cv2
import os

images_pth = "/home/eherrin@ad.ufl.edu/code/gitlab_dev/raiteclassify/data/archive/test_sets/special_cases/attack_on_autonomyPark/images"
labels_pth = "/home/eherrin@ad.ufl.edu/code/gitlab_dev/raiteclassify/data/archive/test_sets/special_cases/attack_on_autonomyPark/labels"

############## Areas for improvement ##############
# color change appeared as an occlusion attack rather than tampering


# 150 total iamges
image_files = sorted(os.listdir(images_pth))

# change 20 frames to be blue

# Case 1: Blue Attack
for idx in range(30, 51): # [277-297] - should be 20 
    image_file = image_files[idx]
    image_path = os.path.join(images_pth, image_file)
    
    img = cv2.imread(image_path)
    if img is not None:
        # Change the image to blue
        img[:, :, 1] = 0  # Set green channel to 0
        img[:, :, 2] = 0  # Set red channel to 0

        # Overwrite the modified image with the same name
        cv2.imwrite(image_path, img)
    else:
        print(f"Could not load image: {image_path}")

# Case 2: red attack
for idx in range(51, 56): # [298-302] - should be 5
    image_file = image_files[idx]
    image_path = os.path.join(images_pth, image_file)
    
    img = cv2.imread(image_path)
    if img is not None:
        # Change the image to blue
        img[:, :, 1] = 0  # Set green channel to 0
        img[:, :, 0] = 0  # Set blue channel to 0

        # Overwrite the modified image with the same name
        cv2.imwrite(image_path, img)
    else:
        print(f"Could not load image: {image_path}")


# Case 3: only blue disabled- red and green
# Still able to detect objects- maybe not shut down in this case.
for idx in range(56, 61): # [303-308] - should be 5
    image_file = image_files[idx]
    image_path = os.path.join(images_pth, image_file)
    
    img = cv2.imread(image_path)
    if img is not None:
        # Change the image to blue
        img[:, :, 0] = 0  # Set blue channel to 0

        # Overwrite the modified image with the same name
        cv2.imwrite(image_path, img)
    else:
        print(f"Could not load image: {image_path}")

# Patch attack [309]
target_image_path  = "/home/eherrin@ad.ufl.edu/code/gitlab_dev/raiteclassify/data/archive/test_sets/special_cases/attack_on_autonomyPark/images/00309.jpg"
# Load the target image (where you want to add the patch)
target_img = cv2.imread(target_image_path)

# Load the patch image (Optimus Prime image)
patch_image_path = '/home/eherrin@ad.ufl.edu/code/gitlab_dev/raiteclassify/data/archive/test_sets/drone/t4_occlusions150/images/occlusions_19.png'
patch_img = cv2.imread(patch_image_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel if available

# Resize the patch to fit the desired size on the target image
patch_height, patch_width = 100, 100  # Define the size you want for the patch
patch_img = cv2.resize(patch_img, (patch_width, patch_height))

# Define the position where the patch will be placed (top-left corner coordinates)
x_offset, y_offset = 50, 50  # Change these values to position the patch

# Get the region of interest (ROI) in the target image where the patch will go
roi = target_img[y_offset:y_offset + patch_height, x_offset:x_offset + patch_width]

# If the patch has an alpha channel (transparency), blend it with the target image
if patch_img.shape[2] == 4:  # Check if the patch has 4 channels (RGBA)
    # Split the patch into its RGB and Alpha channels
    patch_rgb = patch_img[:, :, :3]
    alpha_mask = patch_img[:, :, 3] / 255.0  # Normalize the alpha channel to [0, 1]

    # Blend the patch with the ROI of the target image
    for c in range(0, 3):
        roi[:, :, c] = roi[:, :, c] * (1.0 - alpha_mask) + patch_rgb[:, :, c] * alpha_mask
else:
    # If no alpha channel, simply replace the region with the patch
    target_img[y_offset:y_offset + patch_height, x_offset:x_offset + patch_width] = patch_img

# Save or display the result
cv2.imwrite(target_image_path, target_img)
cv2.imshow('Patched Image', target_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



