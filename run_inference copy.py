# from dataset import RAITEDataset
from torchvision import transforms
from centerpoint_tracker import CentroidTracker
from vision_attack_detection import AttackDetector
import matplotlib.pyplot as plt
import torch
import os
import cv2
import numpy as np
import pandas as pd

# for fusion
radar = False
camera = True
show_visualization = False # Set this to True to enable visualization

def draw_and_rescale_boxes(i, boxes, scores, original_images, original_height, original_width):
    rescaled_boxes = []
    for box, score in zip(boxes, scores):
        if score > 0.2:
            x1, y1, x2, y2 = map(int, box)
            x1 = int(x1 * (original_width / 200))
            y1 = int(y1 * (original_height / 200))
            x2 = int(x2 * (original_width / 200))
            y2 = int(y2 * (original_height / 200))
            rescaled_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(original_images[i], (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"Score: {score:.2f}"
            cv2.putText(original_images[i], label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 5)
    return rescaled_boxes

def draw_and_rescale_centroids(i, centroids, original_images, original_height, original_width):
    rescaled_centroids = []
    centroid = centroids[i]
    u = int(centroid[0] * original_width)
    v = int(centroid[1] * original_height)
    rescaled_centroids.append((u, v))
    cv2.circle(original_images[i], (u, v), 5, (0, 0, 255), -1)
    cv2.putText(original_images[i], "RADAR", (u, v - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
    return rescaled_centroids

def run_inference():
    device = "cuda"
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((200, 200)),
        transforms.ToTensor()
    ])

    dir_path = "tests"
    all_images = sorted([f for f in os.listdir(dir_path) if f.endswith(".jpg") or f.endswith(".png")])

    test_images = []
    original_images = []
    for image_name in all_images:
        image_path = os.path.join(dir_path, image_name)
        image = cv2.imread(image_path)
        original_images.append(image.copy())
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = transform(image_rgb)
        test_images.append(image_tensor)

    test_images = [image.to(device) for image in test_images]

    centerTracker = CentroidTracker(maxDisappeared=30)
    anomalyDetector = AttackDetector(brightness_threshold=50)

    #model = torch.load("models/drones/fasterrcnn_resnet50_fpn_drones_robust_v1.pth")
    model = torch.load("models/drones/fasterrcnn_resnet50_fpn_drones_robust_v1.pth")
    model.to(device)
    model.eval()

    predictions = []
    if camera:
        batch_size = 4
        camera_predictions = []
        with torch.no_grad():
            for i in range(0, len(test_images), batch_size):
                batch = test_images[i:i + batch_size]
                batch = [img.to(device) for img in batch]
                preds = model(batch)
                camera_predictions.extend(preds)
        predictions = camera_predictions

    if radar:
        arr2 = np.loadtxt('multimodal_test2/radar_uvb.csv', delimiter=',')
        radar_predictions = np.delete(arr2, [2], axis=1)
        predictions = radar_predictions

    attacks_list = []
    prev_centroid = None
    output_rows = []

    for i in range(len(original_images)):
        original_height, original_width = original_images[i].shape[:2]

        if camera:
            boxes = camera_predictions[i]["boxes"].cpu().numpy()
            labels = camera_predictions[i]["labels"].cpu().numpy()
            scores = camera_predictions[i]["scores"].cpu().numpy()
            rescaled_boxes = draw_and_rescale_boxes(i, boxes, scores, original_images, original_height, original_width)

        if radar:
            rescaled_centroids = draw_and_rescale_centroids(i, radar_predictions, original_images, original_height, original_width)

        #current_centroids = []

        if radar and camera:
            averaged_centroids = []

            if i == 0:
                box_centroids = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    cX = int((x1 + x2) / 2.0 * (original_width / 200))
                    cY = int((y1 + y2) / 2.0 * (original_height / 200))
                    box_centroids.append((cX, cY))
                if len(boxes) > 0:
                    prev_centroid = box_centroids[0]
                else:
                    prev_centroid = rescaled_centroids[0]
                
            if len(boxes) == 0 or scores[0]<0.2:
                radar_u, radar_v = rescaled_centroids[0]
                averaged_centroids.append((radar_u, radar_v))
            else:
                box_centroids = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    cX = int((x1 + x2) / 2.0 * (original_width / 200))
                    cY = int((y1 + y2) / 2.0 * (original_height / 200))
                    box_centroids.append((cX, cY))
    
                dists = [np.linalg.norm(np.array([cX, cY]) - np.array(prev_centroid)) for (cX, cY) in box_centroids]
                best_idx = np.argmin(dists)
 
                best_cX, best_cY = box_centroids[best_idx]
                dist_to_bbox = np.linalg.norm(np.array([ best_cX,  best_cY]) - np.array(prev_centroid))
                radar_u, radar_v = rescaled_centroids[0]
                dist_to_radar = np.linalg.norm(np.array([radar_u, radar_v]) - np.array(prev_centroid))
                
                if 10 < dist_to_bbox < 100:
                    averaged_centroids.append(box_centroids[best_idx])
                elif dist_to_radar > dist_to_bbox:
                    averaged_centroids.append(box_centroids[best_idx])
                else: 
                    averaged_centroids.append(rescaled_centroids[0])

            centerTracker.update(boxes=None, centroids=averaged_centroids)
            # update previous centroid
            prev_centroid = averaged_centroids
            
            #current_centroids = averaged_centroids
            
            # if i == 0:
            #     radar_u, radar_v = rescaled_centroids[0]
            #     prev_centroid = (radar_u, radar_v)

        elif not radar and camera:
            centerTracker.update(boxes=rescaled_boxes)
            #current_centroids = [(int((x1 + x2)/2), int((y1 + y2)/2)) for x1, y1, x2, y2 in rescaled_boxes]

        else:
            centerTracker.update(boxes=None, centroids=rescaled_centroids)
            #current_centroids = rescaled_centroids

        for objectID, centroid in centerTracker.objects.items():
            cx, cy = centroid
            cv2.circle(original_images[i], (int(cx), int(cy)), 7, (0, 100, 255), -1)
            cv2.putText(original_images[i], f"ID: {objectID}", (int(cx), int(cy) - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 5)
            output_rows.append({"frame": i, "id": objectID, "u": int(cx), "v": int(cy)})

    # Save results to CSV
    mode_name = f"camera_{int(camera)}_radar_{int(radar)}"
    df = pd.DataFrame(output_rows)
    df.to_csv(f"tracking_output_{mode_name}.csv", index=False)

    if show_visualization:
        for i, img in enumerate(original_images):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10, 10))
            plt.imshow(img_rgb)
            plt.title(f"Image {i+1} with Predicted Bounding Boxes")
            plt.axis("off")
            plt.show()

    print(attacks_list)

if __name__ == "__main__":
    run_inference()
