from torchvision import transforms
from centerpoint_tracker import CentroidTracker
from vision_attack_detection import AttackDetector
import matplotlib.pyplot as plt
import torch
import os
import cv2
import numpy as np
import pandas as pd
import json
from collections import defaultdict

# Set global font to Times New Roman
# plt.rcParams["font.family"] = "Times New Roman"

# Settings
radar = False
camera = True
show_visualization = False  # Set to True to view individual image frames

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
        test_images.append(image_tensor.to(device))

    centerTracker = CentroidTracker(maxDisappeared=30)
    anomalyDetector = AttackDetector(brightness_threshold=50)

    model = torch.load("models/drones/fasterrcnn_resnet50_fpn_drone_comp_v2.pth")
    model.to(device)
    model.eval()

    camera_predictions = []
    if camera:
        with torch.no_grad():
            for img in test_images:
                pred = model([img])[0]
                camera_predictions.append(pred)

    confidence_log = defaultdict(list)
    output_rows = []

    for i in range(len(original_images)):
        original_height, original_width = original_images[i].shape[:2]

        if camera:
            boxes = camera_predictions[i]["boxes"].cpu().numpy()
            scores = camera_predictions[i]["scores"].cpu().numpy()
            rescaled_boxes = draw_and_rescale_boxes(i, boxes, scores, original_images, original_height, original_width)

        if radar:
            rescaled_centroids = draw_and_rescale_centroids(i, radar_predictions, original_images, original_height, original_width)

        if not radar and camera:
            centerTracker.update(boxes=rescaled_boxes)
        elif not camera and radar:
            centerTracker.update(boxes=None, centroids=rescaled_centroids)

        for objectID, centroid in centerTracker.objects.items():
            cx, cy = centroid

            # Normalize new object IDs in last segment
            true_objectID = objectID
            if (i + 1) >= 201 and objectID not in [0, 1]:
                true_objectID = 0

            cv2.circle(original_images[i], (int(cx), int(cy)), 7, (0, 100, 255), -1)
            cv2.putText(original_images[i], f"ID: {true_objectID}", (int(cx), int(cy) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 5)

            output_rows.append({"frame": i + 1, "id": true_objectID, "u": int(cx), "v": int(cy)})

            if camera:
                for box, score in zip(boxes, scores):
                    if score > 0.2:
                        cX = int((box[0] + box[2]) / 2.0 * (original_width / 200))
                        cY = int((box[1] + box[3]) / 2.0 * (original_height / 200))
                        dist = np.linalg.norm(np.array([cX, cY]) - np.array([cx, cy]))
                        if dist < 20:
                            confidence_log[true_objectID].append({"frame": i + 1, "score": float(score)})

    mode_name = f"camera_{int(camera)}_radar_{int(radar)}"
    pd.DataFrame(output_rows).to_csv(f"tracking_output_{mode_name}.csv", index=False)
    with open(f"confidence_log_{mode_name}.json", "w") as f:
        json.dump(confidence_log, f, indent=2)

    # Plot
    plt.figure(figsize=(14, 7))
    segment_bounds = [(1, 100), (101, 200), (201, 300)]
    segment_labels = ['No Laser', 'Medium Laser', 'High Laser']
    segment_colors = ['#D0F0C0', '#FFFACD', '#FADADD']

    for (start, end), color, label in zip(segment_bounds, segment_colors, segment_labels):
        plt.axvspan(start, end, color=color, alpha=0.3, label=label)

    for objectID, entries in confidence_log.items():
        scores = [entry["score"] for entry in entries]
        frames = [entry["frame"] for entry in entries]
        scores_series = pd.Series(scores, index=frames).sort_index()
        rolling_mean = scores_series.rolling(window=10, min_periods=1).mean()

        plt.scatter(scores_series.index, scores_series.values, alpha=0.2, s=12)
        plt.plot(rolling_mean.index, rolling_mean.values, label=f"ID {objectID}", linewidth=2.5)
        
        # Draw vertical line where detections stop for this ID
        if frames:
            last_frame = max(frames)
            plt.axvline(x=last_frame, color='gray', linestyle='--', linewidth=1.2)
            plt.text(last_frame + 2, 0.05 + 0.05 * objectID, f"ID {objectID} end",
                    fontsize=8, rotation=90, color='gray')


        for seg_idx, (seg_start, seg_end) in enumerate(segment_bounds):
            seg_scores = [entry["score"] for entry in entries if seg_start <= entry["frame"] <= seg_end]
            if seg_scores:
                avg_seg = sum(seg_scores) / len(seg_scores)
                x_pos = seg_start + (seg_end - seg_start) // 2
                y_pos = 0.85 - 0.05 * objectID - 0.05 * seg_idx
                plt.annotate(f"ID {objectID} Avg: {avg_seg:.2f} (n={len(seg_scores)})",
             (x_pos, y_pos), ha='center', fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))


    plt.xlim([1, 300])
    plt.ylim([0, 1])
    plt.xlabel("Frame", fontsize=12)
    plt.ylabel("Confidence Score", fontsize=12)
    plt.title("Drone Detection Confidence Over Time\n"
              "Model: fasterrcnn_resnet50_fpn_drone_comp_v2\n"
              "ID 1: DJI Phantom, ID 2: DJI Mavic", fontsize=14) #, fontweight='bold')
    plt.legend(title="Laser Condition / Drone ID")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"confidence_plot_{mode_name}.png")
    plt.show()

    if show_visualization:
        for i, img in enumerate(original_images):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10, 10))
            plt.imshow(img_rgb)
            plt.title(f"Image {i+1} with Predicted Bounding Boxes")
            plt.axis("off")
            plt.show()

if __name__ == "__main__":
    run_inference()