# RAITE Object detection evaluator

We provide software for evaluating object detection model performance under several scenarios for detecting both drones and several ground robots.

Performance is evaluated using mAP @ IOU of 50% and 75% overlap between ground truth and predicted bounding boxes.

We provide the following test cases:
- autonomy park
- 

to evaluate the results of a model from only a json file, use command:
- python evaluate.py --json scratch/fasterrcnn_detections_with_targets.json

to evaluate on all test sets for a drone or ugv model,
- this command


## How to use for model evaluation:


- contains several test sets with ground truth to evaluate models
- 


### Questions
* where should we host test set data?
* do we have all the tests we want for the paper?

# RAITE Scenario 1 Dataset

This dataset was collected over three days at RAITE 2024 by the Blue Team (University of Florida). It includes data from **51 attacks** to the vision system, spanning **20 unique attack scenarios** (e.g., "smoke," "laser," "blanket on robot," "cotton balls"). A detailed overview of the attacks can be found in [this Google Sheet](https://docs.google.com/spreadsheets/d/1XFUwoL-b02kC95PayJgFLPnCwcbyNPex/edit?usp=sharing&ouid=108502102907068954075&rtpof=true&sd=true). The same Google Sheet is also included in the dataset directory for convenience.

Scenario 1 took place at the MuTC Soccer Field, with cameras stationed on the side of the field nearest the MuTC facility entry point. The targets throughout the scenario were drones and ground robots. The Spot and Clearpath Warthog ground robots were controlled by employees of MITRE, and two Clearpath Jackals were controlled by on-site staff. Drones were also controlled by on-site staff. There were two drones: a DJI Mavic and a DJI Phantom.

The goal of Scenario 1 was to use multiple cameras and radar data to detect and track drones and ground robots with high confidence while being robust to attacks on the vision-based object detection system. The dataset also includes indicators of when attacks were occurring.

Radar data was collected but not used for detection and tracking during the event; this dataset primarily includes object detection results for each camera for the duration of each attack.

---

## Table of Contents

- [Data Overview](#data-overview)
- [Folder Structure](#folder-structure)
- [Dataset Details](#dataset-details)
- [Object Detection Models](#object-detection-models)
- [Class Labels](#class-labels)
- [Dataset Potential Issues](#dataset-potential-issues)

---

## Data Overview

The dataset is separated into folders, one for each **day** (e.g., `20241022`) of data collection. There are three total days. Each day's folder includes a folder for each attack recorded. Within each of these folders is the data generated **for each of the cameras** used *for that attack* (different quantities of cameras were used for attacks throughout the week, ranging from 1 to 3).

### Dataset Statistics

- **Total images**: 119,457
  - **Day 1 (20241022)**: 16,496 images
  - **Day 2 (20241023)**: 19,712 images
  - **Day 3 (20241024)**: 83,249 images

### Additional Files

An additional folder is included with the `.pth` files for each object detection model used during the event. These models can be re-run, and results can be evaluated using the following codebase: [RAITE Classify](https://gitlab.com/bakerherrin/raiteclassify).

If you need assistance accessing the code, please contact Baker Herrin at: [eherrin@ufl.edu](mailto:eherrin@ufl.edu).

Each folder includes the following data for each camera:

- **`input_stream.mkv`**: A video of the attacked frames
- **`output_stream.mkv`**: A video of the attacked frames with bounding boxes visualized
- **`input_frames`**: A folder of `.png` files for all frames from `input_stream.mkv`
- **`output_frames`**: A folder of `.png` files for all frames from `output_stream.mkv`
- **`detections.json`**: A JSON file including the data associated with each individual camera frame. Specifics on the JSON file can be found in [Dataset Details](#dataset-details).

---

## Folder Structure

Below is an example outline of the directory structure:

```
├── 20241022
│   ├── bottle
│   │   ├── 20241022T1729624271_camera1_no_target
│   │   │   ├── detections.json
│   │   │   ├── input_frames
│   │   │   ├── input_stream.mkv
│   │   │   ├── output_frames
│   │   │   └── output_stream.mkv
│   │   ├── 20241022T1729625494_camera1_spot_and_pumba
│   │   │   ├── detections.json
│   │   │   ├── input_frames
│   │   │   ├── input_stream.mkv
│   │   │   ├── output_frames
│   │   │   └── output_stream.mkv
│   │   └── 20241022T1729625854_camera1_pumba_corner
│   │       ├── detections.json
│   │       ├── input_frames
│   │       ├── input_stream.mkv
│   │       ├── output_frames
│   │       └── output_stream.mkv
```

For more information, see `filetree.txt` included in the dataset.

---

## Dataset Details

Each `detections.json` file includes all data associated with each frame of a particular attack. For example:

```json
{
  "frame_0": {
    "model_type": "YOLO",
    "boxes": [[806.49, 852.56, 1076.27, 1062.85]],
    "scores": [0.87],
    "labels": [6],
    "track_ids": [1]
  }
}
```

### Fields in `detections.json`:
- **`model_type`**: Specifies whether the object detection model was Faster R-CNN or YOLO.
- **`boxes`**: List of bounding box coordinates for all detections in the frame, in pixel space.
- **`scores`**: List of confidence scores associated with each bounding box prediction (0 to 1).
- **`labels`**: List of class labels associated with each bounding box prediction.
- **`track_ids`**: List of integer ID values assigned to each bounding box for tracking. IDs persist until an object is occluded or disappears for too many consecutive frames.

---

## Object Detection Models

Two object detection architectures were used during the competition:

### Faster R-CNN
- Backbone: ResNet-50
- Head: Feature Pyramid Network (FPN)
- Details:
  - Two separate models were trained: one for ground robot detection (UGVs) and one for drone detection.
  - The model treats all targets equally, predicting only whether a target is a UGV or a drone. Therefore, all Faster R-CNN predictions in this dataset have a class label of `1`.

### YOLOv11
- A multi-class detector that differentiates between specific classes. Class labels are listed below.

---

## Class Labels

### YOLOv11 Labels

| Label | Class                               |
|-------|-------------------------------------|
| 1     | Miscellaneous ground robot         |
| 2     | Drone (DJI Mavic or Phantom)       |
| 3     | Clearpath Jackal                   |
| 4     | Boston Dynamics Spot               |
| 6     | Clearpath Warthog (Pumba)          |

### Faster R-CNN Labels

| Label | Class                     |
|-------|---------------------------|
| 1     | Ground robot (UGV) or Drone |

---

## Dataset Potential Issues

Below are some unresolved issues with the dataset:

- **Missing Labels**: Occasionally, some Faster R-CNN bounding box predictions have missing associated class labels (e.g., 6 bounding boxes but only 4 class labels). This issue is believed to be due to the model’s predictions and may affect downstream analysis.