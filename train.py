import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from RAITEDataset import RAITEDataset


def send_email(hours: int, minutes: int, avg_time_per_epoch: float) -> None:
    """ """
    sender_email = "bakerherrin2@gmail.com"
    receiver_email = "bakerherrin2@gmail.com"
    password = "cdvi gqha lund weld"  # Replace with your Gmail password or app-specific password

    subject = "Training Complete - Faster R-CNN"
    body = f"Training is complete.\n\nTotal Training Time: {hours} hours and {minutes} minutes\nAverage Time per Epoch: {avg_time_per_epoch:.2f} seconds"

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        print("Email notification sent!")
    except Exception as e:
        print(f"Failed to send email notification: {e}")


def train():
    pass


if __name__ == "__main__":
    # Start timing
    start_time = time.time()

    # Custom dataset loader for your own data
    transform = transforms.Compose([transforms.ToTensor()])
    classes = 2
    dir_path = "/home/eherrin@ad.ufl.edu/code/gitlab_dev/raiteclassify/2024-raite-ml/data/drone_dataset_revised_20241023/train/images"
    ann_path = "/home/eherrin@ad.ufl.edu/code/gitlab_dev/raiteclassify/2024-raite-ml/data/drone_dataset_revised_20241023/train/labels"

    width = 640
    height = 640
    train_dataset = RAITEDataset(
        dir_path, ann_path, width, height, classes, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Load pre-trained Faster R-CNN
    detection_model = fasterrcnn_resnet50_fpn(pretrained=True)

    # Replace the head of the network for your number of classes (e.g., 2: drone or not drone)
    in_features = detection_model.roi_heads.box_predictor.cls_score.in_features
    detection_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, classes)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    detection_model.to(device)

    optimizer = optim.SGD(
        detection_model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005
    )

    detection_model.train()

    num_epochs = 20
    loss_per_epoch = []
    epoch_times = []

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_start_time = time.time()

        for images, targets, _, _ in train_loader:
            targets["boxes"] = targets["boxes"].squeeze(0)
            targets["labels"] = targets["labels"].squeeze(0)

            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in targets.items()}]

            optimizer.zero_grad()
            loss_dict = detection_model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

        average_loss = running_loss / len(train_loader)
        loss_per_epoch.append(average_loss)

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}, Time: {epoch_time:.2f} seconds"
        )
        torch.save(
            detection_model,
            "/home/eherrin@ad.ufl.edu/code/gitlab_dev/raiteclassify/models/ugvs/fasterrcnn_resnet50_fpn_comp_vfinal.pth",
        )
    total_training_time = time.time() - start_time
    avg_time_per_epoch = sum(epoch_times) / num_epochs

    hours, minutes = divmod(total_training_time // 60, 60)
    send_email(hours, minutes, avg_time_per_epoch)

    torch.save(
        detection_model,
        "/home/eherrin@ad.ufl.edu/code/gitlab_dev/raiteclassify/models/ugvs/fasterrcnn_resnet50_fpn_comp_vfinal.pth",
    )

    # Plotting the loss per epoch
    plt.figure()
    plt.plot(range(1, num_epochs + 1), loss_per_epoch, marker="o")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("loss_per_epoch.png")
    plt.close()
