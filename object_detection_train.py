import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import time
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from dataset import RAITEDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import matplotlib.pyplot as plt

# Function to send email notification
def send_email(hours, minutes, avg_time_per_epoch):
    sender_email = "bakerherrin2@gmail.com"
    receiver_email = "bakerherrin2@gmail.com"
    password = "cdvi gqha lund weld"  # Replace with your Gmail password or app-specific password

    subject = "Training Complete - Faster R-CNN"
    body = f"Training is complete.\n\nTotal Training Time: {hours} hours and {minutes} minutes\nAverage Time per Epoch: {avg_time_per_epoch:.2f} seconds"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        print("Email notification sent!")
    except Exception as e:
        print(f"Failed to send email notification: {e}")

# Start timing
start_time = time.time()

# Custom dataset loader for your own data
transform = transforms.Compose([transforms.ToTensor()])
classes = 2
dir_path = "/home/eherrin@ad.ufl.edu/code/gitlab_dev/raiteclassify/2024-raite-ml/data/drone_dataset_revised_20241023/train/images"
ann_path = "/home/eherrin@ad.ufl.edu/code/gitlab_dev/raiteclassify/2024-raite-ml/data/drone_dataset_revised_20241023/train/labels" 

width = 640
height = 640
train_dataset = RAITEDataset(dir_path, ann_path, width, height, classes, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True) 

# Load pre-trained Faster R-CNN
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Replace the head of the network for your number of classes (e.g., 2: drone or not drone)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

model.train()

num_epochs = 20
loss_per_epoch = []
epoch_times = []

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    epoch_start_time = time.time()
    
    for images, targets, _, _ in train_loader:
        targets['boxes'] = targets['boxes'].squeeze(0)
        targets['labels'] = targets['labels'].squeeze(0)

        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in targets.items()}]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass and optimization step
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    # Calculate average loss for the epoch
    average_loss = running_loss / len(train_loader)
    loss_per_epoch.append(average_loss)

    # Track the time taken for each epoch
    epoch_time = time.time() - epoch_start_time
    epoch_times.append(epoch_time)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}, Time: {epoch_time:.2f} seconds")
    torch.save(model, '/home/eherrin@ad.ufl.edu/code/gitlab_dev/raiteclassify/models/ugvs/fasterrcnn_resnet50_fpn_comp_vfinal.pth')
# Calculate total training time and average time per epoch
total_training_time = time.time() - start_time
avg_time_per_epoch = sum(epoch_times) / num_epochs

# Send email notification
hours, minutes = divmod(total_training_time // 60, 60)
send_email(hours, minutes, avg_time_per_epoch)

# Save the model after training
torch.save(model, '/home/eherrin@ad.ufl.edu/code/gitlab_dev/raiteclassify/models/ugvs/fasterrcnn_resnet50_fpn_comp_vfinal.pth')

# Plotting the loss per epoch
plt.figure()
plt.plot(range(1, num_epochs + 1), loss_per_epoch, marker='o')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('loss_per_epoch.png')
plt.close()
