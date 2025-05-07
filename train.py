
import random

import wandb
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import random_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader

from RAITEDataset import RAITEDataset
  
def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")   


def train(model, loader, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    example_ct = 0
    for epoch in tqdm(range(config['epochs'])):
        for images, targets, _, _ in loader:
            targets["boxes"] = targets["boxes"].squeeze(0)
            targets["labels"] = targets["labels"].squeeze(0)

            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in targets.items()}]
            example_ct +=  len(images)

            optimizer.zero_grad()
            
            loss_dict = model(images, targets)
            loss = criterion(loss_dict)
            #losses = sum(loss for loss in loss_dict.values())
            loss.backward()
            optimizer.step()
        
        train_log(loss, example_ct, epoch)
    
        
def get_data(dataset_path: str, classes: dict[int, int]) -> RAITEDataset:
    ''' Retrieves corresponding image and bbox label data from a folder.
    
    Args:
        dataset_path: path to either train or test data folder with 'images' and 'labels' subfolders.
        classes: dictionary with keys as current existing class labels, values as desired class labels.
    Returns:
        A RATIEDataset object.
    '''
    transform = transforms.Compose([transforms.ToTensor()])
    dir_path = f"{dataset_path}/images"
    ann_path = f"{dataset_path}/labels"
    width = 640
    height = 640
    
    dataset = RAITEDataset(
        dir_path, ann_path, width, height, classes, transform=transform
    )
    return dataset


def make_loader(dataset, batch_size=1):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,  pin_memory=True, num_workers=2)
    return loader

        
def make(config):
    """ 
    
    """
    full_dataset = get_data(config['dataset_path'], config['classes'])
    
    total_len = len(full_dataset)
    train_len = int(config['train_ratio'] * total_len)
    test_len = total_len - train_len
    
    train, test = random_split(full_dataset, [train_len, test_len], generator=torch.Generator().manual_seed(42))
    train_loader = make_loader(train, batch_size=config['batch_size'])
    test_loader = make_loader(test, batch_size=config['batch_size'])

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Replaces the head of the network with the number of classes in the dataset
    # (e.g., 2 when one class label; + 1 for the null class)
     
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(config['classes'])+1)
    
    model.to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay']
    )
    criterion = lambda loss_dict: sum(loss_dict.values()) # model(images, targets)

    return model, train_loader, test_loader, criterion, optimizer        


def model_pipeline(hyperparameters):

    # tell wandb to get started
    with wandb.init(project="pytorch-RAITE", config=hyperparameters):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      model, train_loader, test_loader, criterion, optimizer = make(config)

      train(model, train_loader, criterion, optimizer, config)

      # test(model, test_loader)
      torch.save(model.state_dict(), "model_weights.pth")
      artifact = wandb.Artifact("fasterrcnn-model", type="model")
      artifact.add_file("model_weights.pth")
      wandb.log_artifact(artifact)

    return model


if __name__ == "__main__":

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    wandb.login()
    
    # TODO: set some way to pass data to the config
    config = dict(
    epochs=1,
    classes={0:0, 1:1},
    batch_size=1,
    learning_rate=0.01,
    momentum=0.9,
    weight_decay=0.0005,
    dataset_path="/home/eherrin@ad.ufl.edu/code/gitlab_dev/raiteclassify/data/archive/ugv_dataset_v7/train",
    train_ratio=0.8,
    architecture="FasterRCNN")
    
    model = model_pipeline(config)
