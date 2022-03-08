import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import cv2
import tensorflow as tf
import wandb
from timm.models import create_model
import convnext
import argparse
wandb.login()

wandb.init(entity="safwennaimi", project="sweep")
#wandb.watch_called = False # Re-run the model without restarting the runtime, unnecessary after our next release

# WandB – Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config          # Initialize config
config.batch_size = 4          # input batch size for training (default: 64)
config.test_batch_size = 10    # input batch size for testing (default: 1000)
config.epochs = 50             # number of epochs to train (default: 10)
config.learning_rate = 0.1               # learning rate (default: 0.01)
config.momentum = 0.1          # SGD momentum (default: 0.5) 
config.no_cuda = False         # disables CUDA training
config.seed = 42               # random seed (default: 42)
config.log_interval = 10     # how many batches to wait before logging training status
config.step_size = 7
config.gamma = .1

use_cuda = not config.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

label_names={'ants', 'bees'}
#label_names={'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(32), #224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        transforms.Resize(46), #256
        transforms.CenterCrop(32), #224
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

data_dir = 'hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=config.batch_size,
                                              shuffle=True, num_workers=0)
              for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes
print(class_names)

def train_model(args, model, criterion, optimizer, scheduler, epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print(f'Epoch {epoch}/{epochs-1}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0.0
        
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss_test = running_loss / dataset_sizes['test']
            epoch_acc_test = running_corrects.double() / dataset_sizes['test']
            epoch_loss_train = running_loss / dataset_sizes['train']
            epoch_acc_train = running_corrects.double() / dataset_sizes['train']
            if phase == 'train':
                print('train: Loss: {:.4f} Acc: {:.4f}'.format(
                epoch_loss_train, epoch_acc_train))
            else:
                print('test: Loss: {:.4f} Acc: {:.4f}'.format(
                epoch_loss_test, epoch_acc_test))

            # deep copy the model
            if phase == 'test' and epoch_acc_test > best_acc:
                best_acc = epoch_acc_test
                best_model_wts = copy.deepcopy(model.state_dict())
            metrics = {'accuracy_test': epoch_acc_test, 'loss_test': epoch_loss_test,
            'accuracy_train': epoch_acc_train, 'loss_train': epoch_loss_train}
        wandb.log(metrics)
        print()

    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))




def main():
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Set random seeds and deterministic pytorch for reproducibility
    # random.seed(config.seed)       # python random seed
    torch.manual_seed(config.seed) # pytorch random seed
    # numpy.random.seed(config.seed) # numpy random seed
    torch.backends.cudnn.deterministic = True
    bs=config.batch_size
    print(bs)



    model = create_model(
        'convnext_tiny', 
        pretrained=False, 
        num_classes=1000, 
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
        )
#### ConvNet as fixed feature extractor ####
# Here, we need to freeze all the network except the final layer.
# We need to set requires_grad == False to freeze the parameters so that the gradients are not computed in backward()
    model.head = nn.Linear(in_features=model.head.in_features,out_features=2,bias=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.

    optimizer_conv = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    wandb.watch(model)
# Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=config.step_size, gamma=config.gamma)

    train_model(config, model, criterion, optimizer_conv,
                         exp_lr_scheduler, epochs=config.epochs)


if __name__ == '__main__':
    main()