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
wandb.login()

hyperparameter_defaults = dict(
    batch_size = 16,
    learning_rate = 0.001,
    momentum = 0.9,
    gamma = 0.1,
    step_size = 7,
    epochs = 10,
    pretrained = True
    )

wandb.init(config=hyperparameter_defaults, project="CIFAR10")
config = wandb.config


print(tf.test.is_gpu_available(cuda_only=False,min_cuda_compute_capability=None))

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

#label_names={'ants', 'bees'}
label_names={'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

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

data_dir = 'CIFAR10'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=config.batch_size,
                                              shuffle=True, num_workers=0)
              for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes
print(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(inp, title):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])


def train_model(model, criterion, optimizer, scheduler, num_epochs=config.epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
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

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model






model_conv = torchvision.models.resnet34(pretrained=False)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 10)

model_conv = model_conv.to(device)
wandb.watch(model_conv)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.

optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=config.learning_rate, momentum=config.momentum)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=config.step_size, gamma=config.gamma)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=config.epochs)



"""
########## TRYING CONVNEXT ##########
from timm.models import create_model
import convnext
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
model.head = nn.Linear(in_features=model.head.in_features,out_features=10,bias=True)
model = model.to(device)
print(model)
wandb.watch(model)
criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.

optimizer_conv = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=config.step_size, gamma=config.gamma)

model_conv = train_model(model, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=config.epochs)

"""