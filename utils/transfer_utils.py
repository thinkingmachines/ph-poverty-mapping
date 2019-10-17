# -*- coding: utf-8 -*-

"""Utility methods for training Night Lights Model"""

from __future__ import print_function, division
import time
import os
import copy
import gc
import shutil

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
from decimal import Decimal
from sklearn import metrics
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, models, transforms
from torchsummary import summary
from torch.autograd import Variable

import logging
logging.basicConfig(level=logging.DEBUG)

SEED = 42
np.random.seed(SEED)

USE_GPU = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(USE_GPU)

IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]

def load_transform_data(data_dir="../data", batch_size=32):
    """ Transforms the training and validation sets.
    Source: https://discuss.pytorch.org/t/questions-about-imagefolder/774/6 
    
    Parameters
    ----------
    data_dir : str
        Directory of the training and validations image sets
    batch_size : int (default is 32)
        Batch size 
    
    Returns
    -------
    dict
        Contains the set images for training and validation set
    list
        Contains dataset sizes
    list
        Contains class names
    """
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize(400),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    IMGNET_MEAN, IMGNET_STD
                ),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(400),
                transforms.ToTensor(),
                transforms.Normalize(
                    IMGNET_MEAN, IMGNET_STD
                ),
            ]
        ),
    }

    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(data_dir, x), data_transforms[x]
        )
        for x in ["train", "val"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {
        x: len(image_datasets[x]) for x in ["train", "val"]
    }
    class_names = image_datasets["train"].classes

    return dataloaders, dataset_sizes, class_names


def imshow(inp, title=None, size=(20, 20)):
    """Imshow for Pytorch tensor.
    
    Parameters
    ----------
    inp : torch.Tensor
        The tensor of the input image
    title : str (default is None)
        Title of the image
    size : tuple (default is (20, 20))
        Size of image: (width, height)
    
    """
    plt.figure(figsize=size)
    inp = inp.numpy().transpose((1, 2, 0))
    inp = IMGNET_STD * inp + IMGNET_MEAN
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)


def save_plot(fig_dir, dict_, metric="loss"):
    """Saves train/ val loss curve as a PNG file
    
    Parameters
    ----------
    fig_dir : str
        File to figures directory
    dict_ : dict
        A dictionary containing phase (train or val) as keys 
        and the series of evaluation metrics until current epoch as values
    metric : str
        Label of error metric
    """
    fig = plt.figure()
    for phase in dict_:
        plt.plot(dict_[phase], label=phase)
        plt.xlabel("epoch")
        plt.ylabel(metric)
    plt.legend()
    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fig.savefig(fig_dir + metric + ".png")
    
    plt.close()


def save_checkpoint(
    state, is_best, filename, checkpoint_dir
):
    """Saves latest model
    
    Parameters
    ----------
    state : dict
        State of the model to be saved
    is_best : boolean
        Whether or not current model is the best model
    checkpoint_dir : str
        Path to models
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(state, checkpoint_dir + filename)
    if is_best:
        shutil.copyfile(
            checkpoint_dir + filename,
            checkpoint_dir + "model_best.pt",
        )
        
def load_checkpoint(
    model_best_path,
    model=None, 
    optimizer=None, 
    scheduler=None,
    epoch = 0,
    evals = (None, None, None)
):
    # Load best model    
    if os.path.isfile(model_best_path):
        # Load states
        checkpoint = torch.load(model_best_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Update settings
        epoch = checkpoint['epoch']
        losses = checkpoint['losses']
        accuracies = checkpoint['accuracies']
        f_ones = checkpoint['f_ones']
        evals = (losses, accuracies, f_ones)
        logging.info("Loaded checkpoint '{}' (epoch {}) successfully.".format(model_best_path, epoch))
    else:
        logging.info("No checkpoint found.")
    return model, optimizer, epoch, evals

def train_model(
    model,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer,
    scheduler,
    num_epochs=100,
    curr_epoch=0,
    curr_evals=(None, None, None),
    checkpoint_dir="models/",
):
    """ Trains Night Lights Model
    
    Parameters
    ----------
    model : NTLModel class
        The pretrained model to be fine-tuned
    dataloaders : 
        Contains the set images for training and validation set
    dataset_sizes : list
        Contains dataset sizes
    criterion 
        Loss function, e.g. cross entropy loss
    optimizer
        Optimization algorithm, e.g. SGD, Adam
    scheduler 
        Learning rate scheduler
    num_epochs : int (default is 25)
        Number of epochs
        
    Returns
    -------   
    NTLModel class
        The fine-tuned model
    
    """
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    loss = 10 ** 5
    best_f1 = 0

    phases = ["train", "val"]
    losses, accuracies, f_ones = curr_evals
    if not losses:
        losses = {phase: [] for phase in phases}
    if not accuracies:
        accuracies = {phase: [] for phase in phases}
    if not f_ones:
        f_ones = {phase: [] for phase in phases}

    for epoch in range(curr_epoch, num_epochs):
        logging.info("Epoch {}/{}".format(epoch, num_epochs - 1))

        # Each epoch has a training and validation phase
        update = {}
        for phase in phases:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            preds_ = []
            labels_ = []
            
            # Iterate over data.
            for idx, (inputs, labels) in tqdm(
                enumerate(dataloaders[phase]),
                total=len(dataloaders[phase]),
                desc="Iteration",
                ncols=2,
            ):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward: track history if only in train
                with torch.set_grad_enabled(
                    phase == "train"
                ):
                    outputs = model(inputs)
                    outputs = outputs.view(
                        outputs.size(0), -1
                    )
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(
                    preds == labels.data
                )
                preds_.extend(preds.cpu().numpy().tolist())
                labels_.extend(
                    labels.data.cpu().numpy().tolist()
                )

            # epoch loss, accuracy, and f1 score
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (
                running_corrects.double()
                / dataset_sizes[phase]
            )
            epoch_f1 = metrics.f1_score(
                preds_, labels_, average="macro"
            )

            losses[phase].append(epoch_loss)
            accuracies[phase].append(epoch_acc)
            f_ones[phase].append(epoch_f1)

            # Print progress
            learning_rate = optimizer.param_groups[0]["lr"]
            logging.info(
                "{} Loss: {:.4f} Accuracy: {:.4f} F1-Score: {:.4f} LR: {:.4E}".format(
                    phase.upper(),
                    epoch_loss,
                    epoch_acc,
                    epoch_f1,
                    Decimal(learning_rate),
                )
            )

            if phase == "val":
                # Update scheduler
                scheduler.step(epoch_loss)
                
                # Check if current model gives the best F1 score
                is_best = False
                if epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    best_model_wts = copy.deepcopy(
                        model.state_dict()
                    )
                    is_best = True
                    
                # Save states dictionary
                state = {
                    "epoch": epoch + 1,
                    "lr": learning_rate,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "losses": losses,
                    "accuracies": accuracies,
                    "f_ones": f_ones,
                }
                    
                # Make filename verbose
                filename = "model_{0:d}_{1:.3f}_{2:.3f}_{3:.3f}.pt".format(
                    epoch,
                    epoch_loss,
                    epoch_f1,
                    epoch_acc
                )
                    
                # Save model checkpoint
                save_checkpoint(
                    state,
                    is_best,
                    filename,
                    checkpoint_dir
                )
            
            # Save loss/acc/f1 curve figures
            #save_plot(fig_dir, losses, metric="loss")
            #save_plot(
            #    fig_dir, accuracies, metric="accuracy"
            #)
            #save_plot(fig_dir, f_ones, metric="f1_score")

        if learning_rate <= 1e-10:
            break
            
        loss = loss.item()

    time_elapsed = time.time() - since
    logging.info(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    logging.info("Best Val Accuracy: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(
    model, dataloaders, class_names, num_images=4, size=(5, 5)
):
    """ Prints the predicted labels for selected images.
    
    Parameters
    ----------
    model : NTLModel class
        The model to be used for prediction
    dataloaders : dict
        Contains images in the validation set
    num_images : int (default is 4)
        Number of images to predict
    """

    was_training = model.training
    model.eval()

    images_so_far = 0
    fig = plt.figure()

    pred_list = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(
            dataloaders["val"]
        ):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                pred = class_names[preds[j]]
                imshow(
                    inputs.cpu().data[j],
                    title=pred,
                    size=size,
                )
                pred_list.append(pred)

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def get_embedding(img_path, model_, size=4096, gpu=False):
    """ Returns vector embedding from PIL image
    Source: https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c
    
    Parameters
    ----------
    img_path : str
        Path to image file
    model_ : NTLModel class
        The model to be used for prediction
    size : int (default is 4096)
        Size of the feature embedding
        
    Returns
    ------- 
    tensor 
        The Pytorch tensor containing the feature embeddings
    """

    model = copy.deepcopy(model_)
    model.eval()
    if gpu:
        model = model.cuda()
    image = Image.open(img_path)
    normalize = transforms.Normalize(
        mean=IMGNET_MEAN, std=IMGNET_STD
    )
    scaler = transforms.Resize(400)
    to_tensor = transforms.ToTensor()
    image = Variable(
        normalize(to_tensor(scaler(image))).unsqueeze(0)
    )

    embedding = torch.zeros(1, size, 1, 1)

    def copy_data(m, i, o):
        embedding.copy_(o.data)

    layer = list(model.classifier.children())[-3]
    h = layer.register_forward_hook(copy_data)
    image = image.to(DEVICE)
    h_x = model(image)
    h.remove()

    return embedding.view(embedding.size(0), -1)

def get_embedding_per_image(report, model):
    """Iterates over each image and computes their corresponding feature embeddings
    
    Parameters
    ----------
    report : pandas DataFrame
        The dataframe containing the file locations per image 
    model : model instance
        The transfer model used to extract feature embeddings
        
    Returns
    ------- 
    pandas Dataframe
        Returns the report with an additional column indicating the extracted feature embeddings per image
    """
    
    embeddings = []
    for index, row in tqdm(report.iterrows(), total=len(report)):
        filename = row['filename']
        embedding = np.array(get_embedding(filename, model, gpu=True))
        embeddings.append(embedding[0]) 
        
    report['embeddings'] = embeddings
    return report

def get_mean_embedding_per_cluster(report):
    """Calculates the mean feature embedding per cluster
    
    Parameters
    ----------
    report : pandas DataFrame
        The dataframe containing the file locations per image with an embeddings columns
    
    Returns
    ------- 
    pandas Dataframe
        A DataFrame containing the mean feature embeddings per cluster
    """
    
    cluster_embeddings = {'cluster': [], 'mean_embedding':[]}
    clusters = report['DHSCLUST'].unique()

    for cluster in tqdm(clusters, total=len(clusters)):
        embeddings = report[report['DHSCLUST'] == cluster]['embeddings'].tolist()
        mean_embedding = np.mean(embeddings, axis=0)
        cluster_embeddings['cluster'].append(cluster)
        cluster_embeddings['mean_embedding'].append(mean_embedding)

    cluster_embeddings = pd.DataFrame(cluster_embeddings)
    cluster_embeddings.head(3)
    
    return cluster_embeddings