import wandb
import os
import sys
import argparse
import logging
sys.path.insert(0, '../utils')

import torchvision
import torch
import torchsummary 
import transfer_utils
from transfer_model import NTLModel

SEED = 42
IMG_DIM = (3, 400, 400)
USE_GPU = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(USE_GPU)
 
def main(args):  
    torch.manual_seed(42)
    
    # Load data
    dataloaders, dataset_sizes, class_names = transfer_utils.load_transform_data(
        data_dir=args.data_dir, batch_size=args.batch_size
    )
    
    # Sanity check
    logging.info(USE_GPU)
    logging.info(dataset_sizes)
    logging.info(class_names)
    
    # Instantiate model
    model = torchvision.models.vgg16(pretrained=True)
    model = NTLModel(model, len(class_names))
    if USE_GPU is not 'cpu':
        model = model.cuda()
    logging.info(torchsummary.summary(model, IMG_DIM))
    wandb.watch(model)
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Load checkpoint, if found
    model, optimizer, curr_epoch = transfer_utils.load_checkpoint(
        args.model_best_dir, model, optimizer
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=args.factor, patience=args.patience
    )
    
    # Commence training
    model = transfer_utils.train_model(
        model, 
        dataloaders, 
        dataset_sizes, 
        class_names,
        criterion, 
        optimizer, 
        scheduler, 
        num_epochs=args.epochs, 
        curr_epoch=curr_epoch,
        checkpoint_dir=args.checkpoint_dir
    )

if __name__ == "__main__":
    wandb.init(project="tm-poverty-prediction")
    logging.basicConfig(level=logging.DEBUG)
    
    parser = argparse.ArgumentParser(description='Philippine Poverty Prediction')
    parser.add_argument(
        '--batch-size', type=int, default=32, metavar='N',
        help='input batch size for training (default: 32)'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-6, metavar='LR',
        help='learning rate (default: 1e-6)'
    )
    parser.add_argument(
        '--epochs', type=int, default=100, metavar='N',
        help='number of epochs to train (default: 100)'
    )
    parser.add_argument(
        '--factor', type=int, default=0.1, metavar='N',
        help='factor to reduce learning rate by on pleateau (default: 0.1)'
    )
    parser.add_argument(
        '--patience', type=int, default=10, metavar='N',
        help='number of iterations before reducing lr (default: 10)'
    )
    parser.add_argument(
        '--data-dir', type=str, default="../data/images/", metavar='S',
        help='data directory (default: "../data/images/")'
    )
    parser.add_argument(
        '--model-best-dir', type=str, default="../models/model.pt", metavar='S',
        help='besy model path (default: "../models/model.pt")'
    )
    parser.add_argument(
        '--checkpoint-dir', type=str, default="../models/", metavar='S',
        help='model directory (default: "../models/")'
    )
    args = parser.parse_args()
    wandb.config.update(args)
    
    main(args)