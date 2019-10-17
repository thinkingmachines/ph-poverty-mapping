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

IMG_DIM = (3, 400, 400)
USE_GPU = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(USE_GPU)
 
def main(args):        
    # Load data
    dataloaders, dataset_sizes, class_names = transfer_utils.load_transform_data(
        data_dir=args.data_dir, batch_size=args.batch_size
    )
    
    # Sanity check
    logging.info(dataset_sizes)
    logging.info(class_names)
    
    # Instantiate model
    model = torchvision.models.vgg16(pretrained=True)
    model = NTLModel(model, len(class_names))
    if USE_GPU is not 'cpu':
        model = model.cuda()
    logging.info(torchsummary.summary(model, IMG_DIM))
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Load checkpoint, if found
    curr_epoch, curr_evals = 0, (None, None, None)
    model, optimizer, curr_epoch, curr_evals = transfer_utils.load_checkpoint(
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
        criterion, 
        optimizer, 
        scheduler, 
        num_epochs=args.epochs, 
        curr_epoch=curr_epoch,
        curr_evals=curr_evals,
        checkpoint_dir=args.checkpoint_dir
    )

if __name__ == "__main__":
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
        help='data directory (default: "../models/model.pt")'
    )
    parser.add_argument(
        '--checkpoint-dir', type=str, default="../models/", metavar='S',
        help='data directory (default: "../models/")'
    )
    args = parser.parse_args()
    
    main(args)