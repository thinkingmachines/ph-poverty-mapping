import os
import sys
sys.path.insert(0, '../utils')

import torchvision
import torch
import torchsummary 
import transfer_utils
from transfer_model import NTLModel

use_gpu = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(use_gpu)
print("Device: ", use_gpu)
 
def main():
    # File locations
    model_best_dir = '../data/model.pt'
    data_dir = '../../data/images/'
    fig_dir = '../output/figures/'
    checkpoint_dir = '../output/models/'
    
    # Settings
    img_dim = (3, 400, 400)
    num_epochs = 100
    lr = 1e-6
    factor = 0.1
    patience = 10
    
    curr_epoch = 0
    curr_evals = (None, None, None)
    
    # Load data
    dataloaders, dataset_sizes, class_names = transfer_utils.load_transform_data(
        data_dir=data_dir, batch_size=32
    )
    
    # Sanity check
    print(dataset_sizes)
    print(class_names)
    
    # Instantiate model
    model = torchvision.models.vgg16(pretrained=True)
    model = NTLModel(model, len(class_names))
    model = model.cuda()
    print(torchsummary.summary(model, img_dim))
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Load checkpoint, if found
    model, optimizer, curr_epoch, curr_evals = transfer_utils.load_checkpoint(model_best_dir, model, optimizer)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience)
    
    # Commence training
    model = transfer_utils.train_model(
        model, 
        dataloaders, 
        dataset_sizes, 
        criterion, 
        optimizer, 
        scheduler, 
        num_epochs=num_epochs, 
        curr_epoch=curr_epoch,
        curr_evals=curr_evals,
        fig_dir=fig_dir, 
        checkpoint_dir=checkpoint_dir
    )

if __name__ == "__main__":
    main()