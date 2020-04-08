import os
import time

import numpy as np
import torch
import copy
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
from torchsummary import summary

from dataset import MRIDataset, ToTensor
from network import R2Plus1DClassifier, R2Plus1DBottleClassifier, PARALLEClassifier
from torchvision import transforms, utils
from inflate_src.i3res import I3ResNet
from ResUnet.RESUNET import ResNetUNet

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

def save_checkpoint(model, optimizer, epoch, path):
    print('Saving Model')
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'opt_dict': optimizer.state_dict(),
        }, path)

def train_model(resnet, num_classes, directory, LR=1e-3, layer_sizes=[3, 4, 6, 3],
                 num_epochs=50, save=True, train_Rijeka=False, path='mrnet.pth.tar', 
                 res_path='./ResUnet/old_model/saved.pth.tar'):
    """Initalizes and the model for a fixed number of epochs, using dataloaders from the specified directory, 
    selected optimizer, scheduler, criterion, defualt otherwise. Features saving and restoration capabilities as well. 
    Adapted from the PyTorch tutorial found here: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

        Args:
            num_classes (int): Number of classes in the data
            directory (str): Directory where the data is to be loaded from
            layer_sizes (list, optional): Number of blocks in each layer. Defaults to [2, 2, 2, 2], equivalent to ResNet18.
            num_epochs (int, optional): Number of epochs to train for. Defaults to 45. 
            save (bool, optional): If true, the model will be saved to path. Defaults to True. 
            path (str, optional): The directory to load a model checkpoint from, and if save == True, save to. Defaults to "model_data.pth.tar".
    """

    # initalize the ResNet 18 version of this model
    # model = R2Plus1DBottleClassifier(num_classes=num_classes, layer_sizes=layer_sizes, CH=32).to(device)
    '''
    resunet = ResNetUNet().to(device)
    if os.path.exists(res_path):
        checkpoint = torch.load(res_path)
        print("Reloading ResNet from previously saved checkpoint")
        resunet.load_state_dict(checkpoint['state_dict'])
    '''
    # resnet = torchvision.models.resnet50(pretrained=True)
    model = I3ResNet(copy.deepcopy(resnet), num_classes).to(device)
    summary(model, input_size=(3, 32, 224, 224))
    criterion = nn.BCELoss() # standard crossentropy loss for classification
    # optimizer = optim.Adam(model.parameters(), lr=LR)
    optimizer = optim.SGD(model.parameters(), lr=LR, 
                        momentum=0.9, weight_decay=1e-4)  # hyperparameters as given in paper sec 4.1
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 
    #                                     step_size=50, gamma=0.9)  # the scheduler divides the lr by 10 every 10 epochs

    # prepare the dataloaders into a dict
    MR_dataset = MRIDataset(directory=directory, rijeka=train_Rijeka, transform=transforms.Compose([ToTensor()]))
    train_dataloader = DataLoader(MR_dataset, batch_size=4, 
                                shuffle=True, num_workers=16)
    # IF training on Kinetics-600 and require exactly a million samples each epoch, 
    # import VideoDataset1M and uncomment the following
    # train_dataloader = DataLoader(VideoDataset1M(directory), batch_size=32, num_workers=4)
    val_dataloader = DataLoader(MRIDataset(directory=directory, mode='valid', rijeka=train_Rijeka, 
                                            transform=transforms.Compose([ToTensor()])), 
                                batch_size=4, num_workers=16)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

    # saves the time the process was started, to compute total time at the end
    start = time.time()
    epoch_resume = 0
    best_loss = 1e6
    MAX_patient = 6
    patient = 0

    # check if there was a previously saved checkpoint
    if os.path.exists(path):
        # loads the checkpoint
        checkpoint = torch.load(path)
        print("Reloading from previously saved checkpoint")
        # restores the model and optimizer state_dicts
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])
        # obtains the epoch the training is to resume from
        epoch_resume = checkpoint["epoch"]

    for epoch in tqdm(range(epoch_resume, num_epochs), unit="epochs", 
                        initial=epoch_resume, total=num_epochs, ascii=True):
        # each epoch has a training and validation step, in that order
        for phase in ['train', 'val']:
            # reset the running loss and corrects
            running_loss = 0.0
            # running_corrects = 0
            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                # scheduler.step()
                model.train()
            else:
                model.eval()

            for sample in dataloaders[phase]:
                # move inputs and labels to the device the training is taking place on
                inputs, labels = sample['buffers'], sample['labels']
                inputs = inputs.to(device, dtype= torch.float)
                labels = labels.to(device, dtype= torch.float)
                # inputs_x = inputs[:,0,:,:,:].unsqueeze_(1)
                # inputs_y = inputs[:,1,:,:,:].unsqueeze_(1)
                # inputs_z = inputs[:,2,:,:,:].unsqueeze_(1)
                # inputs_x = inputs_x.to(device, dtype= torch.float)
                # inputs_y = inputs_y.to(device, dtype= torch.float)
                # inputs_z = inputs_z.to(device, dtype= torch.float)
                optimizer.zero_grad()

                # keep intermediate states iff backpropagation will be performed. If false, 
                # then all intermediate states will be thrown away during evaluation, to use
                # the least amount of memory possible.
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    # we're interested in the indices on the max values, not the values themselves
                    # _, preds = torch.max(outputs, 1)  
                    loss = criterion(outputs, labels)

                    # Backpropagate and optimize iff in training mode, else there's no intermediate
                    # values to backpropagate with and will throw an error.
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()   

                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f"{phase} Loss: {epoch_loss}")

            if phase == 'val' and epoch_loss > best_loss:
                patient +=1
                if patient >= MAX_patient and LR > 1e-5:
                    print("decay loss from " + str(LR) + " to " +
                        str(LR * 0.1) + " as no improvement in val loss")
                    LR = LR * 0.1
                    # optimizer = optim.Adam(model.parameters(), lr=LR)
                    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
                    print("created new optimizer with LR " + str(LR))
                    patient = 0
            
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                save_checkpoint(model, optimizer, epoch, path)
                patient = 0     

    # print the total time needed, HH:MM:SS format
    time_elapsed = time.time() - start    
    print(f"Training complete in {time_elapsed//3600}h {(time_elapsed%3600)//60}m {time_elapsed %60}s")
