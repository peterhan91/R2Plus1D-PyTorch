import torch.nn.functional as F
import tensorlayer as tl
import torch
import torch.nn as nn
import numpy as np
import copy
import time
import os

def train_model(model, dataloaders, optimizer, 
                scheduler, num_epochs=100, 
                path='saved.pth.tar'):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 9999
    criterion = nn.L1Loss()
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
    snapshot_FOLDER = './snapshot/'
    tl.files.exists_or_mkdir(snapshot_FOLDER)

    if os.path.exists(path):
        checkpoint = torch.load(path)
        print("Reloading from previously saved checkpoint")
        model.load_state_dict(checkpoint['state_dict'])

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            running_loss = 0.0
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr']) 
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            for sample in dataloaders[phase]:
                inputs, labels = sample['inputs'], sample['targets']
                inputs = inputs.to(device, dtype= torch.float)
                labels = labels.to(device, dtype= torch.float)             
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / dataset_sizes[phase]
            print(f"{phase} Loss: {epoch_loss}")

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print('saving best model')
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({'state_dict': model.state_dict()}, path)

                print('saving snapshot')
                for i, sample in enumerate(dataloaders[phase]):
                    inputs, labels = sample['inputs'], sample['targets']
                    inputs = inputs.to(device, dtype= torch.float)
                    labels = labels.to(device, dtype= torch.float) 
                    outputs = model(inputs)
                    if i == 0:
                        # print(outputs.shape)
                        tl.vis.save_images(np.moveaxis(outputs.detach().cpu().numpy(), 1, -1),
                                            [4, int(outputs.shape[0]//4)], 
                                            snapshot_FOLDER+'snapshot_out_%d.png' % epoch)
                        '''
                        tl.vis.save_images(np.moveaxis(inputs.detach().cpu().numpy(), 1, -1),
                                            [4, int(inputs.shape[0]//4)], 
                                            snapshot_FOLDER+'snapshot_in_%d.png' % epoch)
                        tl.vis.save_images(np.moveaxis(labels.detach().cpu().numpy(), 1, -1),
                                            [4, int(labels.shape[0]//4)], 
                                            snapshot_FOLDER+'snapshot_target_%d.png' % epoch)
                        '''
                        break
        
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model