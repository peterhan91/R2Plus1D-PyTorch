import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from Train_UNET import train_model
from RESUNET import ResNetUNet
from torchvision import transforms
from dataloader import FASTMRIDataset, ToTensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = ResNetUNet().to(device)

# freeze backbone layers
# Comment out to finetune further
for l in model.base_layers:
    for param in l.parameters():
        param.requires_grad = False

directory = '/media/tianyu.han/mri-scratch/DeepLearning/FastMRI/Image_rss/'
dataset = FASTMRIDataset(folder=directory, 
                        transform=transforms.Compose([ToTensor()]))
train_dataloader = DataLoader(dataset, batch_size=16, 
                                shuffle=True, num_workers=16)
val_dataloader = DataLoader(FASTMRIDataset(folder=directory, 
                                        mode='val', 
                                        transform=transforms.Compose([ToTensor()])), 
                                        batch_size=16, num_workers=16)
dataloaders = {'train': train_dataloader, 'val': val_dataloader}

optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, 
                           model.parameters()), lr=1e-4)
# optimizer_ft = optim.Adam(lr=1e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, 
                                        step_size=20, gamma=0.1)        
model = train_model(model, dataloaders, optimizer_ft, exp_lr_scheduler, 
                        num_epochs=150)