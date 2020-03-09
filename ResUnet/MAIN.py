import torch
import torch.optim as optim
import tensorlayer as tl
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from Train_UNET import train_model
from RESUNET import ResNetUNet
from torchvision import transforms
from dataloader import FASTMRIDataset, ToTensor

# directory = '/media/tianyu.han/mri-scratch/DeepLearning/FastMRI/Image_rss/'
save_path = './old_model/MRNet/Res50_inpaint/'
tl.files.exists_or_mkdir(save_path)
directory = '/media/tianyu.han/mri-scratch/DeepLearning/MRKnee/images/'
dataset = FASTMRIDataset(folder=directory, 
                        transform=transforms.Compose([ToTensor()]))
train_dataloader = DataLoader(dataset, batch_size=16, 
                                shuffle=True, num_workers=32)
val_dataloader = DataLoader(FASTMRIDataset(folder=directory, 
                                        mode='val', 
                                        transform=transforms.Compose([ToTensor()])), 
                                        batch_size=16, num_workers=32)
dataloaders = {'train': train_dataloader, 'val': val_dataloader}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = ResNetUNet().to(device)

# freeze backbone layers
# Comment out to finetune further
for l in model.base_layers:
    for param in l.parameters():
        param.requires_grad = True
# optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, 
#                            model.parameters()), lr=1e-4)
optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)  
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, 
                                        step_size=30, gamma=0.1)        
model = train_model(model, dataloaders, optimizer_ft, exp_lr_scheduler, 
                    num_epochs=40, path=save_path+'saved.pth.tar')

# finetune backbone layers
'''
for l in model.base_layers:
    for param in l.parameters():
        param.requires_grad = True
optimizer_new = optim.Adam(model.parameters(), lr=1e-4)   
exp_lr_scheduler_ = lr_scheduler.StepLR(optimizer_new, 
                                         step_size=20, gamma=0.1)     
model = train_model(model, dataloaders, optimizer_new, exp_lr_scheduler_, 
                    num_epochs=40)
'''