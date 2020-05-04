import sys
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms 
import torch.optim as optim
from torch.optim import lr_scheduler 
from ABE_M_model import ABE_M 
from my_model import se_resnet50
from resnet import resnet50
from dataset import SingleData
from torch.utils.data import DataLoader
from loss_func import ABE_loss, ContrastiveLoss, Ms_loss
from torchsummary import summary



def get_data_list(data_path, ratio=0.1):
    img_list = []
    for root, dirs, files in os.walk(data_path):
        if files == []:
            class_name = dirs
        elif dirs == []:
            for f in files:
                img_path = os.path.join(root, f)
                img_list.append(img_path)

    np.random.seed(1)
    train_img_list = np.random.choice(img_list, size=int(len(img_list)*(1-ratio)), replace=False)
    #print(img_list, train_img_list)
    eval_img_list = list(set(img_list) - set(train_img_list))
    return class_name, train_img_list, eval_img_list 

def train_epoch(train_loader, model, loss_fn, optimizer, device):
    model.train()
    losses = []
    total_loss = 0
    for batch_idx, (data, target, _) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        target = target

        loss_outputs = loss_fn(output, target)

        losses.append(loss_outputs.item())
        total_loss += loss_outputs.item()
        if loss_outputs.requires_grad is True:
            loss_outputs.backward()
            optimizer.step()

        if batch_idx % 20 == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch_idx * target.size(0), len(train_loader.dataset), 100. * batch_idx / len(train_loader), np.mean(losses))
            print(message)
            losses = []

    print('total loss {:.6f}'.format(total_loss/(batch_idx+1)))


def eval_epoch(eval_loader, model, loss_fn, device, best, model_name, loss_name):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        for batch_idx, (data, target, _) in enumerate(eval_loader):
            data = data.to(device)
            target = target.to(device)

            output = model(data)

            loss_outputs = loss_fn(output, target)
            #print(loss_outputs)
            val_loss += loss_outputs.item()

    print('val loss {:.6f}'.format(val_loss/(batch_idx+1)))
    if best > val_loss/(batch_idx+1):
        best = val_loss/(batch_idx+1)
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), './model/{}_{}_{:.4f}.pth'.format(model_name, loss_name, best))
        else:
            torch.save(model.state_dict(), './model/{}_{}_{:.4f}.pth'.format(model_name, loss_name, best))
    return best 


if __name__ == '__main__':
    
    arg_len = len(sys.argv)
    if arg_len != 3:
        raise Exception("Invalid argvs!")
    model_name = sys.argv[1]
    loss_name = sys.argv[2]
    print('model is {}, loss function is {}'.format(model_name, loss_name))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    epochs = 100
    batch_size = 60    # 64
    lr = 0.001
    #num_learner = 4
    model_dict = {'ABE_M':ABE_M, 'se_resnet':se_resnet50, 'resnet50':resnet50}
    loss_dict = {'ABELoss':ABE_loss, 'ContrastiveLoss':ContrastiveLoss, 'MsLoss':Ms_loss}


    model = model_dict[model_name](attention=True)
    #model = model_dict[model_name]()  #################### remove attention
    if torch.cuda.device_count() > 1:
        print("Let's use {} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    #optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_fn = loss_dict[loss_name]() ########################################

    scheduler = lr_scheduler.StepLR(optimizer, step_size=50)
    
    data_path = './KIMIA_data/train/'
    
    class_name, train_img_list, eval_img_list = get_data_list(data_path)
    train_transform = transforms.Compose([ 
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor()
    ])
    eval_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    
    #========================== single data ================================
    train_dataset = SingleData(class_name, train_img_list, train_transform)
    eval_dataset = SingleData(class_name, eval_img_list, eval_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    

    best = 10000
    for epoch in range(epochs):
        print('epoch {}/{}'.format(epoch, epochs))
        scheduler.step()
        train_epoch(train_dataloader, model, loss_fn, optimizer, device)
        best = eval_epoch(eval_dataloader, model, loss_fn, device, best, model_name, loss_name)

        



