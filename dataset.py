# modified
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms 

class SingleData(Dataset):
    def __init__(self, class_name, img_list, transform=None):
        """
        build dataset for model test
        """
        self.img_list = img_list
        self.class_name = class_name
        print(self.class_name)
        self.label_list = []
        for i in range(len(img_list)):
            pre_fix = self.img_list[i].split('/')[-1].split('_')[0]
            self.label_list.append(self.class_name.index(pre_fix))
        self.label_list = np.array(self.label_list)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_list[index])
        target = self.label_list[index]
        if self.transform != None:
            img = self.transform(img)
        else:
            transform = transforms.Compose([transforms.CenterCrop(224),
                transforms.ToTensor()
                ]) 
            img = transform(img)
        
        return img, target, self.img_list[index] 

    def __len__(self):
        return len(self.img_list)







