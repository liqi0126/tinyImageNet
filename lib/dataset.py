# Tiny ImangeNet Dataloader
import os
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
import random

def default_loader(path):
    return Image.open(path).convert('RGB')


def special_transfrom(img):
    # 'train/0/0_0.jpg' - 'train/9/9_999.jpg'
    other_class = random.randint(0, 99)
    other_index = random.randint(0, 999)
    other_img_name = 'train/' + str(other_class) + '/' + str(other_class) + '_' + str(other_index) + '.jpg'
    
    root = 'data'
    other_img = default_loader(os.path.join(root, other_img_name))
        
    # 25x25
    smaller_L = 25
    
    # shrink    
    other_img = other_img.resize((smaller_L, smaller_L))
    
    # shift
    corner_x = random.randint(0, 64 - smaller_L)
    corner_y = random.randint(0, 64 - smaller_L)
    
    # attach
    img.paste(other_img, (corner_x, corner_y))
    
    return img


class TinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root, data_list, transform=None, loader=default_loader):
        # root: your_path/TinyImageNet/
        # data_list: your_path/TinyImageNet/train.txt etc.
        images = []
        labels = open(data_list).readlines()
        for line in labels:
            items = line.strip('\n').split()
            img_name = items[0]

            # test list contains only image name
            test_flag = True if len(items) == 1 else False
            label = None if test_flag else np.array(int(items[1]))

            if os.path.isfile(os.path.join(root, img_name)):
                images.append((img_name, label))
            else:
                print(os.path.join(root, img_name) + 'Not Found.')

        self.root = root
        self.images = images
        self.transform = transform
        self.loader = loader
        self.special_transfrom = special_transfrom

    def __getitem__(self, index):
        img_name, label = self.images[index]
        img = self.loader(os.path.join(self.root, img_name))
        if self.transform is not None:
            # img = self.special_transfrom(img) # attach another picture on the picture for training
            img = self.transform(img)
        return (img, label) if label is not None else img

    def __len__(self):
        return len(self.images)


def get_loader(root, data_list, batch_size, workers=4, train=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.257, 0.276])
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            #transforms.RandomAffine(90),
            #transforms.RandomGrayscale(),
            #transforms.RandomPerspective(),
            transforms.ToTensor(),
            #transforms.RandomErasing(),
            normalize
        ])
    else:
        size = int(64 * 1.15)
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            normalize,
        ])
    dataset = TinyImageNetDataset(root, data_list, transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train,
                                         num_workers=workers, pin_memory=True)
    return loader
