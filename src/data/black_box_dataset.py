import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import math
import pdb
from PIL import Image
import pandas as pd
import numpy as np

class GaussianLayer(nn.Module):
    def __init__(self, sigma=8):
        super(GaussianLayer, self).__init__()
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(10),
            nn.Conv2d(1, 1, 21, stride=1, padding=0, bias=None)
            )
        self.weights_init(sigma)
        #self.seq = self.seq.cuda()
  

    def forward(self, x): 
        return self.seq(x)

    def weights_init(self, sigma):
        n = np.zeros((21, 21))
        n[10, 10] = 1 
        k = scipy.ndimage.gaussian_filter(n, sigma=sigma)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))
            f.requires_grad = False

class BBoxDataset(Dataset):

    def __init__(self, csv_path, dataset_dir, mode='train', transforms=None, flag=0, debug=False, config=None):
        self.mode = mode
        if debug:
            dataset_dir = dataset_dir + 'debug'
        print(dataset_dir)
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        self.flag = flag
        self.args = config
        self.annotations = pd.read_csv(csv_path)
       
        print('start loading %s dataset'%mode)
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get sample
        sample = self.annotations.iloc[idx]

        # Get image ID
        #img_id = sample["id"]

        # Get image path
        
        # CORDA # img_path = os.path.join(self.dataset_dir, sample["path"] + '.png')
        # SkinCon # img_path = sample["img_path"]
        img_path = os.path.join(self.dataset_dir, sample["ID"] + '.jpg')
        img = Image.open(img_path).convert('RGB')

        # Get malignancy
        # CORDA # label = int(sample["covid"])
        # SkinCon # label = int(sample["benign_malignant"])
        ALL_CLASSES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
        class_label = sample[ALL_CLASSES].values.tolist().index(1)
        label = int(class_label)
        
        if self.transforms is not None:
            img = self.transforms(img)

        return img, label

