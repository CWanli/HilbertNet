import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from .augmentation import *

__all__ = ['TrainModelNet', 'TestModelNet']

class TrainModelNet(Dataset):
    def __init__(self, csv_path, augmentation=None):
        """
        Args:
            csv_path (string): path to csv file
            augmentation (Boolean): data augmentation
        """
        # Augmentations
        self.augmentation = augmentation
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the points
        self.point_arr = np.asarray(self.data_info.iloc[:, 1])
        # Third column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 2])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        single_point_name = self.point_arr[index]
        single_label = self.label_arr[index]

        img_np = np.load(single_image_name)
        pc_np = np.load(single_point_name)[:,0:3]

        if self.augmentation:
            # rotate by 0/90/180/270 degree over z axis
            # pc_np = rotate_point_cloud_90(pc_np)
            # rotation perturbation
            # if self.rot_horizontal:
            #     pc_np = rotate_point_cloud(pc_np)
            # if self.opt.rot_perturbation:
            pc_np = rotate_perturbation_point_cloud(pc_np)

            # random jittering
            pc_np = jitter_point_cloud(pc_np)

            # random scale
            scale = np.random.uniform(low=0.8, high=1.2)
            pc_np = pc_np * scale

            # random shift
            # if self.opt.translation_perturbation:
            shift = np.random.uniform(-0.1, 0.1, (1,3))
            pc_np += shift

            # random flip
            img_np = flip_image(img_np, axis=1)

        img_as_tensor = torch.Tensor(img_np).unsqueeze(0)
        point_as_tensor = torch.Tensor(pc_np).permute(1,0).contiguous()

        return img_as_tensor, point_as_tensor, single_label

    def __len__(self):
        return self.data_len

class TestModelNet(Dataset):
    def __init__(self, csv_path):
        """
        Args:
            csv_path (string): path to csv file
        """
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the points
        self.point_arr = np.asarray(self.data_info.iloc[:, 1])
        # Third column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 2])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        single_point_name = self.point_arr[index]
        single_label = self.label_arr[index]

        img_np = np.load(single_image_name)
        pc_np = np.load(single_point_name)[:,0:3]

        img_as_tensor = torch.Tensor(img_np).unsqueeze(0)
        point_as_tensor = torch.Tensor(pc_np).permute(1,0).contiguous()

        return img_as_tensor, point_as_tensor, single_label

    def __len__(self):
        return self.data_len



if __name__ == '__main__':

    training_loader = TrainModelNet('./train_modelnet.csv',False)
    testing_loader = TestModelNet('./test_modelnet.csv')

    mn_dataset_loader = torch.utils.data.DataLoader(dataset=training_loader,
                                                    batch_size=16,
                                                    shuffle=True)

    for i, (images, points, labels) in enumerate(mn_dataset_loader):
        assert images.size()[2] == 4096 and images.size()[3] == 64
