import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import os
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

UPPER_LIMIT = 500000


def generate_data(
    img, data_dir, leaves_subpath, width=8, initial_leaves_count=0
):
    leaves_count = initial_leaves_count
    if img.shape[0] >= width and img.shape[1] >= width:
        for i in np.arange(0, img.shape[0] - width, 1):
            for j in np.arange(0, img.shape[1] - width, 1):
                img_window = img[i:i+width, j:j+width, :]
                fname_path = './{}/{}/{}.png'.format(data_dir, leaves_subpath, leaves_count)
                plt.imsave(fname_path, img_window)
                if leaves_count >= UPPER_LIMIT:
                    return leaves_count
                leaves_count += 1
    print('Generation Done for Class {}! Ending At Count Index {}'.format(leaves_subpath, leaves_count))
    return leaves_count

def generate_data_in_dir(dir_path, file_type, target_class, data_dir, target_width=8):
    count_index = 0
    for img_file in os.listdir(dir_path):
        print(img_file)
        if img_file.endswith(file_type):
            img_path = dir_path + '/' + img_file
            print(img_path)
            img = cv2.imread(img_path)
            count_index = generate_data(img, data_dir, target_class, target_width, count_index)

# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

def generate_samplers(data_dir, batch_size=128, valid_prop=0.2):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    # obtain training indices that will be used for validation
    num_train = len(dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_prop * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, valid_loader


