import torch
import torch.nn as nn
from torch.utils.data import  TensorDataset

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader


# make train, test dataset to dataloader.
def get_dataloader(DATA_PATH, LABEL_PATH, classes):
    file_list = os.listdir(DATA_PATH)
    answer_file_list = os.listdir(LABEL_PATH)

    # dataset
    dataset = [0 for _ in range(len(file_list))]
    for i in range(len(file_list)):
        img = np.array(Image.open(DATA_PATH + file_list[i])) # h,w,c
        img = torch.tensor(img, dtype = torch.float32, requires_grad = True)
        img = img.permute(2,0,1)
        c,h,w = img.shape
        img = img.view(1, c, h, w)
        dataset[i] = img

    dataset = torch.cat(dataset, dim = 0)


    # label data
    label = [0 for _ in range(len(file_list))]
    print('label file counts: ', len(label))

    for i in range(len(file_list)):
        label[i] = torch.zeros((classes, h, w))

        img = np.array(Image.open(LABEL_PATH + answer_file_list[i]))
        for j in range(h):
            for k in range(w):
                cls = img[j][k]
                label[i][cls][j][k] = 1

        label[i] = label[i].view(1, classes, h, w)

        if i % 10 == 0:
            print( i / len(file_list) * 100, "% complete")

    train_label = torch.cat(label, dim=0)

    # make dataset
    dataset = TensorDataset(dataset, train_label)
    return dataset

def init_weight(model):
    def init_weight(model):
        if isinstance(model, nn.Linear):
            torch.nn.init.xavier_uniform(model.weight)
            model.bias.data.fill_(0)

        if isinstance(model, nn.Conv2d):
            torch.nn.init.xavier_uniform(model.weight)
            model.bias.data.fill_(0)

        if isinstance(model, nn.ConvTranspose2d):
            torch.nn.init.xavier_uniform(model.weight)
            model.bias.data.fill_(0)