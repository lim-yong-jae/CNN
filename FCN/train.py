import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt

import model
import utils
import hyper

import os

# FCN DNN state dict.
FILENAME = 'model.pt'
FILEPATH = "./" + FILENAME


# Train, Test data.
TRAIN_DATA_PATH = './dataset1/dataset1/images_prepped_train/'
TRAIN_LABEL_PATH = './dataset1/dataset1/annotations_prepped_train/'
TEST_DATA_PATH = './dataset1/dataset1/images_prepped_test/'
TEST_LABEL_PATH = './dataset1/dataset1/annotations_prepped_test/'

TRAIN_DATASET = 'train_dataset.pt'
TEST_DATASET = 'test_dataset.pt'


# designate device
device = torch.device('cpu')
if torch.cuda.is_available() == True:
    device = torch.cuda.current_device()
    torch.cuda.set_device(device)
    device = torch.device('cuda', device)
print("device: ", device)


# classes
classes = 11 + 1 # class(N) + background(1)
print("classes: ", classes)



# dataloader
train_dataset = None
if os.path.isfile('./' + TRAIN_DATASET):
    train_dataset = torch.load('./' + TRAIN_DATASET)
else:
    train_dataset = utils.get_dataloader(TRAIN_DATA_PATH, TRAIN_LABEL_PATH, classes)
    torch.save(train_dataset, './' + TRAIN_DATASET)
    print("save train dataset")

test_dataset = None
if os.path.isfile('./' + TEST_DATASET):
    test_dataset = torch.load('./' + TEST_DATASET)
else:
    test_dataset = utils.get_dataloader(TEST_DATA_PATH, TEST_LABEL_PATH, classes)
    torch.save(test_dataset, './' + TEST_DATASET)
    print("save test dataset")

train_dataloader = DataLoader(train_dataset, batch_size = hyper.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size = hyper.batch_size, shuffle=True)



# model
fcn = model.FCN8(classes)
if os.path.isfile(FILEPATH) == False:
    # get pretrained VGG16 for transfer learning.
    VGG16 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    VGG16.eval()

    # get pretrained VGG16 weight and set it to FCN weight
    params = []
    for param in VGG16.parameters():
        params.append(param.view(-1))
    params = torch.cat(params)

    fcn.set_weight(params)
    torch.save(fcn.state_dict(), FILEPATH)
else:
    fcn.load_state_dict(torch.load(FILEPATH, map_location = device))

fcn = fcn.to(device)
print("is model on cuda? ", next(fcn.parameters()).is_cuda) # True


# resize
h = 224
w = 224
composed = torchvision.transforms.Compose([
    torchvision.transforms.Resize([h,w], interpolation = torchvision.transforms.InterpolationMode.NEAREST)
])

# train
device = torch.device("cpu")
FCNoptim = optim.RMSprop(params = fcn.parameters(), lr = hyper.lr, momentum = hyper.momentum, weight_decay = hyper.weight_decay)

IMG = "./img/"
img_cnt = 80

criterion = nn.BCEWithLogitsLoss()
for i in range(hyper.epochs):
    cnt = 0

    for x, y in train_dataloader:
        cnt += 1

        x = x.to(device)
        y = y.to(device)

        x = composed(x)
        y = composed(y)

        pred = fcn(x)

        # custom loss function
        true_log_pred = torch.log(pred)
        false_log_pred = torch.log(torch.ones_like(pred) - pred)
        loss1 = -(y * true_log_pred).mean()
        loss2 = -((1.001 * torch.ones_like(y)-y) * (false_log_pred)).mean()
        loss = loss1 + 2 * loss2
        print("epochs: ", i, "cnt: ", cnt, "true_loss: ", loss1, "false_loss: ", loss2)

        FCNoptim.zero_grad()
        loss.backward()
        FCNoptim.step()

        # save data
        if(cnt == 1):
            img = y[0]
            img = img.argmax(dim=0)
            img = img.detach().numpy()
            IMG_PATH = IMG + str(img_cnt) + " cmp.png"
            plt.imsave(IMG_PATH, img)

            img2 = pred[0]
            img2 = img2.argmax(dim=0)
            img2 = img2.detach().numpy()
            IMG_PATH = IMG + str(img_cnt) + ".png"
            img_cnt += 1
            plt.imsave(IMG_PATH, img2)
        else:
            img = y[0]
            img = img.argmax(dim=0)
            img2 = pred[0]
            img2 = img2.argmax(dim=0)

            info = [-1]
            count = 0
            for a in range(128, h):
                for b in range(128, w):
                    if img[a][b] != img2[a][b]:
                        info = [a, b, img[a][b], img2[a][b], pred[0][img[a][b]][a][b], pred[0][img2[a][b]][a][b]] # pos_x, pos_y, true_class, pred_class, pred_prob
                        break
                if info[0] > -1:
                    break

            print("pos_x, pos_y, true_class, pred_class, true prob, pred_prob: ", info)


        # save model.
        torch.save(fcn.state_dict(), FILEPATH)
    print("----------------------------------------------------------------------")