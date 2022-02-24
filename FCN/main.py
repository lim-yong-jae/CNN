import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
test_dataset = None
if os.path.isfile('./' + TEST_DATASET):
    test_dataset = torch.load('./' + TEST_DATASET)
else:
    test_dataset = utils.get_dataloader(TEST_DATA_PATH, TEST_LABEL_PATH, classes)
    torch.save(test_dataset, './' + TEST_DATASET)
    print("save test dataset")

test_dataloader = DataLoader(test_dataset, batch_size = hyper.batch_size, shuffle=True)



# model
fcn = model.FCN8(classes)
if os.path.isfile(FILEPATH) == False:
    fcn.apply(utils.init_weight)

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
    fcn.load_state_dict(torch.load(FILEPATH))

fcn = fcn.to(device)
print("is model on cuda? ", next(fcn.parameters()).is_cuda) # True


h = 224
w = 224
# resize
composed = torchvision.transforms.Compose([
    torchvision.transforms.Resize([h, w], interpolation = torchvision.transforms.InterpolationMode.NEAREST)
])

# test
device = torch.device("cpu")
FCNoptim = optim.RMSprop(params = fcn.parameters(), lr = hyper.lr, momentum = hyper.momentum, weight_decay = hyper.weight_decay)

IMG = "./ans_img/"
img_cnt = 0


for x, y in test_dataloader:
    x = x.to(device)
    y = y.to(device)

    x = composed(x)
    y = composed(y)

    pred = fcn(x)
    N, c, h, w = y.shape

    if img_cnt == 0:
        img = y[0].argmax(dim = 0)
        img2 = pred[0].argmax(dim = 0)

        img = img.detach().numpy()
        img2 = img2.detach().numpy()
        ret = img - img2

        info = []
        for a in range(h):
            for b in range(w):
                if ret[a][b] != 0:
                    info.append([a, b, ret[a][b]])

    for i in range(N):
        img = x[i] / 255
        img = img.detach().numpy()
        img = img.transpose((1,2,0))
        IMG_PATH = IMG + str(img_cnt) + "_original.png"
        plt.imsave(IMG_PATH, img)

        img = y[i]
        # prob_y = y[0].max(dim = 1)
        img = img.argmax(dim=0)
        img = img.detach().numpy()
        IMG_PATH = IMG + str(img_cnt) + "_true.png"
        plt.imsave(IMG_PATH, img)

        img2 = pred[i]
        # prob_pred = pred[0].max(dim = 1)
        img2 = img2.argmax(dim=0)
        img2 = img2.detach().numpy()
        IMG_PATH = IMG + str(img_cnt) + "_predict.png"
        img_cnt += 1
        plt.imsave(IMG_PATH, img2)

