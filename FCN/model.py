import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import utils

TORCH_CUDA_ARCH_LIST = 3.5

class PRE_VGG16(nn.Module):
    def __init__(self):
        super(PRE_VGG16, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3,3), stride = (1,1), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),

            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding = 0, dilation = 1, ceil_mode = False)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation= 1, ceil_mode=False)
        )

        self.layer3 = nn.Sequential(
            # total kernel size = 7 by 7
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride = 2, padding=0, dilation= 1, ceil_mode=False)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation= 1, ceil_mode=False)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation= 1, ceil_mode=False)
        )

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]

    def set_weight(self, weights):
        idx = 0

        for k in range(len(self.layers)):
            for params in self.layers[k].parameters():
                params_length = len(params.view(-1))

                new_param = weights[idx: idx + params_length]
                new_param = new_param.view(params.size())

                params.data.copy_(new_param)
                idx += params_length

        return


    def forward(self, input):
        ret = [0 for _ in range(len(self.layers))]
        x = input

        ret[0] = self.layer1(x)
        x = ret[0]
        ret[1] = self.layer2(x)
        x = ret[1]
        ret[2] = self.layer3(x)
        x = ret[2]
        ret[3] = self.layer4(x)
        x = ret[3]
        ret[4] = self.layer5(x)
        x = ret[4]

        return ret




class FCN8(nn.Module):
    def __init__(self, num_classes):
        super(FCN8, self).__init__()

        self.classes = num_classes
        self.VGG16 = PRE_VGG16()

        self.classifier = nn.Sequential(
            #conv 6
            nn.Conv2d(512, 4096, kernel_size = (1, 1)),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.2),

            # conv 7
            nn.Conv2d(4096, self.classes, kernel_size=(1, 1)),
            nn.BatchNorm2d(self.classes),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.2)
        )


        # 4x conv7
        self.conv7_upsample = nn.Sequential(
            nn.ConvTranspose2d(self.classes, self.classes, kernel_size=(3, 3), stride=(2, 2), output_padding=(1,1), padding = (1,1)),
            nn.BatchNorm2d(self.classes),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.classes, self.classes, kernel_size=(3, 3), stride = (2,2), output_padding=(1,1), padding = (1,1)),
            nn.BatchNorm2d(self.classes),
            nn.ReLU(inplace=True)
        )


        # reduce dimension
        ## 2x pool4
        self.Reduce_class_and_upsample_pool4 = nn.Sequential(
            nn.Conv2d(512, self.classes, kernel_size=(1, 1)),
            nn.BatchNorm2d(self.classes),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.classes, self.classes, kernel_size=(3, 3), stride = (2,2), output_padding=(1,1), padding = (1,1)),
            nn.BatchNorm2d(self.classes),
            nn.ReLU(inplace=True)
        )

        ## pool3
        self.Reduce_class_pool3 = nn.Sequential(
            nn.Conv2d(256, self.classes, kernel_size=(1, 1)),
            nn.BatchNorm2d(self.classes),
            nn.ReLU(inplace=True)
        )


        # upsample
        self.FCN = nn.Sequential(
            nn.ConvTranspose2d(self.classes * 3, self.classes, kernel_size = (3,3), stride = (2, 2), output_padding=(1,1),
                               padding = (1,1)),
            nn.BatchNorm2d(self.classes),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.classes, self.classes, kernel_size=(3, 3), stride=(2, 2), output_padding=(1, 1),
                               padding=(1, 1)),
            nn.BatchNorm2d(self.classes),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.classes, self.classes, kernel_size=(3, 3), stride=(2, 2), output_padding=(1, 1),
                               padding=(1, 1))
        )


    def set_weight(self, weights):
        self.VGG16.set_weight(weights)
        return


    def forward(self, input):
        # VGG16
        x = self.VGG16(input)

        for i in range(len(x)):
            x[i] = x[i].clone().detach().requires_grad_(True)

        # x[-1] = pool5, x[-2] = pool4, x[-1] = pool3
        y1 = self.conv7_upsample(self.classifier(x[-1])) # 4x conv7
        y2 = self.Reduce_class_and_upsample_pool4(x[-2]) # 2x pool 4
        y3 = self.Reduce_class_pool3(x[-3]) # pool3
        y = [y1, y2, y3]
        y = torch.cat(y, dim = 1)

        y = self.FCN(y)
        return F.softmax(y)



"""
h = 256
w = 256
test = FCN8(11)
test_input = torch.randn((1, 3, h, w))
x = test(test_input)
print(x.shape) # answer: batch_size, classes, h, w

_, a, _, _ = x.shape
total = 0
for i in range(a):
    total += x[0][i][0][0]
print(total) # answer: 1
"""