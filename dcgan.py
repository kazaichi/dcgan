# -*- coding: utf-8 -*-

import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

if torch.cuda.is_available():
    print("cuda")
    import torch.cuda as t
else:
    print("cpu")
    import torch as t

import torchvision
from torchvision import datasets, models, transforms, utils
import torchvision.utils as vutils

import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt

bs = 103  # バッチサイズ
sz = 64  # 画像のサイズ
nz = 100  # 潜在変数
ngf = 64  # generatorにおける画像のチャネル数
ndf = 64  # discriminaterにおける画像のチャネル数
nc = 1  # 入力チャネル数（白黒、グレースケール＝1、RGBカラー＝３、etc）

#データセットの作成
dataloader = torch.utils.data.DataLoader(
    datasets.ImageFolder(root = './image/all',
                   transform=transforms.Compose([
                       transforms.Grayscale(),
                       transforms.Scale(sz),
                       transforms.ToTensor(),
                       transforms.Normalize(mean=(0.5, ), std=(0.5, ))
                   ])),
    batch_size=bs
)

'''Discriminater'''
class netD(nn.Module):
    def __init__(self):
        super(netD, self).__init__()
        self.netD_1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.netD_2 = nn.LeakyReLU(0.2, inplace=True)
        self.netD_3 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.netD_4 = nn.BatchNorm2d(ndf * 2)
        self.netD_5 = nn.LeakyReLU(0.2, inplace=True)
        self.netD_6 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.netD_7 = nn.BatchNorm2d(ndf * 4)
        self.netD_8 = nn.LeakyReLU(0.2, inplace=True)
        self.netD_9 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.netD_10 = nn.BatchNorm2d(ndf * 8)
        self.netD_11 = nn.LeakyReLU(0.2, inplace=True)
        self.netD_12 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        self.netD_13 = nn.Sigmoid()

    def forward(self, x):
        x = self.netD_1(x)
        x = self.netD_2(x)
        x = self.netD_3(x)
        x = self.netD_4(x)
        x = self.netD_5(x)
        x = self.netD_6(x)
        x = self.netD_7(x)
        x = self.netD_8(x)
        x = self.netD_9(x)
        x = self.netD_10(x)
        x = self.netD_11(x)
        x = self.netD_12(x)
        x = self.netD_13(x)
        return x

'''Generator'''
class netG(nn.Module):
    def __init__(self):
        super(netG, self).__init__()
        self.netG_1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
        self.netG_2 = nn.BatchNorm2d(ngf * 8)
        self.netG_3 = nn.ReLU(True)
        self.netG_4 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.netG_5 = nn.BatchNorm2d(ngf * 4)
        self.netG_6 = nn.ReLU(True)
        self.netG_7 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.netG_8 = nn.BatchNorm2d(ngf * 2)
        self.netG_9 = nn.ReLU(True)
        self.netG_10 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.netG_11 = nn.BatchNorm2d(ngf)
        self.netG_12 = nn.ReLU(True)
        self.netG_13 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)
        self.netG_14 = nn.Tanh()

    def forward(self, x):
        x = self.netG_1(x)
        x = self.netG_2(x)
        x = self.netG_3(x)
        x = self.netG_4(x)
        x = self.netG_5(x)
        x = self.netG_6(x)
        x = self.netG_7(x)
        x = self.netG_8(x)
        x = self.netG_9(x)
        x = self.netG_10(x)
        x = self.netG_11(x)
        x = self.netG_12(x)
        x = self.netG_13(x)
        x = self.netG_14(x)
        return x

criteion = nn.BCELoss()#loss関数の指定
net_D = netD()
net_G = netG()

if torch.cuda.is_available():
    D = net_D.cuda()
    G = net_G.cuda()
    criteion = criteion.cuda()

#最適化法の指定
optimizerD = optim.Adam(net_D.parameters(), lr = 5e-5)
optimizerG = optim.Adam(net_G.parameters(), lr = 5e-5)

input = t.FloatTensor(bs, nc, sz, sz)
noise = t.FloatTensor(normal(0, 1, (bs, 100, 1, 1)))
fixed_noise = t.FloatTensor(bs, 100, 1, 1).normal_(0, 1)
label = t.FloatTensor(bs)

real_label = 1
fake_label = 0

input = Variable(input)
label = Variable(label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)

niter = 1000 #試行回数

lossD_ave = []
lossG_ave = []
condition = 0 #既定のloss値の連続回数計測
for epoch in range(niter):
    DGoutput = 0
    lossD_sum = 0
    lossG_sum = 0
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        net_D.zero_grad()
        real, _ = data
        input.resize_(real.size()).copy_(real)
        label.resize_(bs).fill_(real_label)
        output = net_D(input)
        errD_real = criteion(output, label)
        errD_real.backward()
        D_x = output.data.mean()

        #train with fake (generated)
        noise.data.resize_(bs, 100, 1, 1)
        noise.data.normal_(0, 1)
        fake = net_G(noise)
        label.data.fill_(fake_label)
        output = net_D(fake.detach())
        errD_fake = criteion(output, label)
        errD_fake.backward()
        D_G_z1 = output.data.mean()

        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        net_G.zero_grad()
        label.data.fill_(real_label)
        output = net_D(fake)
        DGoutput += output
        errG = criteion(output, label)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()
        
        #loss計測
        lossD_sum += errD.data.item()
        lossG_sum += errG.data.item()
        #################

    lossD_ave.append(lossD_sum / (len(dataloader)))
    lossG_ave.append(lossG_sum / (len(dataloader)))
    print("[%d/%d] loss_D：%f loss_G: %f" % (epoch+1, niter, lossD_sum / (len(dataloader)), lossG_sum / (len(dataloader))))
    
    #モデルの保存
    if epoch % 50 == 0:
        fake = net_G(fixed_noise)
        vutils.save_image(fake.data, './results/image/fake_samples_epoch_%03d.jpg' % (epoch),normalize=True)
        torch.save(net_D.state_dict(), './results/weight/weightD_%03d.pth' % (epoch/50))
        torch.save(net_G.state_dict(), './results/weight/weightG_%03d.pth' % (epoch/50))
print("学習終了")

torch.save(net_D.state_dict(), './results/weight/lastD.pth')
torch.save(net_G.state_dict(), './results/weight/lastG.pth')

x = np.arange(0, niter, 1)
y1 = lossD_ave
plt.plot(x, y1, label='Discriminater_loss')
y2 = lossG_ave
plt.plot(x, y2, label='Generator_loss')
plt.xlabel("x") 
plt.xlabel("y")
plt.legend()
plt.show
plt.savefig('loss.png')
