#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import models
import matplotlib.pyplot as plt

layer_names = [ 'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
                'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
                'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5']
content_layers = ['relu3_1', 'relu4_1', 'relu5_1']
vgg_layers = models.vgg19(pretrained=True).features

class DFCVAE(nn.Module):
    def __init__(self, nz, imsz):
        super(DFCVAE, self).__init__()
        self.imsz = imsz
        self.encoder = nn.Sequential(
                        nn.Conv2d(3, 32, 4, 2, 1),#, bias=False),
                        nn.BatchNorm2d(32),
                        nn.LeakyReLU(0.2, True),
                        nn.Conv2d(32, 64, 4, 2, 1),#, bias=False),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(0.2, True),
                        nn.Conv2d(64, 128, 4, 2, 1),#, bias=False),
                        nn.BatchNorm2d(128),
                        nn.LeakyReLU(0.2, True),
                        nn.Conv2d(128, 256, 4, 2, 1),#, bias=False),
                        nn.BatchNorm2d(256),
                        nn.LeakyReLU(0.2, True),
                        nn.Conv2d(256, 256, 4, 2, 1),#, bias=False),
                        nn.BatchNorm2d(256))
        self.fce1 = nn.Linear(self.imsz * self.imsz // 4, nz)
        self.fce2 = nn.Linear(self.imsz * self.imsz // 4, nz)
        self.fcd = nn.Linear(nz, self.imsz * self.imsz // 2)
        self.decoder = nn.Sequential(
                        nn.ReLU(),
                        nn.UpsamplingNearest2d(scale_factor=2),
                        nn.modules.ReplicationPad2d(1),
                        nn.Conv2d(512, 256, 3),#, bias=False),
                        nn.BatchNorm2d(256, 1e-3),
                        nn.LeakyReLU(0.2, True),
                        nn.UpsamplingNearest2d(scale_factor=2),
                        nn.modules.ReplicationPad2d(1),
                        nn.Conv2d(256, 128, 3),#, bias=False),
                        nn.BatchNorm2d(128, 1e-3),
                        nn.LeakyReLU(0.2, True),
                        nn.UpsamplingNearest2d(scale_factor=2),
                        nn.modules.ReplicationPad2d(1),
                        nn.Conv2d(128, 64, 3),#, bias=False),
                        nn.BatchNorm2d(64, 1e-3),
                        nn.LeakyReLU(0.2, True),
                        nn.UpsamplingNearest2d(scale_factor=2),
                        nn.modules.ReplicationPad2d(1),
                        nn.Conv2d(64, 32, 3),#, bias=False),
                        nn.BatchNorm2d(32, 1e-3),
                        nn.LeakyReLU(0.2, True),
                        nn.UpsamplingNearest2d(scale_factor=2),
                        nn.modules.ReplicationPad2d(1),
                        nn.Conv2d(32, 3, 3))#, bias=False))

    def sample_z(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(torch.randn(std.size()).cuda())
        # eps = torch.cuda.FloatTensor(std.size()).normal_()
        # eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def encode(self, x):
        h1 = self.encoder(x).view(-1, self.imsz * self.imsz // 4)
        mu, logvar = self.fce1(h1), self.fce2(h1)
        return self.sample_z(mu, logvar)

    def decode(self, z):
        h2 = self.fcd(z)
        recon = self.decoder(h2.view(1, 512, self.imsz // 32, self.imsz // 32))
        return recon

    def forward(self, x):
        h1 = self.encoder(x).view(-1, self.imsz * self.imsz // 4)
        mu = self.fce1(h1)
        logvar = F.softplus(self.fce2(h1))
        z = self.sample_z(mu, logvar)
        h2 = self.fcd(z)
        recon = self.decoder(h2.view(x.size(0), 512, self.imsz // 32, self.imsz // 32))
        return recon, mu, logvar

class ContentLoss(nn.Module):
    def __init__(self, strength, name, target):
        super(ContentLoss, self).__init__()
        self.strength = strength
        self.name = name
        self.target = target
        self.crit = nn.MSELoss()

    def forward(self, x):
        zero = Variable(torch.zeros(x.size()).cuda())
        return self.crit(x - self.target, zero) * self.strength


class VGG(nn.Module):
    def __init__(self, desc=False):
        super(VGG, self).__init__()
        next_content_idx = 0
        self.features = nn.Sequential()
        self.content = []
        for i, module in enumerate(vgg_layers):
            if next_content_idx < len(content_layers):
                name = layer_names[i]
                if name.startswith('conv'):
                    module.weight.requires_grad = False
                    module.bias.requires_grad = False
                self.features.add_module(name, module)
                if name == content_layers[next_content_idx]:
                    if desc:
                        target = torch.Tensor()
                        self.content.append(ContentLoss(1.5 / len(content_layers), name, target))
                    next_content_idx += 1


    def forward(self, x):
        content_out = []
        loss = 0
        next_content_idx = 0
        h = x.clone()
        for name, module in self.features._modules.items():
            h = module(h)
            if name in content_layers:
                content_out.append(h)
                if self.content:
                    loss += self.content[next_content_idx](h) #.append(self.content[next_content_idx](h))
                    next_content_idx += 1
        return content_out, loss
