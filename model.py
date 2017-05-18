#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.autograd import Variable
from torchvision import models

layer_names = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
               'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
               'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
               'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
               'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5']
default_content_layers = ['relu3_1', 'relu4_1', 'relu5_1']


class VGG(nn.Module):

    def __init__(self, content_layers=default_content_layers):
        super(VGG, self).__init__()

        features = models.vgg19(pretrained=True).features

        self.content_layers = content_layers
        self.features = nn.Sequential()
        for i, module in enumerate(features):
            name = layer_names[i]
            self.features.add_module(name, module)

    def forward(self, x):
        batch_size = x.size(0)
        output = []
        for name, module in self.features.named_children():
            x = module(x)
            if name in self.content_layers:
                x_flat = x.view(batch_size, -1)
                output.append(x_flat)
        return output


class Encoder(nn.Module):

    def __init__(self, nc, nef, latent_variable_size, norm='batch'):
        assert norm in [
            'batch', 'instance'], "Norm type should be 'batch' or 'instance'."
        if norm == 'batch':
            Normalize = nn.BatchNorm2d
        else:
            Normalize = nn.InstanceNorm2d

        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, nef, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            Normalize(nef),

            nn.Conv2d(nef, nef * 2, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            Normalize(nef * 2),

            nn.Conv2d(nef * 2, nef * 4, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            Normalize(nef * 4),

            nn.Conv2d(nef * 4, nef * 8, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True),
            Normalize(nef * 8)
        )

        self.mean = nn.Linear(nef * 8 * 4 * 4, latent_variable_size)
        self.logvar = nn.Linear(nef * 8 * 4 * 4, latent_variable_size)

    def forward(self, x):
        x = self.encoder(x).view(x.size(0), -1)
        mean, logvar = self.mean(x), self.logvar(x)
        return mean, logvar


class Decoder(nn.Module):

    def __init__(self, nc, ndf, latent_variable_size, norm='batch'):
        assert norm in [
            'batch', 'instance'], "Norm type should be 'batch' or 'instance'."
        if norm == 'batch':
            Normalize = nn.BatchNorm2d
        else:
            Normalize = nn.InstanceNorm2d

        super(Decoder, self).__init__()

        self.ndf = ndf

        self.decoder_dense = nn.Sequential(
            nn.Linear(latent_variable_size, ndf * 8 * 4 * 4),
            nn.ReLU(True)
        )

        self.decoder_conv = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 8, ndf * 4, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            Normalize(ndf * 4, 1e-3),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 4, ndf * 2, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            Normalize(ndf * 2, 1e-3),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 2, ndf, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            Normalize(ndf, 1e-3),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf, nc, 3, padding=1)
        )

    def forward(self, x):
        x = self.decoder_dense(x).view(x.size(0), self.ndf * 8, 4, 4)
        x = self.decoder_conv(x)
        return x


class Sampler(nn.Module):

    def __init__(self):
        super(Sampler, self).__init__()

    def forward(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mean)
