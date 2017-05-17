#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from time import time, sleep, strftime
from pycrayon import CrayonClient
import os

import model



class DFCVAE():
    _Datasets_path = "/home/sven/Documents/Stage/Datasets/"
    _Default_trainset = "CelebA"
    _Default_testset = "CelebA"

    def __init__(self, starting_lr=5e-4, lr_decay=0.5, batchsize=100, imsize=64, nz=100, trainset=None, testset=None, crayon=False, crayon_ip='128.199.55.16'):
        assert(imsize % 4 == 0)
        self.epoch = 0
        self.lr = starting_lr
        self.decay = lr_decay
        self.batchsize = batchsize
        self.imsize = imsize
        self.nz = nz
        if trainset:
            self.trainset = trainset
        else:
            self.trainset = self._Default_trainset
        if testset:
            self.testset = testset
        else:
            self.testset = self._Default_testset
        self.crayon = crayon
        if self.crayon:
            self.cc = CrayonClient(crayon_ip)
            self.exp = self.cc.open_experiment('DFCVAE')
        self.model = model.DFCVAE(self.nz, self.imsize).cuda()
        self.vgg = model.VGG().cuda()
        self.descriptor = model.VGG(True).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))
        #self.optimizer = optim.Adamax(self.model.parameters(), self.lr, betas=(0.9, 0.99))

    def __str__(self):
        return self.model.__str__()

    def __repr__(self):
        return self.__str__()

    def get_trainset(self):
        print('[+] Loading training data ...')
        self.train_data = self._get_dataloader(self.trainset)
        print('[+] Training data loaded successfully')

    def get_testset(self):
        print('[+] Loading test data ...')
        self.test_data = self._get_dataloader(self.testset)
        print('[+] Test data loaded successfully')

    def _get_dataloader(self, root):
        """
        Returns a preprocessed DataLoader from root folder
        """
        transform = transforms.Compose([
                        transforms.Scale(self.imsize),
                        transforms.CenterCrop(self.imsize),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])
        data = ImageFolder(root=self._Datasets_path+root, transform=transform)
        return DataLoader(data, shuffle=True, batch_size=self.batchsize, num_workers=4)

    def get_img(self, idx, test=False):
        if test:
            return self.test_data.dataset[idx][0]
        else:
            return self.train_data.dataset[idx][0]

    def save(self, path):
        print('[+] Saving model ...')
        torch.save({
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'train_loss': self.train_loss
                #'test_loss': self.test_loss
            }, path)
        print('[+] Model saved successfully')

    def load(self, path):
        if os.path.isfile(path):
            print('[+] Loading model ...')
            checkpoint = torch.load(path)
            self.epoch = checkpoint['epoch']
            self.train_loss = checkpoint['train_loss']
            #self.test_loss = checkpoint['test_loss']
            self.model.load_state_dict(checkpoint['state_dict'])
            print('[+] Model loaded successfully')

    def vgg_deprocess(self, tensor):
        means = torch.cuda.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
        std = torch.cuda.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
        return tensor.mul(std.expand_as(tensor)).add_(means.expand_as(tensor))

    def disp(self, img):
        if type(img) == list:
            old = list(img)
            img = torch.cuda.FloatTensor()
            for i in old:
                if type(i) == Variable:
                    i = i.data
                if len(i.size()) == 4:
                    i = i[0]
                img = torch.cat([img, i.unsqueeze(0)])
        elif type(img) == Variable:
            img = img.data
        torchvision.utils.save_image(self.vgg_deprocess(img), 'disp/image.jpg')


    def train(self, log_step=10):
        self.epoch += 1
        self.model.train()
        self.train_loss = 0
        for batch_idx, (img,_) in enumerate(self.train_data):
            cur_batchsize = len(img)
            img = Variable(img.cuda(), requires_grad=True)

            self.optimizer.zero_grad()

            recon, mu, logvar = self.model(img)

            loss = self.loss_func(recon, img, mu, logvar)
            loss.backward()

            self.train_loss += loss.data[0]
            if not batch_idx:
                self.running_loss = self.train_loss
            self.running_loss *= 0.6
            self.running_loss += 0.4*loss.data[0]
            self.optimizer.step()

            # Logging
            if batch_idx % log_step == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.epoch, batch_idx * cur_batchsize, len(self.train_data.dataset),
                    100. * batch_idx / len(self.train_data),
                    loss.data[0] / len(img)))
                self.disp([img, recon])
                if self.crayon:
                    step = self.epoch * len(self.train_data.dataset) / cur_batchsize + batch_idx
                    self.exp.add_scalar_value('loss', loss.data[0] / cur_batchsize, wall_time=time(), step=step)
                #if batch_idx:
                #    dist = self.running_loss - loss.data[0]
                #    print(dist*dist)
                #    if dist*dist < self.lr:
                #        self.lr *= self.decay
                #        self.optimizer.param_groups[0]['lr'] = self.lr

        print('\n====> Epoch: {} Average loss: {:.4f}'.format(
              self.epoch, self.train_loss / len(self.train_data.dataset)))
        if self.epoch % 2 == 0:
            self.lr *= self.decay
            self.optimizer.param_groups[0]['lr'] = self.lr

    def test(self):
        pass

    def weights_init(self):
        classname = self.model.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def loss_func(self, recon, img, mu, logvar):
        mseloss = torch.nn.MSELoss()
        zero = Variable(torch.zeros(recon.size()).cuda())
        MSE = mseloss(recon-img, zero)
        targets,_ = self.vgg(img)
        for i, mse_layer in enumerate(self.descriptor.content):
            mse_layer.target = targets[i]
        _, FPL = self.descriptor(recon)
        KLD_element = mu.pow(2).add(logvar.exp()).mul(-1).add(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        return MSE + KLD + FPL

    def loss_func2(self, recon, img, mu, logvar):
        mseloss = torch.nn.MSELoss()
        zero = Variable(torch.zeros(recon.size()).cuda())
        MSE = mseloss(recon-img, zero)
        KLD_element = mu.pow(2).add(logvar.exp()).mul(-1).add(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        return MSE + KLD

    def loss_func3(self, recon, img, mu, logvar):
        targets,_ = self.vgg(img)
        for i, mse_layer in enumerate(self.descriptor.content):
            mse_layer.target = targets[i]
        _, FPL = self.descriptor(recon)
        KLD_element = mu.pow(2).add(logvar.exp()).mul(-1).add(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        return FPL + KLD

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Deep Feature Consistent Variational Autoencoder')
    parser.add_argument('-b', '--batchsize', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=5, metavar='N', help='number of epochs to train (default: 5)')
    parser.add_argument('-i', '--imsize', type=int, default=64, metavar='N', help='fixed image width and height (default: 64)')
    parser.add_argument('-z', '--len_z', type=int, default=100, metavar='N', help='length of the latent vector z (default: 100)')
    parser.add_argument('-l', '--learningrate', type=float, default=5e-4, metavar='N', help='starting learning rate, halved at each epoch (default: 0.0005)')
    parser.add_argument('-t', '--trainset', type=str, default='CelebA', metavar='N', help='name of the folder containing the dataset')
    args = parser.parse_args()

    date = strftime('%y%m%d_%H%M%S')
    m = DFCVAE(batchsize=args.batchsize, imsize=args.imsize, trainset=args.trainset, starting_lr=args.learningrate)
    m.weights_init()

    for i in range(0, args.epochs):
        m.get_trainset()
        m.train()
        m.save('snapshots/{}_{}_{}'.format(m.trainset, date, m.epoch))
