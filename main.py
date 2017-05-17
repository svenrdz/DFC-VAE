#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from time import time
# from pycrayon import CrayonClient

import model

# Variables
batchsize = 64
imsize = 16
cropsize = 16
len_z = 500
trainset = "/home/sven/Documents/Stage/Datasets/NUS-WIDE"
testset = "/home/sven/Documents/Stage/Datasets/Landmarks"
trainset = testset

# cc = CrayonClient('128.199.55.16')
# exp = cc.open_experiment('DFCVAE')

DFCVAE = model.DFCVAE(len_z, imsize).cuda()
vgg_conv = model.VGG().cuda()
descriptor_net = model.VGG(True).cuda()

# Dataset
def get_data(root=trainset):
    """
    Returns a preprocessed DataLoader from root folder
    """
    def rgb2bgr(img):
        img = torch.np.asarray(img)
        return img[:,:,::-1]
    def sub_mean(img):
        means_bgr = torch.Tensor([103.939, 116.779, 123.68]).unsqueeze(1).unsqueeze(2)
        return img - means_bgr.expand_as(img)
    custom_transform = transforms.Compose([
                        transforms.Lambda(rgb2bgr),
                        transforms.ToPILImage(),
                        transforms.Scale(imsize),
                        transforms.CenterCrop(cropsize),
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x.mul(255)),
                        transforms.Lambda(sub_mean)])
    data = ImageFolder(root=root, transform=custom_transform)
    return DataLoader(data, shuffle=True, batch_size=batchsize, num_workers=1)

def loss_func(recon, targets, mu, logvar):
    for i, mse_layer in enumerate(descriptor_net.content):
        mse_layer.target = targets[i]
    _, content_losses = descriptor_net(recon)
    FPL = sum(content_losses)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul(-0.5)
    return FPL + KLD

optimizer = optim.Adam(DFCVAE.parameters(), lr=5e-4, betas=(0.5, 0.999))

def train(epoch):
    DFCVAE.train()
    train_loss = 0
    part_loss = 0
    for batch_idx, (img,_) in enumerate(train_data):
        img = Variable(img.cuda())

        optimizer.zero_grad()

        targets,_ = vgg_conv(img.clone())
        recon, mu, logvar,_ = DFCVAE(img)

        loss = loss_func(recon, targets, mu, logvar)
        loss.backward()

        train_loss += loss.data[0]
        optimizer.step()

        # Logging
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(img), len(train_data.dataset),
                100. * batch_idx / len(train_data),
                loss.data[0] / len(img)))#, end='\r', flush=True)
            # exp.add_scalar_value('loss', loss.data[0] / len(img), wall_time=time(), step=epoch*len(train_data.dataset)/batchsize+batch_idx)
    print('\n====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_data.dataset)))
    return img, recon, train_loss

def test(epoch):
    DFCVAE.eval()
    test_loss = 0
    for img,_ in test_data:
        img = Variable(img.cuda(), volatile=True)

        targets,_ = vgg_conv(img.clone())
        recon, mu, logvar,_ = DFCVAE(img)

        loss = loss_func(recon, targets, mu, logvar)
        test_loss += loss.data[0]
    test_loss /= len(test_data.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

def save(path, epoch, train_loss, test_loss):
    torch.save({
            'epoch': epoch + 1,
            'state_dict': DFCVAE.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss
        }, path)

def load(model, path):
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(path)
        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        test_loss = checkpoint['test_loss']
        model.load_state_dict(checkpoint['state_dict'])
        return epoch, train_loss, test_loss

def imshow(data):
    """
    Unprocess and show the image
    !!! Input is a tensor !!!
    """
    def bgr2rgb(img):
        img = torch.np.asarray(img)
        return img[:,:,::-1]
    def add_mean(img):
        means_bgr = torch.Tensor([103.939, 116.779, 123.68]).unsqueeze(1).unsqueeze(2)
        return img + means_bgr.expand_as(img)
    unloader = transforms.Compose([
                transforms.Lambda(add_mean),
                transforms.Lambda(lambda x: x.mul(1/255.)),
                transforms.ToPILImage(),
                transforms.Lambda(bgr2rgb),
                transforms.ToPILImage()])
    img = data.clone().cpu().resize_(3,cropsize,cropsize)
    plt.imshow(unloader(img))
    plt.show()

train_data = get_data()
test_data = get_data(testset)

for epoch in range(10):
    img, recon, train_loss = train(epoch+1)
    imshow(img.data)
    imshow(recon.data)
    test_loss = test(epoch)
    save('dfcvae_{}_{}'.format(epoch+1, time()), epoch+1, train_loss, test_loss)
    optimizer.param_groups[0]['lr'] *= 0.5
