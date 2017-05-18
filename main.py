from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

layer_names = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
               'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
               'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
               'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
               'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5']
default_content_layers = ['relu3_1', 'relu4_1', 'relu5_1']

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, required=True,
                    help='path to dataset folder (must follow PyTorch ImageFolder structure)')
parser.add_argument('--workers', type=int,
                    help='number of data loading workers, default=2', default=2)
parser.add_argument('--batch_size', type=int,
                    default=64, help='input batch size, default=64')
parser.add_argument('--image_size', type=int, default=64,
                    help='height/width length of the input images, default=64')
parser.add_argument('--nz', type=int, default=100,
                    help='size of the latent vector z, default=100')
parser.add_argument('--nef', type=int, default=32,
                    help='number of output channels for the first encoder layer, default=32')
parser.add_argument('--ndf', type=int, default=32,
                    help='number of output channels for the first decoder layer, default=32')
parser.add_argument('--instance_norm', action='store_true',
                    help='use instance norm layer instead of batch norm')
parser.add_argument('--content_layers', type=str, nargs='?', default=None,
                    help='name of the layers to be used to compute the feature perceptual loss')
parser.add_argument('--niter', type=int, default=10,
                    help='number of epochs to train for, default=10')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate, default=0.0005')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--encoder', default='',
                    help="path to encoder (to continue training)")
parser.add_argument('--decoder', default='',
                    help="path to decoder (to continue training)")
parser.add_argument('--outf', default='./output',
                    help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')


args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Dataset loading
# Normalization mean and standard deviation are set accordingly to the ones used
# to train the vgg19 in torchvision model zoo
# https://github.com/pytorch/vision
transform = transforms.Compose([
    transforms.Scale(args.image_size),
    transforms.CenterCrop(args.image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))])
datafolder = dset.ImageFolder(root=args.dataroot, transform=transform)
dataloader = torch.utils.data.DataLoader(
    datafolder, shuffle=True, batch_size=args.batch_size, num_workers=args.workers, drop_last=True)

ngpu = int(args.ngpu)
nz = int(args.nz)
nef = int(args.nef)
ndf = int(args.ndf)
nc = 3
if args.instance_norm:
    Normalize = nn.InstanceNorm2d
else:
    Normalize = nn.BatchNorm2d
if args.content_layers is None:
    content_layers = default_content_layers


# custom weights initialization called on netG and netD
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight.data, a=0.01)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        init.normal(m.weight.data, std=0.015)
        m.bias.data.zero_()


class _VGG(nn.Module):

    def __init__(self, ngpu):
        super(_VGG, self).__init__()

        self.ngpu = ngpu
        features = models.vgg19(pretrained=True).features

        self.features = nn.Sequential()
        for i, module in enumerate(features):
            name = layer_names[i]
            self.features.add_module(name, module)

    def forward(self, input):
        batch_size = input.size(0)
        all_outputs = []
        output = input
        for name, module in self.features.named_children():
            if isinstance(output.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                output = nn.parallel.data_parallel(
                    module, output, range(self.ngpu))
            else:
                output = module(output)
            if name in content_layers:
                all_outputs.append(output.view(batch_size, -1))
        return all_outputs


descriptor = _VGG(ngpu)
print(descriptor)


class _Encoder(nn.Module):

    def __init__(self, ngpu):
        super(_Encoder, self).__init__()
        self.ngpu = ngpu
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
        self.mean = nn.Linear(nef * 8 * 4 * 4, nz)
        self.logvar = nn.Linear(nef * 8 * 4 * 4, nz)

    def sampler(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        if args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def forward(self, input):
        batch_size = input.size(0)
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            hidden = nn.parallel.data_parallel(
                self.encoder, input, range(self.ngpu))
            hidden = hidden.view(batch_size, -1)
            mean = nn.parallel.data_parallel(
                self.mean, hidden, range(self.ngpu))
            logvar = nn.parallel.data_parallel(
                self.logvar, hidden, range(self.ngpu))
        else:
            hidden = self.encoder(input)
            hidden = hidden.view(batch_size, -1)
            mean, logvar = self.mean(hidden), self.logvar(hidden)
        latent_z = self.sampler(mean, logvar)
        return latent_z


encoder = _Encoder(ngpu)
encoder.apply(weights_init)
if args.encoder != '':
    encoer.load_state_dict(torch.load(args.encoder))
print(encoder)


class _Decoder(nn.Module):

    def __init__(self, ngpu):
        super(_Decoder, self).__init__()
        self.ngpu = ngpu
        self.decoder_dense = nn.Sequential(
            nn.Linear(nz, ndf * 8 * 4 * 4),
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

    def forward(self, input):
        batch_size = input.size(0)
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            hidden = nn.parallel.data_parallel(
                self.decoder_dense, input, range(self.ngpu))
            hidden = hidden.view(batch_size, ndf * 8, 4, 4)
            output = nn.parallel.data_parallel(
                self.decoder_conv, input, range(self.ngpu))
        else:
            hidden = self.decoder_dense(input).view(
                batch_size, ndf * 8, 4, 4)
            output = self.decoder_conv(hidden)
        return output


decoder = _Decoder(ngpu)
decoder.apply(weights_init)
if args.decoder != '':
    decoder.load_state_dict(torch.load(args.decoder))
print(decoder)


mse = nn.MSELoss()
def fpl_criterion(recon_features, targets):
    fpl = 0
    for f, target in zip(recon_features, targets):
        fpl += mse(f, target.detach())#.div(f.size(1))
    return fpl

kld_criterion = nn.KLDivLoss()


input = torch.FloatTensor(
    args.batch_size, nc, args.image_size, args.image_size)
latent_labels = torch.FloatTensor(args.batch_size, nz).fill_(1)

if args.cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    descriptor = descriptor.cuda()
    input = input.cuda()
    latent_labels = latent_labels.cuda()

input = Variable(input)
latent_labels = Variable(latent_labels)

# setup optimizer
parameters = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(parameters, lr=args.lr, betas=(args.beta1, 0.999))

encoder.train()
decoder.train()

train_loss = 0
for epoch in range(args.niter):
    for i, (batch, _) in enumerate(dataloader):
        optimizer.zero_grad()
        input.data.copy_(batch)

        latent_z = encoder(input)
        targets = descriptor(input)
        kld = kld_criterion(F.log_softmax(latent_z), latent_labels)
        kld.backward(retain_variables=True)

        recon = decoder(latent_z)
        recon_features = descriptor(recon)
        fpl = fpl_criterion(recon_features, targets)
        fpl.backward()

        loss = kld + fpl
        train_loss += loss.data[0]
        optimizer.step()
        print('[{}/{}][{}/{}] FPL: {:.4f} KLD: {:.4f}'.format(
              epoch, args.niter, i, len(dataloader),
              fpl.data[0], kld.data[0]))
        if i % 100 == 0:
            vutils.save_image(input.data,
                              '{}/inputs.png'.format(args.outf),
                              normalize=True)
            vutils.save_image(recon.data,
                              '{}/reconstructions_epoch_{:03d}.png'.format(
                                  args.outf, epoch),
                              normalize=True)

    # do checkpointing
    torch.save(encoder.state_dict(), '{}/encoder_epoch_{}.pth'.format(args.outf, epoch))
    torch.save(decoder.state_dict(), '{}/decoder_epoch_{}.pth'.format(args.outf, epoch))
