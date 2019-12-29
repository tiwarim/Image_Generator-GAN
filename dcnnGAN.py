#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 21:13:43 2019

@author: mrinal
"""


# importing the libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# setting hyperparameters
batchSize = 64
imageSize = 64 # size of generated images will be 64x64

# creating the transformations
# We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.
transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# loading the datasets

 # We download the training set in the ./data folder and we apply the previous transformations on each image.
dataset = dset.CIFAR10(root='./data', download = True, transform = transform)
# We use dataLoader to get the images of the training set batch by batch.
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2)

#defining the weights_init fnc that takes as input a nn m and taht will initialize all its weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 :
        m.weight.data.normal_(0.0, 0.02)  # assign weights of 0.0 and 0.02 for the convoolutional modules
        
    elif classname.find('BatchNorm') != -1 :
        m.weight.data.normal_(1.0, 0.02) # initialize the weights to batchNorm layers to 1.0 and 0.02
        m.bias.data.fill_(0)
        
# defining the arcitecture of the generator nn
class G(nn.Module): # we inherit modules from nn
    
    def __init__(self):
        super(G, self).__init__() # activating the inheritance. G inhering from nn.module
        # sequential is a clas, self.main is a object of the class
        self.main = nn.Sequential(    # this will be metamodule containing sequence opf layers
            # main contains the layers of different modules
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False), #input of inv cnn is a vector of size 100, number of feature map is 512, kernels will be squres of size 4x4, with no bias
            nn.BatchNorm2d(512), # normalizing the batch
            nn.ReLU(True), # applying reLu to increase non-linearity
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(True), # applying rectification
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 3, 4, 2, 1,bias =  False), # there would be three channels for the fake image output
            nn.Tanh(), # applying tanh rectification to increase non linearity and make sure the output is between -1 and 1 becuase we want the same standard as the image of the dataset
            )

    # this will forward propogate signal inside the neural network of G
    def forward(self, input): # input is the input of nn of generator. Random vector of size 100 representing noise
        output = self.main(input)
        return output
    
# creating the generator object
netG = G() # neural network of the generator
netG.apply(weights_init) # initializing the weights to the generator neural network


# defining the architecture of the discriminator nn
class D(nn.Module): 
    
    def __init__(self):
        super(D, self).__init__() # activating the inheritance
        
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),   # we use leaky relu for convolution as it works better
            
            nn.Conv2d(64, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(512, 1, 4, 1, 0, bias = False), # discrimantor returns a value of dimension 1 to give the discriminating probability
            nn.Sigmoid() # sigmoid gives a value between 0 and 1 giving us the probability we want
            )
        
    def forward(self, input):# input is the random image constructed by the generator, Output is the discriminating probability between 0 and 1
       output = self.main(input)
       return output.view(-1)     # flatten the result of the convolutions from 2d top 1 column
   
# creating the discriminator object
netD = D() # neural network of the discriminator
netD.apply(weights_init) # initializing the weights to the discriminator neural network


# Training the DcGAN 

criterion = nn.BCELoss() # binary cross entropy loss
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999)) # optimizer of discriminator
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999)) # optimizer of generator

for epoch in range(25): # we go through all the images in dataset 25 times
    for i, data in enumerate(dataloader, 0): # going through all te images in the dataset. data is a minibatch of images
        # 1st step updating the weights of the nn of the discriminator
        netD.zero_grad() # initialize gradients wrt to weights
        
        # training the discriminator with a real image from dataset
        real, _ = data # real image from minibatch
        input = Variable(real) # torch variable as these are compatible with the nn
        target = Variable(torch.ones(input.size()[0])) # tensor of ones wrapped in a torch variable
        output = netD(input) # probability between 0 a 1
        errD_real = criterion(output, target)  # computes the loss erro between output and target
        
        # training the discriminator with a fake image generated by the generator
        # create random value vector of size 100 that acts as input for generator
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1))# we create a minibatch (multiple vector of size 100) so that we get minibatch of fake images. Each of the 100 elements will be a matrix of size 1x1
        fake = netG(noise) # we get the fake images wich acts as the input to the discriminator
        target = Variable(torch.zeros(input.size()[0]))
        output = netD(fake.detach()) # we detach the gradient as we are not gonna use the gradient of the output wrt the weights of the generator
        errD_fake = criterion(output, target)
        
        # backpropogating the total error in the D nn
        errD = errD_real + errD_fake
        errD.backward() # back propogating the error
        optimizerD.step() # applying stochastic gradient descent to update the weights of the nn based on how much they are responsible for the total error
        
        
        # 2nd Step updating the weights of the nn of the generator
        netG.zero_grad()
        target = Variable(torch.ones(input.size()[0]))
        output = netD(fake) # we are gonna get the discriminating number and also the grdient of the fake wrt generator weights
        errG = criterion(output, target) # loss error
        
        errG.backward()  # backpropogating the error in the G nn
        optimizerG.step()
        
        
        # 3rd step : Printing the losses and savingthe real images and the generated images
        print('[%d/%d] [%d/%d] Loss_D: %4f  Loss_G: %4f' %  (epoch, 25, i, len(dataloader), errD.data, errG.data))
        if(i %100 == 0): # every 100 steps
            vutils.save_image(real, '%s/real_samples.png' % "./results", normalize = True)
            fake = netG(noise)  # getting the fake images
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True)
            
        
        
    

        
    

    
    
    
    