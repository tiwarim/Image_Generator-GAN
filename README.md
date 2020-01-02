# Image_Generator-GAN
 This computer vision model learns from real images and generates fake images using Genrative neural network. This requires a Generator to use Deconvolutional neural network which takes a noise vector of size 100 and outputs a fake image. This image is taken as input by Discriminator's convoliutional neural network which compares it to the ground truth training data and returns a cost function which gets propogated in the neural network. For the training of Discriminator, we compute cost function against a target of 0 while for Generator we compute it against target of 1. The model is trained for 25 epochs.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

you need to install

```
Anaconda
Python 3.6
Pytorch
CIFAR-10 dataset
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Anaconda :
https://docs.anaconda.com/anaconda/install/
Python 3.6 :
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-python.html
Pytorch :
https://anaconda.org/pytorch/pytorch
CIFAR-10 dataset:
https://www.cs.toronto.edu/~kriz/cifar.html
```


## Running the APP
run the app and wait for the model to get trained and generate images. Once finished, images should be generated in a separate folder called "results". This folder would containing an image labellel "real_samples.png" which contains 64 tiny sample sample images. The model learns from these images, generates fake images and then gets better at it. This process repeats for 25 epochs. At the end we get a fake image generated at each epoch (see results folder).





## Built With



## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

