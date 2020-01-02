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
<br />

**Sample Image** <br />
![real_samples](https://user-images.githubusercontent.com/41305591/71657759-3c70b480-2d0f-11ea-8ba6-9656d16a102b.png) <br />


**Results** <br />
epoch 0 : <br />
![fake_samples_epoch_000](https://user-images.githubusercontent.com/41305591/71657863-a8531d00-2d0f-11ea-9b94-961960a62760.png) <br />

epoch 4: <br />
![fake_samples_epoch_004](https://user-images.githubusercontent.com/41305591/71657914-d5073480-2d0f-11ea-9c86-9b90d9df3659.png) <br />

epoch 8 : <br />
![fake_samples_epoch_008](https://user-images.githubusercontent.com/41305591/71657962-07b12d00-2d10-11ea-98ec-2cab78eb7792.png) <br />

epoch 12: <br />
![fake_samples_epoch_012](https://user-images.githubusercontent.com/41305591/71657991-28798280-2d10-11ea-9596-a915f0a483a6.png) <br />

epoch 15 : <br />
![fake_samples_epoch_015](https://user-images.githubusercontent.com/41305591/71658028-552d9a00-2d10-11ea-8ec1-85ce07c8acd7.png) <br />

epoch 18 : <br />
![fake_samples_epoch_018](https://user-images.githubusercontent.com/41305591/71658436-0e40a400-2d12-11ea-91eb-6f25fcf9f03d.png) <br />

<br />
<br />
<br />
**Issue** <br />
There is a bug in the code which caused the model to take a noise as input in its 19th epoch and thus it started again from square one. <br />

epoch 19: <br />
![fake_samples_epoch_019](https://user-images.githubusercontent.com/41305591/71658562-8b6c1900-2d12-11ea-9883-24d27f4cb374.png) <br />


epoch 21 : <br />
![fake_samples_epoch_021](https://user-images.githubusercontent.com/41305591/71658058-6d9db480-2d10-11ea-9e7f-0fd56a2aca1f.png) <br />

epoch 24 : <br />
![fake_samples_epoch_024](https://user-images.githubusercontent.com/41305591/71658084-860dcf00-2d10-11ea-90dd-fc459394b1df.png)

## Acknowledgments

* Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009
* Udemy
* Hadelin de Ponteves
* Kirill Eremenko

