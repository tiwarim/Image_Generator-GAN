# Image_Generator-GAN
 This is a computer vision model that learns from real images and generates fake images using Genrative neural network. This model is trained for 25 epochs images generated in each epoch is better relative to the previous epoch due to training. This requires a Generator to use Deconvolutional neural network which takes a noise vector of size 100 and outputs a fake imgage. This image is taken as input by Discriminator's convoliutional neural network which compares it to the ground truth training data and returns a cost function which gets propogated in the neural network. For the training of Discriminator, we compute cost function against a target of 0 while for Generator we compute it against target of 1.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

you need to install

```
Anaconda
Python 3.6
Pytorch


```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds


## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

