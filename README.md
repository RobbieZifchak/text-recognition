## Handwritten Text Recognition

#### A multi-class classification problem for images of handwriting.

### The Problem:

* Can images of handwritten text be properly interpreted by a computer?


### Proposed Objective:
    
* Train a Convolutional Neural Network to extract features from training images, and correctly classify images of handwritten words utilizing Keras.


### Data Details
#### Source: IAM Handwriting Database - FKI Research Group

* 657 unique writers handwriting from 1,539 pages of scanned text containing 12,224 unique words
* Image samples are a variety of dimensions
* Some writers handwriting is messy and hard to classify, even to the human eye.
* Nested file structures are not conducive to word classification
* After removing damaged samples and imbalances in class distributions, overall selected modeling data contains 21,308 samples, and 78 unique words.

### Preprocessing

* For properly generating images for classification utilizing Keras ImageDataGenerator, they must must be sorted and stored in unique sub-directories based on their class labels.

        example: 
                text-recognition 
                        │         
                        │
                        └───train-words
                                │      
                                │
                                └───word_1
                                │  -word1 image files
                                │  
                                └───word_2
                                    -word2 image files
                                ...

* Images are then converted to grayscale, and reshaped to (64, 64, 1)
  (1 representing color channel (grayscale = 1, RGB = 3))

#### High-level model overview 

<img src= '.../images/slides/cnn-slide.png' width = '100' >



