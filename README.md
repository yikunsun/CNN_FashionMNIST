# CNN for Image Classification (FashionMNIST Dataset)

Recently I have been learning to use **PyTorch** to build neural networks for **computer vision** tasks. There are many types of computer vision tasks, such as image classification, object detection, and semantic segmentation. Since I was asked to build a model to label an image at work, I started to look into models that are good at classifying images. During my research, I have found **convolutional neural network (CNN)**, a type of neural network, is good at image classification. The model I built is called TinyVGG (https://poloclub.github.io/cnn-explainer/), which is a CNN architecture, so it consists of many of the same layers as a CNN model but on a smaller scale. Below is an overview of CNN.

CNN consists of many **convolutional blocks** to learn patterns from images. Inside each convolutional block, there are multiple **layers** working together to extract features from images. In my understanding, there are 3 important layers in CNN:
- **Convolutional layer**:
  - Convolutional layer consists of neurons performing a dot product of a kernel learned by the network and the output of the previous layer's neuron.
  - The goal of adding a convolutional layer is to convert all receptive pixels in a kernel to a single number, which can be then used for calculating the output of a convolutional neuron.
- **ReLU layer**:
  - ReLU, standing for rectified linear unit, applies an element-wise non-linear transformation to input data and decides which pixel in the output of a convolutional neuron gets activated.
  - ReLU(x) = max(0, x), so when an input value is larger than 0, it will be returned, otherwise 0 will be returned.
- **Max pooling layer**
  - Max pooling layer returns the maximum value in a kernel.
  - The goal of adding a max pooling layer is to reduce the spatial extent of the network, thus making the TinyVGG more computationally efficient and reducing overfitting.

After all convolutional blocks, a **flatten layer** is usually applied to reshape the dimensions of the output tensor, which will then be passed into one or more **linear layers**. Lastly, a **softmax activation function** is applied to scale outputs from logits to prediction probabilities. **Softmax activation function** makes sure the sum of outputs is equal to 1, so the label with the highest probability is the model's prediction. 

## *Table of Contents*
### Section 1: Get and process the Fashion MNIST dataset
- Download the Fashion MNIST dataset using torchvision.datasets module
- Turn datasets into dataloaders 

### Section 2: Build a baseline model for image classification
- Baseline model has only 2 layers, one flatten layer and one linear layer
- Set up loss function, optimizer, and accuracy metric
- Create loops to train and test the model
- Get model results

### Section 3: TinyVGG model for image classification
- TinyVGG model has 2 convolutional blocks consisted of convolutional, relu, and max pooling layers.

### Section 4: Experiment with different approaches to improve model performance
**Approaches:**
- Data augmentation
  - Apply **RandAugment** in Pytorch to apply several augmentation transformations sequentially
- Train model for more epochs
- Change optimizer's learning rate

## *Summary*
The TinyVGG model trained with 5 epochs with each epoch containing over 1800 batches of 32 images can achieve an accuracy of almost 90% on testing data with 10000 images, indicating very strong model performance. 

Out of all the approaches attempted to improve model performance, training with more epochs leads to the best results.
