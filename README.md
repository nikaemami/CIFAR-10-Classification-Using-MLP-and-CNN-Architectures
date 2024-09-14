# CIFAR-10 Classification Using MLP and CNN Architectures

This project involves classifying the CIFAR-10 dataset using both MLP and CNN architectures, exploring the effects of various hyperparameters.

## Dataset Overview

The CIFAR-10 dataset consists of 60,000 color images categorized into 10 classes. Below are some sample images from the dataset:

<p float="left">
  <img src="images/1.png" width="200" />
  <img src="images/2.png" width="200" />
  <img src="images/3.png" width="200" />
</p>

---

## Part 1: MLP Model

We start by training a classifier using a Multilayer Perceptron (MLP) network. The key steps include:

- Using **stochastic mini-batch** for training.
- Defining 2 hidden layers.
- Optimizing hyperparameters (loss function, learning rate, etc.) through trial and error.

### Data Preprocessing

- Reshape images from (32x32x3) to 3072-dimensional vectors.
- Normalize pixel values to the [0-1] range for faster calculations.

### Model Parameters

- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Loss Function**: Categorical Crossentropy
- **Learning Rate**: 0.01
- **Momentum**: 0.9

We evaluated the model's accuracy and error across various epochs on both training and validation datasets.

### Batch Size Experiment

We tested three different batch sizes (32, 64, 256) to evaluate their impact on model performance:

**Batch size = 32:**
<p float="left">
  <img src="images/4.png" width="300" />
  <img src="images/5.png" width="300" />
</p>

**Batch size = 64:**
<p float="left">
  <img src="images/6.png" width="300" />
  <img src="images/7.png" width="300" />
</p>

**Batch size = 256:**
<p float="left">
  <img src="images/8.png" width="300" />
  <img src="images/9.png" width="300" />
</p>

From the results, we found that the largest batch size (256) provided the best performance in terms of both speed and accuracy.

### Activation Function Comparison

We experimented with different activation functions for the hidden layers to observe their effect on network accuracy.

**Activation functions: eLU, eLU, Softmax**
<p float="left">
  <img src="images/10.png" width="300" />
  <img src="images/11.png" width="300" />
</p>
Confusion Matrix:
<img src="images/12.png" width="300"/>

**Activation functions: tanh, tanh, Softmax**
<p float="left">
  <img src="images/13.png" width="300" />
  <img src="images/14.png" width="300" />
</p>
Confusion Matrix:
<img src="images/15.png" width="300"/>

**Activation functions: sigmoid, sigmoid, Softmax**
<p float="left">
  <img src="images/16.png" width="300" />
  <img src="images/17.png" width="300" />
</p>
Confusion Matrix:
<img src="images/18.png" width="300"/>

As seen above, ReLU provides the best accuracy, while sigmoid underperforms significantly.

### Loss Function Experiment

We tested different loss functions to compare their effect on model performance:

**Loss Function: MeanSquaredError**
<p float="left">
  <img src="images/19.png" width="300" />
  <img src="images/20.png" width="300" />
</p>
Confusion Matrix:
<img src="images/21.png" width="300"/>

**Loss Function: Hinge Loss**
<p float="left">
  <img src="images/22.png" width="300" />
  <img src="images/23.png" width="300" />
</p>
Confusion Matrix:
<img src="images/24.png" width="300"/>

Categorical cross-entropy proved to be the most effective loss function.

### Optimizer Comparison

We compared the performance of different optimizers:

**Optimizer: Adam**
<p float="left">
  <img src="images/25.png" width="300" />
  <img src="images/26.png" width="300" />
</p>

**Optimizer: RMSprop**
<p float="left">
  <img src="images/27.png" width="300" />
  <img src="images/28.png" width="300" />
</p>

RMSprop performed worse compared to SGD in terms of accuracy.

### Best Model Summary

After various experiments, the best-performing model was configured as follows:

- **Batch size**: 256
- **Optimizer**: SGD
- **Loss Function**: Categorical Crossentropy
- **Activation Functions**: ReLU, ReLU, Softmax

<p float="left">
  <img src="images/29.png" width="300" />
  <img src="images/30.png" width="300" />
</p>
Confusion Matrix:
<img src="images/31.png" width="300"/>

Additionally, we report the F1 Score and Precision:

```
F1 Score = 0.5298
Precision Score = 0.5394
```


---

## Part 2: MLP + CNN Model

To enhance the classifier’s performance, we added convolutional layers to the best MLP model obtained in Part 1.

### Model with Convolutional Layers

We incorporated pooling and batch normalization layers to further improve performance:

<p float="left">
  <img src="images/32.png" width="300" />
  <img src="images/33.png" width="300" />
</p>
Confusion Matrix:
<img src="images/34.png" width="300"/>

### Dropout Layer

Adding dropout layers improved accuracy and reduced loss:

<p float="left">
  <img src="images/35.png" width="300" />
  <img src="images/36.png" width="300" />
</p>
Confusion Matrix:
<img src="images/37.png" width="300"/>

### Early Stopping

We implemented early stopping to prevent overfitting:

<p float="left">
  <img src="images/38.png" width="300" />
  <img src="images/39.png" width="300" />
</p>
Confusion Matrix:
<img src="images/40.png" width="300"/>

---

The final model achieved an accuracy of 0.70 on the test data and 0.99 on the training data.
