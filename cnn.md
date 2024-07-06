# CNN

## Structure proposal

This is just a proposal.
Maybe a slightly different structure will work better.

### Traits & Classes

#### Model (class)

Fields:

- `layers`: `Array1<Layer>`

Methods:

- `new(layers)`
- `fit(training_dataset, valid_dataset, targets, learning_rate, batch_size, epochs, loss, optimizer)`
- `predict(input)`

#### Layer (trait)

Layer trait will have inheritors classes for each type of layer (Dense,
convolutional, ...).

During the backward pass, we need to compute the gradients of the layer's
weights twice: one including the bias in order to update the current layer's
weights, and one excluding the bias to be propagate back to the previous 
layer.
This last step is necessary because in our implementation, we include the bias
in weights for making the computations & structure simpler.
With this technique, the inputs passed to each layer doesn't include the bias
(which is simply an input equal to one with a weight).
So when we pass gradients to a previous layer, they must match the input
dimensions (so without bias in our case).

Fields:

- `weights`
- `output`
- `gradients`
- `activation`

Methods:

- `forward(inputs)`
- `backward(gradients)`

#### Loss (trait)

Find a way to make it generic for ALL losses (MSE, binary cross-entropy,
categorical cross-entropy, sparse categorical cross-entropy, ...)

Methods:

- `compute_loss(predictions, targets)`
- `compute_gradients(predictions, targets)`

#### Optimizer (trait)

Find a way to make it generic for ALL optimizers (Adam, SGD, ...).

Methods:

- `update_weights(weights, gradients)`

#### Activation (trait)

Methods:

- `forward(inputs)`
- `backward(gradients)`: gradient of the activation function for the backprop.

## Layers

### 2D convolutional layer

Applies convolution operations to the input data using a set of learnable
kernels (filters).
Each kernel slides over the input data, performing element-wise multiplication
and summing the results to produce feature maps.

Parameter:

- `input`: `&[Vec<f64>]`: input;
- `filters`: `usize`: number of filters;
- `kernel`: `Option<&[Vec<f64>]>`: kernel defined by the user.
  If this parameter is set, the `kernel_size` parameter will be ignored, the
  weights will not be trainable, and all input channels will have this same
  kernel;
- `kernel_size`: `Option<(usize, usize)>`: shape of the convolution window.
  If the `kernel` parameter is set, this one will be ignored;
- `strides`: `(usize, usize)`: stride length of the convolution for each dimension;
- `padding`: `enum::{Same, Valid}`: `Valid` means no padding, and `Same` means
  padding evenly to the left/right or up/down;
- `activation`: `Option<enum::{ReLU, Logistic, ...}>`: activation function to use.
  If `None`, identity activation function is used.

Output: `Vec<Vec<f64>>`

### 2D max pooling layer

Reduces the spatial dimensions of the input by taking the maximum value over a
defined window.

Parameters:

- `input`: `&[Vec<f64>]`: input;
- `pool_size`: `(usize, usize)`: shape of the pooling window.
- `strides`: `(usize, usize)`: stride length of the pooling for each dimension;
- `padding`: `enum::{Same, Valid}`: `Valid` means no padding, and `Same` means
  padding evenly to the left/right or up/down;

Output: `Vec<Vec<f64>>`

### Flatten layer

Reshapes the input data into a one-dimensional array, preparing it for input
into a dense (fully connected) layer.

Parameters:

- `input`: `&[f64]`: input;

Output: `Vec<f64>`

### Dense layer

Applies a linear transformation to the input data followed by an optional
activation function.

Parameters:

- `input`: `&[f64]`: input;
- `units`: `usize`: number of neurons;
- `activation`: `Option<enum::{ReLU, Logistic, ...}>`: activation function to use.
  If `None`, identity activation function is used.

Output: `Vec<f64>`

## Fitting

### History struct

Fields:

- `epochs`: `Vec<usize>`;
- `train_loss`: `Vec<f64>`;
- `train_accuracy`: `Vec<f64>`;
- `val_loss`: `Vec<Option<f64>>`;
- `val_accuracy`: `Vec<Option<f64>>`;

### Fit function

Parameters:

- `optimizer`: `enum::{Adam, ...}`;
- `loss`: `enum::{MSE, CategoricalCrossEntropy, ...}`;
- `training_dataset`: `&[Vec<f64>]`;
- `validation_dataset`: `Option<&[Vec<f64>]>`;
- epochs: `usize`

Output: `History`

## Full process

1. Forward propagation: Feed the network with an input and get predictions;
2. Loss computation: Compare the predictions with the targets using a loss
   function (MSE, crossentropy, ...);
3. Backpropagation: Compute the gradients of the loss function for each layer
   with their weights and outputs;
4. Update weights: Adjust the weights in the opposite direction of the gradients
   in order to minimize the loss (gradient descent), using an optimizer (Adam,
   SGD, ...).
