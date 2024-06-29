# CNN

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
  If the `kernel`` parameter is set, this one will be ignored;
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
