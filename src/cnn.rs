use crate::conv2d::Conv2D;
use crate::dense::Dense;
use crate::fit::sparse_categorical_crossentropy;
use crate::maxpool2d::MaxPool2D;
use ndarray::{Array1, Array3};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};

#[derive(Serialize, Deserialize)]
pub struct CNNWeights {
    conv1: Conv2DWeights,
    conv2: Conv2DWeights,
    conv3: Conv2DWeights,
    dense1: DenseWeights,
    dense2: DenseWeights,
}

#[derive(Serialize, Deserialize)]
struct Conv2DWeights {
    kernel: Vec<f32>,
    kernel_shape: Vec<usize>,
    bias: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct DenseWeights {
    weights: Vec<f32>,
    weights_shape: Vec<usize>,
    bias: Vec<f32>,
}

pub struct CNN {
    pub conv1: Conv2D,
    pub conv2: Conv2D,
    pub conv3: Conv2D,
    pub dense1: Dense,
    pub dense2: Dense,
}

impl CNN {
    pub fn new(input_shape: (usize, usize, usize)) -> Self {
        let (height, width, channels) = input_shape;

        CNN {
            conv1: Conv2D::new(channels, 64, 3),
            conv2: Conv2D::new(64, 128, 3),
            conv3: Conv2D::new(128, 256, 3),
            dense1: Dense::new(256 * (height / 8) * (width / 8), 8, |x| x.max(0.0)), // ReLU
            dense2: Dense::new(8, 4, |x| x), // Linear activation, we'll apply softmax later
        }
    }

    pub fn forward(&self, input: &Array3<f32>) -> Array1<f32> {
        let mut x = self.conv1.forward(input);
        let pool = MaxPool2D::new(2);
        x = pool.forward(&x);
        x = self.conv2.forward(&x);
        x = pool.forward(&x);
        x = self.conv3.forward(&x);
        x = pool.forward(&x);

        let flat = x.clone().into_shape(x.len()).unwrap();
        let x = self.dense1.forward(&flat);
        let x = self.dense2.forward(&x);

        // Apply softmax
        let exp = x.mapv(|a| a.exp());
        let sum = exp.sum() + 1e-10;
        exp / sum
    }

    pub fn backward(&mut self, input: &Array3<f32>, target: usize) -> (f32, Vec<Array1<f32>>) {
        // Forward pass
        let mut x = self.conv1.forward(input);
        let x_conv1 = x.clone();
        x = MaxPool2D::new(2).forward(&x);
        let x_pool1 = x.clone();

        x = self.conv2.forward(&x);
        let x_conv2 = x.clone();
        x = MaxPool2D::new(2).forward(&x);
        let x_pool2 = x.clone();

        x = self.conv3.forward(&x);
        let x_conv3 = x.clone();
        x = MaxPool2D::new(2).forward(&x);
        let x_pool3 = x.clone();

        let flat = x.clone().into_shape(x.len()).unwrap();
        let x_dense1 = self.dense1.forward(&flat);
        let output = self.dense2.forward(&x_dense1);

        // Calculate loss
        let mut y_true = Array1::<f32>::zeros(4);
        if target < 4 {
            y_true[target] = 1.0;
        } else {
            y_true[0] = 1.0; // Set a default target
        }
        let loss = sparse_categorical_crossentropy(target.min(3), &output);

        // Backward pass
        let mut grad = output - y_true;
        grad = self.dense2.backward(&x_dense1, &grad);
        grad = self.dense1.backward(&flat, &grad);

        let mut grad = grad.into_shape(x_pool3.dim()).unwrap();
        grad = MaxPool2D::new(2).backward(&x_conv3, &grad);
        grad = self.conv3.backward(&x_pool2, &grad);

        grad = MaxPool2D::new(2).backward(&x_conv2, &grad);
        grad = self.conv2.backward(&x_pool1, &grad);

        grad = MaxPool2D::new(2).backward(&x_conv1, &grad);
        self.conv1.backward(input, &grad);

        let grads = vec![
            self.conv1
                .kernel
                .clone()
                .into_shape(self.conv1.kernel.len())
                .unwrap(),
            self.conv1.bias.clone(),
            self.conv2
                .kernel
                .clone()
                .into_shape(self.conv2.kernel.len())
                .unwrap(),
            self.conv2.bias.clone(),
            self.conv3
                .kernel
                .clone()
                .into_shape(self.conv3.kernel.len())
                .unwrap(),
            self.conv3.bias.clone(),
            self.dense1
                .weights
                .clone()
                .into_shape(self.dense1.weights.len())
                .unwrap(),
            self.dense1.bias.clone(),
            self.dense2
                .weights
                .clone()
                .into_shape(self.dense2.weights.len())
                .unwrap(),
            self.dense2.bias.clone(),
        ];

        (loss, grads)
    }

    pub fn save_weights(&self, filename: &str) -> std::io::Result<()> {
        let weights = CNNWeights {
            conv1: Conv2DWeights {
                kernel: self
                    .conv1
                    .kernel
                    .mapv(|x| if x.is_finite() { x } else { 0.0 })
                    .as_slice()
                    .unwrap()
                    .to_vec(),
                kernel_shape: self.conv1.kernel.shape().to_vec(),
                bias: self
                    .conv1
                    .bias
                    .mapv(|x| if x.is_finite() { x } else { 0.0 })
                    .to_vec(),
            },
            conv2: Conv2DWeights {
                kernel: self
                    .conv2
                    .kernel
                    .mapv(|x| if x.is_finite() { x } else { 0.0 })
                    .as_slice()
                    .unwrap()
                    .to_vec(),
                kernel_shape: self.conv2.kernel.shape().to_vec(),
                bias: self
                    .conv2
                    .bias
                    .mapv(|x| if x.is_finite() { x } else { 0.0 })
                    .to_vec(),
            },
            conv3: Conv2DWeights {
                kernel: self
                    .conv3
                    .kernel
                    .mapv(|x| if x.is_finite() { x } else { 0.0 })
                    .as_slice()
                    .unwrap()
                    .to_vec(),
                kernel_shape: self.conv3.kernel.shape().to_vec(),
                bias: self
                    .conv3
                    .bias
                    .mapv(|x| if x.is_finite() { x } else { 0.0 })
                    .to_vec(),
            },
            dense1: DenseWeights {
                weights: self
                    .dense1
                    .weights
                    .mapv(|x| if x.is_finite() { x } else { 0.0 })
                    .as_slice()
                    .unwrap()
                    .to_vec(),
                weights_shape: self.dense1.weights.shape().to_vec(),
                bias: self
                    .dense1
                    .bias
                    .mapv(|x| if x.is_finite() { x } else { 0.0 })
                    .to_vec(),
            },
            dense2: DenseWeights {
                weights: self
                    .dense2
                    .weights
                    .mapv(|x| if x.is_finite() { x } else { 0.0 })
                    .as_slice()
                    .unwrap()
                    .to_vec(),
                weights_shape: self.dense2.weights.shape().to_vec(),
                bias: self
                    .dense2
                    .bias
                    .mapv(|x| if x.is_finite() { x } else { 0.0 })
                    .to_vec(),
            },
        };

        let serialized = serde_json::to_string(&weights)?;
        let mut file = File::create(filename)?;
        file.write_all(serialized.as_bytes())?;
        Ok(())
    }

    pub fn load_weights(filename: &str) -> std::io::Result<Self> {
        let mut file = File::open(filename)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        let weights: CNNWeights = serde_json::from_str(&contents)?;

        Ok(CNN {
            conv1: Conv2D::from_weights(
                weights.conv1.kernel,
                weights.conv1.kernel_shape,
                weights.conv1.bias,
            ),
            conv2: Conv2D::from_weights(
                weights.conv2.kernel,
                weights.conv2.kernel_shape,
                weights.conv2.bias,
            ),
            conv3: Conv2D::from_weights(
                weights.conv3.kernel,
                weights.conv3.kernel_shape,
                weights.conv3.bias,
            ),
            dense1: Dense::from_weights(
                weights.dense1.weights,
                weights.dense1.weights_shape,
                weights.dense1.bias,
                |x| x.max(0.0), // ReLU activation
            ),
            dense2: Dense::from_weights(
                weights.dense2.weights,
                weights.dense2.weights_shape,
                weights.dense2.bias,
                |x| x, // Linear activation
            ),
        })
    }
}
