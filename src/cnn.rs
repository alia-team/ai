use crate::conv2d::Conv2D;
use crate::dense::Dense;
use crate::fit::sparse_categorical_crossentropy;
use crate::maxpool2d::MaxPool2D;
use ndarray::{Array1, Array3, ArrayBase, IxDyn, OwnedRepr};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};

#[derive(Serialize, Deserialize)]
pub struct CNNWeights {
    conv1: Conv2DWeights,
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
    pub dense1: Dense,
    pub dense2: Dense,
}

impl CNN {
    pub fn new(input_shape: (usize, usize, usize)) -> Self {
        let (height, width, channels) = input_shape;
        let conv_output_size = (height / 2) * (width / 2) * 64; // Assuming stride 1 and valid padding

        CNN {
            conv1: Conv2D::new(channels, 64, 3),
            dense1: Dense::new(conv_output_size, 128, |x| x.max(0.0)), // ReLU
            dense2: Dense::new(128, 10, |x| x), // Linear (softmax applied later)
        }
    }

    fn softmax(&self, x: &Array1<f32>) -> Array1<f32> {
        let max_val = x.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp = x.mapv(|a| (a - max_val).exp());
        let sum = exp.sum();
        exp.mapv(|a| a / (sum + 1e-10)) // Add small epsilon to prevent division by zero
    }

    fn log_softmax(&self, x: &Array1<f32>) -> Array1<f32> {
        let max_val = x.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp = x.mapv(|a| (a - max_val).exp());
        let log_sum = exp.sum().ln();
        x.mapv(|a| a - max_val - log_sum)
    }

    pub fn forward(&mut self, input: &Array3<f32>) -> Array1<f32> {
        let mut x = self.conv1.forward(input);
        x = x.mapv(|v| v.max(0.0)); // ReLU
        let pool = MaxPool2D::new(2);
        x = pool.forward(&x);

        let flat = x.clone().into_shape(x.len()).unwrap();
        let x = self.dense1.forward(&flat);
        let x = x.mapv(|v| v.max(0.0)); // ReLU
        let x = self.dense2.forward(&x);

        x
    }

    pub fn backward(
        &mut self,
        input: &Array3<f32>,
        target: usize,
    ) -> Result<(f32, Vec<ArrayBase<OwnedRepr<f32>, IxDyn>>), String> {
        // Forward pass (store intermediate results)
        let x_conv1 = self.conv1.forward(input);
        let x_relu1 = x_conv1.mapv(|v| v.max(0.0));
        let x_pool1 = MaxPool2D::new(2).forward(&x_relu1);
        let flat = x_pool1.clone().into_shape(x_pool1.len()).unwrap();
        let x_dense1 = self.dense1.forward(&flat);
        let x_relu2 = x_dense1.mapv(|v| v.max(0.0));
        let output = self.dense2.forward(&x_relu2);

        // Calculate loss. Softmax is applied in the following function
        let loss = sparse_categorical_crossentropy(target, &output, true);

        // Check for infinite or NaN loss
        if !loss.is_finite() {
            return Err(format!("Loss is not finite: {}", loss));
        }

        // Backward pass
        let mut grad = output;
        grad[target] -= 1.0;

        // Dense2 backward
        let (grad_dense2, grad_weights_dense2, grad_bias_dense2) =
            self.dense2.backward(&x_relu2, &grad);

        // Dense1 backward
        let (grad_dense1, grad_weights_dense1, grad_bias_dense1) =
            self.dense1.backward(&flat, &grad_dense2);

        // Reshape gradient back to 3D
        let mut grad = grad_dense1.into_shape(x_pool1.dim()).unwrap();

        // MaxPool backward
        grad = MaxPool2D::new(2).backward(&x_relu1, &grad);

        // Conv1 backward
        let (_grad, grad_kernel_conv1, grad_bias_conv1) = self.conv1.backward(input, &grad);

        // Collect all gradients
        let grads: Vec<ArrayBase<OwnedRepr<f32>, IxDyn>> = vec![
            grad_kernel_conv1.into_dyn(),
            grad_bias_conv1.into_dyn(),
            grad_weights_dense1.into_dyn(),
            grad_bias_dense1.into_dyn(),
            grad_weights_dense2.into_dyn(),
            grad_bias_dense2.into_dyn(),
        ];

        Ok((loss, grads))
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
