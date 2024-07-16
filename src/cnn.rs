use crate::batchnorm::BatchNorm;
use crate::conv2d::Conv2D;
use crate::dense::Dense;
use crate::fit::sparse_categorical_crossentropy;
use crate::maxpool2d::MaxPool2D;
use log::{debug, error, info};
use ndarray::{Array1, Array3, ArrayBase, ArrayViewMut1, IxDyn, OwnedRepr};
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
    pub bn1: BatchNorm,
    pub conv2: Conv2D,
    pub bn2: BatchNorm,
    pub conv3: Conv2D,
    pub bn3: BatchNorm,
    pub dense1: Dense,
    pub bn4: BatchNorm,
    pub dense2: Dense,
}

impl CNN {
    pub fn new(input_shape: (usize, usize, usize)) -> Self {
        let (height, width, channels) = input_shape;

        CNN {
            conv1: Conv2D::new(channels, 64, 3),
            bn1: BatchNorm::new(64),
            conv2: Conv2D::new(64, 128, 3),
            bn2: BatchNorm::new(128),
            conv3: Conv2D::new(128, 256, 3),
            bn3: BatchNorm::new(256),
            dense1: Dense::new(256 * (height / 8) * (width / 8), 8, |x| x.max(0.0)), // ReLU
            bn4: BatchNorm::new(8),
            dense2: Dense::new(8, 10, |x| x),
        }
    }

    pub fn forward(&mut self, input: &Array3<f32>) -> Array1<f32> {
        let mut x = self.conv1.forward(input);
        x = self.bn1.forward(&x, true);
        x = x.mapv(|v| v.max(0.0)); // ReLU
        let pool = MaxPool2D::new(2);
        x = pool.forward(&x);

        x = self.conv2.forward(&x);
        x = self.bn2.forward(&x, true);
        x = x.mapv(|v| v.max(0.0)); // ReLU
        x = pool.forward(&x);

        x = self.conv3.forward(&x);
        x = self.bn3.forward(&x, true);
        x = x.mapv(|v| v.max(0.0)); // ReLU
        x = pool.forward(&x);

        let flat = x.clone().into_shape(x.len()).unwrap();
        let x = self.dense1.forward(&flat);
        let x = self
            .bn4
            .forward(&x.clone().into_shape((1, 1, x.len())).unwrap(), true)
            .into_shape(x.len())
            .unwrap();
        let x = x.mapv(|v| v.max(0.0)); // ReLU
        let x = self.dense2.forward(&x);

        // Log pre-softmax output
        println!("Pre-softmax output: {:?}", x);

        // Apply softmax with improved numerical stability
        let max_val = x.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp = x.mapv(|a| (a - max_val).exp());
        let sum = exp.sum();

        // Add a small epsilon to avoid division by zero
        const EPSILON: f32 = 1e-10;
        let softmax_output = exp / (sum + EPSILON);

        // Log post-softmax output
        println!("Post-softmax output: {:?}", softmax_output);

        softmax_output
    }

    pub fn backward(
        &mut self,
        input: &Array3<f32>,
        target: usize,
    ) -> Result<(f32, Vec<ArrayBase<OwnedRepr<f32>, IxDyn>>), String> {
        info!("Starting backward pass");
        debug!("Input shape: {:?}", input.shape());

        // Forward pass (store intermediate results)
        let x_conv1 = self.conv1.forward(input);
        debug!("Conv1 output shape: {:?}", x_conv1.shape());

        let x_bn1 = self.bn1.forward(&x_conv1, true);
        let x_relu1 = x_bn1.mapv(|v| v.max(0.0));
        let x_pool1 = MaxPool2D::new(2).forward(&x_relu1);
        debug!("Pool1 output shape: {:?}", x_pool1.shape());

        let x_conv2 = self.conv2.forward(&x_pool1);
        debug!("Conv2 output shape: {:?}", x_conv2.shape());

        let x_bn2 = self.bn2.forward(&x_conv2, true);
        let x_relu2 = x_bn2.mapv(|v| v.max(0.0));
        let x_pool2 = MaxPool2D::new(2).forward(&x_relu2);
        debug!("Pool2 output shape: {:?}", x_pool2.shape());

        let x_conv3 = self.conv3.forward(&x_pool2);
        debug!("Conv3 output shape: {:?}", x_conv3.shape());

        let x_bn3 = self.bn3.forward(&x_conv3, true);
        let x_relu3 = x_bn3.mapv(|v| v.max(0.0));
        let x_pool3 = MaxPool2D::new(2).forward(&x_relu3);
        debug!("Pool3 output shape: {:?}", x_pool3.shape());

        let flat = x_pool3.clone().into_shape(x_pool3.len()).unwrap();
        debug!("Flattened shape: {:?}", flat.shape());

        let x_dense1 = self.dense1.forward(&flat);
        debug!("Dense1 output shape: {:?}", x_dense1.shape());

        let x_bn4 = self
            .bn4
            .forward(
                &x_dense1.clone().into_shape((1, 1, x_dense1.len())).unwrap(),
                true,
            )
            .into_shape(x_dense1.len())
            .unwrap();
        let x_relu4 = x_bn4.mapv(|v| v.max(0.0));
        let output = self.dense2.forward(&x_relu4);
        debug!("Final output shape: {:?}", output.shape());

        // Calculate loss
        let loss = sparse_categorical_crossentropy(target, &output);
        info!("Loss: {}", loss);

        // Backward pass
        let mut one_hot = Array1::<f32>::zeros(10);
        if target >= 10 {
            return Err(format!("Target {} is out of bounds for 10 classes", target));
        }
        one_hot[target] = 1.0;
        let grad = &output - &one_hot;

        // Dense2 backward
        let (grad_dense2, grad_weights_dense2, grad_bias_dense2) =
            self.dense2.backward(&x_relu4, &grad);
        debug!(
            "Dense2 gradient shapes: {:?}, {:?}, {:?}",
            grad_dense2.shape(),
            grad_weights_dense2.shape(),
            grad_bias_dense2.shape()
        );

        // BN4 + ReLU4 backward
        let grad_bn4 = grad_dense2 * x_relu4.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
        let grad_bn4 = self.bn4.backward(
            &x_dense1.clone().into_shape((1, 1, x_dense1.len())).unwrap(),
            &grad_bn4.clone().into_shape((1, 1, grad_bn4.len())).unwrap(),
        );
        let grad_bn4 = grad_bn4.into_shape(x_dense1.len()).unwrap();
        debug!("BN4 gradient shape: {:?}", grad_bn4.shape());

        // Dense1 backward
        let (grad_dense1, grad_weights_dense1, grad_bias_dense1) =
            self.dense1.backward(&flat, &grad_bn4);
        debug!(
            "Dense1 gradient shapes: {:?}, {:?}, {:?}",
            grad_dense1.shape(),
            grad_weights_dense1.shape(),
            grad_bias_dense1.shape()
        );

        // Reshape gradient back to 3D
        let mut grad = grad_dense1.into_shape(x_pool3.dim()).unwrap();
        debug!("Reshaped gradient shape: {:?}", grad.shape());

        // MaxPool3 backward
        grad = MaxPool2D::new(2).backward(&x_relu3, &grad);
        debug!("MaxPool3 gradient shape: {:?}", grad.shape());

        // BN3 + ReLU3 backward
        grad = grad * x_relu3.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
        grad = self.bn3.backward(&x_conv3, &grad);
        debug!("BN3 gradient shape: {:?}", grad.shape());

        // Conv3 backward
        let (mut grad, grad_kernel_conv3, grad_bias_conv3) = self.conv3.backward(&x_pool2, &grad);
        debug!(
            "Conv3 gradient shapes: {:?}, {:?}, {:?}",
            grad.shape(),
            grad_kernel_conv3.shape(),
            grad_bias_conv3.shape()
        );

        // MaxPool2 backward
        grad = MaxPool2D::new(2).backward(&x_relu2, &grad);
        debug!("MaxPool2 gradient shape: {:?}", grad.shape());

        // BN2 + ReLU2 backward
        grad = grad * x_relu2.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
        grad = self.bn2.backward(&x_conv2, &grad);
        debug!("BN2 gradient shape: {:?}", grad.shape());

        // Conv2 backward
        let (mut grad, grad_kernel_conv2, grad_bias_conv2) = self.conv2.backward(&x_pool1, &grad);
        debug!(
            "Conv2 gradient shapes: {:?}, {:?}, {:?}",
            grad.shape(),
            grad_kernel_conv2.shape(),
            grad_bias_conv2.shape()
        );

        // MaxPool1 backward
        grad = MaxPool2D::new(2).backward(&x_relu1, &grad);
        debug!("MaxPool1 gradient shape: {:?}", grad.shape());

        // BN1 + ReLU1 backward
        grad = grad * x_relu1.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
        grad = self.bn1.backward(&x_conv1, &grad);
        debug!("BN1 gradient shape: {:?}", grad.shape());

        // Conv1 backward
        let (_grad, grad_kernel_conv1, grad_bias_conv1) = self.conv1.backward(input, &grad);
        debug!(
            "Conv1 gradient shapes: {:?}, {:?}",
            grad_kernel_conv1.shape(),
            grad_bias_conv1.shape()
        );

        // Collect all gradients
        let grads: Vec<ArrayBase<OwnedRepr<f32>, IxDyn>> = vec![
            grad_kernel_conv1.into_dyn(),
            grad_bias_conv1.into_dyn(),
            self.bn1.gamma.clone().into_dyn(),
            self.bn1.beta.clone().into_dyn(),
            grad_kernel_conv2.into_dyn(),
            grad_bias_conv2.into_dyn(),
            self.bn2.gamma.clone().into_dyn(),
            self.bn2.beta.clone().into_dyn(),
            grad_kernel_conv3.into_dyn(),
            grad_bias_conv3.into_dyn(),
            self.bn3.gamma.clone().into_dyn(),
            self.bn3.beta.clone().into_dyn(),
            grad_weights_dense1.into_dyn(),
            grad_bias_dense1.into_dyn(),
            self.bn4.gamma.clone().into_dyn(),
            self.bn4.beta.clone().into_dyn(),
            grad_weights_dense2.into_dyn(),
            grad_bias_dense2.into_dyn(),
        ];

        info!("Number of gradient arrays: {}", grads.len());
        for (i, grad) in grads.iter().enumerate() {
            debug!("Gradient {} shape: {:?}", i, grad.shape());
        }

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
            bn1: BatchNorm::new(64),
            conv2: Conv2D::from_weights(
                weights.conv2.kernel,
                weights.conv2.kernel_shape,
                weights.conv2.bias,
            ),
            bn2: BatchNorm::new(128),
            conv3: Conv2D::from_weights(
                weights.conv3.kernel,
                weights.conv3.kernel_shape,
                weights.conv3.bias,
            ),
            bn3: BatchNorm::new(256),
            dense1: Dense::from_weights(
                weights.dense1.weights,
                weights.dense1.weights_shape,
                weights.dense1.bias,
                |x| x.max(0.0), // ReLU activation
            ),
            bn4: BatchNorm::new(8),
            dense2: Dense::from_weights(
                weights.dense2.weights,
                weights.dense2.weights_shape,
                weights.dense2.bias,
                |x| x, // Linear activation
            ),
        })
    }
}
