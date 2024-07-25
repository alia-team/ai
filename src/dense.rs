use crate::activation::Activation;
use crate::optimizer::{Optimizer, Optimizer2D};
use crate::util::outer;
use crate::weights_init::init_biases;
use crate::weights_init::{init_dense_weights, WeightsInit};
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Dense {
    input_size: usize,
    pub output_size: usize,
    input: Array1<f64>,
    pub output: Array1<f64>,
    biases: Array1<f64>,
    weights: Array2<f64>,
    bias_changes: Array1<f64>,
    weight_changes: Array2<f64>,
    activation: Box<dyn Activation>,
    pub transition_shape: (usize, usize, usize),
    optimizer: Optimizer2D,
    dropout: Option<f64>,
    dropout_mask: Array1<f64>,
}

impl Dense {
    pub fn zero(&mut self) {
        self.bias_changes = Array1::<f64>::zeros(self.output_size);
        self.weight_changes = Array2::<f64>::zeros((self.output_size, self.input_size));
        self.output = Array1::<f64>::zeros(self.output_size);
    }

    /// Create a new fully connected layer with the given parameters
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: Box<dyn Activation>,
        optimizer_alg: Optimizer,
        dropout: Option<f64>,
        transition_shape: (usize, usize, usize),
        weights_init: WeightsInit,
    ) -> Dense {
        let weights = init_dense_weights(weights_init, input_size, output_size);
        let biases = init_biases(output_size);

        let optimizer = Optimizer2D::new(optimizer_alg, input_size, output_size);

        let layer: Dense = Dense {
            input_size,
            output_size,
            input: Array1::<f64>::zeros(input_size),
            output: Array1::<f64>::zeros(output_size),
            biases,
            weights,
            bias_changes: Array1::<f64>::zeros(output_size),
            weight_changes: Array2::<f64>::zeros((output_size, input_size)),
            activation,
            transition_shape,
            optimizer,
            dropout,
            dropout_mask: Array1::<f64>::zeros(output_size),
        };

        layer
    }

    pub fn forward(&mut self, input: Array1<f64>, training: bool) -> Array1<f64> {
        if training && self.dropout.is_some() {
            let dropout = self.dropout.unwrap();
            let mut rng = rand::thread_rng();
            self.dropout_mask =
                Array1::<f64>::from_shape_fn((self.output_size,), |_| rng.gen::<f64>());
            self.dropout_mask = self
                .dropout_mask
                .mapv(|x| if x < dropout { 0.0 } else { 1.0 });
            let logits: Array1<f64> = self.weights.dot(&input) + &self.biases;
            self.output = self.activation.forward(logits);
            self.output *= &self.dropout_mask;
            self.input = input;
            self.output.clone()
        } else {
            let logits: Array1<f64> = self.weights.dot(&input) + &self.biases;
            self.output = self.activation.forward(logits);
            self.input = input;
            self.output.clone()
        }
    }

    pub fn backward(&mut self, error: Array1<f64>, training: bool) -> Array1<f64> {
        let mut error = error;
        if self.dropout.is_some() && training {
            error *= &self.dropout_mask;
        }
        error *= &self.activation.backward(self.output.clone());
        let prev_error = self.weights.t().dot(&error);
        self.weight_changes -= &(outer(error.clone(), self.input.clone()));
        self.bias_changes -= &error;

        prev_error
    }

    pub fn update(&mut self, minibatch_size: usize) {
        self.weight_changes /= minibatch_size as f64;
        self.bias_changes /= minibatch_size as f64;
        self.weights += &self.optimizer.weight_changes(&self.weight_changes);
        self.biases += &self.optimizer.bias_changes(&self.bias_changes);
        self.weight_changes = Array2::<f64>::zeros((self.output_size, self.input_size));
        self.bias_changes = Array1::<f64>::zeros(self.output_size);
    }
}
