use super::activation::Activation;
use super::data::*;
use super::optimizer::Optimizer;
use super::weights_init::WeightsInit;
use super::{conv2d::Conv2D, dense::Dense, layer::LayerType, maxpool2d::MaxPool2D};
use core::panic;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array1, Array3};
use std::default::Default;
use std::time::SystemTime;

pub struct Hyperparameters {
    pub batch_size: usize,
    pub epochs: usize,
    pub optimizer: Optimizer,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Hyperparameters {
            batch_size: 32,
            epochs: 10,
            optimizer: Optimizer::Adam(0.9, 0.999, 1e-8),
        }
    }
}

pub struct CNN {
    layers: Vec<LayerType>,
    layer_order: Vec<String>,
    data: Dataset3D,
    minibatch_size: usize,
    creation_time: SystemTime,
    training_history: Vec<f32>,
    testing_history: Vec<f32>,
    time_history: Vec<usize>,
    optimizer: Optimizer,
    epochs: usize,
    input_shape: (usize, usize, usize),
}

impl CNN {
    pub fn new(data: Dataset3D, params: Hyperparameters) -> CNN {
        let creation_time: SystemTime = SystemTime::now();

        let cnn: CNN = CNN {
            layers: vec![],
            layer_order: vec![],
            data,
            minibatch_size: params.batch_size,
            creation_time,
            training_history: vec![],
            testing_history: vec![],
            time_history: vec![],
            optimizer: params.optimizer,
            epochs: params.epochs,
            input_shape: (0, 0, 0),
        };

        cnn
    }

    pub fn set_input_shape(&mut self, input_shape: Vec<usize>) {
        let mut iter = input_shape.into_iter();
        self.input_shape = (
            iter.next().unwrap(),
            iter.next().unwrap_or(1),
            iter.next().unwrap_or(1),
        )
    }

    pub fn add_conv2d_layer(&mut self, num_filters: usize, kernel_size: usize) {
        if self.input_shape.0 == 0 {
            panic!("Input shape not set, use cnn.set_input_shape()");
        }
        let input_size: (usize, usize, usize) = match self.layers.last() {
            Some(LayerType::Conv2D(conv_layer)) => conv_layer.output_size,
            Some(LayerType::MaxPool2D(mxpl_layer)) => mxpl_layer.output_size,
            Some(LayerType::Dense(_)) => panic!("Convolutional Layer cannot follow a Dense Layer"),
            None => self.input_shape,
        };
        let conv_layer: Conv2D =
            Conv2D::new(input_size, kernel_size, 1, num_filters, self.optimizer);
        self.layers.push(LayerType::Conv2D(conv_layer));
        self.layer_order.push(String::from("conv"));
    }

    pub fn add_maxpool2d_layer(&mut self, kernel_size: usize) {
        if self.input_shape.0 == 0 {
            panic!("Input shape not set, use cnn.set_input_shape()");
        }
        let input_size: (usize, usize, usize) = match self.layers.last() {
            Some(LayerType::Conv2D(conv_layer)) => conv_layer.output_size,
            Some(LayerType::MaxPool2D(mxpl_layer)) => mxpl_layer.output_size,
            Some(LayerType::Dense(_)) => panic!("Max Pooling Layer cannot follow a Dense Layer"),
            None => self.input_shape,
        };
        let mxpl_layer: MaxPool2D = MaxPool2D::new(input_size, kernel_size, 2);
        self.layers.push(LayerType::MaxPool2D(mxpl_layer));
        self.layer_order.push(String::from("mxpl"));
    }

    pub fn add_dense_layer(
        &mut self,
        output_size: usize,
        activation: Box<dyn Activation>,
        dropout: Option<f32>,
        weights_init: WeightsInit,
    ) {
        if self.input_shape.0 == 0 {
            panic!("Input shape not set, use cnn.set_input_shape()");
        }
        // Find last layer's output size
        let transition_shape: (usize, usize, usize) = match self.layers.last() {
            Some(LayerType::Conv2D(conv_layer)) => conv_layer.output_size,
            Some(LayerType::MaxPool2D(mxpl_layer)) => mxpl_layer.output_size,
            Some(LayerType::Dense(dense_layer)) => (dense_layer.output_size, 1, 1),
            None => self.input_shape,
        };
        let input_size: usize = transition_shape.0 * transition_shape.1 * transition_shape.2;
        let dense_layer: Dense = Dense::new(
            input_size,
            output_size,
            activation,
            self.optimizer,
            dropout,
            transition_shape,
            weights_init,
        );
        self.layers.push(LayerType::Dense(dense_layer));
        self.layer_order.push(String::from("dense"));
    }

    pub fn forward(&mut self, image: Array3<f32>, training: bool) -> Array1<f32> {
        let mut output: Array3<f32> = image;
        let mut flat_output: Array1<f32> = output.clone().into_shape(output.len()).unwrap();
        for layer in &mut self.layers {
            match layer {
                LayerType::Conv2D(conv_layer) => {
                    output = conv_layer.forward(output);
                    flat_output = output.clone().into_shape(output.len()).unwrap();
                }
                LayerType::MaxPool2D(mxpl_layer) => {
                    output = mxpl_layer.forward(output);
                    flat_output = output.clone().into_shape(output.len()).unwrap();
                }
                LayerType::Dense(dense_layer) => {
                    flat_output = dense_layer.forward(flat_output, training);
                }
            }
        }

        flat_output
    }

    pub fn last_layer_error(&mut self, label: usize) -> Array1<f32> {
        let size: usize = match self.layers.last().unwrap() {
            LayerType::Dense(dense_layer) => dense_layer.output_size,
            _ => panic!("Last layer must be a dense layer."),
        };
        let desired: Array1<f32> =
            Array1::<f32>::from_shape_fn(size, |i| (label == i) as usize as f32);
        self.output() - desired
    }

    pub fn backward(&mut self, label: usize, training: bool) {
        let mut flat_error: Array1<f32> = self.last_layer_error(label);
        let mut error: Array3<f32> = flat_error
            .clone()
            .into_shape((1, 1, flat_error.len()))
            .unwrap();
        for layer in self.layers.iter_mut().rev() {
            match layer {
                LayerType::Conv2D(conv_layer) => {
                    error = conv_layer.backward(error);
                }
                LayerType::MaxPool2D(mxpl_layer) => {
                    error = mxpl_layer.backward(error);
                }
                LayerType::Dense(dense_layer) => {
                    flat_error = dense_layer.backward(flat_error, training);
                    error = flat_error
                        .clone()
                        .into_shape(dense_layer.transition_shape)
                        .unwrap();
                }
            }
        }
    }

    pub fn update(&mut self, minibatch_size: usize) {
        for layer in &mut self.layers {
            match layer {
                LayerType::Conv2D(conv_layer) => conv_layer.update(minibatch_size),
                LayerType::MaxPool2D(_) => {}
                LayerType::Dense(dense_layer) => dense_layer.update(minibatch_size),
            }
        }
    }

    pub fn output(&self) -> Array1<f32> {
        match self.layers.last().unwrap() {
            LayerType::Conv2D(_) => panic!("Last layer is a Conv2D"),
            LayerType::MaxPool2D(_) => panic!("Last layer is a MaxPool2D"),
            LayerType::Dense(dense_layer) => dense_layer.output.clone(),
        }
    }

    pub fn get_accuracy(&self, label: usize) -> f32 {
        let mut max: f32 = 0.0;
        let mut max_idx: usize = 0;
        let output: Array1<f32> = self.output();
        for j in 0..output.len() {
            if output[j] > max {
                max = output[j];
                max_idx = j;
            }
        }

        (max_idx == label) as usize as f32
    }

    pub fn fit(&mut self) {
        for epoch in 0..self.epochs {
            let progress_bar: ProgressBar =
                ProgressBar::new((self.data.trn_size / self.minibatch_size) as u64);
            progress_bar.set_style(
                ProgressStyle::default_bar()
                    .template(&format!(
                        "Epoch {}: [{{bar}}] {{pos}}/{{len}} - Accuracy: {{msg}}",
                        epoch + 1
                    ))
                    .unwrap()
                    .progress_chars("#-"),
            );

            let mut avg_acc = 0.0;
            for i in 0..self.data.trn_size {
                let (image, label) = self.data.get_random_sample().unwrap();
                let label = *self.data.classes.get(&label).unwrap();
                self.forward(image, true);
                self.backward(label, true);

                avg_acc += self.get_accuracy(label);

                if i % self.minibatch_size == self.minibatch_size - 1 {
                    self.update(self.minibatch_size);
                    progress_bar.inc(1);
                    progress_bar.set_message(format!("{:.1}%", avg_acc / (i + 1) as f32 * 100.0));
                }
            }

            avg_acc /= self.data.trn_size as f32;
            progress_bar.set_message(format!("{:.1}% - Testing...", avg_acc * 100.0));

            // Testing
            let mut avg_test_acc = 0.0;
            for _i in 0..self.data.tst_size {
                let (image, label) = self.data.get_random_test_sample().unwrap();
                let label = *self.data.classes.get(&label).unwrap();
                self.forward(image, false);

                avg_test_acc += self.get_accuracy(label);
            }

            avg_test_acc /= self.data.tst_size as f32;
            progress_bar.finish_with_message(format!(
                "{:.1}% - Test accuracy: {:.1}%",
                avg_acc * 100.0,
                avg_test_acc * 100.0
            ));

            self.training_history.push(avg_acc);
            self.testing_history.push(avg_test_acc);
            let duration = SystemTime::now()
                .duration_since(self.creation_time)
                .unwrap();
            self.time_history.push(duration.as_secs() as usize);
        }
    }

    pub fn zero(&mut self) {
        for layer in &mut self.layers {
            match layer {
                LayerType::Conv2D(conv_layer) => conv_layer.zero(),
                LayerType::MaxPool2D(mxpl_layer) => mxpl_layer.zero(),
                LayerType::Dense(dense_layer) => dense_layer.zero(),
            }
        }
    }
}
