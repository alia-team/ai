use crate::activation::Activation;
use crate::data::*;
use crate::optimizer::Optimizer;
use crate::weights_init::WeightsInit;
use crate::{conv2d::Conv2D, dense::Dense, layer::LayerType, maxpool2d::MaxPool2D};
use core::panic;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array1, Array3};
use serde::{Deserialize, Serialize};
use std::default::Default;
use std::fs::File;
use std::time::{SystemTime, UNIX_EPOCH};

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

#[derive(Serialize, Deserialize)]
pub struct MLP {
    layers: Vec<Dense>,
    layer_order: Vec<String>,
    data: Dataset1D,
    minibatch_size: usize,
    creation_time: SystemTime,
    training_history: Vec<f64>,
    testing_history: Vec<f64>,
    time_history: Vec<usize>,
    optimizer: Optimizer,
    epochs: usize,
    input_size: usize,
}

impl MLP {
    pub fn new(data: Dataset1D, input_size: usize, params: Hyperparameters) -> Self {
        let creation_time: SystemTime = SystemTime::now();

        let mlp: MLP = MLP {
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
            input_size,
        };

        mlp
    }

    pub fn add_layer(
        &mut self,
        output_size: usize,
        activation: Box<dyn Activation>,
        dropout: Option<f64>,
        weights_init: WeightsInit,
    ) {
        // Find last layer's output size
        let input_size: usize = match self.layers.last() {
            Some(layer) => layer.output_size,
            None => self.input_size,
        };

        let layer: Dense = Dense::new(
            input_size,
            output_size,
            activation,
            self.optimizer,
            dropout,
            (input_size, 1, 1),
            weights_init,
        );
        self.layers.push(layer);
        self.layer_order.push(String::from("dense"));
    }

    pub fn forward(&mut self, input: Array1<f64>, training: bool) -> Array1<f64> {
        let mut output: Array1<f64> = input;
        for layer in &mut self.layers {
            output = layer.forward(output, training);
        }

        output
    }

    pub fn predict(&mut self, input: Array1<f64>) -> Array1<f64> {
        self.forward(input, false)
    }

    pub fn last_layer_error(&mut self, label: u8) -> Array1<f64> {
        let size: usize = self.layers.last().unwrap().output_size;
        let desired: Array1<f64> =
            Array1::<f64>::from_shape_fn(size, |i| (label == i as u8) as usize as f64);
        self.output() - desired
    }

    pub fn backward(&mut self, label: u8, training: bool) {
        let mut error: Array1<f64> = self.last_layer_error(label);
        for layer in self.layers.iter_mut().rev() {
            error = layer.backward(error, training);
        }
    }

    pub fn update(&mut self, minibatch_size: usize) {
        for layer in &mut self.layers {
            layer.update(minibatch_size);
        }
    }

    pub fn output(&self) -> Array1<f64> {
        self.layers.last().unwrap().output.clone()
    }

    pub fn get_accuracy(&self, label: u8) -> f64 {
        let mut max: f64 = f64::NEG_INFINITY;
        let mut max_idx: u8 = 0;
        let output: Array1<f64> = self.output();

        for (j, &value) in output.iter().enumerate() {
            if value > max {
                max = value;
                max_idx = j as u8;
            }
        }

        (max_idx == label) as u8 as f64
    }

    pub fn fit(&mut self) {
        for epoch in 0..self.epochs {
            let progress_bar: ProgressBar =
                ProgressBar::new((self.data.training_size / self.minibatch_size) as u64);
            progress_bar.set_style(
                ProgressStyle::default_bar()
                    .template(&format!(
                        "Epoch {}/{}: [{{bar}}] Batch {{pos}}/{{len}} - Accuracy: {{msg}}",
                        epoch + 1,
                        self.epochs
                    ))
                    .unwrap()
                    .progress_chars("#-"),
            );

            let mut avg_acc = 0.0;
            for i in 0..self.data.training_size {
                let (sample, label) = self.data.get_random_training_sample().unwrap();
                let label = *self.data.classes.get(&label).unwrap();
                self.forward(sample, true);
                self.backward(label, true);

                avg_acc += self.get_accuracy(label);

                if i % self.minibatch_size == self.minibatch_size - 1 {
                    self.update(self.minibatch_size);
                    progress_bar.inc(1);
                    progress_bar.set_message(format!("{:.1}%", avg_acc / (i + 1) as f64 * 100.0));
                }
            }

            avg_acc /= self.data.training_size as f64;
            if self.data.testing_size == 0 {
                progress_bar.finish_with_message(format!("{:.1}%", avg_acc * 100.0));
            } else {
                progress_bar.set_message(format!("{:.1}% - Testing...", avg_acc * 100.0));

                // Testing
                let mut avg_test_acc = 0.0;
                for _i in 0..self.data.testing_size {
                    let (image, label) = self.data.get_random_testing_sample().unwrap();
                    let label = *self.data.classes.get(&label).unwrap();
                    self.forward(image, false);

                    avg_test_acc += self.get_accuracy(label);
                }

                avg_test_acc /= self.data.testing_size as f64;
                progress_bar.finish_with_message(format!(
                    "{:.1}% - Test accuracy: {:.1}%",
                    avg_acc * 100.0,
                    avg_test_acc * 100.0
                ));
                self.testing_history.push(avg_test_acc);
            }

            self.training_history.push(avg_acc);
            let duration = SystemTime::now()
                .duration_since(self.creation_time)
                .unwrap();
            self.time_history.push(duration.as_secs() as usize);
        }
    }

    pub fn zero(&mut self) {
        for layer in &mut self.layers {
            layer.zero();
        }
    }

    pub fn save(&self, path: &str, model_name: &str) -> String {
        let timestamp: u128 = self
            .creation_time
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();

        let full_path: String = format!("{}{}_{}.json", path, model_name, timestamp);
        let model_file = std::fs::File::create(full_path.clone()).unwrap();
        serde_json::to_writer(model_file, &self).unwrap();

        full_path
    }

    pub fn load(model_file_name: &str) -> MLP {
        let model_file = File::open(model_file_name).unwrap();
        let model: MLP = serde_json::from_reader(model_file).unwrap();

        model
    }
}

#[derive(Serialize, Deserialize)]
pub struct CNN {
    layers: Vec<LayerType>,
    layer_order: Vec<String>,
    data: Dataset3D,
    minibatch_size: usize,
    creation_time: SystemTime,
    training_history: Vec<f64>,
    testing_history: Vec<f64>,
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
        dropout: Option<f64>,
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

    pub fn forward(&mut self, image: Array3<f64>, training: bool) -> Array1<f64> {
        let mut output: Array3<f64> = image;
        let mut flat_output: Array1<f64> = output.clone().into_shape(output.len()).unwrap();
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

    pub fn predict(&mut self, input: Array3<f64>) -> Array1<f64> {
        self.forward(input, false)
    }

    pub fn last_layer_error(&mut self, label: usize) -> Array1<f64> {
        let size: usize = match self.layers.last().unwrap() {
            LayerType::Dense(dense_layer) => dense_layer.output_size,
            _ => panic!("Last layer must be a dense layer."),
        };
        let desired: Array1<f64> =
            Array1::<f64>::from_shape_fn(size, |i| (label == i) as usize as f64);
        self.output() - desired
    }

    pub fn backward(&mut self, label: usize, training: bool) {
        let mut flat_error: Array1<f64> = self.last_layer_error(label);
        let mut error: Array3<f64> = flat_error
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

    pub fn output(&self) -> Array1<f64> {
        match self.layers.last().unwrap() {
            LayerType::Conv2D(_) => panic!("Last layer is a Conv2D"),
            LayerType::MaxPool2D(_) => panic!("Last layer is a MaxPool2D"),
            LayerType::Dense(dense_layer) => dense_layer.output.clone(),
        }
    }

    pub fn get_accuracy(&self, label: usize) -> f64 {
        let mut max: f64 = 0.0;
        let mut max_idx: usize = 0;
        let output: Array1<f64> = self.output();
        for j in 0..output.len() {
            if output[j] > max {
                max = output[j];
                max_idx = j;
            }
        }

        (max_idx == label) as u8 as f64
    }

    pub fn fit(&mut self) {
        for epoch in 0..self.epochs {
            let progress_bar: ProgressBar =
                ProgressBar::new((self.data.training_size / self.minibatch_size) as u64);
            progress_bar.set_style(
                ProgressStyle::default_bar()
                    .template(&format!(
                        "Epoch {}/{}: [{{bar}}] Batch {{pos}}/{{len}} - Accuracy: {{msg}}",
                        epoch + 1,
                        self.epochs
                    ))
                    .unwrap()
                    .progress_chars("#-"),
            );

            let mut avg_acc = 0.0;
            for i in 0..self.data.training_size {
                let (image, label) = self.data.get_random_sample().unwrap();
                let label = *self.data.classes.get(&label).unwrap();
                self.forward(image, true);
                self.backward(label, true);

                avg_acc += self.get_accuracy(label);

                if i % self.minibatch_size == self.minibatch_size - 1 {
                    self.update(self.minibatch_size);
                    progress_bar.inc(1);
                    progress_bar.set_message(format!("{:.1}%", avg_acc / (i + 1) as f64 * 100.0));
                }
            }

            avg_acc /= self.data.training_size as f64;
            progress_bar.set_message(format!("{:.1}% - Testing...", avg_acc * 100.0));

            // Testing
            let mut avg_test_acc = 0.0;
            for _i in 0..self.data.testing_size {
                let (image, label) = self.data.get_random_test_sample().unwrap();
                let label = *self.data.classes.get(&label).unwrap();
                self.forward(image, false);

                avg_test_acc += self.get_accuracy(label);
            }

            avg_test_acc /= self.data.testing_size as f64;
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

    pub fn save(&self, path: &str, model_name: &str) -> String {
        let timestamp: u128 = self
            .creation_time
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();

        let full_path: String = format!("{}{}_{}.json", path, model_name, timestamp);
        let model_file = std::fs::File::create(full_path.clone()).unwrap();
        serde_json::to_writer(model_file, &self).unwrap();

        full_path
    }

    pub fn load(model_path: &str) -> CNN {
        let model_file = File::open(model_path).unwrap();
        let model: CNN = serde_json::from_reader(model_file).unwrap();

        model
    }
}
