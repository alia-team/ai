use crate::cnn::activation::Activation;
use crate::cnn::optimizer::Optimizer;
use crate::cnn::util::*;
use crate::cnn::{conv::ConvLayer, dense::DenseLayer, layer::Layer, mxpl::MxplLayer};
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
    layers: Vec<Layer>,
    layer_order: Vec<String>,
    data: TrainingData,
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
    pub fn new(data: TrainingData, params: Hyperparameters) -> CNN {
        let creation_time = std::time::SystemTime::now();

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

    pub fn add_conv_layer(&mut self, num_filters: usize, kernel_size: usize) {
        if self.input_shape.0 == 0 {
            panic!("Input shape not set, use cnn.set_input_shape()");
        }
        let input_size: (usize, usize, usize) = match self.layers.last() {
            Some(Layer::Conv(conv_layer)) => conv_layer.output_size,
            Some(Layer::Mxpl(mxpl_layer)) => mxpl_layer.output_size,
            Some(Layer::Dense(_)) => panic!("Convolutional Layer cannot follow a Dense Layer"),
            None => self.input_shape,
        };
        let conv_layer: ConvLayer =
            ConvLayer::new(input_size, kernel_size, 1, num_filters, self.optimizer);
        self.layers.push(Layer::Conv(conv_layer));
        self.layer_order.push(String::from("conv"));
    }

    pub fn add_mxpl_layer(&mut self, kernel_size: usize) {
        if self.input_shape.0 == 0 {
            panic!("Input shape not set, use cnn.set_input_shape()");
        }
        let input_size: (usize, usize, usize) = match self.layers.last() {
            Some(Layer::Conv(conv_layer)) => conv_layer.output_size,
            Some(Layer::Mxpl(mxpl_layer)) => mxpl_layer.output_size,
            Some(Layer::Dense(_)) => panic!("Max Pooling Layer cannot follow a Dense Layer"),
            None => self.input_shape,
        };
        let mxpl_layer: MxplLayer = MxplLayer::new(input_size, kernel_size, 2);
        self.layers.push(Layer::Mxpl(mxpl_layer));
        self.layer_order.push(String::from("mxpl"));
    }

    pub fn add_dense_layer(
        &mut self,
        output_size: usize,
        activation: Box<dyn Activation>,
        dropout: Option<f32>,
    ) {
        if self.input_shape.0 == 0 {
            panic!("Input shape not set, use cnn.set_input_shape()");
        }
        // Find last layer's output size
        let transition_shape: (usize, usize, usize) = match self.layers.last() {
            Some(Layer::Conv(conv_layer)) => conv_layer.output_size,
            Some(Layer::Mxpl(mxpl_layer)) => mxpl_layer.output_size,
            Some(Layer::Dense(dense_layer)) => (dense_layer.output_size, 1, 1),
            None => self.input_shape,
        };
        let input_size = transition_shape.0 * transition_shape.1 * transition_shape.2;
        let fcl_layer: DenseLayer = DenseLayer::new(
            input_size,
            output_size,
            activation,
            self.optimizer,
            dropout,
            transition_shape,
        );
        self.layers.push(Layer::Dense(fcl_layer));
        self.layer_order.push(String::from("dense"));
    }

    pub fn forward(&mut self, image: Array3<f32>, training: bool) -> Array1<f32> {
        let mut output: Array3<f32> = image;
        let mut flat_output: Array1<f32> = output.clone().into_shape(output.len()).unwrap();
        for layer in &mut self.layers {
            match layer {
                Layer::Conv(conv_layer) => {
                    output = conv_layer.forward(output);
                    flat_output = output.clone().into_shape(output.len()).unwrap();
                }
                Layer::Mxpl(mxpl_layer) => {
                    output = mxpl_layer.forward(output);
                    flat_output = output.clone().into_shape(output.len()).unwrap();
                }
                Layer::Dense(dense_layer) => {
                    flat_output = dense_layer.forward(flat_output, training);
                }
            }
        }

        flat_output
    }

    pub fn last_layer_error(&mut self, label: usize) -> Array1<f32> {
        let size: usize = match self.layers.last().unwrap() {
            Layer::Dense(dense_layer) => dense_layer.output_size,
            _ => panic!("Last layer is not a DenseLayer"),
        };
        let desired = Array1::<f32>::from_shape_fn(size, |i| (label == i) as usize as f32);
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
                Layer::Conv(conv_layer) => {
                    error = conv_layer.backward(error);
                }
                Layer::Mxpl(mxpl_layer) => {
                    error = mxpl_layer.backward(error);
                }
                Layer::Dense(dense_layer) => {
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
                Layer::Conv(conv_layer) => conv_layer.update(minibatch_size),
                Layer::Mxpl(_) => {}
                Layer::Dense(dense_layer) => dense_layer.update(minibatch_size),
            }
        }
    }

    pub fn output(&self) -> Array1<f32> {
        match self.layers.last().unwrap() {
            Layer::Conv(_) => panic!("Last layer is a ConvLayer"),
            Layer::Mxpl(_) => panic!("Last layer is a MxplLayer"),
            Layer::Dense(dense_layer) => dense_layer.output.clone(),
        }
    }

    pub fn get_accuracy(&self, label: usize) -> f32 {
        let mut max = 0.0;
        let mut max_idx = 0;
        let output = self.output();
        for j in 0..output.len() {
            if output[j] > max {
                max = output[j];
                max_idx = j;
            }
        }

        (max_idx == label) as usize as f32
    }

    pub fn train(&mut self) {
        for epoch in 0..self.epochs {
            let pb = ProgressBar::new((self.data.trn_size / self.minibatch_size) as u64);
            pb.set_style(ProgressStyle::default_bar()
                    .template(&format!("Epoch {}: [{{bar:.cyan/blue}}] {{pos}}/{{len}} - ETA: {{eta}} - acc: {{msg}}", epoch))
                    .unwrap()
                    .progress_chars("#>-"));

            let mut avg_acc = 0.0;
            for i in 0..self.data.trn_size {
                let (image, label) = get_random_image(&self.data);
                let label = *self.data.classes.get(&label).unwrap();
                self.forward(image, true);
                self.backward(label, true);

                avg_acc += self.get_accuracy(label);

                if i % self.minibatch_size == self.minibatch_size - 1 {
                    self.update(self.minibatch_size);
                    pb.inc(1);
                    pb.set_message(format!("{:.1}%", avg_acc / (i + 1) as f32 * 100.0));
                }
            }

            avg_acc /= self.data.trn_size as f32;
            pb.set_message(format!("{:.1}% - Testing...", avg_acc));

            // Testing
            let mut avg_test_acc = 0.0;
            for _i in 0..self.data.tst_size {
                // let image: Array3<f32> = self.data.tst_img[i].clone();
                let (image, label) = get_random_test_image(&self.data);
                let label = *self.data.classes.get(&label).unwrap();
                self.forward(image, false);

                avg_test_acc += self.get_accuracy(label);
            }

            avg_test_acc /= self.data.tst_size as f32;
            pb.finish_with_message(format!(
                "{:.1}% - Test: {:.1}%",
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
                Layer::Conv(conv_layer) => conv_layer.zero(),
                Layer::Mxpl(mxpl_layer) => mxpl_layer.zero(),
                Layer::Dense(dense_layer) => dense_layer.zero(),
            }
        }
    }
}
