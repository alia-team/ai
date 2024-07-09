use crate::activation::{enum_to_activation, Activation, ActivationEnum};
use ndarray::{arr1, concatenate, s, Array1, Array2, Array3, Axis};
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use std::any::Any;
use std::fmt::Debug;

pub trait Layer: Debug {
    fn forward(&mut self, input: &dyn Any) -> Box<dyn Any>;
    fn backward(&mut self, gradients: &dyn Any) -> Box<dyn Any>;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn output_size(&self) -> usize;
    fn input_size(&self) -> usize;
}

#[derive(Debug)]
pub struct Dense {
    pub units: usize,
    pub weights: Option<Array2<f64>>, // Initially None
    pub output: Option<Array1<f64>>,
    pub input: Option<Array1<f64>>,
    pub weight_gradients: Option<Array2<f64>>,
    pub activation: Box<dyn Activation>,
}

impl Dense {
    pub fn new(units: usize, activation: ActivationEnum) -> Self {
        let activation_fn: Box<dyn Activation> = enum_to_activation(activation);
        Dense {
            units,
            weights: None,
            output: None,
            input: None,
            weight_gradients: None,
            activation: activation_fn,
        }
    }

    fn initialize_parameters(&mut self, input_size: usize) {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        self.weights = Some(Array2::from_shape_fn((self.units, input_size + 1), |_| {
            normal.sample(&mut rng)
        })); // Additional column for biases
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: &dyn Any) -> Box<dyn Any> {
        let input = input.downcast_ref::<Array1<f64>>().unwrap();

        if self.weights.is_none() {
            self.initialize_parameters(input.len());
        }

        let input_with_bias = concatenate![Axis(0), input.view(), arr1(&[1.0])];
        self.input = Some(input.clone());
        let weights = self.weights.as_ref().unwrap();
        let linear_output = weights.dot(&input_with_bias);
        let activated_output = self.activation.forward(&linear_output);
        self.output = Some(activated_output.clone());

        Box::new(activated_output)
    }

    fn backward(&mut self, gradients: &dyn Any) -> Box<dyn Any> {
        let gradients = gradients.downcast_ref::<Array1<f64>>().unwrap();
        let input = self.input.as_ref().unwrap();
        let input_with_bias = concatenate![Axis(0), input.view(), arr1(&[1.0])];
        let delta = self.activation.backward(gradients);
        let grad_w = delta.view().into_shape((self.units, 1)).unwrap().dot(
            &input_with_bias
                .view()
                .into_shape((1, input_with_bias.len()))
                .unwrap(),
        );
        self.weight_gradients = Some(grad_w);
        let weights = self.weights.as_ref().unwrap();

        // Return gradients with respect to input, which should have the same shape as the input
        Box::new(weights.slice(s![.., ..-1]).t().dot(&delta))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn output_size(&self) -> usize {
        self.units
    }

    fn input_size(&self) -> usize {
        match &self.weights {
            Some(w) => w.ncols() - 1, // Subtract 1 for bias
            None => panic!("Weights not initialized"),
        }
    }
}

#[derive(Debug, Default)]
pub struct Flatten {
    pub input_shape: Option<(usize, usize, usize)>, // (height, width, channels)
    pub output_size: usize,
    pub input: Option<Array3<f64>>,
    pub output: Option<Array1<f64>>,
}

impl Flatten {
    pub fn new() -> Self {
        Flatten {
            input_shape: None,
            output_size: 0,
            input: None,
            output: None,
        }
    }
}

impl Layer for Flatten {
    fn forward(&mut self, input: &dyn Any) -> Box<dyn Any> {
        let input = input.downcast_ref::<Array3<f64>>().unwrap();

        if self.input_shape.is_none() {
            let shape = input.shape();
            self.input_shape = Some((shape[0], shape[1], shape[2]));
            self.output_size = shape[0] * shape[1] * shape[2];
        }

        self.input = Some(input.clone());
        let output = input.clone().into_shape(self.output_size).unwrap();
        self.output = Some(output.clone());

        Box::new(output)
    }

    fn backward(&mut self, gradients: &dyn Any) -> Box<dyn Any> {
        let gradients = gradients.downcast_ref::<Array1<f64>>().unwrap();
        let (height, width, channels) = self.input_shape.unwrap();
        let reshaped_gradients = gradients
            .clone()
            .into_shape((height, width, channels))
            .unwrap();

        Box::new(reshaped_gradients)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn output_size(&self) -> usize {
        self.output_size
    }

    fn input_size(&self) -> usize {
        match self.input_shape {
            Some((h, w, c)) => h * w * c,
            None => panic!("Input shape not initialized"),
        }
    }
}
