use crate::activation::{enum_to_activation, Activation, ActivationEnum};
use ndarray::{arr1, concatenate, s, Array1, Array2, Axis};
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

pub trait Layer {
    fn forward(&mut self, input: &Array1<f64>) -> Array1<f64>;
    fn backward(&mut self, gradients: &Array1<f64>) -> Array1<f64>;
}

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
        // Initialize weights with small random values
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        self.weights = Some(Array2::from_shape_fn((self.units, input_size + 1), |_| {
            normal.sample(&mut rng)
        })); // Additional column for biases
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        // If weights are not initialized, initialize them
        if self.weights.is_none() {
            self.initialize_parameters(input.len());
        }

        // Add bias term to the input
        let input_with_bias = concatenate![Axis(0), input.view(), arr1(&[1.0])];

        // Store input for use in backpropagation
        self.input = Some(input.clone());

        // Perform the linear transformation
        let weights = self.weights.as_ref().unwrap();
        let linear_output = weights.dot(&input_with_bias);
        println!("weights dot {:?}", linear_output);

        // Apply the activation function
        let activated_output = self.activation.forward(&linear_output);

        // Store output for use in backpropagation
        self.output = Some(activated_output.clone());

        activated_output
    }

    fn backward(&mut self, gradients: &Array1<f64>) -> Array1<f64> {
        // Retrieve the stored input
        let input = self.input.as_ref().unwrap();
        let input_with_bias = concatenate![Axis(0), input.view(), arr1(&[1.0])];

        // Compute the gradient of the activation function
        let delta = self.activation.backward(gradients);

        // Compute gradients w.r.t weights
        let grad_w = delta.view().into_shape((self.units, 1)).unwrap().dot(
            &input_with_bias
                .view()
                .into_shape((1, input_with_bias.len()))
                .unwrap(),
        );

        // Update gradients field
        self.weight_gradients = Some(grad_w);

        // Compute gradients w.r.t input (excluding bias term)
        let weights = self.weights.as_ref().unwrap();
        let grad_input = weights.slice(s![.., ..-1]).t().dot(&delta);

        grad_input
    }
}
