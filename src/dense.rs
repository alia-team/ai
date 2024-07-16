use ndarray::{Array, Array1, Array2, ArrayViewMut, IxDyn};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

pub struct Dense {
    pub weights: Array2<f32>,
    pub bias: Array1<f32>,
    pub activation: fn(f32) -> f32,
}

fn outer_product(a: &Array1<f32>, b: &Array1<f32>) -> Array2<f32> {
    let mut result = Array2::zeros((a.len(), b.len()));
    for i in 0..a.len() {
        for j in 0..b.len() {
            result[[i, j]] = a[i] * b[j];
        }
    }
    result
}

impl Dense {
    pub fn new(input_size: usize, output_size: usize, activation: fn(f32) -> f32) -> Self {
        // He init
        let std_dev = (2.0 / input_size as f32).sqrt();
        let weights = Array::random(
            (input_size, output_size),
            Normal::new(0.0, std_dev).unwrap(),
        );

        let bias = Array::zeros(output_size);

        Dense {
            weights,
            bias,
            activation,
        }
    }

    pub fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        let output = input.dot(&self.weights) + &self.bias;
        output.mapv(self.activation)
    }

    pub fn backward(
        &mut self,
        input: &Array1<f32>,
        grad_output: &Array1<f32>,
    ) -> (Array1<f32>, Array2<f32>, Array1<f32>) {
        let grad_input = grad_output.dot(&self.weights.t());
        let mut grad_weights = outer_product(input, grad_output);
        let mut grad_bias = grad_output.clone();

        // Gradient norm clipping
        clip_gradient_norm(&mut grad_weights.view_mut(), 5.0, 5.0);
        clip_gradient_norm(&mut grad_bias.view_mut(), 5.0, 5.0);

        // Update weights and biases
        self.weights -= &grad_weights;
        self.bias -= &grad_bias;

        (grad_input, grad_weights, grad_bias)
    }

    pub fn from_weights(
        weights: Vec<f32>,
        weights_shape: Vec<usize>,
        bias: Vec<f32>,
        activation: fn(f32) -> f32,
    ) -> Self {
        let weights_array = Array::from_shape_vec(IxDyn(&weights_shape), weights).unwrap();
        Dense {
            weights: weights_array
                .into_shape((weights_shape[0], weights_shape[1]))
                .unwrap(),
            bias: Array1::from(bias),
            activation,
        }
    }
}

fn clip_gradient_norm<D>(grad: &mut ArrayViewMut<f32, D>, max_norm: f32, max_value: f32)
where
    D: ndarray::Dimension,
{
    // Value clipping
    grad.mapv_inplace(|x| x.clamp(-max_value, max_value));

    // Norm clipping
    let norm = grad.mapv(|x| x * x).sum().sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        grad.mapv_inplace(|x| x * scale);
    }
}
