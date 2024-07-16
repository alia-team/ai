use ndarray::{s, Array1, Array3};

pub struct BatchNorm {
    pub gamma: Array1<f32>,
    pub beta: Array1<f32>,
    pub moving_mean: Array1<f32>,
    pub moving_var: Array1<f32>,
    pub epsilon: f32,
}

impl BatchNorm {
    pub fn new(num_features: usize) -> Self {
        BatchNorm {
            gamma: Array1::ones(num_features),
            beta: Array1::zeros(num_features),
            moving_mean: Array1::zeros(num_features),
            moving_var: Array1::ones(num_features),
            epsilon: 1e-5,
        }
    }

    pub fn forward(&mut self, input: &Array3<f32>, training: bool) -> Array3<f32> {
        let num_features = self.gamma.len();

        if training {
            let mut mean = Array1::zeros(num_features);
            let mut var = Array1::zeros(num_features);

            for c in 0..num_features {
                let channel_data = input.slice(s![.., .., c]);
                mean[c] = channel_data.mean().unwrap();
                var[c] = channel_data.var(0.0);
            }

            // Update moving statistics
            self.moving_mean = &self.moving_mean * 0.9 + &mean * 0.1;
            self.moving_var = &self.moving_var * 0.9 + &var * 0.1;

            // Normalize
            let mut normalized = input.clone();
            for c in 0..num_features {
                let mut channel = normalized.slice_mut(s![.., .., c]);
                channel -= mean[c];
                channel /= (var[c] + self.epsilon).sqrt();
            }

            // Scale and shift
            for c in 0..num_features {
                let mut channel = normalized.slice_mut(s![.., .., c]);
                channel *= self.gamma[c];
                channel += self.beta[c];
            }

            normalized
        } else {
            // Use moving statistics for inference
            let mut normalized = input.clone();
            for c in 0..num_features {
                let mut channel = normalized.slice_mut(s![.., .., c]);
                channel -= self.moving_mean[c];
                channel /= (self.moving_var[c] + self.epsilon).sqrt();
                channel *= self.gamma[c];
                channel += self.beta[c];
            }

            normalized
        }
    }

    pub fn backward(&mut self, input: &Array3<f32>, grad_output: &Array3<f32>) -> Array3<f32> {
        let (n, h, w) = input.dim();
        let num_features = self.gamma.len();

        let mut mean = Array1::zeros(num_features);
        let mut var = Array1::zeros(num_features);

        for c in 0..num_features {
            let channel_data = input.slice(s![.., .., c]);
            mean[c] = channel_data.mean().unwrap();
            var[c] = channel_data.var(0.0);
        }

        let mut normalized = input.clone();
        for c in 0..num_features {
            let mut channel = normalized.slice_mut(s![.., .., c]);
            channel -= mean[c];
            channel /= (var[c] + self.epsilon).sqrt();
        }

        let mut grad_gamma = Array1::<f32>::zeros(num_features);
        let mut grad_beta = Array1::<f32>::zeros(num_features);
        let mut grad_input = Array3::<f32>::zeros((n, h, w));

        for c in 0..num_features {
            let grad_output_c = grad_output.slice(s![.., .., c]);
            let normalized_c = normalized.slice(s![.., .., c]);

            grad_gamma[c] = (&grad_output_c * &normalized_c).sum();
            grad_beta[c] = grad_output_c.sum();

            let grad_normalized = &grad_output_c * self.gamma[c];
            let std_dev = (var[c] + self.epsilon).sqrt();
            let grad_var = (-0.5 * &grad_normalized * &normalized_c / std_dev).sum();
            let grad_mean = (-&grad_normalized / std_dev).sum() - 2.0 * grad_var * mean[c];

            let mut grad_input_c = grad_input.slice_mut(s![.., .., c]);
            grad_input_c += &(&grad_normalized / std_dev
                + 2.0 * grad_var * &normalized_c / (n * h * w) as f32
                + grad_mean / (n * h * w) as f32);
        }

        // Update gamma and beta
        self.gamma += &grad_gamma;
        self.beta += &grad_beta;

        grad_input
    }
}
