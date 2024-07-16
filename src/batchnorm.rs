use ndarray::{s, Array1, Array3, Axis};

pub struct BatchNorm {
    pub gamma: Array1<f32>,
    pub beta: Array1<f32>,
    pub moving_mean: Array1<f32>,
    pub moving_var: Array1<f32>,
    pub epsilon: f32,
    pub momentum: f32,
}

impl BatchNorm {
    pub fn new(num_features: usize, epsilon: f32, momentum: f32) -> Self {
        BatchNorm {
            gamma: Array1::ones(num_features),
            beta: Array1::zeros(num_features),
            moving_mean: Array1::zeros(num_features),
            moving_var: Array1::ones(num_features),
            epsilon,
            momentum,
        }
    }

    pub fn forward(&mut self, input: &Array3<f32>, training: bool) -> Array3<f32> {
        let num_features = self.gamma.len();

        if training {
            let mean = input
                .mean_axis(Axis(0))
                .unwrap()
                .mean_axis(Axis(0))
                .unwrap();
            let var = input.var_axis(Axis(0), 0.0).mean_axis(Axis(0)).unwrap(); // Unwrap here

            // Update moving statistics
            self.moving_mean = &self.moving_mean * (1.0 - self.momentum) + &mean * self.momentum;
            self.moving_var = &self.moving_var * (1.0 - self.momentum) + &var * self.momentum;

            // Normalize
            let normalized = (input - &mean.view().insert_axis(Axis(0)).insert_axis(Axis(0)))
                / (&var + self.epsilon)
                    .mapv(f32::sqrt)
                    .view()
                    .insert_axis(Axis(0))
                    .insert_axis(Axis(0));

            // Scale and shift
            &normalized * &self.gamma.view().insert_axis(Axis(0)).insert_axis(Axis(0))
                + &self.beta.view().insert_axis(Axis(0)).insert_axis(Axis(0))
        } else {
            // Use moving statistics for inference
            (input
                - &self
                    .moving_mean
                    .view()
                    .insert_axis(Axis(0))
                    .insert_axis(Axis(0)))
                / (&self.moving_var + self.epsilon)
                    .mapv(f32::sqrt)
                    .view()
                    .insert_axis(Axis(0))
                    .insert_axis(Axis(0))
                * &self.gamma.view().insert_axis(Axis(0)).insert_axis(Axis(0))
                + &self.beta.view().insert_axis(Axis(0)).insert_axis(Axis(0))
        }
    }

    pub fn backward(
        &self,
        input: &Array3<f32>,
        grad_output: &Array3<f32>,
    ) -> (Array3<f32>, Array1<f32>, Array1<f32>) {
        let (n, h, w) = input.dim();
        let num_features = self.gamma.len();

        let mean = input
            .mean_axis(Axis(0))
            .unwrap()
            .mean_axis(Axis(0))
            .unwrap();
        let var = input.var_axis(Axis(0), 0.0).mean_axis(Axis(0)).unwrap(); // Unwrap here

        let normalized = (input - &mean.view().insert_axis(Axis(0)).insert_axis(Axis(0)))
            / (&var + self.epsilon)
                .mapv(f32::sqrt)
                .view()
                .insert_axis(Axis(0))
                .insert_axis(Axis(0));

        let grad_gamma = (&normalized * grad_output)
            .sum_axis(Axis(0))
            .sum_axis(Axis(0));
        let grad_beta = grad_output.sum_axis(Axis(0)).sum_axis(Axis(0));

        let std_dev = (&var + self.epsilon).mapv(f32::sqrt);
        let grad_normalized =
            grad_output * &self.gamma.view().insert_axis(Axis(0)).insert_axis(Axis(0));

        let grad_var = (-0.5 * &grad_normalized * &normalized
            / &std_dev.view().insert_axis(Axis(0)).insert_axis(Axis(0)))
            .sum_axis(Axis(0))
            .sum_axis(Axis(0));
        let grad_mean = (-&grad_normalized
            / &std_dev.view().insert_axis(Axis(0)).insert_axis(Axis(0)))
            .sum_axis(Axis(0))
            .sum_axis(Axis(0))
            - 2.0 * &grad_var * &mean / (n * h * w) as f32;

        let grad_input = &grad_normalized
            / &std_dev.view().insert_axis(Axis(0)).insert_axis(Axis(0))
            + 2.0 * &grad_var.view().insert_axis(Axis(0)).insert_axis(Axis(0)) * &normalized
                / (n * h * w) as f32
            + &grad_mean.view().insert_axis(Axis(0)).insert_axis(Axis(0)) / (n * h * w) as f32;

        (grad_input, grad_gamma, grad_beta)
    }
}
