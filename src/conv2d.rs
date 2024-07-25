use crate::optimizer::{Optimizer, Optimizer4D};
use ndarray::{s, Array3, Array4};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
pub struct Conv2D {
    input_size: (usize, usize, usize),
    kernel_size: usize,
    pub output_size: (usize, usize, usize),
    input: Array3<f64>,
    output: Array3<f64>,
    num_filters: usize,
    kernels: Array4<f64>,
    gradients: Array4<f64>,
    optimizer: Optimizer4D,
}

impl Conv2D {
    pub fn zero(&mut self) {
        self.gradients = Array4::<f64>::zeros((
            self.num_filters,
            self.kernel_size,
            self.kernel_size,
            self.input_size.2,
        ));
        self.output = Array3::<f64>::zeros(self.output_size);
    }

    /// Create a new max pooling layer with the given parameters
    pub fn new(
        input_size: (usize, usize, usize),
        kernel_size: usize,
        stride: usize,
        num_filters: usize,
        optimizer_alg: Optimizer,
    ) -> Conv2D {
        let output_width: usize = ((input_size.0 - kernel_size) / stride) + 1;
        let output_size = (output_width, output_width, num_filters);
        let mut kernels =
            Array4::<f64>::zeros((num_filters, kernel_size, kernel_size, input_size.2));
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Initialize kernels
        for f in 0..num_filters {
            for kd in 0..input_size.2 {
                for ky in 0..kernel_size {
                    for kx in 0..kernel_size {
                        // He weights initialization
                        kernels[[f, ky, kx, kd]] = normal.sample(&mut rand::thread_rng())
                            * (2.0 / (input_size.0.pow(2)) as f64).sqrt();
                    }
                }
            }
        }

        let optimizer = Optimizer4D::new(
            optimizer_alg,
            (num_filters, kernel_size, kernel_size, input_size.2),
        );

        let layer: Conv2D = Conv2D {
            input_size,
            kernel_size,
            output_size,
            output: Array3::<f64>::zeros(output_size),
            input: Array3::<f64>::zeros(input_size),
            num_filters,
            kernels,
            gradients: Array4::<f64>::zeros((num_filters, kernel_size, kernel_size, input_size.2)),
            optimizer,
        };

        layer
    }

    pub fn forward(&mut self, input: Array3<f64>) -> Array3<f64> {
        // Store the input for later use in backpropagation
        self.input = input;

        // Iterate over each filter
        for f in 0..self.output_size.2 {
            let kernel_slice = self.kernels.slice(s![f, .., .., ..]);
            for y in 0..self.output_size.1 {
                for x in 0..self.output_size.0 {
                    let input_slice =
                        self.input
                            .slice(s![x..x + self.kernel_size, y..y + self.kernel_size, ..]);

                    // Convolution + ReLU
                    self.output[[x, y, f]] = (&input_slice * &kernel_slice).sum().max(0.0);
                }
            }
        }

        self.output.clone()
    }

    pub fn backward(&mut self, error: Array3<f64>) -> Array3<f64> {
        assert_eq!(
            error.shape(),
            [self.output_size.0, self.output_size.1, self.output_size.2],
            "Error shape mismatch"
        );

        let mut prev_error = Array3::<f64>::zeros(self.input_size);
        let (out_x, out_y, out_z) = self.output_size;
        let (in_x, in_y, in_z) = (self.input_size.0, self.input_size.1, self.input_size.2);

        for f in 0..out_z {
            for y in 0..out_y {
                for x in 0..out_x {
                    // Skip
                    if self.output[[x, y, f]] <= 0.0 {
                        continue;
                    }

                    let x_end = (x + self.kernel_size).min(in_x);
                    let y_end = (y + self.kernel_size).min(in_y);

                    // Update prev_error
                    for kx in x..x_end {
                        for ky in y..y_end {
                            for kz in 0..in_z {
                                prev_error[[kx, ky, kz]] +=
                                    error[[x, y, f]] * self.kernels[[f, kx - x, ky - y, kz]];
                            }
                        }
                    }

                    // Update gradients
                    for kx in 0..(x_end - x) {
                        for ky in 0..(y_end - y) {
                            for kz in 0..in_z {
                                self.gradients[[f, kx, ky, kz]] -=
                                    error[[x, y, f]] * self.input[[x + kx, y + ky, kz]];
                            }
                        }
                    }
                }
            }
        }
        prev_error
    }

    pub fn update(&mut self, minibatch_size: usize) {
        self.gradients /= minibatch_size as f64;
        self.kernels += &self.optimizer.weight_changes(&self.gradients);
        self.gradients = Array4::<f64>::zeros((
            self.num_filters,
            self.kernel_size,
            self.kernel_size,
            self.input_size.2,
        ));
    }
}
