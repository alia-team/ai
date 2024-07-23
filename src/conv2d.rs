use crate::optimizer::{Optimizer, Optimizer4D};
use ndarray::{s, Array3, Array4};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::ops::{AddAssign, SubAssign};

#[derive(Deserialize, Serialize)]
pub struct Conv2D {
    input_size: (usize, usize, usize),
    kernel_size: usize,
    pub output_size: (usize, usize, usize),
    input: Array3<f64>,
    output: Array3<f64>,
    num_filters: usize,
    kernels: Array4<f64>,
    kernel_changes: Array4<f64>,
    optimizer: Optimizer4D,
}

impl Conv2D {
    pub fn zero(&mut self) {
        self.kernel_changes = Array4::<f64>::zeros((
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

        for f in 0..num_filters {
            for kd in 0..input_size.2 {
                for ky in 0..kernel_size {
                    for kx in 0..kernel_size {
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
            kernel_changes: Array4::<f64>::zeros((
                num_filters,
                kernel_size,
                kernel_size,
                input_size.2,
            )),
            optimizer,
        };

        layer
    }

    pub fn forward(&mut self, input: Array3<f64>) -> Array3<f64> {
        self.input = input;
        for f in 0..self.output_size.2 {
            let kernel_slice = self.kernels.slice(s![f, .., .., ..]);
            for y in 0..self.output_size.1 {
                for x in 0..self.output_size.0 {
                    let input_slice =
                        self.input
                            .slice(s![x..x + self.kernel_size, y..y + self.kernel_size, ..]);
                    self.output[[x, y, f]] = (&input_slice * &kernel_slice).sum().max(0.0);
                }
            }
        }

        self.output.clone()
    }

    pub fn backward(&mut self, error: Array3<f64>) -> Array3<f64> {
        let mut prev_error: Array3<f64> = Array3::<f64>::zeros(self.input_size);
        for f in 0..self.output_size.2 {
            for y in 0..self.output_size.1 {
                for x in 0..self.output_size.0 {
                    if self.output[[x, y, f]] <= 0.0 {
                        continue;
                    }
                    prev_error
                        .slice_mut(s![x..x + self.kernel_size, y..y + self.kernel_size, ..])
                        .add_assign(&(error[[x, y, f]] * &self.kernels.slice(s![f, .., .., ..])));

                    let input_slice =
                        self.input
                            .slice(s![x..x + self.kernel_size, y..y + self.kernel_size, ..]);
                    self.kernel_changes
                        .slice_mut(s![f, .., .., ..])
                        .sub_assign(&(error[[x, y, f]] * &input_slice));
                }
            }
        }

        prev_error
    }

    pub fn update(&mut self, minibatch_size: usize) {
        self.kernel_changes /= minibatch_size as f64;
        self.kernels += &self.optimizer.weight_changes(&self.kernel_changes);
        self.kernel_changes = Array4::<f64>::zeros((
            self.num_filters,
            self.kernel_size,
            self.kernel_size,
            self.input_size.2,
        ));
    }
}
