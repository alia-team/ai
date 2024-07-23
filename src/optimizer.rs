use ndarray::{Array1, Array2, Array4};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Clone, Copy)]
pub enum Optimizer {
    SGD(f64),            // Learning rate
    Momentum(f64, f64),  // Learning rate & momentum factor
    RMSProp(f64, f64),   // Learning rate & decay rate
    Adam(f64, f64, f64), // Learning rate, beta1 & beta2
}

// Optimizer struct for dense layer
#[derive(Serialize, Deserialize)]
pub struct Optimizer2D {
    pub alg: Optimizer,
    pub momentum1: Array2<f64>, // First moment for Adam, velocity for Momentum
    pub momentum2: Array2<f64>, // Second moment for Adam, velocity squared for RMSProp
    pub t: i32,                 // Timestep for Adam
    pub beta1_done: bool,       // Flag to optimize Adam computation
    pub beta2_done: bool,       // Flag to optimize Adam computation
}

impl Optimizer2D {
    pub fn new(alg: Optimizer, input_size: usize, output_size: usize) -> Optimizer2D {
        let momentum1 = Array2::<f64>::zeros((output_size, input_size));
        let momentum2 = Array2::<f64>::zeros((output_size, input_size));
        let t = 0;
        let beta1_done = false;
        let beta2_done = false;

        Optimizer2D {
            alg,
            momentum1,
            momentum2,
            t,
            beta1_done,
            beta2_done,
        }
    }

    // Calculate weight changes based on gradients and the chosen optimization algorithm
    pub fn weight_changes(&mut self, gradients: &Array2<f64>) -> Array2<f64> {
        match self.alg {
            Optimizer::SGD(lr) => gradients * lr,
            Optimizer::Momentum(lr, mu) => {
                self.momentum1 = &self.momentum1 * mu + gradients;
                &self.momentum1 * lr
            }
            Optimizer::RMSProp(lr, rho) => {
                self.momentum1 = &self.momentum1 * rho;
                self.momentum1 += &(gradients.mapv(|x| x.powi(2)) * (1.0 - rho));
                gradients * lr / (self.momentum1.mapv(|x| x.sqrt()) + 1e-8)
            }
            Optimizer::Adam(lr, beta1, beta2) => {
                self.t += 1;
                // Update biased first moment estimate
                self.momentum1 = &self.momentum1 * beta1;
                self.momentum1 += &(gradients.mapv(|x| x * (1.0 - beta1)));
                // Update biased second raw moment estimate
                self.momentum2 = &self.momentum2 * beta2;
                self.momentum2 += &(gradients.mapv(|x| x.powi(2) * (1.0 - beta2)));

                // Compute bias correction factors
                let biased_beta1 = if self.beta1_done {
                    0.0
                } else {
                    let pow = beta1.powi(self.t);
                    if pow < 0.001 {
                        self.beta1_done = true;
                    }
                    pow
                };
                let biased_beta2 = if self.beta2_done {
                    0.0
                } else {
                    let pow = beta2.powi(self.t);
                    if pow < 0.001 {
                        self.beta2_done = true;
                    }
                    pow
                };

                // Compute bias-corrected moment estimates
                let weight_velocity_corrected = &self.momentum1 / (1.0 - biased_beta1);
                let weight_velocity2_corrected = &self.momentum2 / (1.0 - biased_beta2);

                // Compute and return the Adam update
                &weight_velocity_corrected * lr
                    / (weight_velocity2_corrected.mapv(|x| x.sqrt()) + 1e-8)
            }
        }
    }

    // Calculate bias changes (simplified version of weight changes for 1D array)
    pub fn bias_changes(&mut self, gradients: &Array1<f64>) -> Array1<f64> {
        match self.alg {
            Optimizer::SGD(lr) => gradients * lr,
            Optimizer::Momentum(lr, _) => gradients * lr,
            Optimizer::RMSProp(lr, _) => gradients * lr,
            Optimizer::Adam(lr, _, _) => gradients * lr,
        }
    }
}

// Optimizer struct for convolutional layer
#[derive(Serialize, Deserialize)]
pub struct Optimizer4D {
    pub alg: Optimizer,
    pub momentum1: Array4<f64>, // First moment for Adam, velocity for Momentum
    pub momentum2: Array4<f64>, // Second moment for Adam, velocity squared for RMSProp
    pub t: i32,                 // Timestep for Adam
    pub beta1_done: bool,       // Flag to optimize Adam computation
    pub beta2_done: bool,       // Flag to optimize Adam computation
}

impl Optimizer4D {
    pub fn new(alg: Optimizer, size: (usize, usize, usize, usize)) -> Optimizer4D {
        let momentum1: Array4<f64> = Array4::<f64>::zeros(size);
        let momentum2: Array4<f64> = Array4::<f64>::zeros(size);
        let t: i32 = 0;
        let beta1_done: bool = false;
        let beta2_done: bool = false;

        Optimizer4D {
            alg,
            momentum1,
            momentum2,
            t,
            beta1_done,
            beta2_done,
        }
    }

    // Calculate weight changes based on gradients and the chosen optimization algorithm
    pub fn weight_changes(&mut self, gradients: &Array4<f64>) -> Array4<f64> {
        match self.alg {
            Optimizer::SGD(lr) => gradients * lr,
            Optimizer::Momentum(lr, mu) => {
                self.momentum1 = &self.momentum1 * mu + gradients;
                &self.momentum1 * lr
            }
            Optimizer::RMSProp(lr, rho) => {
                self.momentum1 = &self.momentum1 * rho;
                self.momentum1 += &(gradients.mapv(|x| x.powi(2)) * (1.0 - rho));
                gradients * lr / (self.momentum1.mapv(|x| x.sqrt()) + 1e-8)
            }
            Optimizer::Adam(lr, beta1, beta2) => {
                self.t += 1;
                // Update biased first moment estimate
                self.momentum1 = &self.momentum1 * beta1;
                self.momentum1 += &(gradients.mapv(|x| x * (1.0 - beta1)));
                // Update biased second raw moment estimate
                self.momentum2 = &self.momentum2 * beta2;
                self.momentum2 += &(gradients.mapv(|x| x.powi(2) * (1.0 - beta2)));

                // Compute bias correction factors
                let biased_beta1 = if self.beta1_done {
                    0.0
                } else {
                    let pow = beta1.powi(self.t);
                    if pow < 0.001 {
                        self.beta1_done = true;
                    }
                    pow
                };
                let biased_beta2 = if self.beta2_done {
                    0.0
                } else {
                    let pow = beta2.powi(self.t);
                    if pow < 0.001 {
                        self.beta2_done = true;
                    }
                    pow
                };

                // Compute bias-corrected moment estimates
                let weight_velocity_corrected = &self.momentum1 / (1.0 - biased_beta1);
                let weight_velocity2_corrected = &self.momentum2 / (1.0 - biased_beta2);

                // Compute and return the Adam update
                &weight_velocity_corrected * lr
                    / (weight_velocity2_corrected.mapv(|x| x.sqrt()) + 1e-8)
            }
        }
    }
}
