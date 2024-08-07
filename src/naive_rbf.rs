extern crate rand;
use crate::activation::{str_to_activation, Activation};
use crate::util::{self, c_str_to_rust_str};
use nalgebra::*;
use ndarray::arr1;
use rand::Rng;
use std::ffi::c_char;
use std::{ptr, slice};

#[derive(Debug, PartialEq)]
pub struct Centroid {
    pub coordinates: Vec<f64>,
}

impl Centroid {
    pub fn new(coordinates: Vec<f64>) -> Self {
        Centroid { coordinates }
    }

    pub fn forward(&self, input: Vec<f64>, gamma: f64) -> f64 {
        let mut vec_sub: Vec<f64> = vec![];
        for (i, value) in input.iter().enumerate() {
            vec_sub.push(value - self.coordinates[i])
        }

        let mut norm: f64 = 0.0;
        for value in vec_sub {
            norm += value.powi(2)
        }
        norm = norm.sqrt();

        (-gamma * norm.powi(2)).exp()
    }
}

pub struct NaiveRBF {
    pub neurons_per_layer: Vec<usize>,
    pub centroids: Vec<Centroid>,
    pub weights: Vec<Vec<Vec<f64>>>,
    pub outputs: Vec<Vec<f64>>,
    pub gamma: f64,
    pub activation: Box<dyn Activation>,
}

impl NaiveRBF {
    pub fn new(
        input_neurons_count: usize,
        outpout_neurons_count: usize,
        activation: &str,
        training_dataset: Vec<Vec<f64>>,
    ) -> Self {
        let activation_fn: Box<dyn Activation> = str_to_activation(activation);

        // Initialize centroids
        let mut centroids: Vec<Centroid> = vec![];
        for sample in training_dataset.clone() {
            centroids.push(Centroid::new(sample.clone()));
        }

        let neurons_per_layer: Vec<usize> = vec![
            input_neurons_count,
            training_dataset.len(),
            outpout_neurons_count,
        ];
        let weights = util::init_weights(neurons_per_layer.clone(), true);
        let outputs = util::init_outputs(neurons_per_layer.clone(), true);
        let gamma = rand::thread_rng().gen_range(0.01..=1.0);

        NaiveRBF {
            neurons_per_layer,
            centroids,
            weights,
            outputs,
            gamma,
            activation: activation_fn,
        }
    }

    pub fn fit(&mut self, training_dataset: Vec<Vec<f64>>, labels: Vec<Vec<f64>>, gamma: f64) {
        self.gamma = gamma;
        let n_samples = training_dataset.len();
        let feature_len = training_dataset[0].len();

        // Construct Phi matrix
        let mut phi = DMatrix::zeros(n_samples, n_samples);
        for i in 0..n_samples {
            for j in 0..n_samples {
                let mut norm: f64 = 0.0;
                for k in 0..feature_len {
                    let diff = training_dataset[i][k] - training_dataset[j][k];
                    norm += diff * diff;
                }
                norm = norm.sqrt();
                phi[(i, j)] = (-self.gamma * norm.powi(2)).exp();
            }
        }

        // Invert Phi matrix
        let phi_inv = match phi.clone().try_inverse() {
            Some(inv) => inv,
            None => panic!("Matrix inversion failed."),
        };

        // Construct Y vector
        let mut y = DMatrix::zeros(labels.len(), labels[0].len());
        for i in 0..labels.len() {
            for j in 0..labels[0].len() {
                y[(i, j)] = labels[i][j];
            }
        }

        // Compute weights
        let weights_matrix = phi_inv * y;

        // Convert the DMatrix to Vec<Vec<T>>
        let mut weights: Vec<Vec<f64>> = Vec::with_capacity(weights_matrix.ncols());
        for i in 0..self.neurons_per_layer[2] {
            let mut row_vec = Vec::with_capacity(weights_matrix.nrows());
            for j in 0..self.neurons_per_layer[1] {
                row_vec.push(weights_matrix[(j, i)]);
            }
            weights.push(row_vec);
        }

        // Assign weights to the last layer
        self.weights[2] = weights;
    }

    pub fn predict(&mut self, input: Vec<f64>) -> Vec<f64> {
        self.outputs[0] = input.clone();

        // Reset hidden layer's outputs
        self.outputs[1] = vec![];

        // Forward pass in hidden layer
        for centroid in &self.centroids {
            self.outputs[1].push(centroid.forward(input.clone(), self.gamma))
        }

        let mut weighted_sum: f64 = 0.0;

        // Forward pass in output layer
        for i in 0..self.neurons_per_layer[2] {
            for j in 0..self.neurons_per_layer[1] {
                weighted_sum += self.weights[2][i][j] * self.outputs[1][j];
            }

            // Activation
            self.outputs[2][i] = self.activation.forward(arr1(&[weighted_sum]))[0];
        }

        self.outputs[2].clone()
    }
}

/// # Safety
///
/// This function assumes that the pointers `neurons_per_layer` and `training_dataset`
/// are valid and that they point to arrays of `layers_count` and `rows` elements respectively.
/// Each element of the `training_dataset` should be a pointer to an array of `cols` elements.
/// The caller must ensure that these conditions are met to avoid undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn new_naive_rbf(
    input_neurons_count: usize,
    output_neurons_count: usize,
    activation: *const c_char,
    training_dataset: *const *const f64,
    rows: usize,
    cols: usize,
) -> *mut NaiveRBF {
    // Convert activation C string into Rust string
    let activation_str: &str = c_str_to_rust_str(activation);

    // Convert training_dataset to Vec<Vec<f64>>
    let mut training_dataset_vec: Vec<Vec<f64>> = Vec::with_capacity(rows);
    for i in 0..rows {
        let row_slice: &[f64] = unsafe { slice::from_raw_parts(*training_dataset.add(i), cols) };
        training_dataset_vec.push(row_slice.to_vec());
    }

    let naive_rbf: NaiveRBF = NaiveRBF::new(
        input_neurons_count,
        output_neurons_count,
        activation_str,
        training_dataset_vec,
    );
    let boxed_naive_rbf: Box<NaiveRBF> = Box::new(naive_rbf);

    Box::leak(boxed_naive_rbf)
}

/// # Safety
///
/// This function assumes that the pointers `training_dataset` and `labels` are valid and
/// point to arrays of `training_dataset_len` and `labels_len` elements respectively.
/// Each element of the `training_dataset` should be a pointer to an array of `samples_len` elements.
/// The caller must ensure that these conditions are met to avoid undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn fit_naive_rbf(
    naive_rbf_ptr: *mut NaiveRBF,
    training_dataset: *const *const f64,
    training_dataset_nrows: usize,
    training_dataset_ncols: usize,
    labels: *const *const f64,
    labels_nrows: usize,
    labels_ncols: usize,
    gamma: f64,
) {
    // Convert training_dataset to Vec<Vec<f64>>
    let mut training_dataset_vec: Vec<Vec<f64>> = Vec::with_capacity(training_dataset_nrows);
    for i in 0..training_dataset_nrows {
        let row_slice: &[f64] =
            unsafe { slice::from_raw_parts(*training_dataset.add(i), training_dataset_ncols) };
        training_dataset_vec.push(row_slice.to_vec());
    }

    // Convert labels to Vec<Vec<f64>>
    let mut labels_vec: Vec<Vec<f64>> = Vec::with_capacity(labels_nrows);
    for i in 0..labels_nrows {
        let row_slice: &[f64] = unsafe { slice::from_raw_parts(*labels.add(i), labels_ncols) };
        labels_vec.push(row_slice.to_vec());
    }

    if let Some(naive_rbf) = unsafe { naive_rbf_ptr.as_mut() } {
        naive_rbf.fit(training_dataset_vec, labels_vec, gamma);
    }
}

/// # Safety
///
/// This function assumes that the pointer `input` is valid and points to an array of `input_len` elements.
/// The caller must ensure that this condition is met to avoid undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn predict_naive_rbf(
    naive_rbf_ptr: *mut NaiveRBF,
    input: *const f64,
    input_len: usize,
) -> *const f64 {
    // Convert input to Vec<f64>
    let input_slice: &[f64] = unsafe { slice::from_raw_parts(input, input_len) };
    let input_vec: Vec<f64> = input_slice.to_vec();

    if let Some(naive_rbf) = unsafe { naive_rbf_ptr.as_mut() } {
        let output = naive_rbf.predict(input_vec);
        let output_ptr = output.as_ptr();
        std::mem::forget(output);
        output_ptr
    } else {
        ptr::null()
    }
}

/// # Safety
///
/// This function assumes that the pointer `naive_rbf_ptr` is valid and points to a valid `NaiveRBF` instance.
/// The caller must ensure that this condition is met to avoid undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn free_naive_rbf(naive_rbf_ptr: *mut NaiveRBF) {
    let _ = unsafe { Box::from_raw(naive_rbf_ptr) };
}
