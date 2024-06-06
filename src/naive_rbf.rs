use std::{ptr, slice};

use crate::activation::sign;
use crate::utils;
extern crate rand;
use nalgebra::*;
use rand::Rng;

#[derive(Debug, PartialEq)]
pub struct Center {
    pub coordinates: Vec<f64>,
}

impl Center {
    pub fn new(coordinates: Vec<f64>) -> Self {
        Center { coordinates }
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
    pub centers: Vec<Center>,
    pub weights: Vec<Vec<Vec<f64>>>,
    pub outputs: Vec<Vec<f64>>,
    pub gamma: f64,
    pub is_classification: bool,
}

impl NaiveRBF {
    pub fn new(
        neurons_per_layer: Vec<usize>,
        is_classification: bool,
        training_dataset: Vec<Vec<f64>>,
    ) -> Self {
        if neurons_per_layer.len() != 3 {
            panic!("A RBF neural network must contain only 3 layers.")
        }

        // Initialize centers
        let mut centers: Vec<Center> = vec![];
        for _ in 0..neurons_per_layer[1] {
            centers.push(Center::new(
                training_dataset[rand::thread_rng().gen_range(0..training_dataset.len())].clone(),
            ));
        }

        let weights = utils::init_weights(neurons_per_layer.clone(), true);
        let outputs = utils::init_outputs(neurons_per_layer.clone(), true);
        let gamma = rand::thread_rng().gen_range(0.01..=1.0);

        NaiveRBF {
            neurons_per_layer,
            centers,
            weights,
            outputs,
            gamma,
            is_classification,
        }
    }

    pub fn fit(&mut self, training_dataset: Vec<Vec<f64>>, labels: Vec<f64>) {
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
        let y = DVector::from_vec(labels);

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

        // Forward pass in hidden layer
        for center in &self.centers {
            self.outputs[1].push(center.forward(input.clone(), self.gamma))
        }

        // Forward pass in output layer
        for i in 0..self.neurons_per_layer[2] {
            let weighted_sum: f64 = self.weights[2][i]
                .iter()
                .zip(input.clone())
                .map(|(w, x)| w * x)
                .sum();

            // Activation
            self.outputs[2][i] = match self.is_classification {
                true => sign(weighted_sum),
                false => weighted_sum,
            }
        }

        self.outputs[2].clone()
    }
}

#[no_mangle]
pub unsafe extern "C" fn new_naive_rbf(
    neurons_per_layer: *const usize,
    layers_count: usize,
    is_classification: bool,
    training_dataset: *const *const f64,
    rows: usize,
    cols: usize,
) -> *mut NaiveRBF {
    // Convert neurons_per_layer to Vec<usize>
    let npl_slice: &[usize] = unsafe { slice::from_raw_parts(neurons_per_layer, layers_count) };
    let npl_vec: Vec<usize> = npl_slice.to_vec();

    // Convert training_dataset to Vec<Vec<f64>>
    let mut training_dataset_vec: Vec<Vec<f64>> = Vec::with_capacity(rows);
    for i in 0..rows {
        let row_slice: &[f64] = unsafe { slice::from_raw_parts(*training_dataset.add(i), cols) };
        training_dataset_vec.push(row_slice.to_vec());
    }

    let naive_rbf: NaiveRBF = NaiveRBF::new(npl_vec, is_classification, training_dataset_vec);
    let boxed_naive_rbf: Box<NaiveRBF> = Box::new(naive_rbf);

    Box::leak(boxed_naive_rbf)
}

#[no_mangle]
pub unsafe extern "C" fn fit_naive_rbf(
    naive_rbf_ptr: *mut NaiveRBF,
    training_dataset: *const *const f64,
    training_dataset_len: usize,
    samples_len: usize,
    labels: *const f64,
    labels_len: usize,
) {
    // Convert training_dataset to Vec<Vec<f64>>
    let mut training_dataset_vec: Vec<Vec<f64>> = Vec::with_capacity(training_dataset_len);
    for i in 0..training_dataset_len {
        let row_slice: &[f64] =
            unsafe { slice::from_raw_parts(*training_dataset.add(i), samples_len) };
        training_dataset_vec.push(row_slice.to_vec());
    }

    // Convert labels to Vec<f64>
    let labels_slice: &[f64] = unsafe { slice::from_raw_parts(labels, labels_len) };
    let labels_vec: Vec<f64> = labels_slice.to_vec();

    if let Some(naive_rbf) = unsafe { naive_rbf_ptr.as_mut() } {
        naive_rbf.fit(training_dataset_vec, labels_vec);
    }
}

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
        naive_rbf.predict(input_vec).as_ptr()
    } else {
        ptr::null()
    }
}
