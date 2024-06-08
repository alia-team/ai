extern crate rand;
use crate::activation::sign;
use crate::utils;
use nalgebra::DMatrix;
use rand::Rng;
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

pub struct RBF {
    pub neurons_per_layer: Vec<usize>,
    pub centroids: Vec<Centroid>,
    pub weights: Vec<Vec<Vec<f64>>>,
    pub outputs: Vec<Vec<f64>>,
    pub gamma: f64,
    pub is_classification: bool,
}

impl RBF {
    pub fn new(
        neurons_per_layer: Vec<usize>,
        is_classification: bool,
        training_dataset: Vec<Vec<f64>>,
    ) -> Self {
        if neurons_per_layer.len() != 3 {
            panic!("A RBF neural network must contain only 3 layers.")
        }

        let mut rng = rand::thread_rng();
        let mut centroids: Vec<Centroid> = vec![];
        for _ in 0..neurons_per_layer[1] {
            let index = rng.gen_range(0..training_dataset.len());
            centroids.push(Centroid::new(training_dataset[index].clone()));
        }

        let weights = utils::init_weights(neurons_per_layer.clone(), true);
        let outputs = utils::init_outputs(neurons_per_layer.clone(), true);
        let gamma = rand::thread_rng().gen_range(0.01..=1.0);

        RBF {
            neurons_per_layer,
            centroids,
            weights,
            outputs,
            gamma,
            is_classification,
        }
    }

    pub fn fit(
        &mut self,
        training_dataset: Vec<Vec<f64>>,
        labels: Vec<Vec<f64>>,
        gamma: f64,
        max_iterations: usize,
    ) {
        self.gamma = gamma;
        self._lloyds_algorithm(training_dataset.clone(), max_iterations);
        self._update_weights(training_dataset, labels);
    }

    pub fn predict(&mut self, input: Vec<f64>) -> Vec<f64> {
        self.outputs[0] = input.clone();

        // Forward pass in hidden layer
        for centroid in &self.centroids {
            self.outputs[1].push(centroid.forward(input.clone(), self.gamma))
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

    pub fn _lloyds_algorithm(&mut self, training_dataset: Vec<Vec<f64>>, max_iterations: usize) {
        let k = self.centroids.len();
        for _ in 0..max_iterations {
            // Assignment step
            let mut clusters: Vec<Vec<Vec<f64>>> = vec![vec![]; k];
            for point in training_dataset.clone() {
                let mut min_dist = f64::MAX;
                let mut min_index = 0;
                for (i, centroid) in self.centroids.iter().enumerate() {
                    let dist =
                        utils::euclidean_distance(point.clone(), centroid.coordinates.clone());
                    if dist < min_dist {
                        min_dist = dist;
                        min_index = i;
                    }
                }
                clusters[min_index].push(point.clone());
            }

            // Update step
            for (i, cluster) in clusters.iter().enumerate() {
                if !cluster.is_empty() {
                    let new_centroid = utils::compute_centroid(cluster.clone());
                    self.centroids[i] = Centroid::new(new_centroid);
                }
            }
        }
    }

    pub fn _update_weights(&mut self, training_dataset: Vec<Vec<f64>>, labels: Vec<Vec<f64>>) {
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

        let phi_transpose = phi.clone().transpose();

        // Invert Phi matrixes
        let mut inv = phi_transpose.clone() * phi;
        inv = match inv.clone().try_inverse() {
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
        let weights_matrix = inv * phi_transpose * y;

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
}

/// # Safety
///
/// This function assumes that the pointers `neurons_per_layer` and
/// `training_dataset` are valid and that they point to arrays of `layers_count`
/// and `training_dataset_nrows` elements respectively.
/// Each element of the `training_dataset` should be a pointer to an array of
/// `training_dataset_ncols` elements.
/// The caller must ensure that these conditions are met to avoid undefined
/// behavior.
#[no_mangle]
pub unsafe extern "C" fn new_rbf(
    neurons_per_layer: *const usize,
    layers_count: usize,
    is_classification: bool,
    training_dataset: *const *const f64,
    training_dataset_nrows: usize,
    training_dataset_ncols: usize,
) -> *mut RBF {
    // Convert neurons_per_layer to Vec<usize>
    let npl_slice: &[usize] = unsafe { slice::from_raw_parts(neurons_per_layer, layers_count) };
    let npl_vec: Vec<usize> = npl_slice.to_vec();

    // Convert training_dataset to Vec<Vec<f64>>
    let mut training_dataset_vec: Vec<Vec<f64>> = Vec::with_capacity(training_dataset_nrows);
    for i in 0..training_dataset_nrows {
        let row_slice: &[f64] =
            unsafe { slice::from_raw_parts(*training_dataset.add(i), training_dataset_ncols) };
        training_dataset_vec.push(row_slice.to_vec());
    }

    let rbf: RBF = RBF::new(npl_vec, is_classification, training_dataset_vec);
    let boxed_rbf: Box<RBF> = Box::new(rbf);

    Box::leak(boxed_rbf)
}

/// # Safety
///
/// This function assumes that the pointers `training_dataset` and `labels` are
/// valid and point to arrays of `training_dataset_nrows` and `labels_len`
/// elements respectively.
/// Each element of the `training_dataset` should be a pointer to an array of
/// `training_dataset_ncols` elements.
/// The caller must ensure that these conditions are met to avoid undefined
/// behavior.
#[no_mangle]
pub unsafe extern "C" fn fit_rbf(
    rbf_ptr: *mut RBF,
    training_dataset: *const *const f64,
    training_dataset_nrows: usize,
    training_dataset_ncols: usize,
    labels: *const *const f64,
    labels_nrows: usize,
    labels_ncols: usize,
    gamma: f64,
    max_iterations: usize,
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

    if let Some(rbf) = unsafe { rbf_ptr.as_mut() } {
        rbf.fit(training_dataset_vec, labels_vec, gamma, max_iterations);
    }
}

/// # Safety
///
/// This function assumes that the pointers `rbf_ptr` and `input` are valid and
/// points to a valid `RBF` instance and to an array of `input_len` elements
/// respectively.
/// The caller must ensure that these conditions are met to avoid undefined
/// behavior.
#[no_mangle]
pub unsafe extern "C" fn predict_rbf(
    rbf_ptr: *mut RBF,
    input: *const f64,
    input_len: usize,
) -> *const f64 {
    // Convert input to Vec<f64>
    let input_slice: &[f64] = unsafe { slice::from_raw_parts(input, input_len) };
    let input_vec: Vec<f64> = input_slice.to_vec();

    if let Some(rbf) = unsafe { rbf_ptr.as_mut() } {
        rbf.predict(input_vec).as_ptr()
    } else {
        ptr::null()
    }
}

/// # Safety
///
/// This function assumes that the pointer `rbf_ptr` is valid and points to a
/// valid `RBF` instance.
/// The caller must ensure that this condition is met to avoid undefined
/// behavior.
#[no_mangle]
pub unsafe extern "C" fn free_rbf(rbf_ptr: *mut RBF) {
    let _ = unsafe { Box::from_raw(rbf_ptr) };
}
