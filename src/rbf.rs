extern crate rand;
use crate::activation::{string_to_activation, Activation};
use crate::utils::{self, c_str_to_rust_str};
use nalgebra::DMatrix;
use ndarray::arr1;
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashMap;
use std::ffi::c_char;
use std::{ptr, slice}; // Importing IteratorRandom trait

#[derive(Debug, PartialEq)]
pub struct Centroid {
    pub coordinates: Vec<f64>,
}

impl Centroid {
    pub fn new(coordinates: Vec<f64>) -> Self {
        Centroid { coordinates }
    }

    pub fn forward(&self, input: &[f64], gamma: f64) -> f64 {
        let vec_sub: Vec<f64> = input
            .iter()
            .zip(&self.coordinates)
            .map(|(i, &c)| i - c)
            .collect();

        let norm: f64 = vec_sub.iter().map(|&v| v.powi(2)).sum::<f64>().sqrt();

        (-gamma * norm.powi(2)).exp()
    }
}

pub struct RBF {
    pub neurons_per_layer: Vec<usize>,
    pub centroids: Vec<Centroid>,
    pub weights: Vec<Vec<Vec<f64>>>,
    pub outputs: Vec<Vec<f64>>,
    pub gamma: f64,
    pub activation: Box<dyn Activation>,
    pub labels: Vec<Vec<f64>>,
}

impl RBF {
    pub fn new(
        neurons_per_layer: &[usize],
        activation: &str,
        training_dataset: &[Vec<f64>],
        labels: &[Vec<f64>],
    ) -> Self {
        if neurons_per_layer.len() != 3 {
            panic!("A RBF neural network must contain only 3 layers.");
        }
        if neurons_per_layer[1] > training_dataset.len() {
            panic!(
                "Cannot have {} centroids for {} samples in dataset.",
                neurons_per_layer[1],
                training_dataset.len()
            );
        }

        let activation_struct: Box<dyn Activation> = string_to_activation(activation);

        let mut rng = rand::thread_rng();
        let mut centroids: Vec<Centroid> = Vec::with_capacity(neurons_per_layer[1]);
        let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();

        // Organize indices by class
        for (index, label) in labels.iter().enumerate() {
            let class = label[0] as i32; // Assuming labels are one-dimensional and castable to i32
            class_indices.entry(class).or_default().push(index);
        }

        // Calculate the number of centroids per class
        let num_classes = class_indices.len();
        let centroids_per_class = neurons_per_layer[1] / num_classes;

        // Randomly select centroids from each class and sort them
        let mut selected_indices: Vec<usize> = Vec::with_capacity(neurons_per_layer[1]);
        for indices in class_indices.values() {
            let mut class_selected_indices: Vec<usize> = indices
                .choose_multiple(&mut rng, centroids_per_class)
                .cloned()
                .collect();
            class_selected_indices.sort_unstable();
            selected_indices.extend(class_selected_indices);
        }

        // If there's any remaining centroids to select (due to integer division rounding)
        if selected_indices.len() < neurons_per_layer[1] {
            let remaining = neurons_per_layer[1] - selected_indices.len();
            let mut remaining_indices: Vec<usize> = (0..training_dataset.len())
                .filter(|index| !selected_indices.contains(index))
                .collect();
            remaining_indices.shuffle(&mut rng);
            selected_indices.extend(remaining_indices.into_iter().take(remaining));
            selected_indices.sort_unstable();
        }

        // Create centroids from the selected indices
        for &index in &selected_indices {
            centroids.push(Centroid::new(training_dataset[index].clone()));
        }

        let weights = utils::init_weights(neurons_per_layer.to_vec(), true);
        let outputs = utils::init_outputs(neurons_per_layer.to_vec(), true);
        let gamma = rand::thread_rng().gen_range(0.01..=1.0);

        RBF {
            neurons_per_layer: neurons_per_layer.to_vec(),
            centroids,
            weights,
            outputs,
            gamma,
            activation: activation_struct,
            labels: labels.to_vec(),
        }
    }

    pub fn fit(&mut self, training_dataset: &[Vec<f64>], gamma: f64, max_iterations: usize) {
        self.gamma = gamma;
        self._lloyds_algorithm(training_dataset, max_iterations);
        let labels_clone = self.labels.clone();
        self._update_weights(training_dataset, &labels_clone);
    }

    pub fn predict(&mut self, input: &[f64]) -> Vec<f64> {
        self.outputs[0] = input.to_vec();

        // Reset hidden layer's outputs
        self.outputs[1].clear();

        // Forward pass in hidden layer
        for centroid in &self.centroids {
            self.outputs[1].push(centroid.forward(input, self.gamma))
        }

        let mut weighted_sum: f64 = 0.0;

        // Forward pass in output layer
        for i in 0..self.neurons_per_layer[2] {
            for j in 0..self.neurons_per_layer[1] {
                weighted_sum += self.weights[2][i][j] * self.outputs[1][j];
            }

            // Activation
            self.outputs[2][i] = self.activation.forward(&arr1(&[weighted_sum]))[0];
        }

        self.outputs[2].clone()
    }

    pub fn _lloyds_algorithm(&mut self, training_dataset: &[Vec<f64>], max_iterations: usize) {
        let k = self.centroids.len();
        for _ in 0..max_iterations {
            // Assignment step
            let mut clusters: Vec<Vec<Vec<f64>>> = vec![vec![]; k];
            for point in training_dataset.iter() {
                let mut min_dist = f64::MAX;
                let mut min_index = 0;
                for (i, centroid) in self.centroids.iter().enumerate() {
                    let dist = utils::euclidean_distance(point, &centroid.coordinates);
                    if dist < min_dist {
                        min_dist = dist;
                        min_index = i;
                    }
                }
                clusters[min_index].push(point.clone());
            }

            // Update step
            for (j, cluster) in clusters.iter().enumerate() {
                if !cluster.is_empty() {
                    let new_centroid = utils::compute_centroid(cluster);
                    self.centroids[j] = Centroid::new(new_centroid);
                }
            }
        }
    }

    pub fn _update_weights(&mut self, training_dataset: &[Vec<f64>], labels: &[Vec<f64>]) {
        let n_samples = training_dataset.len();
        let n_centroids: usize = self.centroids.len();
        let feature_len = training_dataset[0].len();

        // Construct Phi matrix
        let mut phi = DMatrix::zeros(n_samples, n_centroids);
        for i in 0..n_samples {
            for j in 0..n_centroids {
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

        // Invert Phi matrix
        let temp = phi_transpose.clone() * phi;
        let inv = match temp.clone().try_inverse() {
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
        let phis = inv * phi_transpose;
        let weights_matrix = phis * y;

        // Convert the DMatrix to Vec<Vec<f64>>
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
    activation: *const c_char,
    training_dataset: *const *const f64,
    training_dataset_nrows: usize,
    training_dataset_ncols: usize,
    labels: *const *const f64,
    labels_nrows: usize,
    labels_ncols: usize,
) -> *mut RBF {
    // Convert neurons_per_layer to &[usize]
    let npl_slice: &[usize] = unsafe { slice::from_raw_parts(neurons_per_layer, layers_count) };

    // Convert activation C string into Rust string
    let activation_str: &str = c_str_to_rust_str(activation);

    // Convert training_dataset to Vec<Vec<f64>>
    let training_dataset_vec: Vec<Vec<f64>> = (0..training_dataset_nrows)
        .map(|i| {
            let row_slice: &[f64] =
                unsafe { slice::from_raw_parts(*training_dataset.add(i), training_dataset_ncols) };
            row_slice.to_vec()
        })
        .collect();

    // Convert labels to Vec<Vec<f64>>
    let labels_vec: Vec<Vec<f64>> = (0..labels_nrows)
        .map(|i| {
            let row_slice: &[f64] = unsafe { slice::from_raw_parts(*labels.add(i), labels_ncols) };
            row_slice.to_vec()
        })
        .collect();

    let rbf = RBF::new(
        npl_slice,
        activation_str,
        &training_dataset_vec,
        &labels_vec,
    );
    let boxed_rbf = Box::new(rbf);

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
    gamma: f64,
    max_iterations: usize,
) {
    // Convert training_dataset to Vec<Vec<f64>>
    let training_dataset_vec: Vec<Vec<f64>> = (0..training_dataset_nrows)
        .map(|i| {
            let row_slice: &[f64] =
                unsafe { slice::from_raw_parts(*training_dataset.add(i), training_dataset_ncols) };
            row_slice.to_vec()
        })
        .collect();

    if let Some(rbf) = unsafe { rbf_ptr.as_mut() } {
        rbf.fit(&training_dataset_vec, gamma, max_iterations);
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
    // Convert input to &[f64]
    let input_slice: &[f64] = unsafe { slice::from_raw_parts(input, input_len) };

    if let Some(rbf) = unsafe { rbf_ptr.as_mut() } {
        let output = rbf.predict(input_slice);
        let output_ptr = output.as_ptr();
        std::mem::forget(output);
        output_ptr
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
