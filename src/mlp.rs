extern crate libc;
extern crate rand;
use libc::{c_char, size_t};
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use std::f64;
use std::ffi::CString;
use std::fs::File;
use std::os::raw::c_double;

use crate::util::c_str_to_rust_str;

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct MLP {
    neurons_per_layer: Vec<usize>,
    weights: Vec<Vec<Vec<f64>>>,
    n_layers: usize,
    outputs: Vec<Vec<f64>>,
    deltas: Vec<Vec<f64>>,
}

impl MLP {
    fn new(neurons_per_layer: Vec<usize>) -> MLP {
        let n_layers = neurons_per_layer.len() - 1;
        let mut weights = vec![];
        weights.push(vec![vec![0.0]; neurons_per_layer[0] + 1]);
        for l in 1..=n_layers {
            weights.push(vec![]);
            for i in 0..=neurons_per_layer[l - 1] {
                weights[l].push(vec![]);
                for j in 0..=neurons_per_layer[l] {
                    if j == 0 {
                        weights[l][i].push(0.0)
                    } else {
                        weights[l][i].push(thread_rng().gen_range(-1.0..1.0));
                    }
                }
            }
        }
        let mut outputs = vec![];
        let mut deltas = vec![];
        for l in 0..=n_layers {
            outputs.push(vec![]);
            deltas.push(vec![]);
            for j in 0..=neurons_per_layer[l] {
                if j == 0 {
                    outputs[l].push(1.0);
                } else {
                    outputs[l].push(0.0);
                }
                deltas[l].push(0.0);
            }
        }

        MLP {
            neurons_per_layer,
            weights,
            n_layers,
            outputs,
            deltas,
        }
    }

    pub fn save(&self, path: &str, model_name: &str) -> String {
        let full_path: String = format!("{}{}.json", path, model_name);
        let model_file = std::fs::File::create(full_path.clone()).unwrap();
        serde_json::to_writer(model_file, &self).unwrap();

        full_path
    }

    pub fn load(model_file_name: &str) -> MLP {
        let model_file = File::open(model_file_name).unwrap();
        let model: MLP = serde_json::from_reader(model_file).unwrap();

        model
    }

    /// # Safety
    ///
    /// This function is unsafe because it dereferences raw pointers.
    /// The caller must ensure that:
    ///
    /// - `npl` is a valid pointer to an array of `usize` values
    /// - `npl_len` accurately represents the length of the array pointed to by `npl`
    /// - The returned pointer must be properly managed and eventually freed using `mlp_free`
    #[no_mangle]
    pub unsafe extern "C" fn mlp_new(npl: *const usize, npl_len: usize) -> *mut MLP {
        let npl_vec: Vec<usize> = unsafe { std::slice::from_raw_parts(npl, npl_len).to_vec() };
        let mlp = Box::new(MLP::new(npl_vec));
        Box::into_raw(mlp)
    }

    fn propagate(&mut self, sample_inputs: Vec<f64>, is_classification: bool) {
        self.outputs[0][1..(sample_inputs.len() + 1)].copy_from_slice(&sample_inputs[..]);

        for l in 1..=self.n_layers {
            for j in 1..=self.neurons_per_layer[l] {
                let mut total = 0.0;
                for i in 0..=self.neurons_per_layer[l - 1] {
                    total += self.weights[l][i][j] * self.outputs[l - 1][i];
                }
                if is_classification || l < self.n_layers {
                    total = total.tanh();
                }
                self.outputs[l][j] = total;
            }
        }
    }

    fn predict(&mut self, sample_inputs: Vec<f64>, is_classification: bool) -> Vec<f64> {
        self.propagate(sample_inputs, is_classification);
        self.outputs[self.n_layers][1..].to_vec()
    }

    fn train(
        &mut self,
        all_samples_inputs: Vec<Vec<f64>>,
        all_samples_expected_outputs: Vec<Vec<f64>>,
        all_tests_inputs: Vec<Vec<f64>>,
        all_tests_expected_outputs: Vec<Vec<f64>>,
        alpha: f64,
        epochs: usize,
        is_classification: bool,
    ) -> Vec<Vec<f64>> {
        let mut loss_values: Vec<Vec<f64>> = vec![];
        let mut percent = 0.0;
        for epoch in 0..epochs {
            if (epoch as f64 / epochs as f64) * 100.0 > percent {
                println!("{}%", percent);
                percent += 1.0;
            }
            if epoch % 10 == 0 {
                let mut total_squared_error_train = 0.0;
                let mut total_squared_error_test = 0.0;
                for iter_test in 0..all_tests_inputs.len() {
                    let values =
                        self.predict(all_tests_inputs[iter_test].clone(), is_classification);
                    for (i, &value) in values.iter().enumerate() {
                        total_squared_error_test +=
                            (all_tests_expected_outputs[iter_test][i] - value).powi(2);
                    }
                }
                total_squared_error_test /= all_tests_inputs.len() as f64;
                for iter_train in 0..all_samples_inputs.len() {
                    let values =
                        self.predict(all_samples_inputs[iter_train].clone(), is_classification);
                    for (i, &value) in values.iter().enumerate() {
                        total_squared_error_train +=
                            (all_samples_expected_outputs[iter_train][i] - value).powi(2);
                    }
                }
                total_squared_error_train /= all_samples_inputs.len() as f64;

                loss_values.push(vec![total_squared_error_train, total_squared_error_test]);
            }

            let mut order = (0..all_samples_inputs.len()).collect::<Vec<usize>>();
            let mut rng = thread_rng();
            order.shuffle(&mut rng);

            for order_index in order.iter() {
                let k = *order_index;
                let sample_inputs = &all_samples_inputs[k];
                let sample_expected_outputs = &all_samples_expected_outputs[k];

                self.propagate(sample_inputs.clone(), is_classification);

                for j in 1..=self.neurons_per_layer[self.n_layers] {
                    self.deltas[self.n_layers][j] =
                        self.outputs[self.n_layers][j] - sample_expected_outputs[j - 1];
                    if is_classification {
                        self.deltas[self.n_layers][j] *=
                            1.0 - self.outputs[self.n_layers][j].powi(2);
                    }
                }

                for l in (1..=self.n_layers).rev() {
                    for i in 1..=self.neurons_per_layer[l - 1] {
                        let mut total = 0.0;
                        for j in 1..=self.neurons_per_layer[l] {
                            total += self.weights[l][i][j] * self.deltas[l][j];
                        }
                        total *= 1.0 - self.outputs[l - 1][i].powi(2);
                        self.deltas[l - 1][i] = total;
                    }
                }

                for l in 1..=self.n_layers {
                    for i in 0..=self.neurons_per_layer[l - 1] {
                        for j in 1..=self.neurons_per_layer[l] {
                            self.weights[l][i][j] -=
                                alpha * self.outputs[l - 1][i] * self.deltas[l][j];
                        }
                    }
                }
            }
        }
        loss_values
    }
}

/// # Safety
///
/// This function is unsafe because it dereferences raw pointers.
/// The caller must ensure that:
///
/// - `mlp` is a valid pointer to an MLP struct
/// - `sample_inputs` is a valid pointer to an array of `sample_inputs_len` f64 values
#[no_mangle]
pub unsafe extern "C" fn mlp_predict(
    mlp: *mut MLP,
    sample_inputs: *const c_double,
    sample_inputs_len: usize,
    is_classification: bool,
) -> *mut c_double {
    let sample_inputs_vec: Vec<f64> =
        unsafe { std::slice::from_raw_parts(sample_inputs, sample_inputs_len).to_vec() };
    let mlp_ref = unsafe { &mut *mlp };
    let mut output = mlp_ref.predict(sample_inputs_vec, is_classification);
    let output_ptr = output.as_mut_ptr();
    std::mem::forget(output);
    output_ptr
}

#[repr(C)]
pub struct TrainResult {
    loss_values_ptr: *mut f64,
    len: usize,
    inner_len: usize,
}

/// # Safety
///
/// This function is unsafe because it dereferences raw pointers.
/// The caller must ensure that:
///
/// - `mlp` is a valid pointer to an MLP struct
/// - `all_samples_inputs`, `all_samples_expected_outputs`, `all_tests_inputs`, and `all_tests_expected_outputs`
///   are valid pointers to arrays of pointers, each pointing to valid f64 arrays
/// - The lengths provided (`samples_count`, `sample_inputs_len`, `tests_count`, `test_inputs_len`)
///   accurately represent the sizes of the respective arrays
#[no_mangle]
pub unsafe extern "C" fn mlp_train(
    mlp: *mut MLP,
    all_samples_inputs: *const *const c_double,
    all_samples_expected_outputs: *const *const c_double,
    all_tests_inputs: *const *const c_double,
    all_tests_expected_outputs: *const *const c_double,
    samples_count: usize,
    sample_inputs_len: usize,
    tests_count: usize,
    test_inputs_len: usize,
    alpha: c_double,
    nb_iter: usize,
    is_classification: bool,
) -> TrainResult {
    // SAMPLE
    let all_samples_inputs_vec: Vec<Vec<f64>> = unsafe {
        std::slice::from_raw_parts(all_samples_inputs, samples_count)
            .iter()
            .map(|&input_ptr| std::slice::from_raw_parts(input_ptr, sample_inputs_len).to_vec())
            .collect()
    };
    let all_samples_expected_outputs_vec: Vec<Vec<f64>> = unsafe {
        std::slice::from_raw_parts(all_samples_expected_outputs, samples_count)
            .iter()
            .map(|&output_ptr| {
                std::slice::from_raw_parts(output_ptr, (*mlp).neurons_per_layer[(*mlp).n_layers])
                    .to_vec()
            })
            .collect()
    };
    // TEST
    let all_tests_inputs_vec: Vec<Vec<f64>> = unsafe {
        std::slice::from_raw_parts(all_tests_inputs, tests_count)
            .iter()
            .map(|&input_ptr| std::slice::from_raw_parts(input_ptr, test_inputs_len).to_vec())
            .collect()
    };
    let all_tests_expected_outputs_vec: Vec<Vec<f64>> = unsafe {
        std::slice::from_raw_parts(all_tests_expected_outputs, tests_count)
            .iter()
            .map(|&output_ptr| {
                std::slice::from_raw_parts(output_ptr, (*mlp).neurons_per_layer[(*mlp).n_layers])
                    .to_vec()
            })
            .collect()
    };
    let mlp_ref = unsafe { &mut *mlp };
    let loss_values = mlp_ref.train(
        all_samples_inputs_vec,
        all_samples_expected_outputs_vec,
        all_tests_inputs_vec,
        all_tests_expected_outputs_vec,
        alpha,
        nb_iter,
        is_classification,
    );
    // Flatten the loss values
    let flat_loss_values: Vec<f64> = loss_values.iter().flat_map(|v| v.clone()).collect();

    // Allocate memory for the flat_loss_values and copy data
    let len = flat_loss_values.len();
    let inner_len = if loss_values.is_empty() {
        0
    } else {
        loss_values[0].len()
    };
    let loss_values_ptr = flat_loss_values.as_ptr();
    std::mem::forget(flat_loss_values); // Prevent Rust from freeing the vector

    TrainResult {
        loss_values_ptr: loss_values_ptr as *mut f64,
        len,
        inner_len,
    }
}

#[no_mangle]
pub unsafe extern "C" fn mlp_neurons_per_layer(model_ptr: *mut MLP) -> *const size_t {
    let mlp: &mut MLP = unsafe { model_ptr.as_mut().expect("Null model pointer.") };
    mlp.neurons_per_layer.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn mlp_nlayers(model_ptr: *mut MLP) -> size_t {
    let mlp: &mut MLP = unsafe { model_ptr.as_mut().expect("Null model pointer.") };
    mlp.n_layers + 1
}

#[no_mangle]
pub extern "C" fn free_train_result(result: TrainResult) {
    unsafe {
        libc::free(result.loss_values_ptr as *mut libc::c_void);
    }
}

/// # Safety
///
/// This function is unsafe because it dereferences raw pointers.
/// The caller must ensure that:
/// - `model_ptr`, `path`, and `model_name` are valid, non-null pointers to null-terminated C strings
/// - The returned pointer must be freed by the caller using an appropriate deallocation function
#[no_mangle]
pub unsafe extern "C" fn mlp_save(
    model_ptr: *mut MLP,
    path: *const c_char,
    model_name: *const c_char,
) -> *const c_char {
    let model = unsafe { model_ptr.as_mut().expect("Null model pointer.") };
    let path: &str = c_str_to_rust_str(path);
    let model_name: &str = c_str_to_rust_str(model_name);
    let full_path: CString =
        CString::new(model.save(path, model_name)).expect("Failed to convert Rust str to C str.");
    full_path.into_raw()
}

/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that:
/// - `model_path` is a valid, non-null pointer to a null-terminated C string
/// - The returned pointer must be freed using `free_cnn` to avoid memory leaks
#[no_mangle]
pub unsafe extern "C" fn mlp_load(model_path: *const c_char) -> *mut MLP {
    Box::leak(Box::new(MLP::load(c_str_to_rust_str(model_path))))
}

/// # Safety
///
/// This function is unsafe because it deallocates memory pointed to by a raw pointer.
/// The caller must ensure that:
///
/// - `mlp` is a valid pointer to an MLP struct that was previously created by this library
/// - `mlp` is not used after this function call
#[no_mangle]
pub unsafe extern "C" fn mlp_free(mlp: *mut MLP) {
    unsafe {
        let _ = Box::from_raw(mlp);
    }
}
