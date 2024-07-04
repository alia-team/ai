extern crate rand;
extern crate libc;
use rand::{Rng, thread_rng};
use rand::seq::SliceRandom;
use std::f64;
use std::os::raw::c_double;
use std::fs::File;
use std::io::prelude::*;

#[repr(C)]
pub struct MLP {
    d: Vec<usize>,
    W: Vec<Vec<Vec<f64>>>,
    L: usize,
    X: Vec<Vec<f64>>,
    deltas: Vec<Vec<f64>>,
}

impl MLP {
    fn new(npl: Vec<usize>) -> MLP {
        let L = npl.len() - 1;
        let mut W = vec![];
        W.push(vec![vec![0.0]; npl[0] + 1]);
        for l in 1..=L {
            W.push(vec![]);
            for i in 0..=npl[l - 1] {
                W[l].push(vec![]);
                for j in 0..=npl[l] {
                    if j == 0 {
                        W[l][i].push(0.0)
                    } else {
                        W[l][i].push(rand::thread_rng().gen_range(-1.0..1.0));
                    }
                }
            }
        }
        let mut X = vec![];
        let mut deltas = vec![];
        for l in 0..=L {
            X.push(vec![]);
            deltas.push(vec![]);
            for j in 0..=npl[l] {
                if j == 0 {
                    X[l].push(1.0);
                } else {
                    X[l].push(0.0);
                }
                deltas[l].push(0.0);
            }
        }
        MLP {
            d: npl,
            W,
            L,
            X,
            deltas,
        }
    }

    #[no_mangle]
    pub extern "C" fn mlp_new(npl: *const usize, npl_len: usize) -> *mut MLP {
        let npl_vec: Vec<usize> = unsafe { std::slice::from_raw_parts(npl, npl_len).to_vec() };
        let mlp = Box::new(MLP::new(npl_vec));
        Box::into_raw(mlp)
    }

    fn propagate(&mut self, sample_inputs: Vec<f64>, is_classification: bool) {
        for j in 0..sample_inputs.len() {
            self.X[0][j + 1] = sample_inputs[j];
        }

        for l in 1..=self.L {
            for j in 1..=self.d[l] {
                let mut total = 0.0;
                for i in 0..=self.d[l - 1] {
                    total += self.W[l][i][j] * self.X[l - 1][i];
                }
                if is_classification || l < self.L {
                    total = total.tanh();
                }
                self.X[l][j] = total;
            }
        }
    }

    fn predict(&mut self, sample_inputs: Vec<f64>, is_classification: bool) -> Vec<f64> {
        self.propagate(sample_inputs, is_classification);
        self.X[self.L][1..].to_vec()
    }

    fn train(
        &mut self,
        all_samples_inputs: Vec<Vec<f64>>,
        all_samples_expected_outputs: Vec<Vec<f64>>,
        all_tests_inputs: Vec<Vec<f64>>,
        all_tests_expected_outputs: Vec<Vec<f64>>,
        alpha: f64,
        nb_iter: usize,
        is_classification: bool,
    ) -> Vec<Vec<f64>> {
        let mut loss_values: Vec<Vec<f64>> = vec![];
        let mut percent = 0.0;
        for iter in 0..nb_iter {
            if (iter as f64/nb_iter as f64)*100 as f64 > percent {
                println!("{}%", percent);
                percent+=1.0;
            }
            if iter % 10 == 0 {
                let mut total_squared_error_train = 0.0;
                let mut total_squared_error_test = 0.0;
                for iter_test in 0.. all_tests_inputs.len(){
                    let values = self.predict(all_tests_inputs[iter_test].clone(), is_classification);
                    for iter_values in 0.. values.len(){
                        total_squared_error_test+= (all_tests_expected_outputs[iter_test][iter_values]-values[iter_values]).powi(2);
                    }
                }
                total_squared_error_test= total_squared_error_test/(all_tests_inputs.len() as f64);
                for iter_train in 0.. all_samples_inputs.len(){
                    let values = self.predict(all_samples_inputs[iter_train].clone(), is_classification);
                    for iter_values in 0.. values.len(){
                        total_squared_error_train+= (all_samples_expected_outputs[iter_train][iter_values]-values[iter_values]).powi(2);
                    }
                }
                total_squared_error_train= total_squared_error_train/(all_samples_inputs.len() as f64);

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


                for j in 1..=self.d[self.L] {
                    self.deltas[self.L][j] = self.X[self.L][j] - sample_expected_outputs[j - 1];
                    if is_classification {
                        self.deltas[self.L][j] *= 1.0 - self.X[self.L][j].powi(2);
                    }
                }

                for l in (1..=self.L).rev() {
                    for i in 1..=self.d[l - 1] {
                        let mut total = 0.0;
                        for j in 1..=self.d[l] {
                            total += self.W[l][i][j] * self.deltas[l][j];
                        }
                        total *= 1.0 - self.X[l - 1][i].powi(2);
                        self.deltas[l - 1][i] = total;
                    }
                }

                for l in 1..=self.L {
                    for i in 0..=self.d[l - 1] {
                        for j in 1..=self.d[l] {
                            self.W[l][i][j] -= alpha * self.X[l - 1][i] * self.deltas[l][j];
                        }
                    }
                }
            }
            
        }
        loss_values
    }
}

#[no_mangle]
pub extern "C" fn mlp_predict(
    mlp: *mut MLP,
    sample_inputs: *const c_double,
    sample_inputs_len: usize,
    is_classification: bool,
) -> *mut c_double {
    let sample_inputs_vec: Vec<f64> =
        unsafe { std::slice::from_raw_parts(sample_inputs, sample_inputs_len).to_vec() };
    let mut mlp_ref = unsafe { &mut *mlp };
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

#[no_mangle]
pub extern "C" fn mlp_train(
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
            .map(|&output_ptr| std::slice::from_raw_parts(output_ptr, (*mlp).d[(*mlp).L]).to_vec())
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
            .map(|&output_ptr| std::slice::from_raw_parts(output_ptr, (*mlp).d[(*mlp).L]).to_vec())
            .collect()
    };
    let mut mlp_ref = unsafe { &mut *mlp };
    let loss_values = mlp_ref.train(
        all_samples_inputs_vec,
        all_samples_expected_outputs_vec,
        all_tests_inputs_vec,
        all_tests_expected_outputs_vec,
        alpha as f64,
        nb_iter,
        is_classification,
    );
    // Flatten the loss values
    let flat_loss_values: Vec<f64> = loss_values.iter().flat_map(|v| v.clone()).collect();

    // Allocate memory for the flat_loss_values and copy data
    let len = flat_loss_values.len();
    let inner_len = if loss_values.is_empty() { 0 } else { loss_values[0].len() };
    let loss_values_ptr = flat_loss_values.as_ptr();
    std::mem::forget(flat_loss_values); // Prevent Rust from freeing the vector

    TrainResult {
        loss_values_ptr: loss_values_ptr as *mut f64,
        len,
        inner_len,
    }
}
#[no_mangle]
pub extern "C" fn free_train_result(result: TrainResult) {
    unsafe {
        libc::free(result.loss_values_ptr as *mut libc::c_void);
    }
}
#[no_mangle]
pub extern "C" fn mlp_free(mlp: *mut MLP) {
    unsafe {
        let _ = Box::from_raw(mlp);
    }
}
