extern crate rand;
use rand::Rng;
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
        alpha: f64,
        nb_iter: usize,
        is_classification: bool,
    ) {
        for _ in 0..nb_iter {
            let k = rand::thread_rng().gen_range(0..all_samples_inputs.len());
            let sample_inputs = &all_samples_inputs[k];
            let sample_expected_outputs = &all_samples_expected_outputs[k];

            self.propagate(sample_inputs.clone(), is_classification);

            for j in 1..=self.d[self.L] {
                self.deltas[self.L][j] = self.X[self.L][j] - sample_expected_outputs[j - 1];
                if is_classification {
                    self.deltas[self.L][j] *= 1.0 - self.X[self.L][j].powi(2);
                }
            }

            for l in (2..self.L).rev() {
                for i in 1..=self.d[l - 1] {
                    let mut total = 0.0;
                    for j in 1..=self.d[l] {
                        total += self.W[l][i][j] * self.X[l - 1][i];
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

#[no_mangle]
pub extern "C" fn mlp_train(
    mlp: *mut MLP,
    all_samples_inputs: *const *const c_double,
    all_samples_expected_outputs: *const *const c_double,
    samples_count: usize,
    sample_inputs_len: usize,
    alpha: c_double,
    nb_iter: usize,
    is_classification: bool,
) {
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

    let mut mlp_ref = unsafe { &mut *mlp };
    mlp_ref.train(
        all_samples_inputs_vec,
        all_samples_expected_outputs_vec,
        alpha as f64,
        nb_iter,
        is_classification,
    );

    save_weights(&mlp_ref, "weights.txt").unwrap();
}

#[no_mangle]
pub extern "C" fn mlp_free(mlp: *mut MLP) {
    unsafe {
        let _ = Box::from_raw(mlp);
    }
}

fn save_weights(model: &MLP, filename: &str) -> std::io::Result<()> {
    let mut file = File::create(filename)?;

    writeln!(file, "{} {}", model.L, model.d[0])?;

    for l in 1..model.L {
        for j in 0..model.d[l] {
            for i in 0..model.d[l - 1] {
                writeln!(file, "{}", model.W[l - 1][j][i])?;
            }
            writeln!(file, "{}", model.W[l][model.d[l - 1]][j])?;
        }
    }

    Ok(())
}
