use ai::loss::*;
use ndarray::array;

#[test]
fn categorical_cross_entropy_compute_loss() {
    let predictions = array![[0.1, 0.9], [0.8, 0.2]];
    let targets = array![[0.0, 1.0], [1.0, 0.0]];
    let loss_fn = CategoricalCrossEntropy::new();
    let loss = loss_fn.compute_loss(&predictions, &targets);
    let expected_loss = (-0.9f64.ln() - 0.8f64.ln()) / 2.0;
    assert!((loss - expected_loss).abs() < 1e-6);
}

#[test]
fn categorical_cross_entropy_compute_gradients() {
    let predictions = array![[0.1, 0.9], [0.8, 0.2]];
    let targets = array![[0.0, 1.0], [1.0, 0.0]];
    let loss_fn = CategoricalCrossEntropy::new();
    let grads = loss_fn.compute_gradients(&predictions, &targets);
    let expected_grads = array![[0.05, -0.05], [-0.1, 0.1]];
    assert!((grads - expected_grads).iter().all(|&x| x.abs() < 1e-6));
}

#[test]
fn mse_compute_loss() {
    let predictions = array![[0.1_f64, 0.9_f64], [0.8_f64, 0.2_f64]];
    let targets = array![[0.0_f64, 1.0_f64], [1.0_f64, 0.0_f64]];
    let loss_fn = MSE::new();
    let loss = loss_fn.compute_loss(&predictions, &targets);
    let expected_loss = ((0.1_f64 - 0.0_f64).powi(2)
        + (0.9_f64 - 1.0_f64).powi(2)
        + (0.8_f64 - 1.0_f64).powi(2)
        + (0.2_f64 - 0.0_f64).powi(2))
        / 4.0_f64;
    assert!((loss - expected_loss).abs() < 1e-6);
}

#[test]
fn mse_compute_gradients() {
    let predictions = array![[0.1_f64, 0.9_f64], [0.8_f64, 0.2_f64]];
    let targets = array![[0.0_f64, 1.0_f64], [1.0_f64, 0.0_f64]];
    let loss_fn = MSE::new();
    let grads = loss_fn.compute_gradients(&predictions, &targets);
    let expected_grads = array![
        [
            2.0 * (0.1_f64 - 0.0_f64) / 2.0_f64,
            2.0 * (0.9_f64 - 1.0_f64) / 2.0_f64
        ],
        [
            2.0 * (0.8_f64 - 1.0_f64) / 2.0_f64,
            2.0 * (0.2_f64 - 0.0_f64) / 2.0_f64
        ]
    ];
    assert!((grads - expected_grads).iter().all(|&x| x.abs() < 1e-6));
}
