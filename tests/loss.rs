use ai::loss::*;
use ndarray::array;

#[test]
fn categorical_crossentropy_forward() {
    let predictions = array![[0.1, 0.9], [0.8, 0.2]];
    let targets = array![[0.0, 1.0], [1.0, 0.0]];
    let loss_fn = CategoricalCrossentropy::new();
    let loss = loss_fn.forward(&predictions, &targets);
    let expected_loss = (-0.9f64.ln() - 0.8f64.ln()) / 2.0;
    assert!((loss - expected_loss).abs() < 1e-6);
}

#[test]
fn categorical_crossentropy_backward() {
    let predictions = array![[0.1, 0.9], [0.8, 0.2]];
    let targets = array![[0.0, 1.0], [1.0, 0.0]];
    let loss_fn = CategoricalCrossentropy::new();
    let grads = loss_fn.backward(&predictions, &targets);
    let expected_grads = array![[0.05, -0.05], [-0.1, 0.1]];
    assert!((grads - expected_grads).iter().all(|&x| x.abs() < 1e-6));
}
