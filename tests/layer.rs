use ai::activation::ActivationEnum;
use ai::layer::*;
use ndarray::{arr1, Array1};

#[test]
fn dense_new() {
    let units: usize = 10;
    let activation: ActivationEnum = ActivationEnum::Identity;
    let layer: Dense = Dense::new(units, activation);

    assert_eq!(layer.units, units);
    assert!(layer.weights.is_none());
    assert!(layer.output.is_none());
    assert!(layer.input.is_none());
    assert!(layer.weight_gradients.is_none());
    assert_eq!(layer.activation.as_str(), "Identity");
}

#[test]
fn dense_forward() {
    let units: usize = 2;
    let activation: ActivationEnum = ActivationEnum::ReLU;
    let mut layer: Dense = Dense::new(units, activation);

    let input: Array1<f64> = arr1(&[0., 0.]);
    let outputs: Array1<f64> = layer.forward(&input);

    assert_eq!(outputs.len(), units);
    assert!(layer.weights.is_some());
    assert!(layer.input.is_some());
    assert!(layer.output.is_some());
    for output in outputs {
        assert!(output >= 0.);
    }
}

#[test]
fn dense_backward() {
    let units: usize = 2;
    let activation: ActivationEnum = ActivationEnum::Logistic;
    let mut layer: Dense = Dense::new(units, activation);

    let input: Array1<f64> = arr1(&[0.5, -1.]);
    layer.forward(&input);

    // Arbitrary gradients for simplicity
    let gradients: Array1<f64> = arr1(&[0.1, -0.2]);
    let grad_input: Array1<f64> = layer.backward(&gradients);

    assert_eq!(grad_input.len(), input.len());
    assert!(layer.weight_gradients.is_some());
}
