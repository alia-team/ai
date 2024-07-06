use ai::activation::{enum_to_activation, ActivationEnum};
use ndarray::arr1;

#[test]
fn sign_forward() {
    let mut activation = enum_to_activation(ActivationEnum::Sign);
    let input = arr1(&[-1.0, 0.0, 1.0]);
    let output = activation.forward(&input);
    assert_eq!(output, arr1(&[-1.0, 1.0, 1.0]));
}

#[test]
fn sign_backward() {
    let mut activation = enum_to_activation(ActivationEnum::Sign);
    let gradients = arr1(&[0.1, 0.1, 0.1]);
    let output = activation.backward(&gradients);
    assert_eq!(output, arr1(&[0.0, 0.0, 0.0])); // Sign function has undefined gradient, using 0
}

#[test]
fn heaviside_forward() {
    let mut activation = enum_to_activation(ActivationEnum::Heaviside);
    let input = arr1(&[-1.0, 0.0, 1.0]);
    let output = activation.forward(&input);
    assert_eq!(output, arr1(&[0.0, 1.0, 1.0]));
}

#[test]
fn heaviside_backward() {
    let mut activation = enum_to_activation(ActivationEnum::Heaviside);
    let gradients = arr1(&[0.1, 0.1, 0.1]);
    let output = activation.backward(&gradients);
    assert_eq!(output, arr1(&[0.0, 0.0, 0.0])); // Heaviside function has undefined gradient, using 0
}

#[test]
fn identity_forward() {
    let mut activation = enum_to_activation(ActivationEnum::Identity);
    let input = arr1(&[-1.0, 0.0, 1.0]);
    let output = activation.forward(&input);
    assert_eq!(output, arr1(&[-1.0, 0.0, 1.0]));
}

#[test]
fn identity_backward() {
    let mut activation = enum_to_activation(ActivationEnum::Identity);
    let gradients = arr1(&[0.1, 0.1, 0.1]);
    let output = activation.backward(&gradients);
    assert_eq!(output, gradients);
}

#[test]
fn logistic_forward() {
    let mut activation = enum_to_activation(ActivationEnum::Logistic);
    let input = arr1(&[-1.0, 0.0, 1.0]);
    let output = activation.forward(&input);
    assert!((output[0] - 0.2689414213699951).abs() < 1e-6);
    assert!((output[1] - 0.5).abs() < 1e-6);
    assert!((output[2] - 0.7310585786300049).abs() < 1e-6);
}

#[test]
fn logistic_backward() {
    let mut activation = enum_to_activation(ActivationEnum::Logistic);
    let input = arr1(&[-1.0, 0.0, 1.0]);
    let forward_output = activation.forward(&input);
    let gradients = arr1(&[0.1, 0.1, 0.1]);
    let expected_output = forward_output.mapv(|y| y * (1.0 - y)) * gradients.clone();
    let output = activation.backward(&gradients);
    for i in 0..output.len() {
        assert!(
            (output[i] - expected_output[i]).abs() < 1e-6,
            "Mismatch at index {}: expected {}, got {}",
            i,
            expected_output[i],
            output[i]
        );
    }
}

#[test]
fn tanh_forward() {
    let mut activation = enum_to_activation(ActivationEnum::TanH);
    let input = arr1(&[-1.0, 0.0, 1.0]);
    let output = activation.forward(&input);
    assert!((output[0] - (-0.7615941559557649)).abs() < 1e-6);
    assert!((output[1] - 0.0).abs() < 1e-6);
    assert!((output[2] - 0.7615941559557649).abs() < 1e-6);
}

#[test]
fn tanh_backward() {
    let mut activation = enum_to_activation(ActivationEnum::TanH);
    let input = arr1(&[-1.0, 0.0, 1.0]);
    let forward_output = activation.forward(&input);
    let gradients = arr1(&[0.1, 0.1, 0.1]);
    let expected_output = gradients.clone() * (1.0 - forward_output.mapv(|x| x.powi(2)));
    let output = activation.backward(&gradients);
    for i in 0..output.len() {
        assert!(
            (output[i] - expected_output[i]).abs() < 1e-6,
            "Mismatch at index {}: expected {}, got {}",
            i,
            expected_output[i],
            output[i]
        );
    }
}

#[test]
fn relu_forward() {
    let mut activation = enum_to_activation(ActivationEnum::ReLU);
    let input = arr1(&[-1.0, 0.0, 1.0]);
    let output = activation.forward(&input);
    assert_eq!(output, arr1(&[0.0, 0.0, 1.0]));
}

#[test]
fn relu_backward() {
    let mut activation = enum_to_activation(ActivationEnum::ReLU);
    let input = arr1(&[-1.0, 0.0, 1.0]);
    activation.forward(&input);
    let gradients = arr1(&[0.1, 0.1, 0.1]);
    let expected_output = arr1(&[0.0, 0.0, 0.1]);
    let output = activation.backward(&gradients);
    assert!((output[0] - expected_output[0]).abs() < 1e-6);
    assert!((output[1] - expected_output[1]).abs() < 1e-6);
    assert!((output[2] - expected_output[2]).abs() < 1e-6);
}
