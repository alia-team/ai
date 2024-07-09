use ai::activation::ActivationEnum;
use ai::layer::*;
use ndarray::{arr1, arr3, Array1, Array3};

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
    let outputs: Array1<f64> = *(layer.forward(&input).downcast::<Array1<f64>>().unwrap());

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
    let grad_input: Array1<f64> = *(layer
        .backward(&gradients)
        .downcast::<Array1<f64>>()
        .unwrap());

    assert_eq!(grad_input.len(), input.len());
    assert!(layer.weight_gradients.is_some());
}

#[test]
fn flatten_new() {
    let layer: Flatten = Flatten::new();
    assert_eq!(layer.input_shape, None);
    assert_eq!(layer.output_size, 0);
    assert_eq!(layer.input, None);
    assert_eq!(layer.output, None);
}

#[test]
fn flatten_forward() {
    let input: Array3<f64> = arr3(&[[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]]]);
    let mut layer: Flatten = Flatten::new();
    layer.forward(&input);
    let expected_output: Array1<f64> = arr1(&[0., 0., 0., 0., 0., 0., 0., 0.]);
    assert_eq!(layer.input_shape, Some((2, 2, 2)));
    assert_eq!(layer.output_size, 8);
    assert_eq!(layer.input, Some(input));
    assert_eq!(layer.output, Some(expected_output));
}

#[test]
fn flatten_backward() {
    let mut layer = Flatten::new();
    let input =
        Array3::from_shape_vec((2, 2, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    layer.forward(&input);
    let gradients = arr1(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);

    let result = layer.backward(&gradients);
    let reshaped_gradients = result.downcast_ref::<Array3<f64>>().unwrap();
    let expected_gradients =
        Array3::from_shape_vec((2, 2, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]).unwrap();

    assert_eq!(reshaped_gradients, &expected_gradients);
    assert_eq!(reshaped_gradients.shape(), &[2, 2, 2]);
}
