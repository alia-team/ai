use ai::activation::*;
use ai::layer::*;
use ai::neuron::Neuron;
use ai::tensor::Tensor;

#[test]
fn new() {
    assert_eq!(
        Layer::new(1, binary_step),
        Layer {
            neurons: vec![Neuron {
                weights: None,
                bias: 0.0
            }],
            activation: binary_step
        }
    );
}

#[test]
fn test_forward() {
    let mut perceptron = Layer::new(1, binary_step);
    assert_eq!(
        perceptron.forward(&Tensor::new(vec![1.0, 2.0], vec![2]).unwrap()),
        Tensor::new(vec![1.0], vec![1]).unwrap()
    );
    assert_eq!(
        perceptron.forward(&Tensor::new(vec![-1.0, -2.0], vec![2]).unwrap()),
        Tensor::new(vec![0.0], vec![1]).unwrap()
    );
}
