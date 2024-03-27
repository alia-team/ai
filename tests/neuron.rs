use ai::neuron::*;
use ai::tensor::Tensor;

#[test]
fn new() {
    let bias = 1.0;
    let neuron = Neuron::new(bias);

    assert_eq!(
        neuron,
        Neuron {
            weights: None,
            bias,
        }
    )
}

#[test]
fn forward() {
    let bias = 1.0;
    let mut neuron = Neuron::new(bias);
    let inputs = Tensor::new(&[1.0, 2.0, 3.0], &[3]).unwrap();
    assert_eq!(
        neuron.forward(&inputs),
        Err(NeuronError::WeightsNotInitialized)
    );

    let weights = Tensor::new(&[1.0, 2.0, 3.0], &[3]).unwrap();
    neuron.weights = Some(weights);
    assert_eq!(neuron.forward(&inputs), Ok(15.0));
}
