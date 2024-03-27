use ai::neuron::*;

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
