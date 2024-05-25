use ai::utils;

#[test]
fn init_outputs() {
    let neurons_per_layer: Vec<usize> = vec![2, 3, 1];
    let outputs: Vec<Vec<f64>> = utils::init_outputs(neurons_per_layer.clone());

    // Check number of layers
    assert_eq!(outputs.len(), 3);

    for layer in 0..outputs.len() {
        // Check output layer separately since it doens't have a bias
        if layer == outputs.len() - 1 {
            assert_eq!(outputs[layer].len(), neurons_per_layer[layer]);
            for output in 0..outputs[layer].len() {
                assert_eq!(outputs[layer][output], 0.0);
            }
            continue;
        }

        // Check other layers
        assert_eq!(outputs[layer].len(), neurons_per_layer[layer] + 1);
        for output in 0..outputs[layer].len() {
            if output == 0 {
                assert_eq!(outputs[layer][output], 1.0); // Bias
                continue;
            }
            assert_eq!(outputs[layer][output], 0.0)
        }
    }
}

#[test]
fn init_weights() {
    let neurons_per_layer: Vec<usize> = vec![2, 3, 1];
    let weights: Vec<Vec<Vec<f64>>> = utils::init_weights(neurons_per_layer.clone());

    // Checking number of layers
    assert_eq!(weights.len(), 3);

    for layer in 0..weights.len() {
        // Checking number of neurons
        if layer == 0 {
            assert_eq!(weights[layer].len(), 0); // Empty vector since input layer has no weight
            continue;
        }
        assert_eq!(weights[layer].len(), neurons_per_layer[layer]);

        // Checking number of weights per neuron
        for neuron in 0..weights[layer].len() {
            assert_eq!(
                weights[layer][neuron].len(),
                neurons_per_layer[layer - 1] + 1 // +1 for bias
            )
        }
    }
}
