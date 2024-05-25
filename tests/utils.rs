use ai::utils;

#[test]
fn init_weights() {
    let neurons_per_layer: Vec<usize> = vec![2, 3, 1];
    let weights: Vec<Vec<Vec<f64>>> = utils::init_weights(neurons_per_layer);

    // Checking number of layers
    assert_eq!(weights.len(), 3);

    for layer in weights {
        // Checking number of neurons
        if layer == 0 {
            assert_eq!(weights[layer].len(), 1) // Empty vector since input layer has no weight
        }
        assert_eq!(weights[layer].len(), neurons_per_layer[layer]);

        // Checking number of weights per neuron
        for neuron in weights[layer] {
            assert_eq!(
                weights[layer][neuron].len(),
                neurons_per_layer[layer - 1] + 1 // +1 for bias
            )
        }
    }
}
