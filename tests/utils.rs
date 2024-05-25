use ai::utils;

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
