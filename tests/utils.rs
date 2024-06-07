use ai::utils;

#[test]
fn init_outputs() {
    let neurons_per_layer: Vec<usize> = vec![2, 3, 1];

    // Check for a non-RBF neural network
    let outputs: Vec<Vec<f64>> = utils::init_outputs(neurons_per_layer.clone(), false);

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

    // Check for a RBF neural network
    let outputs: Vec<Vec<f64>> = utils::init_outputs(neurons_per_layer.clone(), true);

    assert_eq!(outputs.len(), 3);
    for layer in 0..outputs.len() {
        // No bias check in a RBF
        assert_eq!(outputs[layer].len(), neurons_per_layer[layer]);
        for output in 0..outputs[layer].len() {
            assert_eq!(outputs[layer][output], 0.0);
        }
    }
}

#[test]
fn init_weights() {
    let neurons_per_layer: Vec<usize> = vec![2, 3, 1];

    // Check for a non-RBF neural network
    let weights: Vec<Vec<Vec<f64>>> = utils::init_weights(neurons_per_layer.clone(), false);

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

    // Check for a RBF neural network
    let weights: Vec<Vec<Vec<f64>>> = utils::init_weights(neurons_per_layer.clone(), true);

    assert_eq!(weights.len(), 3);

    // The two first layers in a RBF shouldn't have any weights
    for i in 0..2 {
        assert_eq!(weights[i].len(), 0);
    }

    for neuron in 0..weights[2].len() {
        assert_eq!(weights[2][neuron].len(), neurons_per_layer[1])
    }
}

#[test]
fn euclidian_distance() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    assert_eq!(utils::euclidean_distance(a, b), 1.4142135623730951);

    let a = vec![1.0, 1.0];
    let b = vec![1.0, 1.0];
    assert_eq!(utils::euclidean_distance(a, b), 0.0);

    let a = vec![1.0, 2.0];
    let b = vec![3.0, 4.0];
    assert_eq!(utils::euclidean_distance(a, b), 2.8284271247461903);
}

#[test]
fn compute_centroid() {
    let a = vec![-1.0, 1.0];
    let b = vec![1.0, 1.0];
    let c = vec![1.0, -1.0];
    let d = vec![-1.0, -1.0];
    let cluster = vec![a, b, c, d];
    assert_eq!(utils::compute_centroid(cluster), vec![0.0, 0.0]);

    let a = vec![0.0, 0.0];
    let b = vec![0.0, 2.0];
    let c = vec![2.0, 2.0];
    let d = vec![2.0, 0.0];
    let cluster = vec![a, b, c, d];
    assert_eq!(utils::compute_centroid(cluster), vec![1.0, 1.0]);
}
