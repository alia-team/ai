extern crate rand;
use ai::utils;
use rand::Rng;

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
    let expect_msg: &str = "output neuron should have weighted inputs";

    assert_eq!(weights.len(), 3);
    // Checking number of weights per neuron only for the output layer
    for neuron in 0..weights.last().expect(expect_msg).len() {
        assert_eq!(
            weights.last().expect(expect_msg)[neuron].len(),
            *neurons_per_layer.last().expect(expect_msg)
        )
    }
}

#[test]
fn init_centers() {
    let mut rng = rand::thread_rng();

    // Build dataset
    let dataset_size: usize = 100;
    let mut set1: Vec<Vec<f64>> = (0..(dataset_size / 2))
        .map(|_| {
            let x = rng.gen::<f64>() * 0.9 + 1.0;
            let y = rng.gen::<f64>() * 0.9 + 1.0;
            vec![x, y]
        })
        .collect();
    let mut set2: Vec<Vec<f64>> = (0..(dataset_size / 2))
        .map(|_| {
            let x = rng.gen::<f64>() * 0.9 + 2.0;
            let y = rng.gen::<f64>() * 0.9 + 2.0;
            vec![x, y]
        })
        .collect();
    let mut dataset = Vec::new();
    dataset.append(&mut set1);
    dataset.append(&mut set2);

    let hidden_layer_neurons_count = dataset_size / 10;

    let centers = utils::init_centers(hidden_layer_neurons_count, dataset);

    assert_eq!(centers.len(), hidden_layer_neurons_count);
}
