extern crate rand;
use rand::Rng;

pub fn init_outputs(neurons_per_layer: Vec<usize>, is_rbf: bool) -> Vec<Vec<f64>> {
    let mut outputs: Vec<Vec<f64>> = vec![];

    // Skip output layer
    for layer in 0..(neurons_per_layer.len() - 1) {
        outputs.push(vec![]);

        // +1 for bias except for RBF
        for neuron in 0..(neurons_per_layer[layer] + (if is_rbf { 0 } else { 1 })) {
            if neuron == 0 && !is_rbf {
                // Bias
                outputs[layer].push(1.0)
            } else {
                // Weight between -1.0 and 1.0
                outputs[layer].push(0.0)
            }
        }
    }

    outputs.push(vec![0.0; *neurons_per_layer.last().unwrap()]);

    outputs
}

pub fn init_weights(neurons_per_layer: Vec<usize>, is_rbf: bool) -> Vec<Vec<Vec<f64>>> {
    let mut weights: Vec<Vec<Vec<f64>>> = vec![];

    for layer in 0..neurons_per_layer.len() {
        weights.push(vec![]);
        if layer == 0 {
            continue;
        }

        if is_rbf {
            continue;
        }

        for neuron in 0..neurons_per_layer[layer] {
            weights[layer].push(vec![]);

            // +1 for bias
            for input in 0..(neurons_per_layer[layer - 1] + 1) {
                if input == 0 && !is_rbf {
                    // Bias
                    weights[layer][neuron].push(0.0)
                } else {
                    // Weight between -1.0 and 1.0
                    weights[layer][neuron].push(rand::thread_rng().gen_range(-1.0..1.0))
                }
            }
        }
    }

    weights
}

pub fn init_centers(hidden_layer_neurons_count: usize, dataset: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut centers: Vec<Vec<f64>> = vec![];

    for _ in 0..hidden_layer_neurons_count {
        centers.push(dataset[rand::thread_rng().gen_range(0..dataset.len())].clone());
    }

    centers
}
