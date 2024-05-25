extern crate rand;
use rand::Rng;

pub fn init_outputs(neurons_per_layer: Vec<usize>) -> Vec<Vec<f64>> {
    let mut outputs: Vec<Vec<f64>> = vec![];

    // Skip output layer
    for layer in 0..(neurons_per_layer.len() - 1) {
        outputs.push(vec![]);

        // +1 for bias
        for neuron in 0..(neurons_per_layer[layer] + 1) {
            if neuron == 0 {
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

pub fn init_weights(neurons_per_layer: Vec<usize>) -> Vec<Vec<Vec<f64>>> {
    let mut weights: Vec<Vec<Vec<f64>>> = vec![];

    for layer in 0..neurons_per_layer.len() {
        weights.push(vec![]);
        if layer == 0 {
            continue;
        }

        for neuron in 0..neurons_per_layer[layer] {
            weights[layer].push(vec![]);

            // +1 for bias
            for input in 0..(neurons_per_layer[layer - 1] + 1) {
                if input == 0 {
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
