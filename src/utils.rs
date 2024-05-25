extern crate rand;
use rand::Rng;

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
