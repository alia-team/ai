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

        if layer == 1 && is_rbf {
            continue;
        }

        for neuron in 0..neurons_per_layer[layer] {
            weights[layer].push(vec![]);

            // +1 for bias if it's not a RBF neural network
            for input in 0..(neurons_per_layer[layer - 1] + (if is_rbf { 0 } else { 1 })) {
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

pub fn euclidean_distance(a: Vec<f64>, b: Vec<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

pub fn compute_centroid(cluster: Vec<Vec<f64>>) -> Vec<f64> {
    let mut centroid = vec![0.0; cluster[0].len()];
    for point in cluster.clone() {
        for (i, value) in point.iter().enumerate() {
            centroid[i] += value;
        }
    }
    for value in centroid.iter_mut() {
        *value /= cluster.len() as f64;
    }
    centroid
}
