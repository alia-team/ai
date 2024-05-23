pub trait Optimizer {
    fn update(&mut self, weight: f64, gradient: f64, learning_rate: f64, layer_index: usize, neuron_index: usize, weight_index: usize) -> f64;
}

struct SGD;

impl Optimizer for SGD {
    fn update(&mut self, weight: f64, gradient: f64, learning_rate: f64, layer_index: usize, neuron_index: usize, weight_index: usize) -> f64 {
        weight - learning_rate * gradient
    }
}

struct BGD {
    accumulated_gradients: std::collections::HashMap<(usize, usize, usize), f64>,
}

impl BGD {
    fn new() -> Self {
        BGD {
            accumulated_gradients: std::collections::HashMap::new(),
        }
    }

    fn accumulate_gradient(&mut self, layer_index: usize, neuron_index: usize, weight_index: usize, gradient: f64) {
        let key = (layer_index, neuron_index, weight_index);
        let entry = self.accumulated_gradients.entry(key).or_insert(0.0);
        *entry += gradient;
    }
}

impl Optimizer for BGD {
    fn update(&mut self, weight: f64, gradient: f64, learning_rate: f64, layer_index: usize, neuron_index: usize, weight_index: usize) -> f64 {
        if let (layer_index, neuron_index, weight_index) = (layer_index, neuron_index, weight_index) {
            let key = (layer_index, neuron_index, weight_index);
            if let Some(&avg_gradient) = self.accumulated_gradients.get(&key) {
                let new_weight = weight - learning_rate * avg_gradient;
                self.accumulated_gradients.remove(&key);
                new_weight
            } else {
                weight
            }
        } else {
            weight
        }
    }
}