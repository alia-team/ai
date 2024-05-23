use rand::Rng;
use crate::activation::Activation;
use crate::optimizer::Optimizer;
use crate::loss::Loss;

pub struct Neuron {
    weights: Vec<f64>,
    output: f64,
    gradient: f64,
    activation: Box<dyn Activation>,
}

impl Neuron {
    fn new(num_inputs: usize, activation: Box<dyn Activation>) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..num_inputs + 1)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        Neuron {
            weights,
            output: 0.0,
            gradient: 0.0,
            activation,
        }
    }

    fn activate(&mut self, inputs: &[f64]) -> f64 {
        self.output = self.weights[0]; // Bias
        for (i, input) in inputs.iter().enumerate() {
            self.output += self.weights[i + 1] * input;
        }
        self.output = self.activation.activate(self.output);
        self.output
    }

    fn calculate_gradient(
        &mut self,
        target: Option<f64>,
        downstream_gradients: Option<&[f64]>,
        downstream_weights: Option<&[f64]>,
        loss: Option<&dyn Loss>,
    ) -> f64 {
        if let Some(target) = target {
            self.gradient = self.activation.derivative(self.output)
                * loss.unwrap().derivative(self.output, target);
        } else {
            self.gradient = downstream_gradients
                .unwrap()
                .iter()
                .zip(downstream_weights.unwrap())
                .map(|(dg, dw)| dg * dw)
                .sum::<f64>();
            self.gradient *= self.activation.derivative(self.output);
        }
        self.gradient
    }

    fn update_weights(
        &mut self,
        inputs: &[f64],
        learning_rate: f64,
        optimizer: &mut dyn Optimizer,
        layer_index: usize,
        neuron_index: usize,
    ) {
        for (i, weight) in self.weights.iter_mut().enumerate() {
            let gradient = if i == 0 {
                self.gradient
            } else {
                self.gradient * inputs[i - 1]
            };
            *weight = optimizer.update(
                *weight,
                gradient,
                learning_rate,
                layer_index,
                neuron_index,
                i,
            );
        }
    }
}
