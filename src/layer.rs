use crate::activation::Activation;
use crate::neuron::Neuron;
use crate::tensor::Tensor;

#[derive(Debug, PartialEq)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub activation: Activation,
}

impl Layer {
    pub fn new(number_of_neurons: u128, activation: Activation) -> Self {
        Layer {
            neurons: (0..number_of_neurons)
                .map(|_| Neuron {
                    weights: None,
                    bias: 0.0,
                })
                .collect(),
            activation,
        }
    }

    pub fn forward(&mut self, inputs: &Tensor) -> Tensor {
        let mut outputs: Vec<f64> = vec![];

        for neuron in self.neurons.iter_mut() {
            if neuron.weights.is_none() {
                let tensor =
                    match Tensor::new(vec![1.0; inputs.data.len()], vec![inputs.data.len()]) {
                        Ok(tensor) => tensor,
                        Err(error) => panic!("{:?}", error),
                    };
                neuron.weights = Some(tensor);
            }

            let output = match neuron.forward(inputs) {
                Ok(output) => output,
                Err(error) => panic!("{:?}", error),
            };

            outputs.push((self.activation)(output))
        }

        match Tensor::new(outputs, vec![self.neurons.len()]) {
            Ok(tensor) => tensor,
            Err(error) => panic!("{:?}", error),
        }
    }
}
