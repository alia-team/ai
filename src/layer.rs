use crate::activation::Activation;
use crate::neuron::Neuron;
use crate::tensor::Tensor;

/// Represents a layer in a neural network.
///
/// A `Layer` is composed of multiple neurons and applies an activation function
/// to the outputs of these neurons.
/// It serves as a basic building block for neural network architectures,
/// facilitating the modeling of complex patterns in data by transforming input
/// tensors into outputs through learned weights and biases.
#[derive(Debug, PartialEq)]
pub struct Layer {
    /// A vector of neurons that constitute the layer.
    pub neurons: Vec<Neuron>,
    /// The activation function applied to the output of each neuron in the
    /// layer.
    pub activation: Activation,
}

impl Layer {
    /// Constructs a new `Layer` with a specified number of neurons and an
    /// activation function.
    ///
    /// Initializes each neuron in the layer with default weights (none) and a
    /// bias of 0.0.
    /// The weights will be later adjusted during the training process.
    ///
    /// # Arguments
    ///
    /// * `number_of_neurons` - The number of neurons to include in the layer.
    /// * `activation` - The activation function to use for neurons in this
    /// layer.
    ///
    /// # Returns
    ///
    /// An instance of `Layer`.
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

    /// Processes input data through the layer, applying neuron transformations
    /// and activation function.
    ///
    /// This function propagates the input `Tensor` through each neuron in the
    /// layer, aggregating the output in a new `Tensor`.
    /// If a neuron's weights are not initialized, it initializes them with a
    /// default tensor.
    /// It then applies the layer's activation function to each neuron's output
    /// before combining these outputs into the final `Tensor`.
    ///
    /// # Arguments
    ///
    /// * `inputs` - A reference to the input `Tensor` for the layer.
    ///
    /// # Returns
    ///
    /// A `Tensor` representing the output of the layer after applying neuron
    /// transformations and the activation function.
    ///
    /// # Panics
    ///
    /// Panics if there is an error creating the output `Tensor` or if neuron
    /// forward propagation fails.
    pub fn forward(&mut self, inputs: &Tensor) -> Tensor {
        let mut outputs: Vec<f32> = vec![];

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
