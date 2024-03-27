use crate::tensor::Tensor;
use std::fmt::{Display, Formatter};

/// Represents a neuron with its parameters.
///
/// Weights are an option because since the number of weights depends on the
/// number of inputs, we can't define the weights at neuron's creation.
/// Weights are initialized during the first forward pass.
#[derive(Debug, PartialEq)]
pub struct Neuron<'a> {
    pub weights: Option<Tensor<'a>>,
    pub bias: f32,
}

#[derive(Debug, PartialEq)]
pub enum NeuronError {
    WeightsNotInitialized,
}

impl<'a> Neuron<'a> {
    /// Creates a new `Neuron` instance from a bias.
    ///
    /// Since the number of weights depends on the number of inputs, we can't
    /// define the weights at neuron's creation.
    /// Weights are initialized during the first forward pass.
    ///
    /// # Argument
    ///
    /// * `bias` - A `f32` letting developer choose its way to initialize it.
    ///
    /// # Returns
    ///
    /// A `Neuron` instance.
    pub fn new(bias: f32) -> Neuron<'a> {
        Neuron {
            weights: None,
            bias,
        }
    }

    /// Performs a forward pass on the neuron with given inputs.
    ///
    /// This method calculates the weighted sum of the inputs and the neuron's
    /// bias.
    /// It requires that the weights of the neuron have been initialized
    /// beforehand.
    /// If the weights are not initialized, it returns a
    /// `NeuronError::WeightsNotInitialized` error.
    ///
    /// # Arguments
    ///
    /// * `inputs` - A reference to a `Tensor` representing the inputs to the
    /// neuron.
    ///
    /// # Returns
    ///
    /// An `Ok(f32)` containing the result of the weighted sum plus the bias if
    /// the weights are initialized.
    /// Otherwise, returns a `NeuronError::WeightsNotInitialized` error.
    pub fn forward(&self, inputs: &Tensor<'a>) -> Result<f32, NeuronError> {
        if self.weights.is_none() {
            return Err(NeuronError::WeightsNotInitialized);
        }

        Ok(self.weights.as_ref().unwrap().dot(inputs).unwrap() + self.bias)
    }
}

impl Display for NeuronError {
    fn fmt(&self, formatter: &mut Formatter) -> std::fmt::Result {
        match *self {
            NeuronError::WeightsNotInitialized => {
                write!(formatter, "Weights are not initialized.")
            }
        }
    }
}
