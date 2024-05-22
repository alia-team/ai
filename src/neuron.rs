use crate::tensor::Tensor;
use std::fmt::{Display, Formatter};

#[derive(Debug, PartialEq)]
pub struct Neuron {
    pub weights: Option<Tensor>,
    pub bias: f64,
}

#[derive(Debug, PartialEq)]
pub enum NeuronError {
    WeightsNotInitialized,
}

impl Neuron {
    pub fn new(bias: f64) -> Neuron {
        Neuron {
            weights: None,
            bias,
        }
    }

    pub fn forward(&self, inputs: &Tensor) -> Result<f64, NeuronError> {
        if self.weights.is_none() {
            return Err(NeuronError::WeightsNotInitialized);
        }

        Ok(self.weights.as_ref().unwrap().dot(inputs).unwrap() + self.bias)
    }
}

impl Display for NeuronError {
    fn fmt(&self, formatter: &mut Formatter) -> std::fmt::Result {
        match self {
            NeuronError::WeightsNotInitialized => {
                write!(formatter, "Weights are not initialized.")
            }
        }
    }
}
