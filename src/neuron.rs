use crate::tensor::Tensor;

/// Represents a neuron with its parameters.
///
/// Weights are initialized during the first forward pass because at this stage,
/// we don't know the needed number of weights. It depend on the number of input.
#[derive(Debug, PartialEq)]
pub struct Neuron<'a> {
    pub weights: Option<Tensor<'a>>,
    pub bias: f32,
}

impl<'a> Neuron<'a> {
    pub fn new(bias: f32) -> Neuron<'a> {
        Neuron {
            weights: None,
            bias,
        }
    }
}
