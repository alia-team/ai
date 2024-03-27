use crate::tensor::Tensor;

/// Represents a neuron with its parameters.
///
/// Weights are initialized during the first forward pass.
pub struct Neuron<'a> {
    pub weights: Option<Tensor<'a>>,
    pub bias: f64,
}
