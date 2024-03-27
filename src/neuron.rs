use crate::tensor::Tensor;

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
}
