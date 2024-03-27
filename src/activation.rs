use std::f32::consts::E;

/// Defines a type alias for activation functions that take a single `f32`
/// argument and return an `f32`.
pub type Activation = fn(f32) -> f32;

/// Implements the logistic (sigmoid) activation function.
///
/// This function is widely used in neural networks, particularly in binary
/// classification problems.
/// It maps any real-valued number into the range (0, 1), making it useful for
/// representing probabilities.
///
/// # Arguments
///
/// * `x` - The input value to the logistic function.
///
/// # Returns
///
/// The output of the logistic function applied to the input value, constrained
/// between 0 and 1.
pub fn logistic(x: f32) -> f32 {
    1.0 / (1.0 + E.powf(-x))
}

/// Alias for the `logistic` function to provide an alternative name commonly
/// used in literature.
///
/// # Arguments
///
/// * `x` - The input value to the sigmoid function.
///
/// # Returns
///
/// The output of the sigmoid function applied to the input value.
pub fn sigmoid(x: f32) -> f32 {
    logistic(x)
}
