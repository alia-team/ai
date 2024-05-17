use std::f32::consts::E;

/// Defines a type alias for activation functions that take a single `f32`
/// argument and return an `f32`.
pub type Activation = fn(f32) -> f32;

/// Implements the binary step activation function.
///
/// It is useful for binary classification tasks where an output needs to be
/// decisively one class or another.
/// However, its use is limited in multi-layer networks due to its
/// non-differentiability at `0`, which prevents gradient-based learning methods
/// from working.
///
/// # Arguments
///
/// * `x` - The input value to the binary step function.
///
/// # Returns
///
/// `1.0` if `x` is greater than or equal to `0`, else `0.0`.
pub fn binary_step(x: f32) -> f32 {
    if x >= 0.0 {
        1.0
    } else {
        0.0
    }
}

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

pub fn sign(x: f64) -> f64 {
    if x >= 0.0 {
        1.0
    } else {
        -1.0
    }
}
