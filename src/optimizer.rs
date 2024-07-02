use ndarray::{Array2, Zip};

pub trait Optimizer {
    fn update_weights(
        &self,
        weights: &mut Array2<f64>,
        gradients: &Array2<f64>,
        learning_rate: f64,
    );
}

#[derive(Default)]
pub struct SGD;

impl SGD {
    pub fn new() -> Self {
        SGD
    }
}

impl Optimizer for SGD {
    fn update_weights(
        &self,
        weights: &mut Array2<f64>,
        gradients: &Array2<f64>,
        learning_rate: f64,
    ) {
        // Perform element-wise subtraction
        Zip::from(weights)
            .and(gradients)
            .for_each(|w, &g| *w -= g * learning_rate);
    }
}
