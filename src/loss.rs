use ndarray::Array2;

pub struct CategoricalCrossentropy;

impl CategoricalCrossentropy {
    pub fn new() -> Self {
        CategoricalCrossentropy
    }

    pub fn forward(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        let mut loss = 0.0;
        for (pred, target) in predictions.outer_iter().zip(targets.outer_iter()) {
            for (&p, &t) in pred.iter().zip(target.iter()) {
                if t == 1.0 {
                    loss -= (p.max(1e-15)).ln(); // Using max to avoid log(0)
                }
            }
        }
        loss / targets.nrows() as f64
    }

    pub fn backward(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64> {
        (predictions - targets) / targets.nrows() as f64
    }
}

impl Default for CategoricalCrossentropy {
    fn default() -> Self {
        Self::new()
    }
}
