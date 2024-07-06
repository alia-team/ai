use ndarray::Array2;

pub trait Loss {
    fn compute_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64;
    fn compute_gradients(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64>;
}

#[derive(Default)]
pub struct CategoricalCrossEntropy;

impl CategoricalCrossEntropy {
    pub fn new() -> Self {
        CategoricalCrossEntropy
    }
}

impl Loss for CategoricalCrossEntropy {
    fn compute_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
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

    fn compute_gradients(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64> {
        (predictions - targets) / targets.nrows() as f64
    }
}

#[derive(Default)]
pub struct MSE;

impl MSE {
    pub fn new() -> Self {
        MSE
    }
}

impl Loss for MSE {
    fn compute_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        let diff = predictions - targets;
        diff.mapv(|x| x.powi(2)).mean().unwrap()
    }

    fn compute_gradients(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64> {
        2.0 * (predictions - targets) / (targets.nrows() as f64)
    }
}
