pub trait Loss {
    fn loss(&self, predicted: f64, target: f64) -> f64;
    fn derivative(&self, predicted: f64, target: f64) -> f64;
}

struct MSE;

impl Loss for MSE {
    fn loss(&self, predicted: f64, target: f64) -> f64 {
        0.5 * (predicted - target).powi(2)
    }

    fn derivative(&self, predicted: f64, target: f64) -> f64 {
        predicted - target
    }
}

struct LogLoss;

impl Loss for LogLoss {
    fn loss(&self, predicted: f64, target: f64) -> f64 {
        const EPSILON: f64 = 1e-15;
        let predicted = predicted.max(EPSILON).min(1.0 - EPSILON);
        -target * predicted.ln() - (1.0 - target) * (1.0 - predicted).ln()
    }

    fn derivative(&self, predicted: f64, target: f64) -> f64 {
        const EPSILON: f64 = 1e-15;
        let predicted = predicted.max(EPSILON).min(1.0 - EPSILON);
        (predicted - target) / (predicted * (1.0 - predicted))
    }
}
