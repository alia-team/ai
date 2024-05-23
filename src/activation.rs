use std::f64::consts::E;

pub trait Activation {
    fn activate(&self, x: f64) -> f64;
    fn derivative(&self, x: f64) -> f64;
}

struct Tanh;

impl Activation for Tanh {
    fn activate(&self, x: f64) -> f64 {
        x.tanh()
    }

    fn derivative(&self, x: f64) -> f64 {
        let tanh_x = x.tanh();
        1.0 - tanh_x.powi(2)
    }
}

struct Logistic;

impl Activation for Logistic {
    fn activate(&self, x: f64) -> f64 {
        1.0 / (1.0 + E.powf(-x))
    }

    fn derivative(&self, x: f64) -> f64 {
        let activated = self.activate(x);
        activated * (1.0 - activated)
    }
}

type Sigmoid = Logistic;
