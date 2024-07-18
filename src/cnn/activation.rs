use ndarray::Array1;

pub trait Activation {
    fn forward(&self, input: Array1<f32>) -> Array1<f32>;
    fn backward(&self, input: Array1<f32>) -> Array1<f32>;
}

pub struct Softmax;
impl Activation for Softmax {
    fn forward(&self, x: Array1<f32>) -> Array1<f32> {
        let max: f32 = x.fold(x[0], |acc, &xi| if xi > acc { xi } else { acc });
        let exps: Array1<f32> = x.mapv(|xi| (xi - max).exp());
        let sum: f32 = exps.sum();
        exps / sum
    }

    fn backward(&self, x: Array1<f32>) -> Array1<f32> {
        Array1::ones(x.len())
    }
}

pub struct Sigmoid;
impl Activation for Sigmoid {
    fn forward(&self, x: Array1<f32>) -> Array1<f32> {
        x.mapv(|xi| 1.0 / (1.0 + (-xi).exp()))
    }

    fn backward(&self, x: Array1<f32>) -> Array1<f32> {
        x.mapv(|xi| xi * (1.0 - xi))
    }
}

pub struct ReLU;
impl Activation for ReLU {
    fn forward(&self, x: Array1<f32>) -> Array1<f32> {
        x.mapv(|xi| if xi > 0.0 { xi } else { 0.0 })
    }

    fn backward(&self, x: Array1<f32>) -> Array1<f32> {
        x.mapv(|xi| if xi > 0.0 { 1.0 } else { 0.0 })
    }
}
