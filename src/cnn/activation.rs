use ndarray::Array1;
use serde::{Deserialize, Serialize};

#[typetag::serde(tag = "type")]
pub trait Activation {
    fn forward(&self, input: Array1<f32>) -> Array1<f32>;
    fn backward(&self, gradients: Array1<f32>) -> Array1<f32>;
}

#[derive(Serialize, Deserialize)]
pub struct Softmax;

#[typetag::serde]
impl Activation for Softmax {
    fn forward(&self, input: Array1<f32>) -> Array1<f32> {
        let max: f32 = input.fold(input[0], |acc, &x| if x > acc { x } else { acc });
        let exps: Array1<f32> = input.mapv(|x| (x - max).exp());
        let sum: f32 = exps.sum();
        exps / sum
    }

    fn backward(&self, gradients: Array1<f32>) -> Array1<f32> {
        Array1::ones(gradients.len())
    }
}

#[derive(Serialize, Deserialize)]
pub struct Sigmoid;

#[typetag::serde]
impl Activation for Sigmoid {
    fn forward(&self, input: Array1<f32>) -> Array1<f32> {
        input.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }

    fn backward(&self, gradients: Array1<f32>) -> Array1<f32> {
        gradients.mapv(|x| x * (1.0 - x))
    }
}

#[derive(Serialize, Deserialize)]
pub struct ReLU;

#[typetag::serde]
impl Activation for ReLU {
    fn forward(&self, input: Array1<f32>) -> Array1<f32> {
        input.mapv(|x| if x > 0.0 { x } else { 0.0 })
    }

    fn backward(&self, gradients: Array1<f32>) -> Array1<f32> {
        gradients.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }
}

#[derive(Serialize, Deserialize)]
pub struct Tanh;

#[typetag::serde]
impl Activation for Tanh {
    fn forward(&self, input: Array1<f32>) -> Array1<f32> {
        input.mapv(|x| x.tanh())
    }

    fn backward(&self, gradients: Array1<f32>) -> Array1<f32> {
        let tanh_values = self.forward(gradients);
        tanh_values.mapv(|x| 1.0 - x.powi(2))
    }
}
