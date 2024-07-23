use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

pub enum ActivationEnum {
    Heaviside,
    Identity,
    Logistic,
    ReLU,
    Sigmoid,
    Sign,
    TanH,
}

// Convert ActivationEnum to the corresponding activation
pub fn enum_to_activation(activation_enum: ActivationEnum) -> Box<dyn Activation> {
    match activation_enum {
        ActivationEnum::Heaviside => Box::new(Heaviside),
        ActivationEnum::Identity => Box::new(Identity),
        ActivationEnum::Logistic => Box::new(Logistic),
        ActivationEnum::ReLU => Box::new(ReLU),
        ActivationEnum::Sigmoid => Box::new(Sigmoid),
        ActivationEnum::Sign => Box::new(Sign),
        ActivationEnum::TanH => Box::new(Tanh),
    }
}

pub fn string_to_activation(string: &str) -> Box<dyn Activation> {
    match string {
        "heaviside" => Box::new(Heaviside),
        "identity" => Box::new(Identity),
        "logistic" => Box::new(Logistic),
        "relu" => Box::new(ReLU),
        "sigmoid" => Box::new(Sigmoid),
        "sign" => Box::new(Sign),
        "tanh" => Box::new(Tanh),
        _ => panic!("Not a supported activation function."),
    }
}

#[typetag::serde(tag = "type")]
pub trait Activation: Debug {
    fn forward(&self, input: Array1<f64>) -> Array1<f64>;
    fn backward(&self, gradients: Array1<f64>) -> Array1<f64>;
    fn as_str(&self) -> &'static str;
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Sign;

#[typetag::serde]
impl Activation for Sign {
    fn forward(&self, input: Array1<f64>) -> Array1<f64> {
        input.mapv(|x| if x >= 0.0 { 1.0 } else { -1.0 })
    }

    fn backward(&self, gradients: Array1<f64>) -> Array1<f64> {
        Array1::zeros(gradients.raw_dim()) // Return zeros of the same shape as gradients
    }

    fn as_str(&self) -> &'static str {
        "Sign"
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Heaviside;

#[typetag::serde]
impl Activation for Heaviside {
    fn forward(&self, input: Array1<f64>) -> Array1<f64> {
        input.mapv(|x| if x >= 0.0 { 1.0 } else { 0.0 })
    }

    fn backward(&self, gradients: Array1<f64>) -> Array1<f64> {
        Array1::zeros(gradients.raw_dim()) // Return zeros of the same shape as gradients
    }

    fn as_str(&self) -> &'static str {
        "Heaviside"
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Identity;

#[typetag::serde]
impl Activation for Identity {
    fn forward(&self, input: Array1<f64>) -> Array1<f64> {
        input.clone()
    }

    fn backward(&self, gradients: Array1<f64>) -> Array1<f64> {
        gradients.clone()
    }

    fn as_str(&self) -> &'static str {
        "Identity"
    }
}

#[derive(Debug, Deserialize, Serialize, Default)]
pub struct Logistic;

#[typetag::serde]
impl Activation for Logistic {
    fn forward(&self, input: Array1<f64>) -> Array1<f64> {
        input.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }

    fn backward(&self, gradients: Array1<f64>) -> Array1<f64> {
        gradients.mapv(|x| x * (1.0 - x))
    }

    fn as_str(&self) -> &'static str {
        "Logistic"
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Softmax;

#[typetag::serde]
impl Activation for Softmax {
    fn forward(&self, input: Array1<f64>) -> Array1<f64> {
        let max: f64 = input.fold(input[0], |acc, &x| if x > acc { x } else { acc });
        let exps: Array1<f64> = input.mapv(|x| (x - max).exp());
        let sum: f64 = exps.sum();
        exps / sum
    }

    fn backward(&self, gradients: Array1<f64>) -> Array1<f64> {
        Array1::ones(gradients.len())
    }

    fn as_str(&self) -> &'static str {
        "Softmax"
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Sigmoid;

#[typetag::serde]
impl Activation for Sigmoid {
    fn forward(&self, input: Array1<f64>) -> Array1<f64> {
        input.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }

    fn backward(&self, gradients: Array1<f64>) -> Array1<f64> {
        gradients.mapv(|x| x * (1.0 - x))
    }

    fn as_str(&self) -> &'static str {
        "Sigmoid"
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ReLU;

#[typetag::serde]
impl Activation for ReLU {
    fn forward(&self, input: Array1<f64>) -> Array1<f64> {
        input.mapv(|x| if x > 0.0 { x } else { 0.0 })
    }

    fn backward(&self, gradients: Array1<f64>) -> Array1<f64> {
        gradients.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }

    fn as_str(&self) -> &'static str {
        "ReLU"
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Tanh;

#[typetag::serde]
impl Activation for Tanh {
    fn forward(&self, input: Array1<f64>) -> Array1<f64> {
        input.mapv(|x| x.tanh())
    }

    fn backward(&self, gradients: Array1<f64>) -> Array1<f64> {
        let tanh_values = self.forward(gradients);
        tanh_values.mapv(|x| 1.0 - x.powi(2))
    }

    fn as_str(&self) -> &'static str {
        "Tanh"
    }
}
