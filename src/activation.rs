use ndarray::Array1;
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

pub trait Activation: Debug {
    fn forward(&mut self, input: &Array1<f64>) -> Array1<f64>;
    fn backward(&mut self, gradients: &Array1<f64>) -> Array1<f64>;
    fn as_str(&self) -> &'static str;
}

// Convert ActivationEnum to the corresponding activation
pub fn enum_to_activation(activation_enum: ActivationEnum) -> Box<dyn Activation> {
    match activation_enum {
        ActivationEnum::Heaviside => Box::new(Heaviside),
        ActivationEnum::Identity => Box::new(Identity),
        ActivationEnum::Logistic => Box::new(Logistic::new()),
        ActivationEnum::ReLU => Box::new(ReLU::new()),
        ActivationEnum::Sigmoid => Box::new(Logistic::new()),
        ActivationEnum::Sign => Box::new(Sign),
        ActivationEnum::TanH => Box::new(TanH::new()),
    }
}

pub fn string_to_activation(string: &str) -> Box<dyn Activation> {
    match string {
        "heaviside" => Box::new(Heaviside),
        "identity" => Box::new(Identity),
        "logistic" => Box::new(Logistic::new()),
        "relu" => Box::new(ReLU::new()),
        "sigmoid" => Box::new(Logistic::new()),
        "sign" => Box::new(Sign),
        "tanh" => Box::new(TanH::new()),
        _ => panic!("Not a supported activation function"),
    }
}

#[derive(Debug)]
pub struct Sign;

impl Activation for Sign {
    fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        input.mapv(|x| if x >= 0.0 { 1.0 } else { -1.0 })
    }

    fn backward(&mut self, gradients: &Array1<f64>) -> Array1<f64> {
        Array1::zeros(gradients.raw_dim()) // Return zeros of the same shape as gradients
    }

    fn as_str(&self) -> &'static str {
        "Sign"
    }
}

#[derive(Debug)]
pub struct Heaviside;

impl Activation for Heaviside {
    fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        input.mapv(|x| if x >= 0.0 { 1.0 } else { 0.0 })
    }

    fn backward(&mut self, gradients: &Array1<f64>) -> Array1<f64> {
        Array1::zeros(gradients.raw_dim()) // Return zeros of the same shape as gradients
    }

    fn as_str(&self) -> &'static str {
        "Heaviside"
    }
}

#[derive(Debug)]
pub struct Identity;

impl Activation for Identity {
    fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        input.clone()
    }

    fn backward(&mut self, gradients: &Array1<f64>) -> Array1<f64> {
        gradients.clone()
    }

    fn as_str(&self) -> &'static str {
        "Identity"
    }
}

#[derive(Debug, Default)]
pub struct Logistic {
    output: Option<Array1<f64>>,
}

impl Logistic {
    pub fn new() -> Self {
        Self { output: None }
    }
}

impl Activation for Logistic {
    fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        let output = input.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        self.output = Some(output.clone());
        output
    }

    fn backward(&mut self, gradients: &Array1<f64>) -> Array1<f64> {
        let sig = self.output.as_ref().unwrap();
        gradients * sig * (1.0 - sig)
    }

    fn as_str(&self) -> &'static str {
        "Logistic"
    }
}

#[derive(Debug, Default)]
pub struct ReLU {
    output: Option<Array1<f64>>,
}

impl ReLU {
    pub fn new() -> Self {
        Self { output: None }
    }
}

impl Activation for ReLU {
    fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        let output = input.mapv(|x| if x <= 0.0 { 0.0 } else { x });
        self.output = Some(output.clone());
        output
    }

    fn backward(&mut self, gradients: &Array1<f64>) -> Array1<f64> {
        let output = self.output.as_ref().unwrap();
        gradients * output.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }

    fn as_str(&self) -> &'static str {
        "ReLU"
    }
}

#[derive(Debug, Default)]
pub struct TanH {
    output: Option<Array1<f64>>,
}

impl TanH {
    pub fn new() -> Self {
        Self { output: None }
    }
}

impl Activation for TanH {
    fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        let output = input.mapv(|x| x.tanh());
        self.output = Some(output.clone());
        output
    }

    fn backward(&mut self, gradients: &Array1<f64>) -> Array1<f64> {
        gradients * (1.0 - self.output.as_ref().unwrap().mapv(|x| x.powi(2)))
    }

    fn as_str(&self) -> &'static str {
        "TanH"
    }
}
