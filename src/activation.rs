use std::f32::consts::E;

pub type Activation = fn(f32) -> f32;

pub fn logistic(x: f32) -> f32 {
    1.0 / (1.0 + E.powf(-x))
}

pub fn sigmoid(x: f32) -> f32 {
    logistic(x)
}
