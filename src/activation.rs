use std::f32::consts::E;

pub type Activation = fn(f32) -> f32;

pub fn binary_step(x: f32) -> f32 {
    if x >= 0.0 {
        1.0
    } else {
        0.0
    }
}

pub fn logistic(x: f32) -> f32 {
    1.0 / (1.0 + E.powf(-x))
}

pub fn sigmoid(x: f32) -> f32 {
    logistic(x)
}

pub fn sign(x: f64) -> f64 {
    if x >= 0.0 {
        1.0
    } else {
        -1.0
    }
}
