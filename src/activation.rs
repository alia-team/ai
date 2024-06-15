pub enum Activation {
    SIGN,
    HEAVISIDE,
    IDENTITY,
}

pub fn sign(x: f64) -> f64 {
    if x >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

pub fn heaviside(x: f64) -> f64 {
    if x >= 0.0 {
        1.0
    } else {
        0.0
    }
}

pub fn identity(x: f64) -> f64 {
    x
}
