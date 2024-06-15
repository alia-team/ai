pub type Activation = fn(f64) -> f64;

pub fn string_to_activation(string: &str) -> Activation {
    match string {
        "sign" => sign,
        "heaviside" => heaviside,
        "identity" => identity,
        "logistic" => logistic,
        "sigmoid" => sigmoid,
        "tanh" => tanh,
        _ => panic!("Not a supported activation function"),
    }
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

pub fn logistic(x: f64) -> f64 {
1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid = logistic;

pub fn tanh(x: f64) -> f64 {
    x.tanh()
}
