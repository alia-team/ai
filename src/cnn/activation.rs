use ndarray::Array1;

pub trait Activation {
    fn forward(&self, input: Array1<f32>) -> Array1<f32>;
    fn backward(&self, gradients: Array1<f32>) -> Array1<f32>;
}

pub struct Softmax;
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

pub struct Sigmoid;
impl Activation for Sigmoid {
    fn forward(&self, input: Array1<f32>) -> Array1<f32> {
        input.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }

    fn backward(&self, gradients: Array1<f32>) -> Array1<f32> {
        gradients.mapv(|x| x * (1.0 - x))
    }
}

pub struct ReLU;
impl Activation for ReLU {
    fn forward(&self, input: Array1<f32>) -> Array1<f32> {
        input.mapv(|x| if x > 0.0 { x } else { 0.0 })
    }

    fn backward(&self, gradients: Array1<f32>) -> Array1<f32> {
        gradients.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }
}

pub struct Tanh;
impl Activation for Tanh {
    fn forward(&self, input: Array1<f32>) -> Array1<f32> {
        input.mapv(|x| x.tanh())
    }

    fn backward(&self, gradients: Array1<f32>) -> Array1<f32> {
        let tanh_values = self.forward(gradients);
        tanh_values.mapv(|x| 1.0 - x.powi(2))
    }
}
