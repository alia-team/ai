use ndarray::{Array1, Array2};
use rand::{rngs::ThreadRng, Rng};
use rand_distr::{Distribution, Normal};

pub enum WeightsInit {
    He,
    NormalizedXavier,
    Xavier,
}

pub fn str_to_weights_init(str: &str) -> WeightsInit {
    match str.to_lowercase().as_str() {
        "he" => WeightsInit::He,
        "normalized_xavier" => WeightsInit::NormalizedXavier,
        "xavier" => WeightsInit::Xavier,
        _ => panic!("Not a supported weights initialization."),
    }
}

fn he(input_size: usize, output_size: usize) -> Array2<f64> {
    let mut thread_rng: ThreadRng = rand::thread_rng();
    let normal: Normal<f64> = Normal::new(0.0, (2.0 / input_size as f64).sqrt()).unwrap();
    Array2::<f64>::from_shape_fn((output_size, input_size), |_| {
        normal.sample(&mut thread_rng)
    })
}

fn xavier(input_size: usize, output_size: usize) -> Array2<f64> {
    let mut thread_rng: ThreadRng = rand::thread_rng();
    let normal: Normal<f64> =
        Normal::new(0.0, (2.0 / (input_size + output_size) as f64).sqrt()).unwrap();
    Array2::<f64>::from_shape_fn((output_size, input_size), |_| {
        normal.sample(&mut thread_rng)
    })
}

fn normalized_xavier(input_size: usize, output_size: usize) -> Array2<f64> {
    let limit: f64 = (6.0 / (input_size + output_size) as f64).sqrt();
    let mut rng: ThreadRng = rand::thread_rng();
    Array2::from_shape_fn((input_size, output_size), |_| rng.gen_range(-limit..limit))
}

pub fn init_biases(output_size: usize) -> Array1<f64> {
    Array1::<f64>::zeros(output_size)
}

pub fn init_dense_weights(init: WeightsInit, input_size: usize, output_size: usize) -> Array2<f64> {
    match init {
        WeightsInit::He => he(input_size, output_size),
        WeightsInit::Xavier => xavier(input_size, output_size),
        WeightsInit::NormalizedXavier => normalized_xavier(input_size, output_size),
    }
}
