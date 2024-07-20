use ndarray::{Array1, Array2};
use rand::{rngs::ThreadRng, Rng};
use rand_distr::{Distribution, Normal};

pub enum WeightsInit {
    He,
    Xavier,
    NormalizedXavier,
}

fn he(input_size: usize, output_size: usize) -> Array2<f32> {
    let mut thread_rng: ThreadRng = rand::thread_rng();
    let normal: Normal<f32> = Normal::new(0.0, (2.0 / input_size as f32).sqrt()).unwrap();
    Array2::<f32>::from_shape_fn((output_size, input_size), |_| {
        normal.sample(&mut thread_rng)
    })
}

fn xavier(input_size: usize, output_size: usize) -> Array2<f32> {
    let mut thread_rng: ThreadRng = rand::thread_rng();
    let normal: Normal<f32> =
        Normal::new(0.0, (2.0 / (input_size + output_size) as f32).sqrt()).unwrap();
    Array2::<f32>::from_shape_fn((output_size, input_size), |_| {
        normal.sample(&mut thread_rng)
    })
}

fn normalized_xavier(input_size: usize, output_size: usize) -> Array2<f32> {
    let limit: f32 = (6.0 / (input_size + output_size) as f32).sqrt();
    let mut rng: ThreadRng = rand::thread_rng();
    Array2::from_shape_fn((input_size, output_size), |_| rng.gen_range(-limit..limit))
}

pub fn init_biases(output_size: usize) -> Array1<f32> {
    Array1::<f32>::zeros(output_size)
}

pub fn init_dense_weights(init: WeightsInit, input_size: usize, output_size: usize) -> Array2<f32> {
    match init {
        WeightsInit::He => he(input_size, output_size),
        WeightsInit::Xavier => xavier(input_size, output_size),
        WeightsInit::NormalizedXavier => normalized_xavier(input_size, output_size),
    }
}
