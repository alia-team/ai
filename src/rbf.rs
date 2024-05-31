use crate::utils;
extern crate rand;
use rand::Rng;

pub struct RBF {
    pub neurons_per_layer: Vec<usize>,
    pub centers: Vec<Vec<f64>>,
    pub weights: Vec<Vec<Vec<f64>>>,
    pub outputs: Vec<Vec<f64>>,
    pub gamma: f64,
    pub is_classification: bool,
}

impl RBF {
    pub fn new(
        neurons_per_layer: Vec<usize>,
        is_classification: bool,
        dataset: Vec<Vec<f64>>,
    ) -> Self {
        if neurons_per_layer.len() != 3 {
            panic!("A RBF neural network must contain only 3 layers.")
        }

        let centers = utils::init_centers(neurons_per_layer[1], dataset);
        let weights = utils::init_weights(neurons_per_layer.clone(), true);
        let outputs = utils::init_outputs(neurons_per_layer.clone(), true);
        let gamma = rand::thread_rng().gen_range(0.01..=1.0);

        RBF {
            neurons_per_layer,
            centers,
            weights,
            outputs,
            gamma,
            is_classification,
        }
    }
}
