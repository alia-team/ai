use crate::utils;
extern crate rand;
use rand::Rng;

#[derive(Debug, PartialEq)]
pub struct Center {
    pub coordinates: Vec<f64>,
}

impl Center {
    pub fn new(coordinates: Vec<f64>) -> Self {
        Center { coordinates }
    }

    pub fn forward(self, input: Vec<f64>, gamma: f64) -> f64 {
        let mut vec_sub: Vec<f64> = vec![];
        for (i, value) in input.iter().enumerate() {
            vec_sub.push(value - self.coordinates[i])
        }

        let mut norm: f64 = 0.0;
        for value in vec_sub {
            norm += value.powi(2)
        }
        norm = norm.sqrt();

        (-gamma * norm.powi(2)).exp()
    }
}

pub struct RBF {
    pub neurons_per_layer: Vec<usize>,
    pub centers: Vec<Center>,
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

        // Initialize centers
        let mut centers: Vec<Center> = vec![];
        for _ in 0..neurons_per_layer[1] {
            centers.push(Center::new(
                dataset[rand::thread_rng().gen_range(0..dataset.len())].clone(),
            ));
        }

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
