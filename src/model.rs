use crate::layer::{Dense, Layer};
use crate::loss::Loss;
use crate::optimizer::Optimizer;
use ndarray::{s, Array1, Array2, Axis};
use ndarray_rand::rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

pub struct Model {
    pub layers: Vec<Box<dyn Layer>>,
}

pub struct History {
    pub epochs: Vec<usize>,
    pub training_loss: Vec<f64>,
    pub training_accuracy: Vec<f64>,
    pub validation_loss: Option<Vec<f64>>,
    pub validation_accuracy: Option<Vec<f64>>,
}

impl Model {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Model { layers }
    }

    pub fn fit(
        &mut self,
        training_dataset: &Array2<f64>,
        validation_dataset: Option<&Array2<f64>>,
        targets: &Array2<f64>,
        learning_rate: f64,
        batch_size: usize,
        epochs: usize,
        optimizer: Box<dyn Optimizer>,
        loss: Box<dyn Loss>,
    ) -> History {
        let mut rng = StdRng::from_entropy();
        let mut history = History {
            epochs: Vec::new(),
            training_loss: Vec::new(),
            training_accuracy: Vec::new(),
            validation_loss: validation_dataset.map(|_| Vec::new()),
            validation_accuracy: validation_dataset.map(|_| Vec::new()),
        };

        let n_samples = training_dataset.nrows();
        let n_batches = n_samples / batch_size;

        for epoch in 0..epochs {
            let mut batch_indices: Vec<usize> = (0..n_samples).collect();
            batch_indices.shuffle(&mut rng);

            for batch in 0..n_batches {
                let start = batch * batch_size;
                let end = start + batch_size;
                let batch_indices = &batch_indices[start..end];

                let batch_inputs = training_dataset.select(Axis(0), batch_indices);
                let batch_targets = targets.select(Axis(0), batch_indices);

                let predictions = self.forward_pass(&batch_inputs);
                let gradients = loss.compute_gradients(&predictions, &batch_targets);

                self.backward_pass(&gradients);
                self.update_weights(&optimizer, learning_rate);
            }

            let train_predictions = self.forward_pass(training_dataset);
            let train_loss = loss.compute_loss(&train_predictions, targets);
            let train_accuracy = self.calculate_accuracy(&train_predictions, targets);

            history.epochs.push(epoch);
            history.training_loss.push(train_loss);
            history.training_accuracy.push(train_accuracy);

            if let Some(val_data) = validation_dataset {
                let val_predictions = self.forward_pass(val_data);
                let val_targets = &targets.slice(s![0..val_data.nrows(), ..]);
                let val_loss =
                    loss.compute_loss(&val_predictions.to_owned(), &val_targets.to_owned());
                let val_accuracy =
                    self.calculate_accuracy(&val_predictions.to_owned(), &val_targets.to_owned());

                history.validation_loss.as_mut().unwrap().push(val_loss);
                history
                    .validation_accuracy
                    .as_mut()
                    .unwrap()
                    .push(val_accuracy);
            }
        }

        history
    }

    pub fn predict(&mut self, input: &Array1<f64>) -> Array1<f64> {
        let mut output = input.clone();

        for layer in self.layers.iter_mut() {
            output = layer.forward(&output);
        }

        output
    }

    fn forward_pass(&mut self, inputs: &Array2<f64>) -> Array2<f64> {
        let mut outputs =
            Array2::<f64>::zeros((inputs.nrows(), self.layers.last().unwrap().output_size()));

        for (i, input) in inputs.outer_iter().enumerate() {
            let mut output = input.to_owned();
            for layer in self.layers.iter_mut() {
                output = layer.forward(&output);
            }
            outputs.row_mut(i).assign(&output);
        }

        outputs
    }

    fn backward_pass(&mut self, gradients: &Array2<f64>) {
        let mut current_gradients = gradients.clone();

        for layer in self.layers.iter_mut().rev() {
            let mut new_gradients =
                Array2::<f64>::zeros((current_gradients.nrows(), layer.input_size()));

            for (i, gradient) in current_gradients.outer_iter().enumerate() {
                let new_gradient = layer.backward(&gradient.to_owned());

                // Ensure the new gradient has the correct shape
                if new_gradient.len() != layer.input_size() {
                    panic!("Layer backward pass returned incorrect gradient shape. Expected: {}, Got: {}", 
                       layer.input_size(), new_gradient.len());
                }

                new_gradients.row_mut(i).assign(&new_gradient);
            }

            current_gradients = new_gradients;
        }
    }

    fn update_weights(&mut self, optimizer: &Box<dyn Optimizer>, learning_rate: f64) {
        for layer in self.layers.iter_mut() {
            if let Some(dense_layer) = layer.as_any_mut().downcast_mut::<Dense>() {
                if let (Some(weights), Some(weight_gradients)) = (
                    dense_layer.weights.as_mut(),
                    dense_layer.weight_gradients.as_ref(),
                ) {
                    optimizer.update_weights(weights, weight_gradients, learning_rate);
                } else {
                    panic!("Layer weights or gradients are not initialized");
                }
            }
        }
    }

    fn calculate_accuracy(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        let mut correct = 0;
        for (pred, target) in predictions.outer_iter().zip(targets.outer_iter()) {
            if Self::argmax(&pred.to_owned()) == Self::argmax(&target.to_owned()) {
                correct += 1;
            }
        }
        correct as f64 / targets.nrows() as f64
    }

    fn argmax(array: &Array1<f64>) -> usize {
        let mut max_index = 0;
        let mut max_value = array[0];
        for (i, &value) in array.iter().enumerate().skip(1) {
            if value > max_value {
                max_value = value;
                max_index = i;
            }
        }
        max_index
    }
}
