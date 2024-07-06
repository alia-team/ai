use ai::activation::ActivationEnum;
use ai::layer::{Dense, Layer};
use ai::loss::MSE;
use ai::model::*;
use ai::optimizer::SGD;
use ndarray::array;

#[test]
fn new() {
    let layer1: Box<dyn Layer> = Box::new(Dense::new(3, ActivationEnum::ReLU));
    let layer2: Box<dyn Layer> = Box::new(Dense::new(2, ActivationEnum::Sigmoid));
    let model: Model = Model::new(vec![layer1, layer2]);
    assert_eq!(model.layers.len(), 2);
}

#[test]
fn fit() {
    let layer1 = Box::new(Dense::new(3, ActivationEnum::ReLU));
    let layer2 = Box::new(Dense::new(2, ActivationEnum::Sigmoid));
    let mut model = Model::new(vec![layer1, layer2]);

    let inputs = array![[0.5, 0.8, 0.2], [0.1, 0.4, 0.6]];
    let targets = array![[1.0, 0.0], [0.0, 1.0]];

    let optimizer = Box::new(SGD::new());
    let loss = Box::new(MSE::new());

    let history = model.fit(&inputs, None, &targets, 0.01, 1, 10, optimizer, loss);

    assert_eq!(history.epochs.len(), 10);
    assert!(history.training_loss.len() > 0);
}

#[test]
fn predict() {
    let layer1 = Box::new(Dense::new(3, ActivationEnum::ReLU));
    let layer2 = Box::new(Dense::new(2, ActivationEnum::Sigmoid));
    let mut model = Model::new(vec![layer1, layer2]);

    let input = array![0.5, 0.8, 0.2];
    let output = model.predict(&input);

    assert_eq!(output.len(), 2);
}
