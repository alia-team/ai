use ai::cnn::activation::{ReLU, Softmax};
use ai::cnn::data::{load_image_dataset, TrainingData};
use ai::cnn::model::*;
use ai::cnn::optimizer::Optimizer;
use ai::cnn::weights_init::WeightsInit;

#[test]
fn alia() {
    println!("Loading dataset...");
    let dataset: TrainingData =
        load_image_dataset("./dataset/", 0.8, Some(100)).expect("Failed to load image dataset.");
    println!(
        "Loaded {} training images and {} testing images.",
        dataset.trn_size, dataset.tst_size
    );

    let hyperparameters: Hyperparameters = Hyperparameters {
        batch_size: 10,
        epochs: 10,
        optimizer: Optimizer::Adam(0.001, 0.9, 0.9),
    };

    println!("Building CNN...");
    let mut cnn: CNN = CNN::new(dataset, hyperparameters);
    cnn.set_input_shape(vec![100, 100, 3]);
    cnn.add_conv_layer(8, 3);
    cnn.add_mxpl_layer(2);
    cnn.add_dense_layer(128, Box::new(ReLU), Some(0.25), WeightsInit::He);
    cnn.add_dense_layer(64, Box::new(ReLU), Some(0.25), WeightsInit::He);
    cnn.add_dense_layer(10, Box::new(Softmax), None, WeightsInit::Xavier);
    println!("CNN built.");

    println!("Fitting...");
    cnn.fit();
    println!("Fitting done.");
}
