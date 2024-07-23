use ai::cnn::activation::*;
use ai::cnn::data::{load_image_dataset, Dataset1D, Dataset3D};
use ai::cnn::model::*;
use ai::cnn::optimizer::Optimizer;
use ai::cnn::weights_init::WeightsInit;
use ndarray::{array, Array1};

// WARNING: Comment the alia test case before push: the repo doesn't contain the required dataset.
#[test]
fn alia() {
    println!("Loading dataset...");
    let dataset: Dataset3D =
        load_image_dataset("./dataset/", 0.8, Some(100)).expect("Failed to load image dataset.");
    println!(
        "Loaded {} training images and {} testing images.",
        dataset.training_size, dataset.testing_size
    );

    let hyperparameters: Hyperparameters = Hyperparameters {
        batch_size: 10,
        epochs: 10,
        optimizer: Optimizer::Adam(0.001, 0.9, 0.9),
    };

    println!("Building CNN...");
    let mut cnn: CNN = CNN::new(dataset, hyperparameters);
    cnn.set_input_shape(vec![100, 100, 3]);
    cnn.add_conv2d_layer(8, 3);
    cnn.add_maxpool2d_layer(2);
    cnn.add_dense_layer(128, Box::new(ReLU), Some(0.25), WeightsInit::He);
    cnn.add_dense_layer(64, Box::new(ReLU), Some(0.25), WeightsInit::He);
    cnn.add_dense_layer(10, Box::new(Softmax), None, WeightsInit::Xavier);
    println!("CNN built.");

    println!("Fitting...");
    cnn.fit();
    println!("Fitting done.");

    println!("Saving model...");
    let path: String = String::from("./models/");
    let model_name: String = String::from("cnn");
    let full_path: String = cnn.save(&path, &model_name);
    println!("Model saved to {}.", full_path);
}

#[test]
fn xor() {
    println!("Loading dataset...");
    let samples: Vec<Array1<f32>> = vec![
        array![0., 0.],
        array![0., 1.],
        array![1., 0.],
        array![1., 1.],
    ];
    let targets: Vec<u8> = vec![0, 1, 1, 0];
    let training_ratio: f32 = 1.;
    let max_samples_per_class: Option<u8> = None;
    let dataset: Dataset1D =
        Dataset1D::new(samples, targets, training_ratio, max_samples_per_class);
    println!(
        "Loaded {} training samples and {} testing samples.",
        dataset.training_size, dataset.testing_size
    );

    let input_size: usize = 2;
    let hyperparameters: Hyperparameters = Hyperparameters {
        batch_size: 1,
        epochs: 100,
        optimizer: Optimizer::SGD(0.001),
    };

    println!("Building MLP...");
    let mut mlp: MLP = MLP::new(dataset, input_size, hyperparameters);
    mlp.add_layer(2, Box::new(Sigmoid), None, WeightsInit::Xavier);
    mlp.add_layer(4, Box::new(Sigmoid), None, WeightsInit::Xavier);
    mlp.add_layer(1, Box::new(Sigmoid), None, WeightsInit::Xavier);
    println!("MLP built.");

    println!("Fitting...");
    mlp.fit();
    println!("Fitting done.");
}
