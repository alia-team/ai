use ai::activation::*;
use ai::data::{load_image_dataset, Dataset1D, Dataset3D};
use ai::model::*;
use ai::optimizer::Optimizer;
use ai::weights_init::WeightsInit;
use ndarray::{array, Array1};

// WARNING: Comment the alia test case before push: the repo doesn't contain the required dataset.
/*
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
    let mut cnn: CNN = CNN::new(hyperparameters);
    cnn.set_input_shape(vec![100, 100, 1]);
    cnn.add_conv2d_layer(8, 3);
    cnn.add_maxpool2d_layer(2);
    cnn.add_dense_layer(128, Box::new(ReLU), Some(0.1), WeightsInit::He);
    cnn.add_dense_layer(3, Box::new(Softmax), None, WeightsInit::Xavier);
    println!("CNN built.");

    println!("Fitting...");
    cnn.fit(dataset);
    println!("Fitting done.");

    println!("Saving model...");
    let path: &str = "./models/";
    let model_name: &str = "cnn";
    let full_path: String = cnn.save(path, model_name);
    println!("Model saved to {}.", full_path);

    println!("Loading model...");
    let mut loaded_model: CNN = CNN::load(&full_path);
    println!("Model loaded.");
    println!("Predicting... It should predicts a Phidippus.");
    let image_path: &str = "dataset/phidippus/835255150-388.png";
    let output: Array1<f64> = loaded_model.predict(image_path);

    // Get predicted class name
    let max_output: usize = output
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
        .unwrap();
    let predicted_class: &str = match max_output {
        0 => "Avicularia",
        1 => "Phidippus",
        2 => "Tegenaria",
        _ => "An error occured.",
    };
    println!("Predicted: {:?}", predicted_class);
    assert_eq!(predicted_class, "Phidippus");
}
*/

#[test]
fn xor() {
    println!("Loading dataset...");
    let samples: Vec<Array1<f64>> = vec![
        array![0., 0.],
        array![0., 1.],
        array![1., 0.],
        array![1., 1.],
    ];
    let targets: Vec<u8> = vec![0, 1, 1, 0];
    let training_ratio: f64 = 1.;
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
    let mut mlp: MLP = MLP::new(input_size, hyperparameters);
    mlp.add_layer(2, Box::new(Sigmoid), None, WeightsInit::Xavier);
    mlp.add_layer(4, Box::new(Sigmoid), None, WeightsInit::Xavier);
    mlp.add_layer(1, Box::new(Sigmoid), None, WeightsInit::Xavier);
    println!("MLP built.");

    println!("Fitting...");
    mlp.fit(dataset);
    println!("Fitting done.");
}
