use ai::cnn::activation::{ReLU, Softmax};
use ai::cnn::data::{TrainImage, TrainingData};
use ai::cnn::model::*;
use ai::cnn::optimizer::Optimizer;
use ai::cnn::weights_init::WeightsInit;
use ndarray::Array3;
use rust_mnist::Mnist;
use std::collections::HashMap;
use std::path::Path;

pub fn load_mnist<T>(mnist_path: T) -> TrainingData
where
    T: AsRef<Path>,
{
    let (rows, cols) = (28, 28);
    let mnist_path = mnist_path.as_ref();
    let mnist = Mnist::new(mnist_path.to_str().unwrap());

    let mut trn_img = Vec::<TrainImage>::new();
    let mut trn_lbl = Vec::<usize>::new();
    let mut tst_img = Vec::<TrainImage>::new();
    let mut tst_lbl = Vec::<usize>::new();

    // Make unpacked folder inside mnist_path
    let mnist_path = mnist_path.join("unpacked");
    if !mnist_path.exists() {
        std::fs::create_dir(mnist_path.as_path()).expect("Failed to create unpacked folder.");
    }

    for i in 0..1000 {
        let mut img: Array3<f32> = Array3::<f32>::zeros((rows, cols, 1));

        for j in 0..rows {
            for k in 0..cols {
                img[[j, k, 0]] = mnist.train_data[i][(j * 28 + k) as usize] as f32 / 255.0;
            }
        }
        trn_img.push(TrainImage::Image(img));
        trn_lbl.push(mnist.train_labels[i] as usize);
    }

    for i in 0..100 {
        let mut img: Array3<f32> = Array3::<f32>::zeros((rows, cols, 1));

        for j in 0..rows {
            for k in 0..cols {
                img[[j, k, 0]] = mnist.test_data[i][(j * 28 + k) as usize] as f32 / 255.0;
            }
        }
        tst_img.push(TrainImage::Image(img));
        tst_lbl.push(mnist.test_labels[i] as usize);
    }

    // 'classes' allows us to only train on a subset of the data
    // Here, we use all 10 classes

    let classes: HashMap<usize, usize> = (0..10).enumerate().collect();

    let training_data: TrainingData = TrainingData {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        rows,
        cols,
        trn_size: 1000,
        tst_size: 100,
        classes,
    };

    training_data
}

#[test]
fn mnist() {
    // Load MNIST dataset
    let data = load_mnist("./data/");

    // Set hyperparameters
    let hyperparameters = Hyperparameters {
        batch_size: 10,
        epochs: 10,
        optimizer: Optimizer::Adam(0.001, 0.9, 0.9),
        ..Hyperparameters::default()
    };

    // Create CNN architecture
    let mut cnn = CNN::new(data, hyperparameters);
    cnn.set_input_shape(vec![28, 28, 3]);
    cnn.add_conv_layer(8, 3);
    cnn.add_mxpl_layer(2);
    cnn.add_dense_layer(128, Box::new(ReLU), Some(0.25), WeightsInit::He);
    cnn.add_dense_layer(64, Box::new(ReLU), Some(0.25), WeightsInit::He);
    cnn.add_dense_layer(10, Box::new(Softmax), None, WeightsInit::Xavier);

    cnn.fit();
}
