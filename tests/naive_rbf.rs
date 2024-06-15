use ai::naive_rbf::*;
extern crate rand;
use rand::Rng;

fn build_dataset(size: usize, clusters_count: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    let mut dataset: Vec<Vec<f64>> = vec![];

    for i in 1..=clusters_count {
        dataset.append(
            &mut (0..(size / clusters_count))
                .map(|_| {
                    let x = rng.gen::<f64>() * 0.9 + i as f64;
                    let y = rng.gen::<f64>() * 0.9 + i as f64;
                    vec![x, y]
                })
                .collect(),
        )
    }

    dataset
}

fn build_labels(size: usize, classes: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut labels: Vec<Vec<f64>> = vec![];
    let classes_count: usize = classes.len();

    for class in classes {
        for _ in 0..(size / classes_count) {
            labels.push(class.clone())
        }
    }

    labels
}

#[test]
fn centroid_new() {
    assert_eq!(
        Centroid::new(vec![1.0, 2.0]),
        Centroid {
            coordinates: vec![1.0, 2.0]
        }
    )
}

#[test]
fn centroid_forward() {
    let centroid: Centroid = Centroid::new(vec![1.0, 2.0]);
    let input: Vec<f64> = vec![2.0, 3.0];
    let gamma: f64 = 0.1;

    assert_eq!(centroid.forward(input, gamma), 0.8187307530779818);
}

#[test]
fn new() {
    let dataset_size: usize = 100;
    let clusters_count: usize = 2;
    let dataset: Vec<Vec<f64>> = build_dataset(dataset_size, clusters_count);
    let input_neurons_count: usize = 3;
    let output_neurons_count: usize = 3;
    let activation: &str = "sign";
    let naive_rbf: NaiveRBF = NaiveRBF::new(
        input_neurons_count,
        output_neurons_count,
        activation,
        dataset,
    );

    // Check outputs
    assert_eq!(naive_rbf.outputs.len(), 3);
    for layer in 0..naive_rbf.outputs.len() {
        // No bias check in a RBF
        assert_eq!(
            naive_rbf.outputs[layer].len(),
            naive_rbf.neurons_per_layer[layer]
        );
        for output in 0..naive_rbf.outputs[layer].len() {
            assert_eq!(naive_rbf.outputs[layer][output], 0.0);
        }
    }

    // Check weights
    assert_eq!(naive_rbf.weights.len(), 3);
    assert_eq!(naive_rbf.weights[2].len(), output_neurons_count);
    for neuron in 0..naive_rbf.weights[2].len() {
        assert_eq!(
            naive_rbf.weights[2][neuron].len(),
            naive_rbf.neurons_per_layer[1]
        )
    }

    // Check other parameters
    assert_eq!(naive_rbf.centroids.len(), dataset_size);
    assert!(naive_rbf.gamma <= 1.0 && naive_rbf.gamma >= 0.01);
}

#[test]
fn fit() {
    let dataset_size: usize = 100;
    let clusters_count: usize = 2;
    let classes: Vec<Vec<f64>> = vec![vec![1.0], vec![-1.0]];
    let training_dataset: Vec<Vec<f64>> = build_dataset(dataset_size, clusters_count);
    let input_neurons_count: usize = 2;
    let output_neurons_count: usize = 1;
    let labels: Vec<Vec<f64>> = build_labels(dataset_size, classes);
    let gamma: f64 = 0.01;
    let activation: &str = "sign";
    let mut naive_rbf: NaiveRBF = NaiveRBF::new(
        input_neurons_count,
        output_neurons_count,
        activation,
        training_dataset.clone(),
    );

    naive_rbf.fit(training_dataset, labels, gamma);

    assert_eq!(naive_rbf.weights.len(), 3);
    assert_eq!(naive_rbf.weights[2].len(), output_neurons_count);
    for neuron in 0..naive_rbf.weights[2].len() {
        assert_eq!(
            naive_rbf.weights[2][neuron].len(),
            naive_rbf.neurons_per_layer[1]
        )
    }
}

#[test]
fn predict() {
    let dataset_size: usize = 100;
    let clusters_count: usize = 2;
    let dataset: Vec<Vec<f64>> = build_dataset(dataset_size, clusters_count);
    let input_neurons_count: usize = 2;
    let output_neurons_count: usize = 1;
    let activation: &str = "sign";
    let mut naive_rbf: NaiveRBF = NaiveRBF::new(
        input_neurons_count,
        output_neurons_count,
        activation,
        dataset,
    );
    let input: Vec<f64> = vec![1.0, 2.0];
    let prediction: Vec<f64> = naive_rbf.predict(input);

    assert_eq!(prediction.len(), 1);
    assert!(prediction[0] == -1.0 || prediction[0] == 1.0)
}

#[test]
fn linear_simple() {
    let training_dataset: Vec<Vec<f64>> = vec![vec![1.0, 1.0], vec![2.0, 3.0], vec![3.0, 3.0]];
    let labels: Vec<Vec<f64>> = vec![vec![1.0], vec![-1.0], vec![-1.0]];
    let input_neurons_count: usize = 2;
    let output_neurons_count: usize = 1;
    let activation: &str = "sign";
    let mut model: NaiveRBF = NaiveRBF::new(
        input_neurons_count,
        output_neurons_count,
        activation,
        training_dataset.clone(),
    );

    println!("Initialization");
    println!("Centroids: {:?}", model.centroids);
    println!("Weights: {:?}", model.weights);
    println!("Outputs: {:?}", model.outputs);

    let gamma: f64 = 0.1;
    model.fit(training_dataset.clone(), labels.clone(), gamma);

    println!("Fitting");
    println!("Centroids: {:?}", model.centroids);
    println!("Weights: {:?}", model.weights);
    println!("Outputs: {:?}", model.outputs);

    for (i, input) in training_dataset.iter().enumerate() {
        let output: Vec<f64> = model.predict(input.to_vec());

        println!("Outputs {}: {:?}", i, output);
        println!("Centroids: {:?}", model.centroids);
        println!("Weights: {:?}", model.weights);
        println!("Outputs: {:?}", model.outputs);
    }
}
