use ai::rbf::*;
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

    assert_eq!(centroid.forward(&input, gamma), 0.8187307530779818);
}

#[test]
fn new() {
    let dataset_size: usize = 100;
    let clusters_count: usize = 2;
    let classes: Vec<Vec<f64>> = vec![vec![1.0], vec![-1.0]];
    let training_dataset: Vec<Vec<f64>> = build_dataset(dataset_size, clusters_count);
    let labels: Vec<Vec<f64>> = build_labels(dataset_size, classes);
    let centroids_count: usize = dataset_size / 10;
    let output_layer_neurons_count: usize = 1;
    let activation: &str = "sign";
    let rbf: RBF = RBF::new(
        &[2, centroids_count, output_layer_neurons_count],
        activation,
        &training_dataset,
        &labels,
    );

    // Check outputs
    assert_eq!(rbf.outputs.len(), 3);
    for layer in 0..rbf.outputs.len() {
        // No bias check in a RBF
        assert_eq!(rbf.outputs[layer].len(), rbf.neurons_per_layer[layer]);
        for output in 0..rbf.outputs[layer].len() {
            assert_eq!(rbf.outputs[layer][output], 0.0);
        }
    }

    // Check weights
    assert_eq!(rbf.weights.len(), 3);
    assert_eq!(rbf.weights[2].len(), output_layer_neurons_count);
    for neuron in 0..rbf.weights[2].len() {
        assert_eq!(rbf.weights[2][neuron].len(), rbf.neurons_per_layer[1])
    }

    // Check other parameters
    assert_eq!(rbf.centroids.len(), centroids_count);
    assert!(rbf.gamma <= 1.0 && rbf.gamma >= 0.01);
}

#[test]
#[should_panic(expected = "A RBF neural network must contain only 3 layers.")]
fn new_too_many_layers() {
    let training_dataset_size: usize = 2;
    let clusters_count: usize = 2;
    let classes: Vec<Vec<f64>> = vec![vec![1.0], vec![-1.0]];
    let training_dataset: Vec<Vec<f64>> = build_dataset(training_dataset_size, clusters_count);
    let labels: Vec<Vec<f64>> = build_labels(training_dataset_size, classes);
    let neurons_per_layer: Vec<usize> = vec![2, 2, 2, 1];
    let activation: &str = "sign";
    RBF::new(&neurons_per_layer, activation, &training_dataset, &labels);
}

#[test]
#[should_panic(expected = "Cannot have 10 centroids for 2 samples in dataset.")]
fn new_too_many_centroids() {
    let training_dataset_size: usize = 2;
    let clusters_count: usize = 2;
    let classes: Vec<Vec<f64>> = vec![vec![1.0], vec![-1.0]];
    let training_dataset: Vec<Vec<f64>> = build_dataset(training_dataset_size, clusters_count);
    let labels: Vec<Vec<f64>> = build_labels(training_dataset_size, classes);
    let neurons_per_layer: Vec<usize> = vec![2, 10, 1];
    let activation: &str = "sign";
    RBF::new(&neurons_per_layer, activation, &training_dataset, &labels);
}

#[test]
fn fit() {
    let dataset_size: usize = 100;
    let clusters_count: usize = 2;
    let classes: Vec<Vec<f64>> = vec![vec![1.0], vec![-1.0]];
    let training_dataset: Vec<Vec<f64>> = build_dataset(dataset_size, clusters_count);
    let labels: Vec<Vec<f64>> = build_labels(dataset_size, classes);
    let gamma: f64 = 0.01;
    let max_iterations: usize = 10;
    let output_layer_neurons_count: usize = 1;
    let activation: &str = "sign";
    let mut rbf: RBF = RBF::new(
        &[2, dataset_size, output_layer_neurons_count],
        activation,
        &training_dataset,
        &labels,
    );

    rbf.fit(&training_dataset, gamma, max_iterations);

    assert_eq!(rbf.weights.len(), 3);
    assert_eq!(rbf.weights[2].len(), output_layer_neurons_count);
    for neuron in 0..rbf.weights[2].len() {
        assert_eq!(rbf.weights[2][neuron].len(), rbf.neurons_per_layer[1])
    }
}

#[test]
fn predict() {
    let dataset_size: usize = 100;
    let clusters_count: usize = 2;
    let classes: Vec<Vec<f64>> = vec![vec![1.0], vec![-1.0]];
    let training_dataset: Vec<Vec<f64>> = build_dataset(dataset_size, clusters_count);
    let labels: Vec<Vec<f64>> = build_labels(dataset_size, classes);
    let output_layer_neurons_count: usize = 1;
    let activation: &str = "sign";
    let mut rbf: RBF = RBF::new(
        &[2, dataset_size, output_layer_neurons_count],
        activation,
        &training_dataset,
        &labels,
    );

    let input: Vec<f64> = vec![1.0, 2.0];
    let prediction: Vec<f64> = rbf.predict(&input);

    assert_eq!(prediction.len(), 1);
    assert!(prediction[0] == -1.0 || prediction[0] == 1.0)
}
