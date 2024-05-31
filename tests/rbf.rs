use ai::rbf::*;
extern crate rand;
use rand::Rng;

#[test]
fn new() {
    let mut rng = rand::thread_rng();

    // Build dataset
    let dataset_size: usize = 100;
    let mut set1: Vec<Vec<f64>> = (0..(dataset_size / 2))
        .map(|_| {
            let x = rng.gen::<f64>() * 0.9 + 1.0;
            let y = rng.gen::<f64>() * 0.9 + 1.0;
            vec![x, y]
        })
        .collect();
    let mut set2: Vec<Vec<f64>> = (0..(dataset_size / 2))
        .map(|_| {
            let x = rng.gen::<f64>() * 0.9 + 2.0;
            let y = rng.gen::<f64>() * 0.9 + 2.0;
            vec![x, y]
        })
        .collect();
    let mut dataset = Vec::new();
    dataset.append(&mut set1);
    dataset.append(&mut set2);

    let hidden_layer_neurons_count: usize = dataset_size / 10;

    let rbf = RBF::new(vec![3, hidden_layer_neurons_count, 3], true, dataset);

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
    let expect_msg: &str = "output neuron should have weighted inputs";
    assert_eq!(rbf.weights.len(), 3);
    for neuron in 0..rbf.weights.last().expect(expect_msg).len() {
        assert_eq!(
            rbf.weights.last().expect(expect_msg)[neuron].len(),
            *rbf.neurons_per_layer.last().expect(expect_msg)
        )
    }

    // Check other parameters
    assert_eq!(rbf.centers.len(), hidden_layer_neurons_count);
    assert!(rbf.gamma <= 1.0 && rbf.gamma >= 0.01);
}
