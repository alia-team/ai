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

#[test]
fn center_new() {
    assert_eq!(
        Center::new(vec![1.0, 2.0]),
        Center {
            coordinates: vec![1.0, 2.0]
        }
    )
}

#[test]
fn center_forward() {
    let center: Center = Center::new(vec![1.0, 2.0]);
    let input: Vec<f64> = vec![2.0, 3.0];
    let gamma: f64 = 0.1;

    assert_eq!(center.forward(input, gamma), 0.8187307530779818);
}

#[test]
fn rbf_new() {
    let dataset_size: usize = 100;
    let clusters_count: usize = 2;
    let dataset: Vec<Vec<f64>> = build_dataset(dataset_size, clusters_count);
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
