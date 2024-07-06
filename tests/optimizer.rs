use ai::optimizer::*;
use ndarray::Array2;

#[test]
fn sgd() {
    let mut weights = Array2::from_elem((2, 2), 0.5);
    let gradients = Array2::from_elem((2, 2), 0.1);
    let learning_rate = 0.01;

    let optimizer = SGD::new();
    optimizer.update_weights(&mut weights, &gradients, learning_rate);

    println!("{:?}", weights);
}
