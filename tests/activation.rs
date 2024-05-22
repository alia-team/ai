use ai::activation;

#[test]
fn binary_step() {
    assert_eq!(activation::binary_step(-1.0), 0.0);
    assert_eq!(activation::binary_step(0.0), 1.0);
    assert_eq!(activation::binary_step(1.0), 1.0);
}

#[test]
fn identity() {
    assert_eq!(activation::identity(-1.0), -1.0);
    assert_eq!(activation::identity(0.0), 0.0);
    assert_eq!(activation::identity(1.0), 1.0);
}

#[test]
fn linear() {
    assert_eq!(activation::linear(-1.0), -1.0);
    assert_eq!(activation::linear(0.0), 0.0);
    assert_eq!(activation::linear(1.0), 1.0);
}

#[test]
fn logistic() {
    assert_eq!(activation::logistic(1.0), 0.7310585786300049);
    assert_eq!(activation::logistic(10.0), 0.9999546021312976);
    assert_eq!(activation::logistic(-5.0), 0.006692850924284857);
}

#[test]
fn sigmoid() {
    assert_eq!(activation::sigmoid(1.0), 0.7310585786300049);
    assert_eq!(activation::sigmoid(10.0), 0.9999546021312976);
    assert_eq!(activation::sigmoid(-5.0), 0.006692850924284857);
}

#[test]
fn sign() {
    assert_eq!(activation::sign(-1.0), -1.0);
    assert_eq!(activation::sign(0.0), 1.0);
    assert_eq!(activation::sign(1.0), 1.0);
}

#[test]
fn softmax() {
    assert_eq!(
        activation::softmax(vec![1.0, 2.0, 3.0]),
        vec![0.09003057317038046, 0.24472847105479767, 0.6652409557748219]
    );
    assert_eq!(
        activation::softmax(vec![-0.1, 3.8, 1.1, -0.3]),
        vec![
            0.018334730910579303,
            0.9057806106734848,
            0.06087345037003522,
            0.015011208045900745
        ]
    );
    assert_eq!(
        activation::softmax(vec![1.3, 5.1, 2.2, 0.7, 1.1]),
        vec![
            0.020190464732580685,
            0.9025376890165726,
            0.04966052987196013,
            0.011080761983386346,
            0.01653055439550022
        ]
    );
}
