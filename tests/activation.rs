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
    assert_eq!(activation::logistic(1.0), 0.7310586);
    assert_eq!(activation::logistic(10.0), 0.9999546);
    assert_eq!(activation::logistic(-5.0), 0.0066928524);
}

#[test]
fn sigmoid() {
    assert_eq!(activation::sigmoid(1.0), 0.7310586);
    assert_eq!(activation::sigmoid(10.0), 0.9999546);
    assert_eq!(activation::sigmoid(-5.0), 0.0066928524);
}

#[test]
fn sign() {
    assert_eq!(activation::sign(-1.0), -1.0);
    assert_eq!(activation::sign(0.0), 1.0);
    assert_eq!(activation::sign(1.0), 1.0);
}
