use ai::activation;

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
