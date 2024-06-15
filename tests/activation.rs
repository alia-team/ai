use ai::activation;

#[test]
fn sign() {
    assert_eq!(activation::sign(-1.0), -1.0);
    assert_eq!(activation::sign(0.0), 1.0);
    assert_eq!(activation::sign(1.0), 1.0);
}

#[test]
fn logistic() {
    assert_eq!(activation::logistic(-1.0), 0.2689414213699951);
    assert_eq!(activation::logistic(0.0), 0.5);
    assert_eq!(activation::logistic(1.0), 0.7310585786300049);
}
