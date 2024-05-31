use ai::activation;

#[test]
fn sign() {
    assert_eq!(activation::sign(-1.0), -1.0);
    assert_eq!(activation::sign(0.0), 1.0);
    assert_eq!(activation::sign(1.0), 1.0);
}
