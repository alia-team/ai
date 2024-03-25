use ai::tensor::*;

#[test]
fn new() {
    let data = [0.0, 1.0, 2.0, 3.0];
    let shape = [2, 2];

    let tensor = Tensor::new(&data, &shape);

    assert_eq!(tensor.data, data);
    assert_eq!(tensor.shape, shape);
}
