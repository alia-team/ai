use ai::tensor::*;

#[test]
fn new() {
    let data = [0.0, 1.0, 2.0, 3.0];
    let shape = [2, 2];
    let tensor = Tensor::new(&data, &shape).unwrap();

    assert_eq!(tensor.data, data);
    assert_eq!(tensor.shape, shape);

    let data = [0.0];
    let shape = [2, 2];
    let tensor = Tensor::new(&data, &shape);
    assert_eq!(
        tensor,
        Err(TensorError::ShapeDataMismatch {
            shape_elements: 4,
            data_elements: 1
        })
    )
}

#[test]
fn flat_index() {
    let data = [0.0, 1.0, 2.0, 3.0];
    let shape = [2, 2];
    let tensor = Tensor::new(&data, &shape).unwrap();

    assert_eq!(tensor.flat_index(&[1, 1]), Ok(3));
    assert_eq!(
        tensor.flat_index(&[1, 2]),
        Err(TensorError::IndexOutOfBounds {
            dimension: 1,
            index: 2,
            bound: 1
        })
    );

    let data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let shape = [2, 2, 2];
    let tensor = Tensor::new(&data, &shape).unwrap();

    assert_eq!(tensor.flat_index(&[0, 1, 0]), Ok(2));
    assert_eq!(
        tensor.flat_index(&[0, 0, 0, 0]),
        Err(TensorError::DimensionMismatch {
            expected: 3,
            found: 4
        })
    );
}

#[test]
fn get() {
    let data = [0.0, 1.0, 2.0, 3.0];
    let shape = [2, 2];
    let tensor = Tensor::new(&data, &shape).unwrap();

    assert_eq!(tensor.get(&[0, 0]), Ok(0.0));
    assert_eq!(tensor.get(&[0, 1]), Ok(1.0));
    assert_eq!(tensor.get(&[1, 0]), Ok(2.0));
    assert_eq!(tensor.get(&[1, 1]), Ok(3.0));
    assert_eq!(
        tensor.get(&[1, 2]),
        Err(TensorError::IndexOutOfBounds {
            dimension: 1,
            index: 2,
            bound: 1
        })
    );
    assert_eq!(
        tensor.get(&[0, 0, 0]),
        Err(TensorError::DimensionMismatch {
            expected: 2,
            found: 3
        })
    );
}
