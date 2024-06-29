use ai::cnn::{self, Activation, Padding};

#[test]
fn max_pool_2d() {
    let input: Vec<Vec<f64>> = vec![
        vec![12.0, 20.0, 30.0, 0.0],
        vec![8.0, 12.0, 2.0, 0.0],
        vec![34.0, 70.0, 37.0, 4.0],
        vec![112.0, 100.0, 25.0, 12.0],
    ];
    let pool_size: (usize, usize) = (2, 2);
    let stride: usize = 2;
    let padding: Padding = Padding::Valid;
    let expected_output: Vec<Vec<f64>> = vec![vec![20.0, 30.0], vec![112.0, 37.0]];
    assert_eq!(
        cnn::max_pool_2d(&input, pool_size, stride, padding),
        expected_output
    );

    let padding: Padding = Padding::Same;
    let expected_output: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0, 0.0, 0.0],
        vec![0.0, 20.0, 30.0, 0.0],
        vec![0.0, 112.0, 37.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0],
    ];
    assert_eq!(
        cnn::max_pool_2d(&input, pool_size, stride, padding),
        expected_output
    );

    let pool_size: (usize, usize) = (3, 2);
    let stride: usize = 1;
    let expected_output: Vec<Vec<f64>> = vec![
        vec![20.0, 30.0, 30.0, 0.0],
        vec![70.0, 70.0, 37.0, 4.0],
        vec![112.0, 100.0, 37.0, 12.0],
        vec![112.0, 100.0, 37.0, 12.0],
    ];
    assert_eq!(
        cnn::max_pool_2d(&input, pool_size, stride, padding),
        expected_output
    );
}

#[test]
fn conv_2d() {
    let input: Vec<Vec<f64>> = vec![
        vec![7.0, 2.0, 3.0, 3.0, 8.0],
        vec![4.0, 5.0, 3.0, 8.0, 4.0],
        vec![3.0, 3.0, 2.0, 8.0, 4.0],
        vec![2.0, 8.0, 7.0, 2.0, 7.0],
        vec![5.0, 4.0, 4.0, 5.0, 4.0],
    ];
    let strides = (1, 1);
    let kernel = Some(vec![
        vec![1.0, 0.0, -1.0],
        vec![1.0, 0.0, -1.0],
        vec![1.0, 0.0, -1.0],
    ]);
    let kernel_size = None;
    let filters = 1;
    let padding = Padding::Valid;
    let activation = Activation::Identity;
    let output = cnn::conv_2d(
        &input,
        strides,
        kernel,
        kernel_size,
        filters,
        padding,
        activation,
    );
    let expected_output: Vec<Vec<f64>> = vec![
        vec![6.0, -9.0, -8.0],
        vec![-3.0, -2.0, -3.0],
        vec![-3.0, 0.0, -2.0],
    ];
    assert_eq!(output, expected_output);
}
