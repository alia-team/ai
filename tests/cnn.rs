use ai::cnn;

#[test]
fn max_pool_2d() {
    let input: Vec<Vec<f64>> = vec![
        vec![12.0, 20.0, 30.0, 0.0],
        vec![8.0, 12.0, 2.0, 0.0],
        vec![34.0, 70.0, 37.0, 4.0],
        vec![112.0, 100.0, 25.0, 12.0],
    ];
    let pool_size: &[usize; 2] = &[2, 2];
    let stride: usize = 2;
    let padding: &str = "valid";
    let expected_output: Vec<Vec<f64>> = vec![vec![20.0, 30.0], vec![112.0, 37.0]];
    assert_eq!(
        cnn::max_pool_2d(&input, pool_size, stride, padding),
        expected_output
    );

    let padding: &str = "same";
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

    let pool_size: &[usize; 2] = &[3, 2];
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
