pub struct LinearRegression {
    pub weights: Vec<f64>,
}

fn invert_matrix(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let size = matrix.len();
    let mut augmented_matrix = vec![vec![0.0; size * 2]; size];

    // Create augmented matrix [matrix | I]
    for i in 0..size {
        for j in 0..size {
            augmented_matrix[i][j] = matrix[i][j];
        }
        augmented_matrix[i][i + size] = 1.0;
    }

    // Perform Gauss-Jordan elimination
    for i in 0..size {
        // Make the diagonal contain all 1's
        let diag_val = augmented_matrix[i][i];
        for j in 0..size * 2 {
            augmented_matrix[i][j] /= diag_val;
        }

        // Make the other columns contain 0's
        for k in 0..size {
            if k != i {
                let val = augmented_matrix[k][i];
                for j in 0..size * 2 {
                    augmented_matrix[k][j] -= val * augmented_matrix[i][j];
                }
            }
        }
    }

    // Extract the inverse matrix
    let mut inverse_matrix = vec![vec![0.0; size]; size];
    for i in 0..size {
        for j in 0..size {
            inverse_matrix[i][j] = augmented_matrix[i][j + size];
        }
    }

    inverse_matrix
}

impl LinearRegression {
    pub fn new(input_size: usize) -> LinearRegression {
        LinearRegression {
            weights: vec![0.0; input_size + 1],
        }
    }

    pub fn fit(&mut self, data_points: &Vec<Vec<f64>>, target_values: &[f64]) {
        let mut augmented_data_points: Vec<Vec<f64>> = Vec::new();
        for data_point in data_points {
            let mut augmented_point = data_point.clone();
            augmented_point.insert(0, 1.0); // Add intercept term
            augmented_data_points.push(augmented_point);
        }

        let x = augmented_data_points;
        let y = target_values;

        // Calculate X^T * X
        let mut x_t_x = vec![vec![0.0; x[0].len()]; x[0].len()];
        for i in 0..x.len() {
            for (j, row) in x_t_x.iter_mut().enumerate().take(x[0].len()) {
                for (k, val) in row.iter_mut().enumerate().take(x[0].len()) {
                    *val += x[i][j] * x[i][k];
                }
            }
        }

        // Calculate (X^T * X)^(-1)
        let x_t_x_inv = invert_matrix(&x_t_x);

        // Calculate X^T * y
        let mut x_t_y = vec![0.0; x[0].len()];
        for i in 0..x.len() {
            for (j, val) in x_t_y.iter_mut().enumerate().take(x[0].len()) {
                *val += x[i][j] * y[i];
            }
        }

        // Calculate weights = (X^T * X)^(-1) * (X^T * y)
        let mut weights = vec![0.0; x[0].len()];
        for (i, weight) in weights.iter_mut().enumerate() {
            *weight = x_t_x_inv[i]
                .iter()
                .zip(x_t_y.iter())
                .map(|(&a, &b)| a * b)
                .sum();
        }

        self.weights = weights;
    }

    pub fn predict(&self, input: &[f64]) -> f64 {
        let mut weighted_sum: f64 = self.weights[0]; // Intercept
        for (i, x) in input.iter().enumerate() {
            weighted_sum += self.weights[i + 1] * x;
        }
        weighted_sum
    }
}

#[no_mangle]
pub extern "C" fn new_linear_regression(input_size: usize) -> *mut LinearRegression {
    Box::into_raw(Box::new(LinearRegression::new(input_size)))
}

/// # Safety
///
/// This function is unsafe because:
///
/// - It dereferences raw pointers (`ptr`, `data_points`, and `target_values`). The caller must ensure that:
///   - `ptr` is a valid, non-null pointer to a `LinearRegression` struct.
///   - `data_points` is a valid, non-null pointer to an array of `n_points` pointers, each pointing to an array of `n_features` f64 values.
///   - `target_values` is a valid, non-null pointer to an array of `n_points` f64 values.
/// - The memory referenced by these pointers must be properly aligned and initialized.
/// - The `n_points` and `n_features` parameters must accurately represent the dimensions of the data.
/// - The lifetimes of the data pointed to by `data_points` and `target_values` must outlive this function call.
/// - No other part of the program should mutate the data referenced by these pointers while this function is executing.
///
/// Calling this function with invalid pointers or incorrect sizes may lead to undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn fit_linear_regression(
    ptr: *mut LinearRegression,
    data_points: *const *const f64,
    target_values: *const f64,
    n_points: usize,
    n_features: usize,
) {
    let linear_regression = unsafe { &mut *ptr };
    let data_points: Vec<Vec<f64>> = unsafe {
        (0..n_points)
            .map(|i| std::slice::from_raw_parts(*data_points.add(i), n_features).to_vec())
            .collect()
    };
    let target_values = unsafe { std::slice::from_raw_parts(target_values, n_points) };

    linear_regression.fit(&data_points, target_values);
}

/// # Safety
///
/// This function is unsafe because:
///
/// - It dereferences raw pointers (`ptr` and `new_point`). The caller must ensure that:
///   - `ptr` is a valid, non-null pointer to a `LinearRegression` struct.
///   - `new_point` is a valid, non-null pointer to an array of `n_features` f64 values.
/// - The memory referenced by these pointers must be properly aligned and initialized.
/// - The `n_features` parameter must accurately represent the number of features in the `new_point` array.
/// - The `n_features` must match the number of features used when the model was trained.
/// - The lifetime of the data pointed to by `new_point` must outlive this function call.
/// - No other part of the program should mutate the data referenced by these pointers while this function is executing.
///
/// Calling this function with invalid pointers or incorrect sizes may lead to undefined behavior.
/// Additionally, using a model that hasn't been properly trained or with mismatched feature counts may lead to incorrect predictions.
#[no_mangle]
pub unsafe extern "C" fn predict_linear_regression(
    ptr: *mut LinearRegression,
    new_point: *const f64,
    n_features: usize,
) -> f64 {
    let linear_regression = unsafe { &mut *ptr };
    let new_point = unsafe { std::slice::from_raw_parts(new_point, n_features) };
    linear_regression.predict(new_point)
}

/// # Safety
///
/// This function is unsafe because:
///
/// - It deallocates memory pointed to by a raw pointer (`ptr`). The caller must ensure that:
///   - `ptr` is either null or a valid pointer to a `LinearRegression` struct that was previously
///     created by a corresponding allocation function (likely `Box::into_raw()`).
///   - This function is called exactly once for each allocated `LinearRegression`.
///   - After this function is called, `ptr` must not be used again, as it now points to deallocated memory.
/// - No other part of the program should be concurrently using or deallocating the `LinearRegression` pointed to by `ptr`.
///
/// Calling this function with an invalid pointer, a pointer to stack-allocated memory, or a pointer
/// that has already been freed may lead to undefined behavior, including memory corruption or crashes.
#[no_mangle]
pub unsafe extern "C" fn free_linear_regression(ptr: *mut LinearRegression) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}
