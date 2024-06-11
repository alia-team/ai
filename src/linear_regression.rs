// Define the LinearRegression struct
pub struct LinearRegression {
    pub weights: Vec<f64>,
}

// Function to invert a matrix
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

// Implement LinearRegression struct
impl LinearRegression {
    pub fn new(input_size: usize) -> LinearRegression {
        LinearRegression {
            weights: vec![0.0; input_size + 1],
        }
    }

    pub fn fit(&mut self, data_points: &Vec<Vec<f64>>, target_values: &Vec<f64>) {
        let mut augmented_data_points: Vec<Vec<f64>> = Vec::new();
        for i in 0..data_points.len() {
            let mut augmented_point = data_points[i].clone();
            augmented_point.insert(0, 1.0); // Add intercept term
            augmented_data_points.push(augmented_point);
        }

        let x = augmented_data_points;
        let y = target_values.clone();

        // Calculate X^T * X
        let mut x_t_x = vec![vec![0.0; x[0].len()]; x[0].len()];
        for i in 0..x.len() {
            for j in 0..x[0].len() {
                for k in 0..x[0].len() {
                    x_t_x[j][k] += x[i][j] * x[i][k];
                }
            }
        }

        // Calculate (X^T * X)^(-1)
        let x_t_x_inv = invert_matrix(&x_t_x);

        // Calculate X^T * y
        let mut x_t_y = vec![0.0; x[0].len()];
        for i in 0..x.len() {
            for j in 0..x[0].len() {
                x_t_y[j] += x[i][j] * y[i];
            }
        }

        // Calculate weights = (X^T * X)^(-1) * (X^T * y)
        let mut weights = vec![0.0; x[0].len()];
        for i in 0..x[0].len() {
            for j in 0..x[0].len() {
                weights[i] += x_t_x_inv[i][j] * x_t_y[j];
            }
        }

        self.weights = weights;
    }

    pub fn predict(&self, input: &Vec<f64>) -> f64 {
        let mut weighted_sum: f64 = self.weights[0]; // Intercept
        for i in 0..input.len() {
            weighted_sum += self.weights[i + 1] * input[i];
        }
        weighted_sum
    }
}

// Expose functions for FFI
#[no_mangle]
pub extern "C" fn new_linear_regression(input_size: usize) -> *mut LinearRegression {
    Box::into_raw(Box::new(LinearRegression::new(input_size)))
}

#[no_mangle]
pub extern "C" fn fit_linear_regression(ptr: *mut LinearRegression, data_points: *const *const f64, target_values: *const f64, n_points: usize, n_features: usize) {
    let linear_regression = unsafe { &mut *ptr };
    let data_points: Vec<Vec<f64>> = unsafe {
        (0..n_points).map(|i| {
            std::slice::from_raw_parts(*data_points.add(i), n_features).to_vec()
        }).collect()
    };
    let target_values = unsafe { std::slice::from_raw_parts(target_values, n_points) };

    linear_regression.fit(&data_points, &target_values.to_vec());
}

#[no_mangle]
pub extern "C" fn predict_linear_regression(ptr: *mut LinearRegression, new_point: *const f64, n_features: usize) -> f64 {
    let linear_regression = unsafe { &mut *ptr };
    let new_point = unsafe { std::slice::from_raw_parts(new_point, n_features) };
    linear_regression.predict(&new_point.to_vec())
}

#[no_mangle]
pub extern "C" fn free_linear_regression(ptr: *mut LinearRegression) {
    if !ptr.is_null() {
        unsafe {
            Box::from_raw(ptr);
        }
    }
}
