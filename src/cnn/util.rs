use ndarray::{Array1, Array2};

/// Computes the outer product of two vectors
pub fn outer(x: Array1<f32>, y: Array1<f32>) -> Array2<f32> {
    let mut result: Array2<f32> = Array2::<f32>::zeros((x.len(), y.len()));
    for i in 0..x.len() {
        for j in 0..y.len() {
            result[[i, j]] = x[i] * y[j];
        }
    }
    result
}
