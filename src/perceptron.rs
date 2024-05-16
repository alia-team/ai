use rand::Rng;
use std::os::raw::c_double;

#[repr(C)]
pub struct Perceptron {
    weights: Vec<c_double>, // Using c_double for FFI compatibility
}

impl Perceptron {
    pub fn new(input_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights: Vec<c_double> = (0..=input_size)
            .map(|_| rng.gen::<c_double>() * 2.0 - 1.0)
            .collect();
        Perceptron { weights }
    }

    pub fn fit(&mut self, data_points: &[Vec<c_double>], class_labels: &[c_double], iterations: usize) {
        let mut rng = rand::thread_rng();
        for _ in 0..iterations {
            let random_index = rng.gen_range(0..data_points.len());
            let target_label = class_labels[random_index];
            let mut input = vec![1.0]; // Bias term
            input.extend(&data_points[random_index]);

            // Calculate weighted sum
            let weighted_sum: c_double = self.weights.iter().zip(&input).map(|(w, x)| w * x).sum();

            // Activation function
            let predicted_label = if weighted_sum >= 0.0 { 1.0 } else { -1.0 };

            // Update weights
            (0..self.weights.len()).for_each(|i| {
                self.weights[i] += 0.001 * (target_label - predicted_label) * input[i];
            });
        }
    }

    pub fn predict(&self, new_point: &[c_double]) -> c_double {
        let mut input = vec![1.0]; // Bias term
        input.extend(new_point);
        let weighted_sum: c_double = self.weights.iter().zip(&input).map(|(w, x)| w * x).sum();
        if weighted_sum >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }
}

#[no_mangle]
pub extern "C" fn create_perceptron(input_size: usize) -> *mut Perceptron {
    Box::into_raw(Box::new(Perceptron::new(input_size)))
}

#[no_mangle]
pub extern "C" fn fit_perceptron(ptr: *mut Perceptron, data_points: *const *const c_double, class_labels: *const c_double, n_points: usize, n_features: usize, iterations: usize) {
    let perceptron = unsafe { &mut *ptr };
    let data_points: Vec<Vec<c_double>> = unsafe {
        (0..n_points).map(|i| {
            std::slice::from_raw_parts(*data_points.add(i), n_features).to_vec()
        }).collect()
    };
    let class_labels = unsafe { std::slice::from_raw_parts(class_labels, n_points) };
    perceptron.fit(&data_points, class_labels, iterations);
}

#[no_mangle]
pub extern "C" fn predict_perceptron(ptr: *mut Perceptron, new_point: *const c_double, n_features: usize) -> c_double {
    let perceptron = unsafe { &mut *ptr };
    let new_point = unsafe { std::slice::from_raw_parts(new_point, n_features) };
    perceptron.predict(new_point)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perceptron() {
        let mut rng = rand::thread_rng();
        let data_points: Vec<Vec<f64>> = (0..100)
            .map(|_| vec![rng.gen::<f64>() * 100.0, rng.gen::<f64>() * 100.0])
            .collect();

        let mut class_labels: Vec<f64> = Vec::new();
        for point in &data_points {
            if point[0] + 3.0 * point[1] - 78.0 * 3.0 >= 0.0 {
                class_labels.push(1.0);
            } else {
                class_labels.push(-1.0);
            }
        }

        let mut perceptron = Perceptron::new(2);
        perceptron.fit(&data_points, &class_labels, 1000);

        let new_point = vec![0.0, 0.0];
        let predicted_class = perceptron.predict(&new_point);
        assert!(predicted_class == -1.0 || predicted_class == 1.0);

        let new_point = vec![50.0, 50.0];
        let predicted_class = perceptron.predict(&new_point);
        assert!(predicted_class == -1.0 || predicted_class == 1.0);

        let new_point = vec![0.0, 100.0];
        let predicted_class = perceptron.predict(&new_point);
        assert!(predicted_class == -1.0 || predicted_class == 1.0);

        let new_point = vec![100.0, 100.0];
        let predicted_class = perceptron.predict(&new_point);
        assert!(predicted_class == -1.0 || predicted_class == 1.0);
    }
}
