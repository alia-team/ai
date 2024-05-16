use rand::Rng;

// Perceptron for an input with two values + 1 for the bias
pub struct Perceptron {
    weights: [f64; 3], // 2 + 1 for the bias
}

impl Perceptron {
    // Initialize the Perceptron with random weights
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let weights = [
            rng.gen::<f64>() * 2.0 - 1.0,
            rng.gen::<f64>() * 2.0 - 1.0,
            rng.gen::<f64>() * 2.0 - 1.0,
        ];
        // 2 + 1 for the bias
        Perceptron { weights }
    }

    // Train the Perceptron with the given data points and labels
    pub fn fit(&mut self, data_points: &[[f64; 2]], class_labels: &[f64], iterations: usize) {
        let mut rng = rand::thread_rng();
        for _ in 0..iterations {
            let random_index = rng.gen_range(0..data_points.len());
            let target_label = class_labels[random_index];
            let input = [
                1.0,
                data_points[random_index][0],
                data_points[random_index][1],
            ];

            // Calculate weighted sum
            let weighted_sum: f64 = self.weights.iter().zip(&input).map(|(w, x)| w * x).sum();

            // Activation function
            let predicted_label = if weighted_sum >= 0.0 { 1.0 } else { -1.0 };

            // Update weights
            for i in 0..self.weights.len() {
                self.weights[i] += 0.001 * (target_label - predicted_label) * input[i];
            }
        }
    }

    // Predict the class of a new data point
    pub fn predict(&self, new_point: &[f64; 2]) -> f64 {
        let input = [1.0, new_point[0], new_point[1]];
        let weighted_sum: f64 = self.weights.iter().zip(&input).map(|(w, x)| w * x).sum();
        if weighted_sum >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perceptron() {
        let mut rng = rand::thread_rng();
        let data_points: Vec<[f64; 2]> = (0..100)
            .map(|_| [rng.gen::<f64>() * 100.0, rng.gen::<f64>() * 100.0])
            .collect();

        let mut class_labels: Vec<f64> = Vec::new();
        for point in &data_points {
            if point[0] + 3.0 * point[1] - 78.0 * 3.0 >= 0.0 {
                class_labels.push(1.0);
            } else {
                class_labels.push(-1.0);
            }
        }

        let mut perceptron = Perceptron::new();
        perceptron.fit(&data_points, &class_labels, 1000);

        let new_point = [0.0, 0.0];
        let predicted_class = perceptron.predict(&new_point);
        assert!(predicted_class == -1.0);

        let new_point = [50.0, 50.0];
        let predicted_class = perceptron.predict(&new_point);
        assert!(predicted_class == 1.0);

        let new_point = [0.0, 100.0];
        let predicted_class = perceptron.predict(&new_point);
        assert!(predicted_class == 1.0);

        let new_point = [100.0, 100.0];
        let predicted_class = perceptron.predict(&new_point);
        assert!(predicted_class == 1.0);
    }
}
