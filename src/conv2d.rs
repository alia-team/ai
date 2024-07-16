use ndarray::{s, Array, Array1, Array3, ArrayD, ArrayViewMut, Axis, IxDyn};
use ndarray_rand::RandomExt;
use rand_distr::Normal;
use rayon::prelude::*;
use std::ops::AddAssign;
use std::sync::Mutex;

pub struct Conv2D {
    pub filters: usize,
    pub kernel: ArrayD<f32>,
    pub bias: Array1<f32>,
}

impl Conv2D {
    pub fn new(input_channels: usize, filters: usize, kernel_size: usize) -> Self {
        // He weights initialization
        let std_dev = (2.0 / (input_channels * kernel_size * kernel_size) as f32).sqrt();
        let kernel = Array::random(
            IxDyn(&[filters, input_channels, kernel_size, kernel_size]),
            Normal::new(0.0, std_dev).unwrap(),
        );
        let bias = Array::zeros(filters);
        Conv2D {
            filters,
            kernel,
            bias,
        }
    }

    pub fn forward(&self, input: &Array3<f32>) -> Array3<f32> {
        let (height, width, _) = input.dim();
        let kernel_size = self.kernel.shape()[2];
        let mut output = Array3::<f32>::zeros((height, width, self.filters));

        for f in 0..self.filters {
            for (idx, window) in input
                .windows((kernel_size, kernel_size, input.shape()[2]))
                .into_iter()
                .enumerate()
            {
                let h = idx / width;
                let w = idx % width;
                let sum: f32 = window
                    .iter()
                    .zip(self.kernel.index_axis(Axis(0), f).iter())
                    .map(|(&x, &k)| x * k)
                    .sum();
                output[[h, w, f]] = (sum + self.bias[f]).max(0.0);
            }
        }

        output
    }

    pub fn backward(
        &mut self,
        input: &Array3<f32>,
        grad_output: &Array3<f32>,
    ) -> (Array3<f32>, ArrayD<f32>, Array1<f32>) {
        let (height, width, _) = input.dim();
        let kernel_shape = self.kernel.shape();
        let (kernel_height, kernel_width) = (kernel_shape[2], kernel_shape[3]);
        let grad_input = Mutex::new(Array3::<f32>::zeros(input.dim()));
        let grad_kernel = Mutex::new(ArrayD::<f32>::zeros(self.kernel.dim()));
        let grad_bias = Mutex::new(Array1::<f32>::zeros(self.filters));

        // Parallelize the outer loop
        (0..self.filters).into_par_iter().for_each(|f| {
            let mut local_grad_kernel = ArrayD::<f32>::zeros(self.kernel.dim());
            let mut local_grad_bias = 0.0;
            let mut local_grad_input = Array3::<f32>::zeros(input.dim());

            for h in 0..height {
                for w in 0..width {
                    let grad_output_val = grad_output[[h, w, f]];
                    local_grad_bias += grad_output_val;

                    for c in 0..self.kernel.shape()[1] {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let h_idx = (h as i32 + kh as i32 - 1).max(0).min(height as i32 - 1)
                                    as usize;
                                let w_idx = (w as i32 + kw as i32 - 1).max(0).min(width as i32 - 1)
                                    as usize;

                                local_grad_kernel[[f, c, kh, kw]] +=
                                    input[[h_idx, w_idx, c]] * grad_output_val;
                                local_grad_input[[h_idx, w_idx, c]] +=
                                    self.kernel[[f, c, kh, kw]] * grad_output_val;
                            }
                        }
                    }
                }
            }

            // Update the shared grad_kernel, grad_bias, and grad_input
            let mut shared_grad_kernel = grad_kernel.lock().unwrap();
            shared_grad_kernel
                .slice_mut(s![f, .., .., ..])
                .add_assign(&local_grad_kernel.slice(s![f, .., .., ..]));

            let mut shared_grad_bias = grad_bias.lock().unwrap();
            shared_grad_bias[f] += local_grad_bias;

            let mut shared_grad_input = grad_input.lock().unwrap();
            shared_grad_input.add_assign(&local_grad_input);
        });

        // Gradient norm clipping
        let mut grad_kernel = grad_kernel.into_inner().unwrap();
        let mut grad_bias = grad_bias.into_inner().unwrap();
        clip_gradient_norm(&mut grad_kernel.view_mut(), 5.0, 5.0);
        clip_gradient_norm(&mut grad_bias.view_mut(), 5.0, 5.0);

        // Update weights and biases
        self.kernel -= &grad_kernel;
        self.bias -= &grad_bias;

        (grad_input.into_inner().unwrap(), grad_kernel, grad_bias)
    }

    pub fn from_weights(kernel: Vec<f32>, kernel_shape: Vec<usize>, bias: Vec<f32>) -> Self {
        Conv2D {
            filters: kernel_shape[0],
            kernel: Array::from_shape_vec(IxDyn(&kernel_shape), kernel).unwrap(),
            bias: Array1::from(bias),
        }
    }
}

fn clip_gradient_norm<D>(grad: &mut ArrayViewMut<f32, D>, max_norm: f32, max_value: f32)
where
    D: ndarray::Dimension,
{
    // Value clipping
    grad.mapv_inplace(|x| x.clamp(-max_value, max_value));

    // Norm clipping
    let norm = grad.mapv(|x| x * x).sum().sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        grad.mapv_inplace(|x| x * scale);
    }
}
