use ndarray::{s, Array3};

pub struct MaxPool2D {
    pool_size: usize,
}

impl MaxPool2D {
    pub fn new(pool_size: usize) -> Self {
        MaxPool2D { pool_size }
    }

    pub fn forward(&self, input: &Array3<f32>) -> Array3<f32> {
        let (height, width, channels) = input.dim();
        let new_height = height / self.pool_size;
        let new_width = width / self.pool_size;

        let mut output = Array3::zeros((new_height, new_width, channels));

        for c in 0..channels {
            for h in 0..new_height {
                for w in 0..new_width {
                    let pool_slice = input.slice(s![
                        h * self.pool_size..(h + 1) * self.pool_size,
                        w * self.pool_size..(w + 1) * self.pool_size,
                        c
                    ]);
                    output[[h, w, c]] =
                        pool_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                }
            }
        }

        output
    }

    pub fn backward(&self, input: &Array3<f32>, grad_output: &Array3<f32>) -> Array3<f32> {
        let (height, width, channels) = input.dim();
        let (out_height, out_width, _) = grad_output.dim();

        let mut grad_input = Array3::<f32>::zeros((height, width, channels));

        for h in 0..out_height {
            for w in 0..out_width {
                for c in 0..channels {
                    let mut max_val = f32::MIN;
                    let mut max_idx = (0, 0);
                    for ph in 0..self.pool_size {
                        for pw in 0..self.pool_size {
                            let val = input[[h * self.pool_size + ph, w * self.pool_size + pw, c]];
                            if val > max_val {
                                max_val = val;
                                max_idx = (ph, pw);
                            }
                        }
                    }
                    grad_input[[
                        h * self.pool_size + max_idx.0,
                        w * self.pool_size + max_idx.1,
                        c,
                    ]] = grad_output[[h, w, c]];
                }
            }
        }
        grad_input
    }
}
