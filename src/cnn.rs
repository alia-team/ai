use rand::Rng;
use rand_distr::{Distribution, Uniform};

pub enum Activation {
    Identity,
    ReLU,
    Sigmoid,
    Sign,
}

#[derive(Copy, Clone)]
pub enum Padding {
    Same,
    Valid,
}

pub fn max_pool_2d(
    input: &[Vec<f64>],
    pool_size: (usize, usize),
    stride: usize,
    padding: Padding,
) -> Vec<Vec<f64>> {
    let (input_height, input_width) = (input.len(), input[0].len());
    let (pool_height, pool_width) = pool_size;

    // Determine padding
    let (pad_h, pad_w) = match padding {
        Padding::Same => {
            let pad_h = ((input_height - 1) * stride + pool_height - input_height) / 2;
            let pad_w = ((input_width - 1) * stride + pool_width - input_width) / 2;
            (pad_h, pad_w)
        }
        Padding::Valid => (0, 0),
    };

    // Apply padding to input
    let padded_input = pad_input(input, pad_h, pad_w);

    // Output dimensions
    let (out_height, out_width) = match padding {
        Padding::Same => (input_height, input_width),
        Padding::Valid => (
            (input_height - pool_height + stride) / stride,
            (input_width - pool_width + stride) / stride,
        ),
    };

    // Initialize the output matrix
    let mut output = vec![vec![f64::MIN; out_width]; out_height];

    // Max pooling
    for (i, output_row) in output.iter_mut().enumerate().take(out_height) {
        for (j, output_val) in output_row.iter_mut().enumerate().take(out_width) {
            for k in 0..pool_height {
                for l in 0..pool_width {
                    let x = i * stride + k;
                    let y = j * stride + l;
                    if x < padded_input.len() && y < padded_input[0].len() {
                        *output_val = output_val.max(padded_input[x][y]);
                    }
                }
            }
        }
    }

    output
}

fn pad_input(input: &[Vec<f64>], pad_h: usize, pad_w: usize) -> Vec<Vec<f64>> {
    let (input_height, input_width) = (input.len(), input[0].len());
    let mut padded_input = vec![vec![0.0; input_width + 2 * pad_w]; input_height + 2 * pad_h];

    for i in 0..input_height {
        for j in 0..input_width {
            padded_input[i + pad_h][j + pad_w] = input[i][j];
        }
    }

    padded_input
}

fn initialize_kernels(
    filters: usize,
    kernel_size: (usize, usize),
    channels: usize,
) -> Vec<Vec<Vec<Vec<f64>>>> {
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(-0.05, 0.05); // Example range for random initialization
    let mut kernels = vec![vec![vec![vec![0.0; channels]; kernel_size.1]; kernel_size.0]; filters];

    for f in 0..filters {
        for i in 0..kernel_size.0 {
            for j in 0..kernel_size.1 {
                for c in 0..channels {
                    kernels[f][i][j][c] = dist.sample(&mut rng);
                }
            }
        }
    }
    kernels
}

pub fn conv_2d(
    input: &Vec<Vec<f64>>,
    strides: (usize, usize),
    kernel: Option<Vec<Vec<f64>>>,
    kernel_size: Option<(usize, usize)>,
    filters: usize,
    padding: Padding,
    activation: Activation,
) -> Vec<Vec<f64>> {
    // Determine kernel size and initialize if not provided
    let (kernel, kernel_size) = match kernel {
        Some(k) => (k.clone(), (k.len(), k[0].len())),
        None => {
            let size = kernel_size.expect("Kernel size must be provided if kernel is not given");
            let initialized_kernels = initialize_kernels(filters, size, 1); // Assuming single channel input for simplicity
            let kernel = initialized_kernels
                .into_iter()
                .map(|f| {
                    f.into_iter()
                        .map(|v| v.into_iter().map(|c| c[0]).collect())
                        .collect()
                })
                .next()
                .unwrap();
            (kernel, size)
        }
    };

    // Function to apply the specified activation
    fn apply_activation(value: f64, activation: &Activation) -> f64 {
        match activation {
            Activation::ReLU => value.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-value).exp()),
            Activation::Identity => value,
            Activation::Sign => {
                if value >= 0.0 {
                    1.0
                } else {
                    -1.0
                }
            }
        }
    }

    // Pad input based on the padding type
    let pad_input = |input: &Vec<Vec<f64>>, kernel_size: (usize, usize), padding: &Padding| -> Vec<Vec<f64>> {
        match padding {
            Padding::Valid => input.clone(),
            Padding::Same => {
                let pad_height = kernel_size.0 / 2;
                let pad_width = kernel_size.1 / 2;
                let mut padded = vec![vec![0.0; input[0].len() + 2 * pad_width]; input.len() + 2 * pad_height];
                for i in 0..input.len() {
                    for j in 0..input[0].len() {
                        padded[i + pad_height][j + pad_width] = input[i][j];
                    }
                }
                padded
            }
        }
    };

    // Pad the input if needed
    let padded_input = pad_input(input, kernel_size, &padding);

    // Calculate output dimensions
    let (output_height, output_width) = match padding {
        Padding::Valid => (
            (input.len() - kernel_size.0 + strides.0) / strides.0,
            (input[0].len() - kernel_size.1 + strides.1) / strides.1,
        ),
        Padding::Same => (
            (input.len() + strides.0 - 1) / strides.0,
            (input[0].len() + strides.1 - 1) / strides.1,
        ),
    };
    let mut output = vec![vec![0.0; output_width]; output_height];

    // Perform the convolution
    for i in 0..output_height {
        for j in 0..output_width {
            let mut sum = 0.0;
            for ki in 0..kernel_size.0 {
                for kj in 0..kernel_size.1 {
                    let input_value = padded_input[i * strides.0 + ki][j * strides.1 + kj];
                    let kernel_value = kernel[ki][kj];
                    sum += input_value * kernel_value;
                }
            }
            output[i][j] = apply_activation(sum, &activation);
        }
    }

    output
}
