pub fn max_pool_2d(
    input: &[Vec<f64>],
    pool_size: &[usize; 2],
    stride: usize,
    padding: &str,
) -> Vec<Vec<f64>> {
    let (input_height, input_width) = (input.len(), input[0].len());
    let (pool_height, pool_width) = (pool_size[0], pool_size[1]);

    // Determine padding
    let (pad_h, pad_w) = match padding {
        "same" => {
            let pad_h = ((input_height - 1) * stride + pool_height - input_height) / 2;
            let pad_w = ((input_width - 1) * stride + pool_width - input_width) / 2;
            (pad_h, pad_w)
        },
        _ => (0, 0),
    };

    // Apply padding to input
    let padded_input = pad_input(input, pad_h, pad_w);

    // Output dimensions
    let (out_height, out_width) = match padding {
        "same" => (input_height, input_width),
        _ => (
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
