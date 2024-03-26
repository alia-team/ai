use std::error::Error;
use std::fmt::{Display, Formatter};

pub struct Tensor<'a> {
    pub data: &'a [f32],
    pub shape: &'a [usize],
}

#[derive(Debug, PartialEq)]
pub enum TensorError {
    DimensionMismatch {
        expected: usize,
        found: usize,
    },
    IndexOutOfBounds {
        dimension: usize,
        index: usize,
        bound: usize,
    },
}

impl<'a> Tensor<'a> {
    pub fn new(data: &'a [f32], shape: &'a [usize]) -> Tensor<'a> {
        Tensor { data, shape }
    }

    pub fn flat_index(&self, shape: &[usize]) -> Result<usize, TensorError> {
        if shape.len() != self.shape.len() {
            return Err(TensorError::DimensionMismatch {
                expected: self.shape.len(),
                found: shape.len(),
            });
        }

        let mut flat_index = 0;
        let mut dimension_stride = 1;

        for (i, &dimension_index) in shape.iter().rev().enumerate() {
            let shape_index = self.shape.len() - 1 - i;

            if dimension_index >= self.shape[shape_index] {
                return Err(TensorError::IndexOutOfBounds {
                    dimension: shape_index,
                    index: dimension_index,
                    bound: self.shape[shape_index] - 1,
                });
            }
            flat_index += dimension_index * dimension_stride;
            dimension_stride *= self.shape[shape_index];
        }

        Ok(flat_index)
    }
}

impl Display for TensorError {
    fn fmt(&self, formatter: &mut Formatter) -> std::fmt::Result {
        match *self {
            TensorError::DimensionMismatch { expected, found } => {
                write!(
                    formatter,
                    "Dimension mismatch: expected {}, found {}",
                    expected, found
                )
            }
            TensorError::IndexOutOfBounds {
                dimension,
                index,
                bound,
            } => {
                write!(
                    formatter,
                    "Index out of bounds: dimension {}, index {}, exceeds bound {}",
                    dimension, index, bound
                )
            }
        }
    }
}

impl Error for TensorError {}
